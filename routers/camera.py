"""
In-clinic camera router. /video_feed serves the MJPEG stream from the webcam
attached to THIS machine (services/mjpeg_camera.py -> cv2.VideoCapture(0)),
which is the only capture path in this build: the server runs on the
clinic desktop next to the camera.
"""
import json
from typing import Dict, Any
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse

from services import mjpeg_camera, game_state
from services.auth.deps import get_optional_user, require_clinical
from services.auth.roles import CLINICAL_ROLES
from services.user_context import UserContext, get_context

router = APIRouter()


def user_ctx(user=Depends(require_clinical)) -> UserContext:
    """This request's own live-session state (metrics, game engine, pose data).
    Keyed by the logged-in user, so doctors at different clinics never share
    counters, exercise settings or game state. require_clinical is cached per
    request by FastAPI, so the login/DB check still runs only once."""
    return get_context(user.id)

# WebSocket routes cannot use the HTTP-only `Depends(require_clinical)` that
# app.py attaches to `router`, so they live on their own router and do the
# same cookie/role check by hand (see camera_ws below).
ws_router = APIRouter()

_MAX_WS_FRAME_BYTES = 2 * 1024 * 1024  # a 640x480 JPEG is ~30-80 KB; 2 MB is a generous cap


@ws_router.websocket("/ws/camera")
async def camera_ws(websocket: WebSocket):
    """Browser-camera pipeline (works when the server has NO camera, i.e.
    any real deployment). The browser opens the patient's webcam with
    getUserMedia, sends each frame here as a binary JPEG, and gets the
    annotated JPEG (skeleton drawn) back. Joint angles / reps / game state
    are still read by the page from /api/pose_data and /api/game/status.

    Protocol: client sends binary JPEG frames (at most 2 unanswered at a time, so
    latency never builds up). Per frame the server sends one JSON text message:
    {"type": "game", "status": {...}} in game mode, {"type": "pose", "data": {...}}
    in exercise mode, then the reply itself. With ?overlay=1 the reply is a small
    {"type": "draw", "l": lines, "d": dots} JSON (the browser draws the skeleton over
    its own live video); without it the reply is the annotated JPEG as before.
    """
    # 1) Same login check as the HTTP API (cookie -> approved clinical user).
    user = get_optional_user(websocket)  # only reads .cookies, works on a WebSocket
    if user is None or user.role not in CLINICAL_ROLES:
        await websocket.close(code=4401)
        return

    # 2) Cookies are sent on cross-site WebSocket handshakes too, so refuse
    #    pages served from a different origin.
    origin = websocket.headers.get("origin")
    if origin and urlparse(origin).netloc != websocket.headers.get("host"):
        await websocket.close(code=4403)
        return

    overlay = websocket.query_params.get("overlay") == "1"
    ctx = get_context(user.id)   # this doctor's own state
    await websocket.accept()
    ctx.remote_connected()
    try:
        while True:
            data = await websocket.receive_bytes()
            if len(data) > _MAX_WS_FRAME_BYTES:
                await websocket.send_text('{"error":"frame too large"}')
                continue
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_text('{"error":"bad frame"}')
                continue
            try:
                # MediaPipe is CPU-heavy: keep it off the asyncio event loop.
                jpg = await run_in_threadpool(mjpeg_camera.process_frame, ctx, frame, overlay)
            except Exception as e:  # never kill the socket over one bad frame
                print(f"WS camera: processing failed: {e}")
                await websocket.send_text('{"error":"processing failed"}')
                continue
            if jpg is None:
                await websocket.send_text('{"error":"encode failed"}')
                continue
            # Game mode: push the live game status over this same socket, right before the
            # frame. The page used to poll /api/game/status over HTTP (a separate request +
            # DB lookup for every poll), which is what made the game lag on the deployed
            # site. Status and frame now travel together, so they are always in step.
            if ctx.game.get_mode() == "game":
                try:
                    status = ctx.game.get_game_status()
                    if status.get("active_game") is not None:
                        await websocket.send_text(
                            json.dumps({"type": "game", "status": status}, default=float))
                except Exception as e:      # never lose the frame over a status problem
                    print(f"WS camera: game status failed: {e}")
            else:
                # Exercise mode: same idea - push the joint angles / reps / scores with the
                # frame instead of the page polling /api/pose_data (an HTTP request + DB
                # lookup every 300 ms, which competed with MediaPipe for the server).
                try:
                    await websocket.send_text(
                        json.dumps({"type": "pose", "data": ctx.pose_data}, default=float))
                except Exception as e:
                    print(f"WS camera: pose data failed: {e}")
            if overlay:
                await websocket.send_text(json.dumps({"type": "draw", "l": jpg["l"], "d": jpg["d"]}))
            else:
                await websocket.send_bytes(jpg)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS camera: closed with error: {e}")
    finally:
        ctx.remote_disconnected()


@router.get("/video_feed")
async def video_feed(ctx: UserContext = Depends(user_ctx)):
    """Server-attached camera feed (local/in-clinic mode only -- the deployed
    site uses /ws/camera instead). <img src='/video_feed'>"""
    return StreamingResponse(
        mjpeg_camera.gen_frames(ctx),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/api/pose_data")
async def api_pose_data(ctx: UserContext = Depends(user_ctx)):
    """This user's latest joint angles + detection flag, updated every frame."""
    return JSONResponse(ctx.pose_data)


@router.post("/api/camera/stop")
async def api_camera_stop(ctx: UserContext = Depends(user_ctx)):
    """Stop / page unload: clear this user's live flags (and release the
    server-attached camera, if that mode is in use)."""
    mjpeg_camera.stop_camera(ctx)
    return JSONResponse({"success": True, "message": "Camera stopped"})


@router.post("/api/exercise_type")
async def set_exercise_type(payload: Dict[str, Any], ctx: UserContext = Depends(user_ctx)):
    """Frontend calls this whenever the exercise dropdown, target ROM, or
    side (left/right/both) selector changes, so the MJPEG stream draws the
    right joints/side and rep-counts against the right threshold."""
    ex   = payload.get("exercise_type")
    rom  = payload.get("target_rom")
    side = payload.get("side")
    # The server remembers each user's Exercise/Game mode between page loads
    # (context lives 30 min), but the page always starts in "Exercise". The page
    # sends its own mode with every exercise sync so a stale "game" mode on the
    # server can never make the camera draw/track the full body for an exercise.
    mode = payload.get("mode")
    if mode in ("exercise", "game"):
        ctx.game.set_mode(mode)
    ctx.metrics.set_exercise_state(exercise_type=ex, target_rom=rom, side=side)
    current_ex, current_rom, current_side = ctx.metrics.get_exercise_state()
    return JSONResponse({
        "success": True,
        "exercise_type": current_ex,
        "target_rom": current_rom,
        "side": current_side,
    })


def _reset_session_state(ctx: UserContext):
    """Runs under ctx.process_lock so it can never interleave with a frame that
    is still being processed. Without this, a slow frame (production: MediaPipe
    on a busier/slower server + network latency) that started BEFORE the reset
    would finish AFTER it and write the previous session's rep count back into
    pose_data, and the page would see "target reps reached" and end the new
    session the moment it started."""
    with ctx.process_lock:
        ctx.metrics.reset_state()
        # Session counter: every pose_data message carries it, so the page can tell a
        # message produced before this reset (still travelling over the socket) from one
        # produced after it, and never lets an old session's rep count end the new one.
        ctx.pose_data["seq"] = ctx.pose_data.get("seq", 0) + 1
        # /api/pose_data serves ctx.pose_data, which is only rewritten when the
        # next frame is processed. Clear the per-session numbers here too,
        # otherwise the old rep count is served until that frame arrives.
        ctx.pose_data["reps"]          = 0
        ctx.pose_data["stability"]     = 100.0
        ctx.pose_data["smoothness"]    = 100.0
        ctx.pose_data["balance"]       = 100.0
        ctx.pose_data["fatigue"]       = 0.0
        ctx.pose_data["primary_angle"] = None


@router.post("/api/session/reset")
async def api_session_reset(ctx: UserContext = Depends(user_ctx)):
    """Call this right before a session starts so rep count + stability
    buffer don't carry over stale data from a previous session/patient."""
    await run_in_threadpool(_reset_session_state, ctx)   # lock may wait for an in-flight frame
    return JSONResponse({"success": True, "seq": ctx.pose_data.get("seq", 0)})


@router.get("/api/camera/status")
async def api_camera_status(ctx: UserContext = Depends(user_ctx)):
    """Quick status check - is THIS user's camera socket open."""
    return JSONResponse({"active": ctx.remote_clients > 0})


# ── Game mode ────────────────────────────────────────────────────────────
# Mirrors the exercise-mode routes above (set_exercise_type / session/reset
# / pose_data), but for the session page's "Game" section. The camera loop
# itself (services/mjpeg_camera.py) already dispatches frames to whichever
# game is selected via services/game_state.py — these routes just let the
# session page drive that state over HTTP.

@router.get("/api/games")
async def api_list_games():
    """[{id, label, description}, ...] to populate the session page's game
    dropdown (game/registry.py — add a game there and it shows up here with
    no other change)."""
    return JSONResponse({"games": game_state.get_available_games()})


@router.post("/api/game/select")
async def api_game_select(payload: Dict[str, Any], ctx: UserContext = Depends(user_ctx)):
    """Frontend calls this when the therapist picks a game from the
    dropdown. Switches the session page into game mode and creates a fresh
    engine instance for game_id — any target/timer set for a previous game
    is cleared, so /api/game/target must be called again before play."""
    game_id = payload.get("game_id")
    if not ctx.game.select_game(game_id):
        raise HTTPException(404, f"Unknown game: {game_id!r}")
    ctx.game.set_mode("game")
    return JSONResponse({
        "success": True,
        "active_game": ctx.game.get_active_game_id(),
    })


@router.post("/api/session/mode")
async def api_session_mode(payload: Dict[str, Any], ctx: UserContext = Depends(user_ctx)):
    """Switch the session page between "exercise" and "game" sections
    without changing which game/exercise is selected (e.g. the therapist
    flips back to Exercise mode after a game session)."""
    mode = payload.get("mode")
    if not ctx.game.set_mode(mode):
        raise HTTPException(400, f"Invalid mode: {mode!r} (expected 'exercise' or 'game')")
    return JSONResponse({"success": True, "mode": mode})


@router.post("/api/game/target")
async def api_game_target(payload: Dict[str, Any], ctx: UserContext = Depends(user_ctx)):
    """Set this game session's finish condition before calibration starts:
    target_steps and/or duration_seconds (either may be omitted/None —
    whichever condition is set fires first; both unset = manual stop
    only)."""
    ctx.game.set_target(
        target_steps=payload.get("target_steps"),
        duration_seconds=payload.get("duration_seconds"),
    )
    return JSONResponse({"success": True})


@router.post("/api/game/calibrate")
async def api_game_calibrate(ctx: UserContext = Depends(user_ctx)):
    """Reset any stale step/lane/crouch state from a previous session and
    start measuring the neutral standing position now (mirrors
    /api/session/reset + the exercise flow's calibration screen)."""
    ctx.game.reset_game()
    ctx.game.request_calibration()
    return JSONResponse({"success": True})


@router.get("/api/game/status")
async def api_game_status(ctx: UserContext = Depends(user_ctx)):
    """Live engine status + target/timer/finished state — the session page
    polls this during play to drive the HUD and detect auto-finish."""
    return JSONResponse(ctx.game.get_game_status())


@router.post("/api/game/stop")
async def api_game_stop(ctx: UserContext = Depends(user_ctx)):
    """Therapist/patient ends the game session early (before target/timer
    is reached). Does not release the camera — that's /api/camera/stop,
    called separately by the frontend same as the exercise flow."""
    ctx.game.finish_game(reason="manual")
    return JSONResponse({"success": True})


@router.get("/api/game/summary")
async def api_game_summary(ctx: UserContext = Depends(user_ctx)):
    """Server-tracked totals (steps, events, target/timer info) for the
    session page to fold into its /api/sessions save payload once the game
    session ends."""
    return JSONResponse(ctx.game.get_game_summary())
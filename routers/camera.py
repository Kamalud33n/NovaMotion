"""
In-clinic camera router. /video_feed serves the MJPEG stream from the webcam
attached to THIS machine (services/mjpeg_camera.py -> cv2.VideoCapture(0)),
which is the only capture path in this build: the server runs on the
clinic desktop next to the camera.
"""
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from services import mjpeg_camera, metrics, game_state

router = APIRouter()


@router.get("/video_feed")
async def video_feed():
    """Primary live camera feed with pose skeleton drawn server-side. <img src='/video_feed'>"""
    return StreamingResponse(
        mjpeg_camera.gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/api/pose_data")
async def api_pose_data():
    """Latest joint angles + detection flag, updated every frame by gen_frames()."""
    return JSONResponse(mjpeg_camera.latest_pose_data)


@router.post("/api/camera/stop")
async def api_camera_stop():
    """Explicitly release the camera device (called on Stop / page unload)."""
    mjpeg_camera.stop_camera()
    return JSONResponse({"success": True, "message": "Camera stopped"})


@router.post("/api/exercise_type")
async def set_exercise_type(payload: Dict[str, Any]):
    """Frontend calls this whenever the exercise dropdown, target ROM, or
    side (left/right/both) selector changes, so the MJPEG stream draws the
    right joints/side and rep-counts against the right threshold."""
    ex   = payload.get("exercise_type")
    rom  = payload.get("target_rom")
    side = payload.get("side")
    metrics.set_exercise_state(exercise_type=ex, target_rom=rom, side=side)
    current_ex, current_rom, current_side = metrics.get_exercise_state()
    return JSONResponse({
        "success": True,
        "exercise_type": current_ex,
        "target_rom": current_rom,
        "side": current_side,
    })


@router.post("/api/session/reset")
async def api_session_reset():
    """Call this right before a session starts so rep count + stability
    buffer don't carry over stale data from a previous session/patient."""
    metrics.reset_state()
    return JSONResponse({"success": True})


@router.get("/api/camera/status")
async def api_camera_status():
    """Quick status check - useful for frontend polling / debugging."""
    return JSONResponse({"active": mjpeg_camera.is_active()})


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
async def api_game_select(payload: Dict[str, Any]):
    """Frontend calls this when the therapist picks a game from the
    dropdown. Switches the session page into game mode and creates a fresh
    engine instance for game_id — any target/timer set for a previous game
    is cleared, so /api/game/target must be called again before play."""
    game_id = payload.get("game_id")
    if not game_state.select_game(game_id):
        raise HTTPException(404, f"Unknown game: {game_id!r}")
    game_state.set_mode("game")
    return JSONResponse({
        "success": True,
        "active_game": game_state.get_active_game_id(),
    })


@router.post("/api/session/mode")
async def api_session_mode(payload: Dict[str, Any]):
    """Switch the session page between "exercise" and "game" sections
    without changing which game/exercise is selected (e.g. the therapist
    flips back to Exercise mode after a game session)."""
    mode = payload.get("mode")
    if not game_state.set_mode(mode):
        raise HTTPException(400, f"Invalid mode: {mode!r} (expected 'exercise' or 'game')")
    return JSONResponse({"success": True, "mode": mode})


@router.post("/api/game/target")
async def api_game_target(payload: Dict[str, Any]):
    """Set this game session's finish condition before calibration starts:
    target_steps and/or duration_seconds (either may be omitted/None —
    whichever condition is set fires first; both unset = manual stop
    only)."""
    game_state.set_target(
        target_steps=payload.get("target_steps"),
        duration_seconds=payload.get("duration_seconds"),
    )
    return JSONResponse({"success": True})


@router.post("/api/game/calibrate")
async def api_game_calibrate():
    """Reset any stale step/lane/crouch state from a previous session and
    start measuring the neutral standing position now (mirrors
    /api/session/reset + the exercise flow's calibration screen)."""
    game_state.reset_game()
    game_state.request_calibration()
    return JSONResponse({"success": True})


@router.get("/api/game/status")
async def api_game_status():
    """Live engine status + target/timer/finished state — the session page
    polls this during play to drive the HUD and detect auto-finish."""
    return JSONResponse(game_state.get_game_status())


@router.post("/api/game/stop")
async def api_game_stop():
    """Therapist/patient ends the game session early (before target/timer
    is reached). Does not release the camera — that's /api/camera/stop,
    called separately by the frontend same as the exercise flow."""
    game_state.finish_game(reason="manual")
    return JSONResponse({"success": True})


@router.get("/api/game/summary")
async def api_game_summary():
    """Server-tracked totals (steps, events, target/timer info) for the
    session page to fold into its /api/sessions save payload once the game
    session ends."""
    return JSONResponse(game_state.get_game_summary())
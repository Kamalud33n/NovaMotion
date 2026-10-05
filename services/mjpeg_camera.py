"""
Live camera pipeline for the session page (<img src="/video_feed">), the
only capture path in this in-clinic build. The browser <img> tag decodes the
multipart MJPEG stream natively, no JS frame-decoding / canvas loop needed.
Runs as a sync generator (FastAPI runs sync generators in a thread pool), so
it does not block the asyncio event loop.

It opens cv2.VideoCapture(index) on the machine physically running this
server (the doctor's in-clinic desktop). By default it auto-detects and
prefers a connected USB webcam over the built-in laptop camera (see
_detect_camera_index() below); set the CAMERA_INDEX env var to force a
specific device instead.
"""
import os
import sys
import time
import threading
import datetime
from typing import Dict, Any, Optional

import cv2
import numpy as np

from config import mp_drawing, KEY_LANDMARKS, get_angle, POSE_CONNECTIONS, set_frame_size, get_angle_2d
from config import HAND_CONNECTIONS as _HAND_CONNECTIONS
from services import metrics
from services import user_context
from services.user_context import UserContext

# Bright, high-visibility drawing specs (default MediaPipe style is dim on a
# 1280x720 frame) — used only as a fallback; primary drawing is done manually
# per-exercise-type via _draw_filtered_skeleton() below.
_MJPEG_LANDMARK_SPEC = (
    mp_drawing.DrawingSpec(color=(0, 230, 120), thickness=1, circle_radius=3)
    if mp_drawing else None
)
_MJPEG_CONNECTION_SPEC = (
    mp_drawing.DrawingSpec(color=(255, 160, 0), thickness=1)
    if mp_drawing else None
)

# Exercise-type → active joints/lines filter
# Instead of drawing the full 33-point skeleton, only draw the connections
# relevant to whichever exercise is currently selected on the session page.
_LS, _RS = "left_shoulder", "right_shoulder"
_LE, _RE = "left_elbow", "right_elbow"
_LW, _RW = "left_wrist", "right_wrist"
_LH, _RH = "left_hip", "right_hip"
_LK, _RK = "left_knee", "right_knee"
_LA, _RA = "left_ankle", "right_ankle"

# NOTE (fix — pose not detecting): visibility threshold used to gate whether
# a joint counts as "visible" for detection + drawing. Lowered from 0.5 -> 0.3
# so partially-visible / poorly-lit joints still register instead of the
# frontend showing "No pose detected — move into frame" indefinitely.
_VISIBILITY_THRESHOLD = 0.3

# Camera device state 
_mjpeg_cap: Optional[cv2.VideoCapture] = None
_mjpeg_active = False
_mjpeg_lock = threading.Lock()

# Background frame reader (fixes growing camera delay)
_mjpeg_latest_frame = None
_mjpeg_frame_id = 0
_mjpeg_frame_lock = threading.Lock()
_mjpeg_reader_thread: Optional[threading.Thread] = None
_mjpeg_reader_running = False



def _rep_debug(ctx, exercise, side, target_rom, primary_angle):
    now = time.time()
    if now - ctx.dbg_last < 1.0:
        return
    ctx.dbg_last = now
    print(f"REPDBG ex={exercise} side={side} target={target_rom} rom={None if primary_angle is None else round(primary_angle, 1)} "
          f"{ctx.metrics.get_rep_debug(target_rom)}")


def _is_hand_exercise(exercise_type: str) -> bool:
    """Hand Grip / Finger Flexion needs MediaPipe Hands (21 finger
    landmarks) instead of — or alongside — Pose, since Pose's landmarks
    stop at the wrist. Kept separate from the old 'Hand Rehab' pose-only
    elbow↔wrist exercise, which stays on the regular Pose path."""
    ex = (exercise_type or "").lower()
    return "grip" in ex or "finger" in ex


def _draw_hand_skeleton(frame, hand_landmarks_list):
    """Draw all detected hands using MediaPipe's own connections — simpler
    than the filtered body-skeleton drawing since we want the full hand."""
    if mp_drawing is None or _HAND_CONNECTIONS is None:
        return
    for hand_landmarks in hand_landmarks_list:
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, _HAND_CONNECTIONS,
            landmark_drawing_spec=_MJPEG_LANDMARK_SPEC,
            connection_drawing_spec=_MJPEG_CONNECTION_SPEC,
        )


# MediaPipe Hands labels each hand "Left"/"Right" assuming a MIRRORED (selfie)
# image. gen_frames() flips the frame BEFORE inference, so the label is already
# the patient's real hand. If Left/Right ever comes out swapped on a particular
# camera setup, flip this one flag to True - nothing else needs to change.
_HAND_LABEL_SWAPPED = False


def _ui_side(side: str) -> str:
    """MetricsState.set_exercise_state() stores the side INVERTED (pose mirror fix).
    Undo that here to get what the doctor actually picked in the UI."""
    s = (side or "both").lower()
    if s == "left":
        return "right"
    if s == "right":
        return "left"
    return "both"


def _select_hands(hand_results, side: str):
    """Return only the hand landmark sets that match the side picked in the UI
    ("both" keeps every detected hand). Previously every detected hand was
    drawn and hand #0 was used for angles no matter which side was selected."""
    lms = list(getattr(hand_results, "multi_hand_landmarks", None) or [])
    if not lms:
        return []
    ui = _ui_side(side)
    if ui == "both":
        return lms
    wanted = ui
    if _HAND_LABEL_SWAPPED:
        wanted = "left" if ui == "right" else "right"
    handed = list(getattr(hand_results, "multi_handedness", None) or [])
    picked = []
    for i, lm in enumerate(lms):
        if i >= len(handed):
            continue
        try:
            label = handed[i].classification[0].label.lower()
        except Exception:
            continue
        if label == wanted:
            picked.append(lm)
    return picked


def _filter_connections_by_side(conns, side: str):
    """Keep only connections where both points belong to the requested side.
    "both" (or an unrecognized value) returns everything unchanged. If
    filtering would remove every connection (e.g. a torso-crossing pair with
    no same-side match), fall back to the unfiltered list rather than
    drawing nothing."""
    s = (side or "both").lower()
    if s not in ("left", "right"):
        return conns
    prefix = "left_" if s == "left" else "right_"
    filtered = [(a, b) for a, b in conns if a.startswith(prefix) and b.startswith(prefix)]
    return filtered if filtered else conns


def _get_active_connections(exercise_type: str, side: str = "both"):
    """Return the small set of (point_a, point_b) name-pairs to draw for the
    currently selected exercise, instead of the full body skeleton. `side`
    further restricts this to just the left or right limb when the patient
    is only exercising one side."""
    ex = (exercise_type or "").lower()

    if "elbow" in ex:
        conns = [(_LS, _LE), (_LE, _LW), (_RS, _RE), (_RE, _RW)]
    elif "hand" in ex:
        conns = [(_LE, _LW), (_RE, _RW)]
    elif "ankle" in ex:
        conns = [(_LK, _LA), (_RK, _RA)]
    elif "knee" in ex or "squat" in ex or "leg" in ex:
        conns = [(_LH, _LK), (_LK, _LA), (_RH, _RK), (_RK, _RA)]
    elif "hip" in ex:
        conns = [(_LS, _LH), (_RS, _RH), (_LH, _LK), (_RH, _RK)]
    elif "shoulder" in ex or "arm" in ex:
        conns = [(_LS, _LE), (_RS, _RE), (_LS, _LH), (_RS, _RH)]
    elif "balance" in ex:
        conns = [
            (_LS, _RS), (_LS, _LH), (_RS, _RH), (_LH, _RH),
            (_LS, _LE), (_LE, _LW), (_RS, _RE), (_RE, _RW),
            (_LH, _LK), (_LK, _LA), (_RH, _RK), (_RK, _RA),
        ]
    else:
        # Fallback: arms + legs, no torso clutter
        conns = [(_LS, _LE), (_LE, _LW), (_RS, _RE), (_RE, _RW),
                 (_LH, _LK), (_LK, _LA), (_RH, _RK), (_RK, _RA)]

    return _filter_connections_by_side(conns, side)


def _relevant_landmarks_visible(landmarks, exercise_type: str, side: str = "both") -> bool:
    active_conns = _get_active_connections(exercise_type, side)
    active_points = set()
    for a, b in active_conns:
        active_points.add(a)
        active_points.add(b)
    if not active_points:
        return True

    visible_count = 0
    for name in active_points:
        idx = KEY_LANDMARKS.get(name)
        if idx is not None and idx < len(landmarks) and landmarks[idx].visibility > _VISIBILITY_THRESHOLD:
            visible_count += 1

    required = max(1, (len(active_points) + 1) // 2)  # majority, rounded up
    return visible_count >= required


def _draw_filtered_skeleton(frame, landmarks, exercise_type: str, side: str = "both"):
    """Draw only the joints/lines relevant to the active exercise (and, when
    a single side is selected, only that limb), at a smaller, cleaner size
    than the MediaPipe default style."""
    h, w = frame.shape[:2]

    def _pt(name):
        idx = KEY_LANDMARKS.get(name)
        if idx is None or idx >= len(landmarks):
            return None
        lmk = landmarks[idx]
        if lmk.visibility < _VISIBILITY_THRESHOLD:
            return None
        return (int(lmk.x * w), int(lmk.y * h))

    active_conns = _get_active_connections(exercise_type, side)
    active_points = set()
    for a, b in active_conns:
        active_points.add(a)
        active_points.add(b)

    # Lines first (so dots sit on top), thin + clean
    for a, b in active_conns:
        pa, pb = _pt(a), _pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, (255, 160, 0), 1, cv2.LINE_AA)

    # Small joint dots
    for name in active_points:
        p = _pt(name)
        if p:
            cv2.circle(frame, p, 3, (0, 230, 120), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 3, (255, 255, 255), 1, cv2.LINE_AA)


def _r(v):
    return round(float(v), 3)


def _items_pose_full(items, landmarks):
    pts = {}
    for i, lmk in enumerate(landmarks):
        if lmk.visibility >= _VISIBILITY_THRESHOLD:
            pts[i] = (_r(lmk.x), _r(lmk.y))
    for a, b in (POSE_CONNECTIONS or ()):
        if a in pts and b in pts:
            items["l"].append([pts[a][0], pts[a][1], pts[b][0], pts[b][1]])
    items["d"].extend([list(p) for p in pts.values()])


def _items_pose_filtered(items, landmarks, exercise_type, side):
    def _pt(name):
        idx = KEY_LANDMARKS.get(name)
        if idx is None or idx >= len(landmarks):
            return None
        lmk = landmarks[idx]
        if lmk.visibility < _VISIBILITY_THRESHOLD:
            return None
        return (_r(lmk.x), _r(lmk.y))

    conns = _get_active_connections(exercise_type, side)
    names = set()
    for a, b in conns:
        names.add(a)
        names.add(b)
        pa, pb = _pt(a), _pt(b)
        if pa and pb:
            items["l"].append([pa[0], pa[1], pb[0], pb[1]])
    for n in names:
        p = _pt(n)
        if p:
            items["d"].append(list(p))


def _items_hands(items, hand_landmarks_list):
    for hl in hand_landmarks_list:
        lm = hl.landmark
        for a, b in (_HAND_CONNECTIONS or ()):
            items["l"].append([_r(lm[a].x), _r(lm[a].y), _r(lm[b].x), _r(lm[b].y)])
        for pt in lm:
            items["d"].append([_r(pt.x), _r(pt.y)])


def _finish(frame, overlay, items):
    if overlay:
        return items
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        print("MJPEG: JPEG encode failed, skipping this frame")
        return None
    return buf.tobytes()


_MAX_CAMERA_PROBE_INDEX = 4  # auto-detect probes indices 0..4


def _open_capture(index: int):
    """Open a single VideoCapture handle for `index` using the right backend
    for this OS. Windows: CAP_DSHOW is far more reliable than the default
    (MSMF) backend for OpenCV running inside a FastAPI/uvicorn worker
    thread — MSMF can silently hang or fail to open in this context even
    though it works fine in a plain standalone script."""
    if sys.platform == "win32":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW), "CAP_DSHOW"
    return cv2.VideoCapture(index), "default"


def _detect_camera_index() -> int:
    """Pick which camera device to open. An explicit CAMERA_INDEX env var
    always wins (manual override). Otherwise, auto-detect: probe indices
    from highest to lowest and use the first one that actually opens AND
    delivers a real frame. A plugged-in USB webcam almost always enumerates
    AFTER the built-in laptop camera (which sits at index 0), so probing
    high-to-low picks the USB webcam automatically whenever one is
    connected, and falls back to the laptop cam (index 0) when it's the
    only camera available — no CAMERA_INDEX env var needed for the common
    case."""
    env_val = os.getenv("CAMERA_INDEX")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            print(f"MJPEG: CAMERA_INDEX env var ('{env_val}') is not a valid integer, auto-detecting instead")

    for idx in range(_MAX_CAMERA_PROBE_INDEX, -1, -1):
        probe, _ = _open_capture(idx)
        ok = probe.isOpened()
        if ok:
            ok, _frame = probe.read()  # confirm it actually delivers frames, not just "opened"
        probe.release()
        if ok:
            if idx > 0:
                print(f"MJPEG: auto-detected external camera at index={idx} "
                      f"(preferred over the laptop cam at index 0)")
            return idx

    return 0  # nothing responded to any probe — fall back to the default index


def _mjpeg_open_camera() -> bool:
    """Open the MJPEG-pipeline camera device. Returns False (with logging) on failure."""
    global _mjpeg_cap

    with _mjpeg_lock:
        if _mjpeg_cap is not None and _mjpeg_cap.isOpened():
            return True

        camera_index = _detect_camera_index()
        cap, backend_name = _open_capture(camera_index)

        # 320x240 made several webcams (built-in laptop cams especially)
        # fall back to a center-cropped sensor region instead of a real
        # downscale, which looks like the feed is "zoomed in" close and
        # can't frame a patient standing back a few meters. Requesting a
        # standard 1280x720 gets the driver's normal full-FOV mode on both
        # laptop cams and external webcams, so a patient up to ~3m away
        # stays fully in frame for MediaPipe to track accurately.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            # Ask the driver to keep only 1 frame buffered (not all backends
            # honor this, but it helps on the ones that do — combined with
            # the reader-thread pattern below, it's belt-and-suspenders
            # against the buffered-delay problem).
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            print(f"MJPEG: could not open camera at index={camera_index} with backend={backend_name} "
                  f"(cv2.VideoCapture({camera_index}).isOpened() == False — no webcam device at that "
                  f"index, already in use by another process, or blocked by OS permissions. Try a "
                  f"different CAMERA_INDEX value — 0, 1, 2... — if you have more than one camera)")
            cap.release()
            _mjpeg_cap = None
            return False

        # Confirm we can actually read a frame, not just that the handle opened
        ok, _test_frame = cap.read()
        if not ok:
            print(f"MJPEG: camera opened (index={camera_index}, backend={backend_name}) but first read() failed")
            cap.release()
            _mjpeg_cap = None
            return False

        _mjpeg_cap = cap
        print(f"MJPEG: camera opened (index={camera_index}, backend={backend_name}, 1280x720 @30fps requested)")

        # Start the background reader thread that keeps _mjpeg_latest_frame
        # fresh — see the comment on the globals above for why this exists.
        global _mjpeg_reader_thread, _mjpeg_reader_running, _mjpeg_latest_frame, _mjpeg_frame_id
        _mjpeg_latest_frame = None
        _mjpeg_frame_id = 0
        _mjpeg_reader_running = True
        _mjpeg_reader_thread = threading.Thread(target=_mjpeg_reader_loop, daemon=True)
        _mjpeg_reader_thread.start()

        return True


def _mjpeg_reader_loop():
    global _mjpeg_latest_frame, _mjpeg_frame_id
    consecutive_failures = 0
    while _mjpeg_reader_running:
        with _mjpeg_lock:
            cap = _mjpeg_cap
        if cap is None:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= 10:
                print("MJPEG reader: camera stopped returning frames, stopping reader thread")
                break
            time.sleep(0.01)
            continue
        consecutive_failures = 0
        with _mjpeg_frame_lock:
            _mjpeg_latest_frame = frame
            _mjpeg_frame_id += 1


def _mjpeg_release_camera(ctx: Optional[UserContext] = None):
    """Release the server-attached camera device (local/in-clinic mode only)
    and reset that user's live pose flags."""
    global _mjpeg_cap, _mjpeg_active, _mjpeg_reader_running, _mjpeg_latest_frame
    _mjpeg_reader_running = False   # signal reader thread to stop
    with _mjpeg_lock:
        _mjpeg_active = False
        if _mjpeg_cap is not None:
            _mjpeg_cap.release()
            print("MJPEG: camera released")
        _mjpeg_cap = None
    _mjpeg_latest_frame = None
    if ctx is not None:
        ctx.reset_pose_data()


def process_frame(ctx: UserContext, frame, overlay: bool = False):
    """Run the full pose / game pipeline on ONE BGR frame (as delivered by a
    camera) and return the annotated JPEG bytes, or None if the frame could
    not be encoded. Shared by the server-camera MJPEG generator (gen_frames)
    and the browser-camera WebSocket (/ws/camera). `ctx` is the calling user's
    own state (metrics, game engine, pose data, MediaPipe trackers), so several
    doctors can stream at once without seeing each other's data. Frames of the
    SAME user are processed one at a time (their trackers are stateful)."""
    with ctx.process_lock:
        ctx.touch()
        return _process_frame_locked(ctx, frame, overlay)


def _process_frame_locked(ctx: UserContext, frame, overlay: bool = False):
    items = {"l": [], "d": []}
    m = ctx.metrics
    g = ctx.game
    frame = cv2.flip(frame, 1)  # mirror, like the reference app
    set_frame_size(frame.shape[1], frame.shape[0])

    # ── Game mode: session page is on the "Game" section, not
    # "Exercise" — skip the exercise-specific rep/ROM/stability path
    # entirely and hand the frame to the currently selected game's
    # engine instead (services/game_state.py -> game/registry.py).
    if g.get_mode() == "game":
        game_results = None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            _pose = ctx.pose
            if _pose is not None:
                game_results = _pose.process(rgb)
            rgb.flags.writeable = True
        except Exception as e:
            print(f"MJPEG: MediaPipe processing failed (game mode): {e}")
            game_results = None

        game_person_found = bool(game_results and game_results.pose_landmarks)
        now = time.time()

        if game_person_found and mp_drawing is not None:
            if overlay:
                _items_pose_full(items, game_results.pose_landmarks.landmark)
            else:
                mp_drawing.draw_landmarks(
                    frame, game_results.pose_landmarks, POSE_CONNECTIONS,
                    landmark_drawing_spec=_MJPEG_LANDMARK_SPEC,
                    connection_drawing_spec=_MJPEG_CONNECTION_SPEC,
                )
            g.process_frame(
                game_results.pose_landmarks.landmark,
                game_results.pose_world_landmarks,
                now,
                frame.shape,
            )
        else:
            g.process_frame(None, None, now, frame.shape)

        ctx.pose_data["detected"] = game_person_found
        ctx.pose_data["ts"] = datetime.datetime.now().isoformat()

        return _finish(frame, overlay, items)

    active_exercise, target_rom, side = m.get_exercise_state()
    hand_mode = _is_hand_exercise(active_exercise)

    angles: Dict[str, float] = {}
    primary_angle: Optional[float] = None
    reps = m.get_rep_count()
    stability  = m.get_stability()
    smoothness = m.get_smoothness()
    balance    = m.get_balance()
    fatigue    = m.get_current_fatigue()

    if hand_mode:
        # ── Hand Grip / Finger Flexion path: MediaPipe Hands ──────
        hand_results = None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            _hands = ctx.hands
            if _hands is not None:
                hand_results = _hands.process(rgb)
            rgb.flags.writeable = True
        except Exception as e:
            print(f"MJPEG: MediaPipe Hands processing failed: {e}")
            hand_results = None

        # Only the hand(s) matching the Left/Right/Both selector are
        # drawn and measured - the other hand is ignored completely.
        selected_hands = _select_hands(hand_results, side) if hand_results else []
        hand_found = bool(selected_hands)
        detected = hand_found

        if hand_found:
            if overlay:
                _items_hands(items, selected_hands)
            else:
                _draw_hand_skeleton(frame, selected_hands)
            # Single-hand exercise: first selected hand drives angles/reps.
            lm0 = selected_hands[0].landmark
            angles = metrics.compute_finger_curl_angles(lm0)
            # Stability for a hands-only frame = wrist steadiness (there
            # are no hips/shoulders in this frame, it used to stay at 100).
            stability = m.update_stability_point(lm0[0].x, lm0[0].y)

            primary_angle = metrics.compute_primary_angle(angles, active_exercise, side)
            reps       = m.update_rep_count(primary_angle, target_rom)
            _rep_debug(ctx, active_exercise, side, target_rom, primary_angle)
            # Balance is body-pose-derived (shoulder sway) and does not
            # apply to a hands-only frame, so it stays at its neutral
            # value; smoothness comes from finger-angle jerk.
            smoothness = m.update_smoothness(primary_angle)
            fatigue    = m.maybe_record_rep_quality(reps, primary_angle, target_rom)

        ctx.pose_data["detected"]      = detected
        ctx.pose_data["angles"]        = angles
        ctx.pose_data["ts"]            = datetime.datetime.now().isoformat()
        ctx.pose_data["reps"]          = reps
        ctx.pose_data["stability"]     = stability
        ctx.pose_data["smoothness"]    = smoothness
        ctx.pose_data["balance"]       = balance
        ctx.pose_data["fatigue"]       = fatigue
        ctx.pose_data["primary_angle"] = primary_angle

        return _finish(frame, overlay, items)

    # ── Regular body-pose path (all other exercise types) ─────────
    results = None
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        _pose = ctx.pose
        if _pose is not None:
            results = _pose.process(rgb)
        rgb.flags.writeable = True
    except Exception as e:
        print(f"MJPEG: MediaPipe processing failed: {e}")
        results = None

    person_found = bool(results and results.pose_landmarks)

    # "detected" now means "the joints THIS exercise needs are
    # visible" rather than "MediaPipe found some person somewhere
    # in frame" — so an Elbow exercise no longer needs the
    # patient's legs in shot to register as connected.
    detected = person_found and _relevant_landmarks_visible(
        results.pose_landmarks.landmark, active_exercise, side
    )

    if person_found and mp_drawing is not None:
        # Skeleton is still drawn whenever MediaPipe found a person
        # at all, even if the exercise-relevant joints aren't all
        # visible yet — the doctor can see the patient adjusting
        # into frame instead of a blank video.
        if overlay:
            _items_pose_filtered(items, results.pose_landmarks.landmark, active_exercise, side)
        else:
            _draw_filtered_skeleton(frame, results.pose_landmarks.landmark, active_exercise, side)

        lm = results.pose_landmarks.landmark
        try:
            if len(lm) > 16:
                angles["l_elbow"] = round(get_angle_2d(lm[11], lm[13], lm[15]), 1)
                angles["r_elbow"] = round(get_angle_2d(lm[12], lm[14], lm[16]), 1)
            if len(lm) > 28:
                angles["l_knee"] = round(get_angle_2d(lm[23], lm[25], lm[27]), 1)
                angles["r_knee"] = round(get_angle_2d(lm[24], lm[26], lm[28]), 1)
            if len(lm) > 26:
                angles["l_hip"] = round(get_angle_2d(lm[11], lm[23], lm[25]), 1)
                angles["r_hip"] = round(get_angle_2d(lm[12], lm[24], lm[26]), 1)
            if len(lm) > 14:
                # Shoulder flexion/abduction: hip → shoulder → elbow,
                # measures how far the arm is raised relative to the torso.
                angles["l_shoulder"] = round(get_angle_2d(lm[23], lm[11], lm[13]), 1)
                angles["r_shoulder"] = round(get_angle_2d(lm[24], lm[12], lm[14]), 1)
            if len(lm) > 32:
                # knee - ankle - foot_index (landmarks 31/32 are the only
                # foot points MediaPipe Pose provides)
                angles["l_ankle"] = round(get_angle_2d(lm[25], lm[27], lm[31]), 1)
                angles["r_ankle"] = round(get_angle_2d(lm[26], lm[28], lm[32]), 1)
        except Exception as e:
            print(f"MJPEG: angle calculation failed: {e}")

        # Real rep counting + stability + smoothness + balance
        # (no randomness, no duplicated/copied metrics)
        primary_angle = metrics.compute_primary_angle(angles, active_exercise, side)
        reps       = m.update_rep_count(primary_angle, target_rom)
        _rep_debug(ctx, active_exercise, side, target_rom, primary_angle)
        stability  = m.update_stability(lm)
        smoothness = m.update_smoothness(primary_angle)
        balance    = m.update_balance(lm)

        # Real fatigue — record rep quality the instant a new rep
        # is detected, then fatigue score reflects the actual
        # early-vs-recent quality trend for this session
        fatigue = m.maybe_record_rep_quality(reps, primary_angle, target_rom)

    ctx.pose_data["detected"]      = detected
    ctx.pose_data["angles"]        = angles
    ctx.pose_data["ts"]            = datetime.datetime.now().isoformat()
    ctx.pose_data["reps"]          = reps
    ctx.pose_data["stability"]     = stability
    ctx.pose_data["smoothness"]    = smoothness
    ctx.pose_data["balance"]       = balance
    ctx.pose_data["fatigue"]       = fatigue
    ctx.pose_data["primary_angle"] = primary_angle

    return _finish(frame, overlay, items)



def gen_frames(ctx: UserContext):
    """Server-attached camera mode (local/in-clinic): frames come from a camera
    plugged into THIS machine. Not used by the deployed browser-camera flow."""
    global _mjpeg_active

    if not _mjpeg_open_camera():
        # Yield one explanatory frame instead of just hanging/breaking silently
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        msg_line1, msg_line2 = "Camera unavailable", "Check device / permissions"
        cv2.putText(blank, msg_line1, (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(blank, msg_line2, (15, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", blank)
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return

    _mjpeg_active = True
    print("MJPEG: stream starting")

    try:
        last_frame_id = -1
        while _mjpeg_active:
            with _mjpeg_lock:
                cap = _mjpeg_cap
            if cap is None:
                print("MJPEG: camera handle gone, stopping generator")
                break

            # Pull whatever's freshest from the reader thread instead of
            # calling cap.read() here directly — this is what prevents the
            # processing loop (MediaPipe is the slow part) from falling
            # behind and showing an increasingly stale/delayed frame.
            with _mjpeg_frame_lock:
                frame = _mjpeg_latest_frame
                frame_id = _mjpeg_frame_id

            if frame is None:
                time.sleep(0.01)   # camera just opened, reader hasn't produced a frame yet
                continue
            if frame_id == last_frame_id:
                time.sleep(0.005)  # no new frame since last loop — avoid reprocessing/duplicating it
                continue
            last_frame_id = frame_id
            frame = frame.copy()  # reader thread may overwrite the shared slot while we work on this one

            jpg = process_frame(ctx, frame)
            if jpg is None:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

    except GeneratorExit:
        # Client closed the <img> / navigated away
        print("MJPEG: client disconnected, closing generator")
    except Exception as e:
        print(f"MJPEG: generator crashed unexpectedly: {e}")
    finally:
        _mjpeg_release_camera(ctx)
        print("MJPEG: stream ended, camera released")


def is_active() -> bool:
    """True if a server-attached camera is streaming OR any user has a
    browser-camera socket open -- used by /api/health."""
    if user_context.any_camera_active():
        return True
    with _mjpeg_lock:
        return _mjpeg_active and _mjpeg_cap is not None and _mjpeg_cap.isOpened()


def stop_camera(ctx: Optional[UserContext] = None):
    """Release the server-attached camera device (no-op in browser-camera mode)
    and clear this user's live pose flags. Called on Stop / page unload."""
    _mjpeg_release_camera(ctx)

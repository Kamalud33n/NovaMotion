"""
Game-mode state for the session page's camera pipeline. Mirrors
services/metrics.py's role for "Exercise" mode, but for the "Game" section:
holds which mode the session page is in, which game is selected, that
game's live engine instance, and the therapist-configured target (target
steps / customizable duration) so mjpeg_camera.py can dispatch frames to
the engine and the session page can poll one place for "is this game
session finished yet".

Not tied to any one game — new games only ever get added to game/registry.py;
nothing here needs to change (each game just declares its own
"target_metric" in the registry — see registry.py).
"""
import threading
import time
from typing import Optional

from game.registry import get_game, list_games, get_target_metric

_lock = threading.Lock()
_mode = "exercise"                    # "exercise" | "game"
_active_game_id: Optional[str] = None
_engine = None                        # live engine instance for _active_game_id
_target_metric = "steps"              # which eng.get_status() key the target below is measured against

# -- target / auto-finish state ---------------------------------------------
# Configured once per game session via set_target(), right after select_game()
# and before request_calibration(). Either value can be None to disable that
# finish condition (e.g. a steps-only session with no time limit, or a timed
# free-play session with no step target). The clock only starts once
# calibration completes — time spent calibrating never counts against the
# patient.
_target_steps: Optional[int] = None
_duration_seconds: Optional[int] = None
_start_time: Optional[float] = None   # time.time() when calibration completed
_finished = False
_finish_reason: Optional[str] = None  # "target" | "time" | None


def get_mode() -> str:
    with _lock:
        return _mode


def set_mode(mode: str) -> bool:
    """Switch the session page between "exercise" and "game". Returns False
    on an unrecognized mode (state left unchanged)."""
    global _mode
    if mode not in ("exercise", "game"):
        return False
    with _lock:
        _mode = mode
    return True


def get_available_games():
    """[{id, label, description}, ...] for the session page's game dropdown."""
    return list_games()


def _reset_target_locked():
    """Clear target/timer/finish state. Caller must hold _lock."""
    global _target_steps, _duration_seconds, _start_time, _finished, _finish_reason
    _target_steps = None
    _duration_seconds = None
    _start_time = None
    _finished = False
    _finish_reason = None


def select_game(game_id: str) -> bool:
    """Switch the active game to game_id, creating a fresh engine instance
    and clearing any previous game's target/timer/finish state (a new game
    always starts unconfigured — the session page must call set_target()
    again before play). Returns False if game_id isn't in the registry
    (state left unchanged)."""
    global _active_game_id, _engine, _target_metric
    entry = get_game(game_id)
    if entry is None:
        return False
    with _lock:
        _active_game_id = game_id
        _engine = entry["engine_cls"]()
        _target_metric = get_target_metric(game_id)
        _reset_target_locked()
    return True


def get_active_game_id() -> Optional[str]:
    with _lock:
        return _active_game_id


def get_engine():
    """Live engine instance for the currently selected game, or None if no
    game has been selected yet."""
    with _lock:
        return _engine


def set_target(target_steps: Optional[int], duration_seconds: Optional[int]):
    """Configure this game session's finish condition. Call after
    select_game(), before reset_game()/request_calibration(). target_steps:
    stop the session once the engine's target metric (e.g. "steps") reaches
    this count. duration_seconds: stop the session once this many seconds
    have elapsed since calibration finished. Pass None for either to
    disable that condition; passing None for both means the session never
    auto-finishes (manual stop only, same as today)."""
    with _lock:
        _reset_target_locked()
        global _target_steps, _duration_seconds
        _target_steps = target_steps
        _duration_seconds = duration_seconds


def reset_game():
    """Call right before a game session (re)starts, so calibration/step/lane
    state doesn't carry over from a previous patient/session (mirrors
    metrics.reset_state()). Also clears the finish/timer state — the target
    values set via set_target() are kept (a retry of the same session keeps
    the same target unless the therapist changes it)."""
    eng = get_engine()
    if eng is not None:
        eng.reset()
    with _lock:
        global _start_time, _finished, _finish_reason
        _start_time = None
        _finished = False
        _finish_reason = None


def request_calibration():
    """Start measuring the neutral standing position now."""
    eng = get_engine()
    if eng is not None:
        eng.request_calibration()


def finish_game(reason: str = "manual"):
    """Mark the current game session finished right now — used by the
    session page's own "Stop" button (therapist/patient ends the session
    before the target/timer condition is reached). No-op if already
    finished (auto-finish already fired, or Stop pressed twice)."""
    with _lock:
        global _finished, _finish_reason
        if not _finished:
            _finished, _finish_reason = True, reason


def _check_progress_locked(status: dict, now: float):
    """Update the finish/timer state from one frame's engine status. Caller
    must hold _lock. The clock only starts once calibration completes, so
    calibration time is never counted toward the duration target."""
    global _start_time, _finished, _finish_reason
    if _finished:
        return
    if not status.get("calibrated"):
        return
    if _start_time is None:
        _start_time = now
        return
    elapsed = now - _start_time
    if _target_steps is not None and status.get(_target_metric, 0) >= _target_steps:
        _finished, _finish_reason = True, "target"
    elif _duration_seconds is not None and elapsed >= _duration_seconds:
        _finished, _finish_reason = True, "time"


def get_game_status() -> dict:
    """The active engine's live status, plus which game is active and the
    target/timer/finish state — this is what the session page polls during
    play to drive the HUD and to know when to show the "session complete"
    screen."""
    eng = get_engine()
    if eng is None:
        return {"active_game": None}
    status = eng.get_status()
    now = time.time()
    with _lock:
        _check_progress_locked(status, now)
        target_steps, duration_seconds = _target_steps, _duration_seconds
        start_time, finished, finish_reason = _start_time, _finished, _finish_reason
        target_metric = _target_metric

    elapsed = (now - start_time) if start_time is not None else 0.0
    remaining = (max(0.0, duration_seconds - elapsed) if duration_seconds is not None else None)

    status["active_game"] = _active_game_id
    status.update({
        "target_metric":     target_metric,
        "target_progress":   status.get(target_metric, 0),
        "target_steps":      target_steps,
        "duration_seconds":  duration_seconds,
        "elapsed_seconds":   round(elapsed, 1),
        "remaining_seconds": (round(remaining, 1) if remaining is not None else None),
        "finished":          finished,
        "finish_reason":     finish_reason,
    })
    return status


def get_game_summary() -> dict:
    """Server-tracked totals at game-session end (steps, events, target/
    timer info), for the session page to fold into the /api/sessions save
    payload."""
    eng = get_engine()
    if eng is None:
        return {}
    summary = eng.get_summary()
    now = time.time()
    with _lock:
        start_time = _start_time
        summary.update({
            "target_metric":    _target_metric,
            "target_steps":     _target_steps,
            "duration_seconds": _duration_seconds,
            "elapsed_seconds":  round((now - start_time), 1) if start_time is not None else 0.0,
            "finished":         _finished,
            "finish_reason":    _finish_reason,
        })
    return summary


def process_frame(landmarks, world_landmarks, now: float, frame_shape=None):
    """Dispatch one camera frame to the active game engine, if any, and
    update the target/timer/finish tracking from the resulting status.
    No-op when no game has been selected (mode can be "game" for a moment
    before the dropdown selection lands)."""
    eng = get_engine()
    if eng is None:
        return
    eng.process_frame(landmarks, world_landmarks, now, frame_shape=frame_shape)
    with _lock:
        _check_progress_locked(eng.get_status(), now)

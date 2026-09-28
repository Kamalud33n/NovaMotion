"""
Per-user game-mode state for the session page's camera pipeline. Mirrors
services/metrics.py's role for "Exercise" mode, but for the "Game" section:
holds which mode the session page is in, which game is selected, that
game's live engine instance, and the therapist-configured target (target
steps / customizable duration) so mjpeg_camera.py can dispatch frames to
the engine and the session page can poll one place for "is this game
session finished yet".

Everything lives on a GameState instance (one per logged-in user, see
services/user_context.py), so two doctors playing at the same time never
share a game engine, calibration, or timer.

Not tied to any one game -- new games only ever get added to game/registry.py;
nothing here needs to change (each game just declares its own
"target_metric" in the registry -- see registry.py).
"""
import threading
import time
from typing import Optional

from game.registry import get_game, list_games, get_target_metric


def get_available_games():
    """[{id, label, description}, ...] for the session page's game dropdown."""
    return list_games()


class GameState:
    def __init__(self):
        self._lock = threading.Lock()
        self._mode = "exercise"                # "exercise" | "game"
        self._active_game_id: Optional[str] = None
        self._engine = None                    # live engine instance for _active_game_id
        self._target_metric = "steps"          # which eng.get_status() key the target is measured against

        # -- target / auto-finish state ---------------------------------------
        # Configured once per game session via set_target(), right after
        # select_game() and before request_calibration(). Either value can be
        # None to disable that finish condition. The clock only starts once
        # calibration completes -- time spent calibrating never counts against
        # the patient.
        self._target_steps: Optional[int] = None
        self._duration_seconds: Optional[int] = None
        self._start_time: Optional[float] = None   # time.time() when calibration completed
        self._finished = False
        self._finish_reason: Optional[str] = None  # "target" | "time" | None

    def get_mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str) -> bool:
        """Switch the session page between "exercise" and "game". Returns False
        on an unrecognized mode (state left unchanged)."""
        if mode not in ("exercise", "game"):
            return False
        with self._lock:
            self._mode = mode
        return True

    def _reset_target_locked(self):
        """Clear target/timer/finish state. Caller must hold self._lock."""
        self._target_steps = None
        self._duration_seconds = None
        self._start_time = None
        self._finished = False
        self._finish_reason = None

    def select_game(self, game_id: str) -> bool:
        """Switch the active game to game_id, creating a fresh engine instance
        and clearing any previous game's target/timer/finish state. Returns
        False if game_id isn't in the registry (state left unchanged)."""
        entry = get_game(game_id)
        if entry is None:
            return False
        with self._lock:
            self._active_game_id = game_id
            self._engine = entry["engine_cls"]()
            self._target_metric = get_target_metric(game_id)
            self._reset_target_locked()
        return True

    def get_active_game_id(self) -> Optional[str]:
        with self._lock:
            return self._active_game_id

    def get_engine(self):
        """Live engine instance for the currently selected game, or None if no
        game has been selected yet."""
        with self._lock:
            return self._engine

    def set_target(self, target_steps: Optional[int], duration_seconds: Optional[int]):
        """Configure this game session's finish condition. Call after
        select_game(), before reset_game()/request_calibration(). Pass None for
        either to disable that condition; None for both = manual stop only."""
        with self._lock:
            self._reset_target_locked()
            self._target_steps = target_steps
            self._duration_seconds = duration_seconds

    def reset_game(self):
        """Call right before a game session (re)starts, so calibration/step/lane
        state doesn't carry over from a previous patient/session. Also clears
        the finish/timer state -- the target values set via set_target() are
        kept."""
        eng = self.get_engine()
        if eng is not None:
            eng.reset()
        with self._lock:
            self._start_time = None
            self._finished = False
            self._finish_reason = None

    def request_calibration(self):
        """Start measuring the neutral standing position now."""
        eng = self.get_engine()
        if eng is not None:
            eng.request_calibration()

    def finish_game(self, reason: str = "manual"):
        """Mark the current game session finished right now. No-op if already
        finished (auto-finish already fired, or Stop pressed twice)."""
        with self._lock:
            if not self._finished:
                self._finished, self._finish_reason = True, reason

    def _check_progress_locked(self, status: dict, now: float):
        """Update the finish/timer state from one frame's engine status. Caller
        must hold self._lock. The clock only starts once calibration completes,
        so calibration time is never counted toward the duration target."""
        if self._finished:
            return
        if not status.get("calibrated"):
            return
        if self._start_time is None:
            self._start_time = now
            return
        elapsed = now - self._start_time
        if self._target_steps is not None and status.get(self._target_metric, 0) >= self._target_steps:
            self._finished, self._finish_reason = True, "target"
        elif self._duration_seconds is not None and elapsed >= self._duration_seconds:
            self._finished, self._finish_reason = True, "time"

    def get_game_status(self) -> dict:
        """The active engine's live status, plus which game is active and the
        target/timer/finish state -- what the session page polls during play."""
        eng = self.get_engine()
        if eng is None:
            return {"active_game": None}
        status = eng.get_status()
        now = time.time()
        with self._lock:
            self._check_progress_locked(status, now)
            target_steps, duration_seconds = self._target_steps, self._duration_seconds
            start_time, finished, finish_reason = self._start_time, self._finished, self._finish_reason
            target_metric = self._target_metric
            active_game = self._active_game_id

        elapsed = (now - start_time) if start_time is not None else 0.0
        remaining = (max(0.0, duration_seconds - elapsed) if duration_seconds is not None else None)

        status["active_game"] = active_game
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

    def get_game_summary(self) -> dict:
        """Server-tracked totals at game-session end (steps, events, target/
        timer info), for the session page to fold into the /api/sessions save
        payload."""
        eng = self.get_engine()
        if eng is None:
            return {}
        summary = eng.get_summary()
        now = time.time()
        with self._lock:
            start_time = self._start_time
            summary.update({
                "target_metric":    self._target_metric,
                "target_steps":     self._target_steps,
                "duration_seconds": self._duration_seconds,
                "elapsed_seconds":  round((now - start_time), 1) if start_time is not None else 0.0,
                "finished":         self._finished,
                "finish_reason":    self._finish_reason,
            })
        return summary

    def process_frame(self, landmarks, world_landmarks, now: float, frame_shape=None):
        """Dispatch one camera frame to the active game engine, if any, and
        update the target/timer/finish tracking from the resulting status.
        No-op when no game has been selected."""
        eng = self.get_engine()
        if eng is None:
            return
        eng.process_frame(landmarks, world_landmarks, now, frame_shape=frame_shape)
        with self._lock:
            self._check_progress_locked(eng.get_status(), now)

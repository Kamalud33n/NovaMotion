"""
Per-user live session context.

Before this module, rep counts, the selected exercise/side/ROM, the game
engine, the latest pose data and the MediaPipe trackers were module-level
globals, so two doctors using the site at the same time (e.g. from different
clinics) overwrote each other's data. Now every logged-in user gets their own
UserContext, looked up by user id:

    ctx = get_context(user.id)
    ctx.metrics      # MetricsState  (reps, stability, exercise, side, ROM ...)
    ctx.game         # GameState     (game engine, calibration, timer ...)
    ctx.pose_data    # dict served by /api/pose_data
    ctx.pose / ctx.hands   # this user's own MediaPipe trackers (lazy)

State is in this process's memory, so run ONE uvicorn process (`python app.py`
already does). Several worker processes would each have their own registry.

The same user logged in on two devices at once shares one context (one doctor
= one live session); different users never share anything.
"""
import threading
import time
from typing import Any, Dict, Optional

import config
from services.metrics import MetricsState
from services.game_state import GameState

# A context nobody has touched for this long (and with no camera socket open)
# is dropped to free its MediaPipe models. 30 min keeps the state alive long
# enough for the page to fetch /api/game/summary and save the session after
# the camera stops.
IDLE_TTL_SECONDS = 30 * 60
_SWEEP_EVERY_SECONDS = 60


def new_pose_data() -> Dict[str, Any]:
    return {
        "detected": False, "angles": {}, "ts": None,
        "reps": 0, "stability": 100.0, "primary_angle": None,
        "smoothness": 100.0, "balance": 100.0, "fatigue": 0.0,
    }


class UserContext:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.metrics = MetricsState()
        self.game = GameState()
        self.pose_data: Dict[str, Any] = new_pose_data()

        # process_frame() for this user runs one frame at a time (MediaPipe
        # trackers + metrics are stateful); other users run in parallel.
        self.process_lock = threading.Lock()
        self.dbg_last = 0.0                 # throttles the REPDBG log line

        self.remote_clients = 0             # browser-camera WebSockets open right now
        self.last_used = time.time()

        self._models_lock = threading.Lock()
        self._pose = None
        self._pose_made = False
        self._hands = None
        self._hands_made = False

    # -- bookkeeping ---------------------------------------------------------
    def touch(self):
        self.last_used = time.time()

    def remote_connected(self):
        self.remote_clients += 1
        self.touch()

    def remote_disconnected(self):
        """A browser-camera WebSocket closed: clear the live pose flags."""
        self.remote_clients = max(0, self.remote_clients - 1)
        self.touch()
        if self.remote_clients == 0:
            self.pose_data["detected"] = False
            self.pose_data["angles"] = {}

    def reset_pose_data(self):
        self.pose_data["detected"] = False
        self.pose_data["angles"] = {}

    # -- MediaPipe trackers (created on first use, one set per user) ----------
    @property
    def pose(self):
        with self._models_lock:
            if not self._pose_made:
                self._pose = config.create_pose()
                self._pose_made = True
            return self._pose

    @property
    def hands(self):
        with self._models_lock:
            if not self._hands_made:
                self._hands = config.create_hands()
                self._hands_made = True
            return self._hands

    def close(self):
        with self._models_lock:
            for m in (self._pose, self._hands):
                try:
                    if m is not None:
                        m.close()
                except Exception:
                    pass
            self._pose = self._hands = None
            self._pose_made = self._hands_made = False


# -- registry -----------------------------------------------------------------
_contexts: Dict[str, UserContext] = {}
_registry_lock = threading.Lock()
_last_sweep = 0.0


def get_context(user_id: str) -> UserContext:
    """This user's context (created on first use)."""
    global _last_sweep
    now = time.time()
    stale = []
    with _registry_lock:
        ctx = _contexts.get(user_id)
        if ctx is None:
            ctx = _contexts[user_id] = UserContext(user_id)
        ctx.last_used = now

        if now - _last_sweep >= _SWEEP_EVERY_SECONDS:
            _last_sweep = now
            for uid, c in list(_contexts.items()):
                if c is not ctx and c.remote_clients == 0 and now - c.last_used > IDLE_TTL_SECONDS:
                    stale.append(_contexts.pop(uid))
    for c in stale:          # free the models outside the registry lock
        c.close()
    return ctx


def any_camera_active() -> bool:
    """True if any user currently has a browser-camera socket open."""
    with _registry_lock:
        return any(c.remote_clients > 0 for c in _contexts.values())


def active_user_count() -> int:
    with _registry_lock:
        return sum(1 for c in _contexts.values() if c.remote_clients > 0)

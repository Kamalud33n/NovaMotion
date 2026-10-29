"""Patient lock for the live pose pipeline.

Locks onto the first stable person in frame and rejects poses that jump away
from them (someone walking through), so reps, game input and metrics stay
tied to the patient.
"""
import math
import os
import threading
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

STATE_OFF = "off"
STATE_SEARCHING = "searching"
STATE_LOCKED = "locked"
STATE_HOLDING = "holding"

_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 11, 12, 23, 24
_SCALE_TORSO = "torso"
_SCALE_SHOULDERS = "shoulders"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.getenv(name) or "").strip())
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or "").strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class FocusConfig:
    enabled: bool = True
    jump_ratio: float = 0.8
    scale_tolerance: float = 1.5
    drift_per_second: float = 1.5
    grace_seconds: float = 1.5
    lock_frames: int = 10
    relock_radius: float = 2.0
    min_visibility: float = 0.3
    hint_seconds: float = 6.0

    @classmethod
    def from_env(cls) -> "FocusConfig":
        d = cls()
        return cls(
            enabled=_env_bool("FOCUS_MODE_ENABLED", d.enabled),
            jump_ratio=_env_float("FOCUS_JUMP_RATIO", d.jump_ratio),
            scale_tolerance=max(1.0, _env_float("FOCUS_SCALE_TOLERANCE", d.scale_tolerance)),
            drift_per_second=_env_float("FOCUS_DRIFT_PER_SECOND", d.drift_per_second),
            grace_seconds=_env_float("FOCUS_GRACE_SECONDS", d.grace_seconds),
            lock_frames=max(1, _env_int("FOCUS_LOCK_FRAMES", d.lock_frames)),
            relock_radius=_env_float("FOCUS_RELOCK_RADIUS", d.relock_radius),
            min_visibility=min(1.0, max(0.0, _env_float("FOCUS_MIN_VISIBILITY", d.min_visibility))),
            hint_seconds=max(0.0, _env_float("FOCUS_HINT_SECONDS", d.hint_seconds)),
        )


@dataclass(frozen=True)
class BodyRef:
    x: float
    y: float
    scale: float
    scale_kind: str


@dataclass(frozen=True)
class FocusResult:
    accept: bool
    state: str
    reset_tracker: bool = False


def _mean(points: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def measure(landmarks, frame_shape, min_visibility: float = 0.3) -> Optional[BodyRef]:
    if landmarks is None or len(landmarks) <= _R_HIP:
        return None
    h, w = frame_shape[:2]

    def point(idx: int) -> Optional[Tuple[float, float]]:
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h) if lm.visibility >= min_visibility else None

    l_sh, r_sh = point(_L_SHOULDER), point(_R_SHOULDER)
    shoulders = [p for p in (l_sh, r_sh) if p]
    hips = [p for p in (point(_L_HIP), point(_R_HIP)) if p]
    sh_mid, hip_mid = _mean(shoulders), _mean(hips)
    centre = _mean(shoulders + hips)
    if centre is None:
        return None

    if sh_mid and hip_mid:
        scale = math.hypot(sh_mid[0] - hip_mid[0], sh_mid[1] - hip_mid[1])
        kind = _SCALE_TORSO
    elif l_sh and r_sh:
        scale = math.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1])
        kind = _SCALE_SHOULDERS
    else:
        return None
    if scale <= 0:
        return None
    return BodyRef(centre[0], centre[1], scale, kind)


def _same_person(prev: BodyRef, cur: BodyRef, elapsed: float, cfg: FocusConfig) -> bool:
    allowed = cfg.jump_ratio * prev.scale * (1.0 + cfg.drift_per_second * max(0.0, elapsed))
    if math.hypot(cur.x - prev.x, cur.y - prev.y) > allowed:
        return False
    if prev.scale_kind == _SCALE_TORSO and cur.scale_kind == _SCALE_TORSO:
        ratio = cur.scale / prev.scale
        if not (1.0 / cfg.scale_tolerance <= ratio <= cfg.scale_tolerance):
            return False
    return True


class FocusLock:
    def __init__(self, config: Optional[FocusConfig] = None):
        self._cfg = config or FocusConfig.from_env()
        self._mutex = threading.Lock()
        self._clear()

    @property
    def state(self) -> str:
        if not self._cfg.enabled:
            return STATE_OFF
        return self._state

    def relock(self) -> None:
        with self._mutex:
            self._clear()

    def update(self, landmarks, frame_shape, now: float) -> FocusResult:
        if not self._cfg.enabled:
            return FocusResult(landmarks is not None, STATE_OFF)
        ref = measure(landmarks, frame_shape, self._cfg.min_visibility)
        with self._mutex:
            if self._state == STATE_SEARCHING:
                return self._search(ref, now)
            return self._track(ref, now)

    def _clear(self) -> None:
        self._state = STATE_SEARCHING
        self._last: Optional[BodyRef] = None
        self._last_seen = 0.0
        self._bad_since: Optional[float] = None
        self._candidate: Optional[BodyRef] = None
        self._candidate_frames = 0
        self._hint: Optional[BodyRef] = None
        self._hint_at = 0.0

    def _near_hint(self, ref: BodyRef, now: float) -> bool:
        if self._hint is not None and now - self._hint_at > self._cfg.hint_seconds:
            self._hint = None
        if self._hint is None:
            return True
        dist = math.hypot(ref.x - self._hint.x, ref.y - self._hint.y)
        if dist > self._cfg.relock_radius * self._hint.scale:
            return False
        if self._hint.scale_kind == _SCALE_TORSO and ref.scale_kind == _SCALE_TORSO:
            ratio = ref.scale / self._hint.scale
            return 1.0 / self._cfg.scale_tolerance <= ratio <= self._cfg.scale_tolerance
        return True

    def _search(self, ref: Optional[BodyRef], now: float) -> FocusResult:
        if ref is None or not self._near_hint(ref, now):
            self._candidate, self._candidate_frames = None, 0
            return FocusResult(False, STATE_SEARCHING)

        if self._candidate and _same_person(self._candidate, ref, 0.0, self._cfg):
            self._candidate_frames += 1
        else:
            self._candidate_frames = 1
        self._candidate = ref

        if self._candidate_frames >= self._cfg.lock_frames:
            self._state = STATE_LOCKED
            self._last, self._last_seen = ref, now
            self._bad_since = None
            self._hint = None
            return FocusResult(True, STATE_LOCKED)
        return FocusResult(True, STATE_SEARCHING)

    def _track(self, ref: Optional[BodyRef], now: float) -> FocusResult:
        if ref and _same_person(self._last, ref, now - self._last_seen, self._cfg):
            self._state = STATE_LOCKED
            self._last, self._last_seen = ref, now
            self._bad_since = None
            return FocusResult(True, STATE_LOCKED)

        if self._bad_since is None:
            self._bad_since = now
        if now - self._bad_since <= self._cfg.grace_seconds:
            self._state = STATE_HOLDING
            return FocusResult(False, STATE_HOLDING)

        hint = self._last
        self._clear()
        self._hint, self._hint_at = hint, now
        return FocusResult(False, STATE_SEARCHING, reset_tracker=True)
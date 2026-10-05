"""
Per-user exercise metrics (rep count, stability, smoothness, balance, fatigue,
active exercise / ROM / side).

Everything that used to be a module-level global lives on a MetricsState
instance now, so two doctors using the site at the same time (different
clinics, different patients) never see each other's numbers. Each logged-in
user gets their own MetricsState through services/user_context.py.

The stateless helpers (angle_to_rom, compute_primary_angle, finger angles)
are still plain module-level functions.
"""
import threading
import collections
import statistics
import time
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Stateless helpers (no per-user state)
# ---------------------------------------------------------------------------
def angle_to_rom(angle: Optional[float], exercise_type: str) -> Optional[float]:
    """Convert a raw joint angle into ROM = how far the joint has MOVED from
    its neutral pose, in degrees (0..180). Rep counting, ROM, accuracy, smoothness
    and fatigue all work on this value, so it must grow as the patient performs
    the movement:

      * shoulder / arm raise : the hip-shoulder-elbow angle already grows as the
                               arm goes up (arm down ~0-20deg) -> use it as-is.
      * elbow / knee / hip / hand grip / finger curl : the joint is ~180deg when
        straight/open and the angle SHRINKS as it flexes/curls -> ROM = 180 - angle.
        (Using the raw angle here is what made reps never count and ROM read
        100% while the limb was resting straight.)
      * ankle : neutral (foot flat) is ~90deg, moving either way is movement
                -> ROM = |angle - 90|.
    """
    if angle is None:
        return None
    ex = (exercise_type or "").lower()
    if "shoulder" in ex or "arm" in ex:
        rom = angle
    elif "ankle" in ex:
        rom = abs(angle - 90.0)
    else:
        rom = 180.0 - angle
    return max(0.0, min(180.0, rom))


def compute_primary_angle(angles: Dict[str, float], exercise_type: str, side: str = "both") -> Optional[float]:
    """Return the primary movement value (ROM in degrees, see angle_to_rom) for
    the active exercise. When `side` is "left" or "right" only that limb is used
    instead of averaging both, so a one-arm/one-leg exercise isn't dragged down
    (or inflated) by the resting side. Returns None when the exercise has no
    joint-angle signal (Balance) or the needed joint isn't visible."""
    ex = (exercise_type or "").lower()
    s  = (side or "both").lower()

    def avg(*vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def pick(l_key: str, r_key: str):
        if s == "left":
            return angles.get(l_key)
        if s == "right":
            return angles.get(r_key)
        return avg(angles.get(l_key), angles.get(r_key))

    if "balance" in ex:
        # Balance is scored from stability/sway only - there is no joint angle,
        # so no ROM/rep counting (previously it averaged elbow+knee angles,
        # which produced meaningless numbers).
        return None

    if "grip" in ex or "finger" in ex:
        # Thumb excluded on purpose (its joint geometry differs); side does not
        # apply, the tracked hand is already chosen in mjpeg_camera.
        raw = avg(angles.get("index"), angles.get("middle"),
                  angles.get("ring"), angles.get("pinky"))
    elif "shoulder" in ex or "arm" in ex:
        raw = pick("l_shoulder", "r_shoulder")
    elif "elbow" in ex or "hand" in ex:
        # Pose-only "Hand Rehab" is tracked through the elbow-wrist chain (same
        # as the skeleton + UI badge group). It used to fall through to a
        # default that averaged elbows AND knees.
        raw = pick("l_elbow", "r_elbow")
    elif "knee" in ex or "squat" in ex or "leg" in ex:
        raw = pick("l_knee", "r_knee")
    elif "hip" in ex:
        raw = pick("l_hip", "r_hip")
    elif "ankle" in ex:
        raw = pick("l_ankle", "r_ankle")
    else:
        raw = pick("l_elbow", "r_elbow")

    return angle_to_rom(raw, ex)


def compute_finger_curl_angles(hand_landmarks) -> Dict[str, float]:
    """Real per-finger curl angle from MediaPipe Hands landmarks (21 pts).
    ~170-180° = finger fully extended (open hand), ~40-70° = fully curled
    (closed fist). Thumb uses CMC-MCP-IP since its joint layout differs
    from the other four fingers."""
    from config import get_angle, HAND_LANDMARKS as HL

    def pt(name):
        return hand_landmarks[HL[name]]

    angles: Dict[str, float] = {}
    try:
        angles["thumb"]  = round(get_angle(pt("thumb_cmc"),  pt("thumb_mcp"),  pt("thumb_ip")), 1)
        angles["index"]  = round(get_angle(pt("index_mcp"),  pt("index_pip"),  pt("index_dip")), 1)
        angles["middle"] = round(get_angle(pt("middle_mcp"), pt("middle_pip"), pt("middle_dip")), 1)
        angles["ring"]   = round(get_angle(pt("ring_mcp"),   pt("ring_pip"),   pt("ring_dip")), 1)
        angles["pinky"]  = round(get_angle(pt("pinky_mcp"),  pt("pinky_pip"),  pt("pinky_dip")), 1)
    except Exception as e:
        print(f"finger curl angle calculation failed: {e}")
    return angles


def compute_grip_angle(finger_angles: Dict[str, float]) -> Optional[float]:
    """Average curl across the 4 main fingers (thumb excluded — its range
    of motion/joint geometry differs enough that mixing it in would skew
    the open/close signal used for rep counting). This is the primary_angle
    fed into update_rep_count() for Hand Grip / Finger Flexion exercises,
    same as an elbow/knee angle is for other exercise types."""
    vals = [finger_angles.get(f) for f in ("index", "middle", "ring", "pinky")]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# Stability / balance look at the last second of movement and smoothness is measured per
# 1/30 s, so the scores no longer depend on how many frames per second the server manages.
_WINDOW_S = 1.0
_SMOOTH_REF_DT = 1.0 / 30.0


class MetricsState:
    """All live metric state for ONE user's session."""

    def __init__(self):
        # Active exercise / target ROM state
        self._exercise_lock = threading.Lock()
        self._current_exercise_type = "Shoulder Rehab"
        self._current_target_rom = 90.0
        self._current_side = "both"   # "left" / "right" / "both" -- which limb to track & draw

        # Rep counting
        self._rep_lock = threading.Lock()
        self._rep_count = 0
        self._rep_stage: Optional[str] = None      # "up" / "down"
        self._rep_peak = 0.0
        self._rom_window: "collections.deque" = collections.deque(maxlen=3)

        # Stability -- hip-center landmark jitter over a rolling window
        self._stability_lock = threading.Lock()
        self._landmark_jitter_buffer: "collections.deque" = collections.deque(maxlen=200)   # (t, x, y), last _WINDOW_S seconds
        self._current_stability_score = 100.0

        # Smoothness -- frame-to-frame angular jerk variance
        self._smoothness_lock = threading.Lock()
        self._angle_velocity_buffer: "collections.deque" = collections.deque(maxlen=15)
        self._last_primary_angle: Optional[float] = None
        self._last_angle_t: Optional[float] = None
        self._current_smoothness_score = 100.0

        # Balance -- shoulder-midpoint lateral sway
        self._balance_lock = threading.Lock()
        self._shoulder_sway_buffer: "collections.deque" = collections.deque(maxlen=200)   # (t, x), last _WINDOW_S seconds
        self._current_balance_score = 100.0

        # Fatigue -- rep-quality decline over the session
        self._fatigue_lock = threading.Lock()
        self._rep_quality_buffer: "collections.deque" = collections.deque(maxlen=30)
        self._current_fatigue_score = 0.0
        self._last_fatigue_rep_count = 0

    # -- Exercise state getters/setters --------------------------------------
    def set_exercise_state(self, exercise_type: Optional[str] = None, target_rom: Optional[float] = None,
                           side: Optional[str] = None):
        if exercise_type:
            with self._exercise_lock:
                self._current_exercise_type = exercise_type
        if target_rom is not None:
            try:
                rom = float(target_rom)
            except (TypeError, ValueError):
                return
            with self._exercise_lock:
                self._current_target_rom = rom
        if side:
            s = side.strip().lower()
            if s in ("left", "right", "both"):
                # The camera frame is mirrored (cv2.flip in mjpeg_camera.py)
                # BEFORE MediaPipe Pose runs on it, so MediaPipe's own
                # left_*/right_* landmark labels end up swapped relative to the
                # patient's actual left/right side. Store the inverted value
                # here -- the one place both the skeleton-line filter
                # (mjpeg_camera.py) and compute_primary_angle() read from -- so
                # "Left" in the UI always means the patient's real left side.
                if s == "left":
                    s = "right"
                elif s == "right":
                    s = "left"
                with self._exercise_lock:
                    self._current_side = s

    def get_exercise_state(self):
        with self._exercise_lock:
            return self._current_exercise_type, self._current_target_rom, self._current_side

    # -- Score getters (thread-safe reads) -----------------------------------
    def get_stability(self) -> float:
        with self._stability_lock:
            return self._current_stability_score

    def get_smoothness(self) -> float:
        with self._smoothness_lock:
            return self._current_smoothness_score

    def get_balance(self) -> float:
        with self._balance_lock:
            return self._current_balance_score

    def get_current_fatigue(self) -> float:
        with self._fatigue_lock:
            return self._current_fatigue_score

    def get_rep_count(self) -> int:
        with self._rep_lock:
            return self._rep_count

    def get_rep_debug(self, target_rom: float) -> str:
        with self._rep_lock:
            med = statistics.median(self._rom_window) if self._rom_window else None
            return (f"median={med} stage={self._rep_stage} peak={round(self._rep_peak, 1)} "
                    f"count={self._rep_count} high={round(target_rom * 0.80, 1)} low={round(target_rom * 0.55, 1)}")

    # -- Updaters ------------------------------------------------------------
    def update_rep_count(self, primary_angle: Optional[float], target_rom: float) -> int:
        """A rep is counted when ROM reaches >= 80% of target, then the joint comes
        back down: either to <= 55% of target, or by at least 35% of target below
        the peak just reached (so a patient whose resting ROM reads high still
        re-arms). A 3-sample median filters single-frame landmark spikes."""
        if primary_angle is None or target_rom <= 0:
            with self._rep_lock:
                return self._rep_count

        high_thresh = target_rom * 0.80
        low_thresh  = target_rom * 0.55
        drop        = target_rom * 0.35

        with self._rep_lock:
            self._rom_window.append(primary_angle)
            value = statistics.median(self._rom_window)
            if self._rep_stage is None:
                self._rep_stage = "up" if value >= high_thresh else "down"
                self._rep_peak = value
            elif self._rep_stage == "down":
                if value >= high_thresh:
                    self._rep_count += 1
                    self._rep_stage = "up"
                    self._rep_peak = value
            else:
                if value > self._rep_peak:
                    self._rep_peak = value
                if value < high_thresh and (value <= low_thresh or value <= self._rep_peak - drop):
                    self._rep_stage = "down"
                    self._rep_peak = 0.0
            return self._rep_count

    def update_stability(self, landmarks) -> float:
        """Real stability score from hip-center landmark jitter over a short
        rolling window -- steadier patient = less x/y variance = higher score."""
        try:
            hip_x = (landmarks[23].x + landmarks[24].x) / 2
            hip_y = (landmarks[23].y + landmarks[24].y) / 2
        except Exception:
            return self._current_stability_score
        return self.update_stability_point(hip_x, hip_y)

    def update_stability_point(self, x: float, y: float) -> float:
        """Same jitter score as update_stability() but for any tracked point.
        Hand-grip exercises use the wrist landmark, because a hands-only frame has
        no hips."""
        with self._stability_lock:
            now = time.monotonic()
            buf = self._landmark_jitter_buffer
            buf.append((now, x, y))
            while buf and now - buf[0][0] > _WINDOW_S:
                buf.popleft()
            if len(buf) >= 5:
                xs = [p[1] for p in buf]
                ys = [p[2] for p in buf]
                jitter = statistics.pstdev(xs) + statistics.pstdev(ys)
                # jitter is in normalized [0,1] frame coords; scale empirically to 0-100
                score = max(0.0, min(100.0, 100.0 - jitter * 4000))
                self._current_stability_score = round(score, 1)
            return self._current_stability_score

    def update_smoothness(self, primary_angle: Optional[float]) -> float:
        if primary_angle is None:
            return self._current_smoothness_score

        with self._smoothness_lock:
            now = time.monotonic()
            if self._last_primary_angle is not None and self._last_angle_t is not None:
                dt = min(0.5, max(0.02, now - self._last_angle_t))
                delta = abs(primary_angle - self._last_primary_angle) * (_SMOOTH_REF_DT / dt)
                self._angle_velocity_buffer.append(delta)
            self._last_primary_angle = primary_angle
            self._last_angle_t = now

            if len(self._angle_velocity_buffer) >= 5:
                jerk_variance = statistics.pstdev(self._angle_velocity_buffer)
                # empirically scaled: small deltas (smooth) -> near 100,
                # large erratic swings -> drops toward 0
                score = max(0.0, min(100.0, 100.0 - jerk_variance * 8))
                self._current_smoothness_score = round(score, 1)
            return self._current_smoothness_score

    def update_balance(self, landmarks) -> float:
        try:
            sh_x = (landmarks[11].x + landmarks[12].x) / 2
        except Exception:
            return self._current_balance_score

        with self._balance_lock:
            now = time.monotonic()
            buf = self._shoulder_sway_buffer
            buf.append((now, sh_x))
            while buf and now - buf[0][0] > _WINDOW_S:
                buf.popleft()
            if len(buf) >= 5:
                sway = statistics.pstdev([p[1] for p in buf])
                # normalized [0,1] frame coords; scale empirically to 0-100
                score = max(0.0, min(100.0, 100.0 - sway * 5000))
                self._current_balance_score = round(score, 1)
            return self._current_balance_score

    def _record_rep_quality(self, peak_angle: Optional[float], target_rom: float):
        """Called once per completed rep. Logs how close that rep's peak angle
        got to the target ROM, then derives fatigue from the drop-off between
        the first half and second half of the session's rep quality."""
        if peak_angle is None or target_rom <= 0:
            return

        quality = max(0.0, min(100.0, (peak_angle / target_rom) * 100))
        with self._fatigue_lock:
            self._rep_quality_buffer.append(quality)
            n = len(self._rep_quality_buffer)
            if n >= 4:
                half   = n // 2
                buf    = list(self._rep_quality_buffer)
                early  = buf[:half]
                recent = buf[half:]
                early_avg  = sum(early)  / len(early)
                recent_avg = sum(recent) / len(recent)
                decline = max(0.0, early_avg - recent_avg)  # >0 if quality dropped
                # scale a 0-40pt quality drop across the session to 0-100 fatigue
                self._current_fatigue_score = round(min(100.0, decline * 2.5), 1)

    def maybe_record_rep_quality(self, reps: int, primary_angle: Optional[float], target_rom: float) -> float:
        """Call once per frame with the latest rep count. Records rep quality
        exactly once per newly-completed rep, then returns the current
        fatigue score."""
        with self._fatigue_lock:
            already_recorded = reps <= self._last_fatigue_rep_count
        if not already_recorded:
            self._record_rep_quality(primary_angle, target_rom)
            with self._fatigue_lock:
                self._last_fatigue_rep_count = reps
        return self.get_current_fatigue()

    def reset_state(self):
        """Call right before a session starts so rep count + stability/smoothness/
        balance/fatigue buffers don't carry over stale data from a previous
        session/patient."""
        with self._rep_lock:
            self._rep_count = 0
            self._rep_stage = None
            self._rep_peak = 0.0
            self._rom_window.clear()
        with self._stability_lock:
            self._landmark_jitter_buffer.clear()
            self._current_stability_score = 100.0
        with self._smoothness_lock:
            self._angle_velocity_buffer.clear()
            self._last_primary_angle = None
            self._last_angle_t = None
            self._current_smoothness_score = 100.0
        with self._balance_lock:
            self._shoulder_sway_buffer.clear()
            self._current_balance_score = 100.0
        with self._fatigue_lock:
            self._rep_quality_buffer.clear()
            self._current_fatigue_score = 0.0
            self._last_fatigue_rep_count = 0

"""
Per-user exercise metrics (rep count, per-rep scores, stability, smoothness, balance,
fatigue, active exercise / ROM / side).

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
from typing import Any, Dict, List, Optional


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


def tracked_joints(exercise_type: str, side: str = "both") -> Dict[str, str]:
    """{angles-dict key: report label} for the joints that matter to this exercise.
    `side` is the value stored by set_exercise_state, i.e. already inverted for the mirrored
    camera, so MediaPipe's l_* key is the patient's RIGHT limb and r_* the patient's LEFT."""
    ex = (exercise_type or "").lower()
    s = (side or "both").lower()
    if "balance" in ex:
        return {}
    if "grip" in ex or "finger" in ex:
        return {"index": "Index finger", "middle": "Middle finger", "ring": "Ring finger",
                "pinky": "Pinky finger", "thumb": "Thumb"}
    if "shoulder" in ex or "arm" in ex:
        joint = "shoulder"
    elif "elbow" in ex or "hand" in ex:
        joint = "elbow"
    elif "knee" in ex or "squat" in ex or "leg" in ex:
        joint = "knee"
    elif "hip" in ex:
        joint = "hip"
    elif "ankle" in ex:
        joint = "ankle"
    else:
        joint = "elbow"
    out: Dict[str, str] = {}
    if s in ("left", "both"):       # MediaPipe-left key == patient's right limb
        out[f"l_{joint}"] = f"Right {joint}"
    if s in ("right", "both"):
        out[f"r_{joint}"] = f"Left {joint}"
    return out


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

# Per-rep score = 0.6 * range + 0.2 * stability + 0.2 * smoothness (all 0-100).
#   range      = peak ROM reached DURING that rep / target ROM
#   stability  = average live stability score while that rep was in progress
#   smoothness = average live smoothness score while that rep was in progress
_REP_W_RANGE  = 0.6
_REP_W_STAB   = 0.2
_REP_W_SMOOTH = 0.2
# A rep "window" opens when ROM rises this far above the resting value, and an
# attempt that falls back near rest without reaching 80% of target is dropped.
_WIN_START_FRAC   = 0.15
_WIN_REST_FRAC    = 0.08
_INCOMPLETE_FRAC  = 0.30     # such an attempt must have reached 30% of target to be reported


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

        # Per-rep scoring (all guarded by _rep_lock)
        self._rest_min: Optional[float] = None     # lowest ROM seen since the last rep ended
        self._win_active = False                    # a rep (or attempt) is in progress
        self._win_counted = False                   # ...and it already reached 80% of target
        self._win_peak = 0.0
        self._win_stab_sum = 0.0
        self._win_smooth_sum = 0.0
        self._win_bal_sum = 0.0
        self._win_n = 0
        self._rep_scores: List[float] = []          # final score of every completed rep
        self._rep_details: List[Dict[str, Any]] = []   # EVERY attempt in order (correct + wrong): done / score / rom / speed / feedback
        self._joint_recent: "collections.deque" = collections.deque(maxlen=10)   # last few resting-frame joint snapshots
        self._win_joint: Dict[str, List[float]] = {}   # label -> [min, max] angle seen during the current attempt
        self._win_t0 = 0.0                          # time.time() when the current rep window opened
        self._live_rep_score: Optional[float] = None   # live while a rep runs, frozen after it
        self._incomplete_reps = 0

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

    def get_rep_scoring(self) -> Dict[str, Any]:
        """Per-rep results for the live page: final score of each completed rep, the live
        score of the rep in progress (frozen at its final score once it ends), and how many
        attempts fell back to rest without reaching the target."""
        with self._rep_lock:
            return {
                "rep_scores": list(self._rep_scores),
                "rep_details": [dict(d) for d in self._rep_details],
                "rep_live": self._live_rep_score,
                "rep_live_rom": round(self._win_peak, 1) if self._win_active else None,
                "rep_incomplete": self._incomplete_reps,
            }

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
        re-arms). A 3-sample median filters single-frame landmark spikes.

        The same pass also scores each rep on its own (see _REP_W_*): a rep window opens
        when the joint leaves rest, collects the peak ROM + live stability/smoothness, and
        closes when the joint comes back down. Call update_stability()/update_smoothness()
        every frame too: the window averages their live values."""
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
                self._rest_min = value
            elif self._rep_stage == "down":
                if value >= high_thresh:
                    self._rep_count += 1
                    self._rep_stage = "up"
                    self._rep_peak = value
                    if not self._win_active:
                        self._open_window(value)
                    self._win_counted = True
                    self._window_add(value, target_rom)
                else:
                    self._track_attempt(value, target_rom)
            else:
                if value > self._rep_peak:
                    self._rep_peak = value
                if value < high_thresh and (value <= low_thresh or value <= self._rep_peak - drop):
                    self._rep_stage = "down"
                    self._rep_peak = 0.0
                    self._close_rep(target_rom)
                    self._rest_min = value
                else:
                    self._window_add(value, target_rom)
            return self._rep_count

    # -- per-rep scoring helpers (call with _rep_lock held) -------------------
    def _open_window(self, value: float):
        self._win_active = True
        self._win_counted = False
        self._win_t0 = time.time()
        # the window opens a few frames AFTER the joint starts moving, so seed the min/max
        # from the last few frames before it to catch the start of the movement
        self._win_joint = {}
        for snap in self._joint_recent:
            for label, v in snap.items():
                cur = self._win_joint.get(label)
                if cur is None:
                    self._win_joint[label] = [v, v]
                else:
                    cur[0] = min(cur[0], v); cur[1] = max(cur[1], v)
        self._joint_recent.clear()
        self._win_peak = value
        self._win_stab_sum = 0.0
        self._win_smooth_sum = 0.0
        self._win_bal_sum = 0.0
        self._win_n = 0

    def _window_score(self, target_rom: float) -> Optional[float]:
        if self._win_n <= 0 or target_rom <= 0:
            return None
        range_pct = min(100.0, self._win_peak / target_rom * 100.0)
        stab      = self._win_stab_sum / self._win_n
        smooth    = self._win_smooth_sum / self._win_n
        score = _REP_W_RANGE * range_pct + _REP_W_STAB * stab + _REP_W_SMOOTH * smooth
        return round(max(0.0, min(100.0, score)), 1)

    def _window_add(self, value: float, target_rom: float):
        if not self._win_active:
            return
        if value > self._win_peak:
            self._win_peak = value
        # live stability/smoothness as of the previous frame (plain float reads)
        self._win_stab_sum += self._current_stability_score
        self._win_smooth_sum += self._current_smoothness_score
        self._win_bal_sum += self._current_balance_score
        self._win_n += 1
        live = self._window_score(target_rom)
        if live is not None:
            self._live_rep_score = live

    def _track_attempt(self, value: float, target_rom: float):
        """Joint is below the rep threshold while waiting for the next rep: decide when
        a rep window opens, and when an attempt that never reached the target is dropped."""
        if self._rest_min is None:
            self._rest_min = value
        if self._win_active:
            self._window_add(value, target_rom)
            if value <= self._rest_min + _WIN_REST_FRAC * target_rom:
                # back at rest without reaching the target -> incomplete attempt
                if self._win_peak >= _INCOMPLETE_FRAC * target_rom:
                    self._incomplete_reps += 1
                    self._rep_details.append(self._window_details(target_rom, done=False))
                self._win_active = False
                self._win_counted = False
                self._live_rep_score = self._rep_scores[-1] if self._rep_scores else None
                self._rest_min = min(self._rest_min, value)
            return
        self._rest_min = min(self._rest_min, value)
        if value >= self._rest_min + _WIN_START_FRAC * target_rom:
            self._open_window(value)
            self._window_add(value, target_rom)

    def _window_details(self, target_rom: float, done: bool = True) -> Dict[str, Any]:
        """One row for the report: the attempt that just ended (call with _rep_lock held).
        rom = highest angle reached in that attempt, speed = that angle / attempt time (deg/s).
        done=False is a wrong attempt that fell back to rest without reaching the target."""
        n = max(1, self._win_n)
        rom = round(self._win_peak, 1)
        secs = max(0.1, time.time() - self._win_t0)
        range_pct = (self._win_peak / target_rom * 100.0) if target_rom > 0 else 0.0
        stab = self._win_stab_sum / n
        smooth = self._win_smooth_sum / n
        score = self._window_score(target_rom)
        if not done:
            fb = f"Wrong rep - reached only {range_pct:.0f}% of target"
        elif range_pct < 90:
            fb = "Raise higher"
        elif stab < 70:
            fb = "Keep body steady"
        elif smooth < 70:
            fb = "Move smoother"
        else:
            fb = "Good rep"
        joints = {name: round(hi - lo, 1) for name, (lo, hi) in self._win_joint.items()}
        # Hands-only frame (hand grip / finger) has no body landmarks -> no balance signal.
        ex = (self._current_exercise_type or "").lower()
        bal = None if ("grip" in ex or "finger" in ex) else round(self._win_bal_sum / n, 1)
        return {"done": done, "score": score if score is not None else 0.0, "rom": rom,
                "speed": round(rom / secs, 1), "duration": round(secs, 1), "feedback": fb,
                "stability": round(stab, 1), "smoothness": round(smooth, 1), "balance": bal,
                "joints": joints}

    def _close_rep(self, target_rom: float):
        """The joint came back down after a counted rep: freeze that rep's final score."""
        if self._win_active and self._win_counted:
            score = self._window_score(target_rom)
            if score is not None:
                self._rep_scores.append(score)
                self._live_rep_score = score
                self._rep_details.append(self._window_details(target_rom))
        self._win_active = False
        self._win_counted = False

    def update_joint_angles(self, angles: Dict[str, float], exercise_type: str, side: str = "both"):
        """Call every frame AFTER update_rep_count(). While a rep/attempt is in progress it keeps the
        lowest and highest angle of each tracked joint, so the report can show how far every joint
        moved in that rep (max - min) instead of one end-of-session snapshot."""
        if not angles:
            return
        tracked = tracked_joints(exercise_type, side)
        with self._rep_lock:
            if not self._win_active:
                snap = {label: angles[key] for key, label in tracked.items() if angles.get(key) is not None}
                if snap:
                    self._joint_recent.append(snap)
                return
            for key, label in tracked.items():
                v = angles.get(key)
                if v is None:
                    continue
                cur = self._win_joint.get(label)
                if cur is None:
                    self._win_joint[label] = [v, v]
                else:
                    if v < cur[0]: cur[0] = v
                    if v > cur[1]: cur[1] = v

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

    def _record_rep_quality(self, quality: float):
        """Called once per completed rep with that rep's final score (0-100). Fatigue is
        the drop-off between the first half and second half of the session's rep scores."""
        quality = max(0.0, min(100.0, float(quality)))
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
        """Call once per frame. Records each newly COMPLETED rep's score exactly once
        (the arguments are kept for the existing callers), then returns the current
        fatigue score."""
        with self._rep_lock:
            done = len(self._rep_scores)
            last = self._rep_scores[-1] if done else None
        with self._fatigue_lock:
            fresh = done > self._last_fatigue_rep_count and last is not None
        if fresh:
            self._record_rep_quality(last)
            with self._fatigue_lock:
                self._last_fatigue_rep_count = done
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
            self._rest_min = None
            self._win_active = False
            self._win_counted = False
            self._win_peak = 0.0
            self._win_stab_sum = 0.0
            self._win_smooth_sum = 0.0
            self._win_bal_sum = 0.0
            self._win_n = 0
            self._rep_scores = []
            self._rep_details = []
            self._joint_recent.clear()
            self._win_joint = {}
            self._live_rep_score = None
            self._incomplete_reps = 0
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
"""
Rehab Runner — game engine 
"""
import math
import threading
from collections import deque
from typing import Dict, Optional

import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


# ----------------------------------------------------------------------------
# CONFIG — detection thresholds (unchanged from the original rehab_runner.py)
# ----------------------------------------------------------------------------

VIS_MIN = 0.4           # min landmark visibility for the strict calibration check (was 0.5)
PLAY_VIS_MIN = 0.3      # min visibility to keep PLAYING - the same threshold the camera pipeline uses
                        # to track/draw a joint (was 0.5, which paused the game whenever a knee/ankle
                        # dipped just below it)
PERSON_GRACE_S = 1.2    # once calibrated, a landmark dropout shorter than this does NOT pause the game

# All movement distances are measured in "torso lengths" (shoulder-centre ->
# hip-centre distance measured during calibration). This makes every
# threshold independent of how far the patient stands from the camera and
# of body size. Everything is measured RELATIVE TO THE CALIBRATED NEUTRAL
# STANDING POSITION.

# --- camera calibration / framing guide ---------------------------------
EDGE_MARGIN = 0.02      # landmarks closer than this to the frame edge count as "cut off"
BODY_H_MAX = 0.82       # nose->ankle span (fraction of frame height) above this = too close
BODY_H_MIN = 0.45       # ... below this = too far
CENTER_TOL = 0.18       # body centre must be within this (fraction of frame width) of the middle
CALIB_SECONDS = 2.0     # neutral-stance measuring time
CALIB_STILL_T = 0.12    # hips may wander at most this (torso lengths) while calibrating
CALIB_LOST_S = 0.4      # body lost this long while calibrating -> restart measuring

# --- smoothing ------------------------------------------------------------
EMA = 0.65              # landmark position smoothing (0..1, higher = less smoothing / less lag)
EMA_ANGLE = 0.6         # knee-angle smoothing

# --- STEP detection (the game is driven by these) --------------------------
# A STEP = ANY movement of a foot: the ankle moves away from where that foot
# was resting (up, sideways, forward - marching, tapping, shuffling,
# side-stepping). The ankle is measured in the camera image, so leaning the
# body alone (feet planted) is NOT a step.
FOOT_MOVE_T = 0.07      # ankle must move this far (torso lengths, ~3-4 cm) from its resting spot = 1 step
FOOT_STILL_D = 0.04     # a foot that moves less than this between frames counts as "still" ...
FOOT_STILL_S = 0.20     # ... after this long the foot is "landed" and its new spot becomes the resting spot
FOOT_EMA = 0.45         # extra smoothing on the ankle position (kills landmark jitter)
STEP_MIN_GAP_S = 0.10   # two steps closer together than this are counted as one
STEP_MAX_RISE = 0.12    # hips this far ABOVE neutral = a jump, not a step
STEP_BLOCK_S = 0.30     # ignore foot movement this long after a crouch starts / ends (feet slide then)
STEP_SETTLE_S = 0.6     # after a foot lands, keep reading the body X for this long (then hold it)

# --- LEFT / RIGHT : position is read only while stepping --------------------
LATERAL_DEAD = 0.05     # body-centre shifts smaller than this are ignored (tracking noise)
LATERAL_RANGE = 0.45    # shift (torso lengths) that puts the player on the outer lane edge
LANE_SWITCH = 0.60      # lane changes when the player is this far (in lane widths) from its lane centre
LANE_FRAMES = 2         # ... and stays there this many frames

# --- CROUCH : both knees bent AND hips lower ---------------------------------
CROUCH_BEND_T = 18.0    # BOTH knee angles must be this many degrees below the neutral knee angles ...
CROUCH_DROP_T = 0.06    # ... AND hips dropped by this much (torso lengths)
CROUCH_DROP_STRONG = 0.16   # a very large hip drop counts even if the knee angle is hard to measure
CROUCH_HOLD_S = 0.06    # must be held this long
CROUCH_FRAMES = 2       # ... and this many frames
CROUCH_REL_BEND = 10.0  # release: knee angle back within this many degrees of neutral ...
CROUCH_REL_DROP = 0.035 # ... or hips back up to within this of neutral
CROUCH_REL_FRAMES = 2   # release must hold this many frames

CALIB_KEYS = ("cx", "cy", "hip_x", "hip_y", "sh_x", "sh_y", "torso",
              "kl", "kr", "knee", "la_y", "ra_y")
PARTS = ("head", "shoulders", "hip", "knees", "feet")


# ----------------------------------------------------------------------------
# Pure math helpers (unchanged)
# ----------------------------------------------------------------------------

def calculate_angle(a, b, c):
    """Angle in degrees at point b formed by the segments b->a and b->c (2D or 3D)."""
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def leg_angle_from_vertical(hip, ankle):
    """LEFT/RIGHT LEG ANGLE: angle between the straight-down direction and the
    hip->ankle line. ~0 deg = leg straight down, grows when the leg is lifted
    forward or moved sideways. (Kept different from the knee angle on purpose.)"""
    straight_down = (hip[0], hip[1] + 1.0)
    return calculate_angle(straight_down, hip, ankle)


def calculate_speed(prev_pt, curr_pt, dt, body_scale):
    """Movement speed in torso-lengths per second (distance between two frames
    divided by time, normalised by body size so it works at any distance)."""
    if prev_pt is None or dt <= 0 or body_scale <= 0:
        return 0.0
    dist = math.hypot(curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
    return dist / body_scale / dt


def lateral_to_norm(lateral):
    """Map the body-centre shift (torso lengths, - = patient's left) to the
    player's horizontal position in lane units: -1 = left lane centre,
    0 = middle lane centre, +1 = right lane centre. Continuous, with a small
    dead-zone so tracking noise does not make the player wobble."""
    a = abs(lateral)
    if a <= LATERAL_DEAD:
        return 0.0
    n = min(1.0, (a - LATERAL_DEAD) / (LATERAL_RANGE - LATERAL_DEAD))
    return math.copysign(n, lateral)


def calculate_response_time(obstacle_event_time, movement_time):
    """Seconds between the obstacle 'requiring' a reaction and the detected movement."""
    if obstacle_event_time is None or movement_time is None:
        return None
    return max(0.0, movement_time - obstacle_event_time)


class RehabRunnerEngine:
    """Per-session game engine. One instance per active game session (same
    lifetime as thero's own metrics.py module-state — reset() between
    patients/sessions).

    Not a Thread: the caller (thero's camera loop) calls process_frame() once
    per frame it already has landmarks for. No camera, no HTTP, no DB here.
    """

    def __init__(self):
        self.lock = threading.Lock()

        self.events = deque(maxlen=200)
        self.event_seq = 0

        self.status = {
            "flip": False,
            "person": "none",           # none | partial | ok  (enough body visible to PLAY)
            "body_ok": False,           # full body visible, centred, right distance
            "parts": {p: False for p in PARTS},
            "guidance": "No person detected. Please stand in front of the camera.",
            "message": "Waiting for camera...",
            "level": "warn",            # ok | warn | error
            "calibrated": False,
            "calibrating": False,
            "calib_progress": 0.0,
            # live movement values
            "left_leg_angle": 0.0, "right_leg_angle": 0.0, "knee_angle": 180.0,
            "movement_speed": 0.0,
            "lateral": 0.0,             # body-centre shift, torso lengths (- = left)
            "x_norm": 0.0,              # player position -1 (left) .. +1 (right), held between steps
            "lane": 1,                  # 0 / 1 / 2
            "rise": 0.0,                # hip height above neutral (torso lengths)
            "bend": 0.0,                # knee bend below neutral (degrees, the smaller of the two knees)
            "crouching": False,
            "steps": 0,                 # steps counted since calibration
            "left_steps": 0,            # left-foot steps counted since calibration
            "right_steps": 0,           # right-foot steps counted since calibration
            "step_foot": "",            # "L" / "R" while that foot is lifted, else ""
        }

        # thero's camera pipeline mirrors the frame (cv2.flip(frame, 1) in
        # mjpeg_camera.py) BEFORE MediaPipe runs, so landmark x already
        # matches on-screen/patient-facing position — no extra flip needed
        # here. Confirmed against the reported "avatar moves opposite to
        # patient" bug: flip was defaulting to True, double-flipping x.
        self.flip = False
        self.aspect = 1280.0 / 720.0

        self.body_ok = False
        self.body_h = None
        self.body_lost_since = None
        self.last_ok_t = 0.0            # last time the body was fully tracked (for PERSON_GRACE_S)
        self.prev_t = 0.0
        self.speed_hist = deque(maxlen=60)
        self._reset_motion()

    # -- state -----------------------------------------------------------
    def _reset_motion(self):
        """Forget the neutral reference and all movement state."""
        self.sp, self.ksm = {}, {}
        self.base = None
        self.prev = None
        self.lane, self.lane_cnt = 1, 0
        self.crouching, self.crouch_since = False, None
        self.crouch_cnt, self.uncrouch_cnt = 0, 0
        self.calib_start, self.calib_samples = None, []
        # step state
        self.step_foot = None           # None / "L" / "R" = which foot is moving right now
        self.fs, self.foot = {}, {}     # smoothed ankle positions / per-foot step state
        self.step_block_until = 0.0
        self.step_t = 0.0               # time of the last counted step
        self.step_active_until = 0.0    # body X is read until this time, then held
        self.x_hold = 0.0               # held player position (lane units)
        self.step_total = 0
        self.step_total_left = 0
        self.step_total_right = 0
        with self.lock:
            self.status.update(x_norm=0.0, lane=1, crouching=False, lateral=0.0,
                                rise=0.0, bend=0.0, calib_progress=0.0,
                                steps=0, left_steps=0, right_steps=0, step_foot="")

    def reset(self):
        """Call right before a new game session starts, so calibration and
        step/lane/crouch state don't carry over from a previous
        patient/session (mirrors services/metrics.py's reset_state())."""
        self._reset_motion()
        self.body_ok = False
        self.body_h = None
        self.body_lost_since = None
        self.last_ok_t = 0.0
        self.prev = None
        self.prev_t = 0.0
        self.speed_hist.clear()
        self.events.clear()
        self.event_seq = 0
        with self.lock:
            self.status.update(calibrated=False, calibrating=False,
                                message="Waiting for camera...", level="warn")

    # -- public API used by the session-page routes -----------------------
    def request_calibration(self):
        """Start measuring the neutral standing position now."""
        self._reset_motion()
        with self.lock:
            self.status.update(calibrated=False, calibrating=True, calib_progress=0.0)

    def reset_calibration(self):
        """Back to 'not calibrated' (used when the calibration screen opens)."""
        with self.lock:
            self.status.update(calibrated=False, calibrating=False, calib_progress=0.0)

    def set_flip(self, flip: bool):
        """Flip the tracked-side mapping (fixes 'player moves the wrong
        way'). The neutral stance must be re-measured because the x axis
        changes sign."""
        with self.lock:
            self.flip = bool(flip)
            self.status["flip"] = self.flip
            had = self.status["calibrating"] or self.status["calibrated"]
        self._reset_motion()
        with self.lock:
            self.status.update(calibrated=False, calibrating=had, calib_progress=0.0)

    def get_status(self) -> dict:
        with self.lock:
            return dict(self.status)

    def get_events_since(self, since: int):
        with self.lock:
            return [e for e in self.events if e["id"] > since]

    def get_event(self, event_id):
        with self.lock:
            for e in self.events:
                if e["id"] == event_id:
                    return dict(e)
        return None

    def get_summary(self) -> dict:
        """Server-tracked totals at session end. Score/lives are a front-end
        (JS game-loop) concept, same as in the original rehab_runner.py —
        those arrive from the client at session-end and get combined with
        this summary when the session gets saved (session-page integration
        step, not done in this file)."""
        with self.lock:
            return {
                "steps": self.step_total,
                "left_steps": self.step_total_left,
                "right_steps": self.step_total_right,
                "calibrated": self.status["calibrated"],
                "events": list(self.events),
            }

    # -- per-frame entry point, called by thero's camera loop -------------
    def process_frame(self, landmarks, world_landmarks, now: float, frame_shape=None):
        """landmarks: results.pose_landmarks.landmark from thero's own
        MediaPipe Pose call (mjpeg_camera.py) for this frame, or None if no
        person was found. world_landmarks: results.pose_world_landmarks
        (optional, improves knee-angle accuracy — falls back to 2D if not
        given). frame_shape: the frame's (h, w, ...) so aspect ratio tracks
        the actual camera resolution."""
        if mp is None:
            with self.lock:
                self.status.update(message="mediapipe is not installed", level="error")
            return

        if frame_shape:
            fh, fw = frame_shape[:2]
            if fh > 0:
                self.aspect = fw / float(fh)

        person, checks = "none", None
        if landmarks is not None:
            checks = self._check_body(landmarks)
            person = "ok" if self._body_visible(landmarks) else "partial"

        self.body_ok = bool(checks and checks["ok"])
        self.body_h = checks["body_h"] if checks else None
        self._track_calib_loss(now)

        if person == "ok":
            self.last_ok_t = now
            try:
                self._handle_frame(landmarks, world_landmarks, now)
            except Exception as exc:
                print(f"[RehabRunnerEngine] frame skipped: {exc}")

        # A single frame (or a fraction of a second) where MediaPipe loses a knee/ankle or
        # returns no pose is normal, especially on a slower server. Once calibrated, report
        # such a dropout as "ok" for PERSON_GRACE_S: the engine just keeps its last state
        # (nothing is processed for that frame), instead of the page pausing the game behind a
        # "Step back into frame" overlay that flickers on and off.
        shown = person
        if person != "ok" and self.last_ok_t and now - self.last_ok_t <= PERSON_GRACE_S:
            with self.lock:
                if self.status["calibrated"]:
                    shown = "ok"
        self._update_message(shown, checks)

    def _update_message(self, person, checks):
        with self.lock:
            st = self.status
            if checks is None:
                parts = {p: False for p in PARTS}
                guidance = "No person detected. Please stand in front of the camera."
            else:
                parts, guidance = checks["parts"], checks["guidance"]
            st.update(person=person, parts=parts, guidance=guidance, body_ok=self.body_ok)
            if person == "none":
                st.update(message="No person detected. Please stand in front of the camera.", level="warn")
            elif st["calibrating"]:
                st.update(message=("Calibrating - stand straight and still..." if self.body_ok else guidance),
                          level="warn")
            elif not st["calibrated"]:
                st.update(message=("Body detected. Ready to calibrate." if self.body_ok else guidance),
                          level="ok" if self.body_ok else "warn")
            elif person == "partial":
                st.update(message="Step back: hips, knees and ankles must be visible.", level="warn")
            else:
                st.update(message="Pose tracking OK", level="ok")

    def _emit(self, mtype, now, m, speed):
        """Publish one detected movement (used for the report + response time)."""
        peak = max((s for t, s in self.speed_hist if now - t <= 0.6), default=speed)
        with self.lock:
            self.event_seq += 1
            self.events.append({"id": self.event_seq, "type": mtype, "t": now,
                                 "left_leg_angle": m["left_leg"], "right_leg_angle": m["right_leg"],
                                 "knee_angle": m["knee"], "movement_speed": peak})

    def _track_calib_loss(self, now):
        """Restart the neutral measurement if the body is lost for a while."""
        if self.body_ok:
            self.body_lost_since = None
            return
        if self.calib_samples or self.calib_start is not None:
            if self.body_lost_since is None:
                self.body_lost_since = now
            elif now - self.body_lost_since > CALIB_LOST_S:
                self.calib_start, self.calib_samples = None, []
                self.body_lost_since = None
                with self.lock:
                    self.status["calib_progress"] = 0.0

    # -- body check (calibration screen) -----------------------------------
    @staticmethod
    def _body_visible(lm):
        """Enough of the body to PLAY (relaxed - the head may leave the frame)."""
        L = mp.solutions.pose.PoseLandmark
        need = [L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP,
                L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE]
        return all(lm[i.value].visibility >= PLAY_VIS_MIN for i in need)

    def _check_body(self, lm):
        """Strict full-body check used by the calibration screen: head,
        shoulders, hip, knees and feet must be visible and inside the frame,
        the patient must be centred and at a sensible distance."""
        L = mp.solutions.pose.PoseLandmark
        groups = {
            "head": (L.NOSE,),
            "shoulders": (L.LEFT_SHOULDER, L.RIGHT_SHOULDER),
            "hip": (L.LEFT_HIP, L.RIGHT_HIP),
            "knees": (L.LEFT_KNEE, L.RIGHT_KNEE),
            "feet": (L.LEFT_ANKLE, L.RIGHT_ANKLE, L.LEFT_FOOT_INDEX, L.RIGHT_FOOT_INDEX),
        }

        def good(i):
            p = lm[i.value]
            return (p.visibility >= VIS_MIN and EDGE_MARGIN <= p.x <= 1 - EDGE_MARGIN
                    and EDGE_MARGIN <= p.y <= 1 - EDGE_MARGIN)

        parts = {name: all(good(i) for i in ids) for name, ids in groups.items()}
        body_h, ok = None, False
        if not all(parts.values()):
            guidance = "Please move backward. Full body is not detected."
        else:
            ankle_y = (lm[L.LEFT_ANKLE.value].y + lm[L.RIGHT_ANKLE.value].y) / 2
            body_h = ankle_y - lm[L.NOSE.value].y
            xs = [lm[i.value].x for i in (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP)]
            cx = sum(xs) / 4.0
            if body_h > BODY_H_MAX:
                guidance = "Please move backward. You are too close."
            elif body_h < BODY_H_MIN:
                guidance = "Please move closer. You are too far."
            elif abs(cx - 0.5) > CENTER_TOL:
                guidance = "Please stand in the center of the frame."
            else:
                guidance, ok = "Full body detected.", True
        return {"parts": parts, "ok": ok, "guidance": guidance, "body_h": body_h}

    # -- pose processing -----------------------------------------------------
    def _points(self, lm):
        """Raw landmark positions for one frame in selfie view (patient's
        left = screen left when flip is on), aspect-corrected so x/y units
        are equal."""
        L = mp.solutions.pose.PoseLandmark
        ids = {"ls": L.LEFT_SHOULDER, "rs": L.RIGHT_SHOULDER, "lh": L.LEFT_HIP, "rh": L.RIGHT_HIP,
               "lk": L.LEFT_KNEE, "rk": L.RIGHT_KNEE, "la": L.LEFT_ANKLE, "ra": L.RIGHT_ANKLE}
        out = {}
        for key, i in ids.items():
            p = lm[i.value]
            x = (1.0 - p.x) if self.flip else p.x
            out[key] = (x * self.aspect, p.y)
        return out

    @staticmethod
    def _world_knees(world):
        """Knee angles (left, right) from the 3D world landmarks, or (None, None)."""
        if world is None:
            return None, None
        try:
            L = mp.solutions.pose.PoseLandmark

            def W(i):
                p = world.landmark[i.value]
                return (p.x, p.y, p.z)
            return (calculate_angle(W(L.LEFT_HIP), W(L.LEFT_KNEE), W(L.LEFT_ANKLE)),
                    calculate_angle(W(L.RIGHT_HIP), W(L.RIGHT_KNEE), W(L.RIGHT_ANKLE)))
        except Exception:
            return None, None

    def _handle_frame(self, lm, world, now):
        kl, kr = self._world_knees(world)
        self._process(now, self._points(lm), kl, kr)

    @staticmethod
    def _derive(sp, ks):
        """Body signals from the SMOOTHED landmark positions."""
        ls, rs, lh, rh = sp["ls"], sp["rs"], sp["lh"], sp["rh"]
        lk, rk, la, ra = sp["lk"], sp["rk"], sp["la"], sp["ra"]
        hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        sh = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        return {
            "cx": (hip[0] + sh[0]) / 2, "cy": (hip[1] + sh[1]) / 2,    # body centre (torso middle)
            "hip_x": hip[0], "hip_y": hip[1], "sh_x": sh[0], "sh_y": sh[1],
            "torso": math.hypot(hip[0] - sh[0], hip[1] - sh[1]),
            "left_leg": leg_angle_from_vertical(lh, la),
            "right_leg": leg_angle_from_vertical(rh, ra),
            "kl": ks["kl"], "kr": ks["kr"], "knee": (ks["kl"] + ks["kr"]) / 2.0,
            "la_y": la[1], "ra_y": ra[1],
            "lk_x": lk[0], "lk_y": lk[1], "rk_x": rk[0], "rk_y": rk[1],
        }

    def _process(self, now, raw, kl_raw=None, kr_raw=None):
        """One frame of movement analysis.
        raw: dict of raw (x, y) points ls rs lh rh lk rk la ra (aspect-corrected)."""
        # 1) smooth landmark positions (EMA)
        sp = self.sp
        for k, (x, y) in raw.items():
            if k not in sp:
                sp[k] = [x, y]
            else:
                sp[k][0] += EMA * (x - sp[k][0])
                sp[k][1] += EMA * (y - sp[k][1])
        if kl_raw is None or kr_raw is None:        # 2D fallback
            kl_raw = calculate_angle(raw["lh"], raw["lk"], raw["la"])
            kr_raw = calculate_angle(raw["rh"], raw["rk"], raw["ra"])
        ks = self.ksm
        for k, v in (("kl", kl_raw), ("kr", kr_raw)):
            ks[k] = v if k not in ks else ks[k] + EMA_ANGLE * (v - ks[k])
        m = self._derive(sp, ks)

        # 2) movement speed = fastest of hip centre / both knees (torso-lengths per s)
        pts = {"hip": (m["hip_x"], m["hip_y"]), "lk": (m["lk_x"], m["lk_y"]), "rk": (m["rk_x"], m["rk_y"])}
        torso = self.base["torso"] if self.base else max(m["torso"], 1e-3)
        speed = 0.0
        if self.prev is not None and now > self.prev_t:
            dt = now - self.prev_t
            speed = max(calculate_speed(self.prev[k], pts[k], dt, torso) for k in pts)
        speed = min(speed, 15.0)
        self.prev, self.prev_t = pts, now
        self.speed_hist.append((now, speed))

        upd = {"left_leg_angle": m["left_leg"], "right_leg_angle": m["right_leg"],
               "knee_angle": m["knee"], "movement_speed": speed}

        with self.lock:
            calibrating = self.status["calibrating"]
            calibrated = self.status["calibrated"]

        # 3) calibration: median of the neutral standing posture
        if calibrating:
            if not self.body_ok:                     # full body must be visible while measuring
                with self.lock:
                    self.status.update(upd)
                return
            if self.calib_start is None:
                self.calib_start = now
            self.calib_samples.append({k: m[k] for k in CALIB_KEYS})
            progress = min(1.0, (now - self.calib_start) / CALIB_SECONDS)
            upd["calib_progress"] = progress
            if progress >= 1.0 and len(self.calib_samples) >= 10:
                cols = {k: np.array([s[k] for s in self.calib_samples]) for k in CALIB_KEYS}
                base = {k: float(np.median(v)) for k, v in cols.items()}
                base["torso"] = max(base["torso"], 1e-3)
                wander = max(float(np.ptp(cols["hip_x"])), float(np.ptp(cols["hip_y"]))) / base["torso"]
                if wander > CALIB_STILL_T:           # patient was moving -> measure again
                    self.calib_start, self.calib_samples = None, []
                    upd["calib_progress"] = 0.0
                else:
                    base["body_h"] = self.body_h
                    self.base = base
                    self.lane, self.lane_cnt = 1, 0
                    self.crouching, self.crouch_since = False, None
                    self.crouch_cnt, self.uncrouch_cnt = 0, 0
                    self.step_foot, self.step_t, self.step_active_until = None, now, 0.0
                    self.fs, self.foot, self.step_block_until = {}, {}, 0.0
                    self.x_hold, self.step_total = 0.0, 0
                    self.step_total_left, self.step_total_right = 0, 0
                    upd.update(calibrating=False, calibrated=True, calib_progress=1.0,
                               x_norm=0.0, lane=1, crouching=False, steps=0,
                               left_steps=0, right_steps=0, step_foot="")
            with self.lock:
                self.status.update(upd)
            return
        if not calibrated or self.base is None:
            with self.lock:
                self.status.update(upd)
            return

        # 4) body signals relative to the calibrated neutral stance
        b = self.base
        lateral = (m["cx"] - b["cx"]) / b["torso"]                 # - = patient's left
        rise = (b["hip_y"] - m["hip_y"]) / b["torso"]              # + = body moved up, - = hips lower
        drop = -rise
        bend = min(b["kl"] - m["kl"], b["kr"] - m["kr"])           # BOTH knees must bend (degrees below neutral)
        # foot height difference vs. neutral: + = LEFT foot higher, - = RIGHT foot higher
        diff = (m["ra_y"] - m["la_y"]) / b["torso"] - (b["ra_y"] - b["la_y"]) / b["torso"]

        # ---- STEP: ANY foot movement (ankle moves away from its resting spot) ----
        if rise >= STEP_MAX_RISE:                    # a jump: the take-off AND the landing are not steps
            self.step_block_until = now + STEP_BLOCK_S
        allowed = now >= self.step_block_until
        started = []
        for k in ("la", "ra"):
            x, y = raw[k]
            if k not in self.fs:
                self.fs[k] = [x, y]
            else:
                self.fs[k][0] += FOOT_EMA * (x - self.fs[k][0])
                self.fs[k][1] += FOOT_EMA * (y - self.fs[k][1])
            pos = (self.fs[k][0] / b["torso"], self.fs[k][1] / b["torso"])
            if self._foot_moved(k, pos, now, allowed):
                started.append(k)
        mv_l, mv_r = self.foot["la"]["moving"], self.foot["ra"]["moving"]
        if mv_l and mv_r:
            self.step_foot = "L" if self.foot["la"]["d"] >= self.foot["ra"]["d"] else "R"
        else:
            self.step_foot = "L" if mv_l else ("R" if mv_r else None)
        for k in started:
            if now - self.step_t >= STEP_MIN_GAP_S:
                self.step_t = now
                self.step_total += 1
                if k == "la":
                    self.step_total_left += 1
                else:
                    self.step_total_right += 1
                self._emit("STEP_" + ("L" if k == "la" else "R"), now, m, speed)
        if self.step_foot is not None or started:
            self.step_active_until = now + STEP_SETTLE_S
        stepping = self.step_foot is not None or now < self.step_active_until

        # follow slow posture drift (camera bump, weight shift) while standing still and neutral
        if (not stepping and not self.crouching and abs(rise) < 0.06 and bend < 15.0
                and abs(lateral) < LATERAL_DEAD):
            for k, cur in (("cx", m["cx"]), ("hip_y", m["hip_y"])):
                b[k] += 0.02 * (cur - b[k])

        # ---- LEFT / RIGHT: the body X is read ONLY while stepping, then held ----
        if stepping:
            self.x_hold = lateral_to_norm(lateral)
        norm = self.x_hold
        cand = self.lane
        if abs(norm - (self.lane - 1)) > LANE_SWITCH:
            cand = min(2, max(0, int(round(norm)) + 1))
        if cand != self.lane:
            self.lane_cnt += 1
            if self.lane_cnt >= LANE_FRAMES:
                mtype = "LEFT" if cand < self.lane else "RIGHT"
                self.lane, self.lane_cnt = cand, 0
                self._emit(mtype, now, m, speed)
        else:
            self.lane_cnt = 0

        # ---- CROUCH: both knees bent AND hips lower, held; release when standing up ----
        crouch_on = drop > CROUCH_DROP_T and (bend > CROUCH_BEND_T or drop > CROUCH_DROP_STRONG)
        crouch_hold = drop > CROUCH_REL_DROP and (bend > CROUCH_REL_BEND or drop > CROUCH_DROP_STRONG * 0.7)
        if not self.crouching:
            if crouch_on:
                self.crouch_cnt += 1
                if self.crouch_since is None:
                    self.crouch_since = now
                if self.crouch_cnt >= CROUCH_FRAMES and now - self.crouch_since >= CROUCH_HOLD_S:
                    self.crouching, self.uncrouch_cnt = True, 0
                    self.step_block_until = now + STEP_BLOCK_S
                    self._emit("CROUCH", now, m, speed)
            else:
                self.crouch_cnt, self.crouch_since = 0, None
        else:
            if crouch_hold:
                self.uncrouch_cnt = 0
            else:
                self.uncrouch_cnt += 1
                if self.uncrouch_cnt >= CROUCH_REL_FRAMES:
                    self.step_block_until = now + STEP_BLOCK_S
                    self.crouching, self.crouch_since = False, None
                    self.crouch_cnt, self.uncrouch_cnt = 0, 0

        upd.update(lateral=lateral, x_norm=norm, lane=self.lane, rise=rise, bend=bend,
                   crouching=self.crouching, steps=self.step_total,
                   left_steps=self.step_total_left, right_steps=self.step_total_right,
                   step_foot=self.step_foot or "")
        with self.lock:
            self.status.update(upd)

    def _foot_moved(self, k, pos, now, allowed):
        """Per-foot step detector. pos = ankle position in torso lengths.
        Returns True on the frame a NEW foot movement (= one step) starts."""
        f = self.foot.get(k)
        if f is None:
            self.foot[k] = {"anchor": pos, "moving": False, "sp": pos, "st": now, "d": 0.0}
            return False
        d = math.hypot(pos[0] - f["anchor"][0], pos[1] - f["anchor"][1])
        f["d"] = d
        if not f["moving"]:
            if d > FOOT_MOVE_T:
                if allowed:
                    f["moving"], f["sp"], f["st"] = True, pos, now
                    return True
                f["anchor"] = pos                    # jump / crouch change: not a step, re-anchor
            else:                                    # resting foot: follow slow drift
                f["anchor"] = (f["anchor"][0] + 0.05 * (pos[0] - f["anchor"][0]),
                               f["anchor"][1] + 0.05 * (pos[1] - f["anchor"][1]))
            return False
        if d < FOOT_MOVE_T * 0.5:                    # foot came back to where it rested -> ready for the next step
            f["moving"] = False
        elif math.hypot(pos[0] - f["sp"][0], pos[1] - f["sp"][1]) > FOOT_STILL_D:
            f["sp"], f["st"] = pos, now
        elif now - f["st"] >= FOOT_STILL_S:          # foot landed somewhere new -> that is its resting spot now
            f["moving"], f["anchor"] = False, pos
        return False
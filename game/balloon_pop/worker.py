"""
Balloon Pop - shoulder rehab game engine.
"""
import math
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
VIS_MIN = 0.45          # min landmark visibility for a tracked arm
REST_ANGLE = 25.0       # arm below this elevation = "at rest"
REST_HOLD = 0.35        # seconds at rest before the next balloon spawns
MAX_DT = 0.25           # a longer gap between frames is treated as this long
COUNTDOWN_S = 3.0

# sizes below were pixels on a 720 px tall canvas in the prototype
_PX = 720.0
M_SIDE, M_TOP, M_BOTTOM = 110 / _PX, 135 / _PX, 90 / _PX
R_MIN, R_MAX = 30 / _PX, 95 / _PX
HAND_R_MIN, HAND_R_MAX = 14 / _PX, 28 / _PX
FLOAT_AWAY_SPEED = 240 / _PX      # scene units / s after a miss
FLOAT_AWAY_S = 0.8

PLANE_NAME = {"abd": "Abduction", "flex": "Flexion"}
MODE_PLANES = {"abd": ["abd"], "flex": ["flex"], "mixed": ["abd", "flex"]}
ARM_SIDES = {"left": ["left"], "right": ["right"], "both_alt": ["left", "right"]}
ORDER_NAMES = ["Random", "Low to high", "High to low"]
N_COLORS = 6            # the browser owns the palette; engine sends an index

DEFAULT_CFG = dict(
    mode="abd", arm="right",
    lvl_low=45, lvl_mid=70, lvl_high=92,    # balloon heights, % of calibrated max
    order=0,                                # 0 random, 1 low->high, 2 high->low
    size=100, reach=100,                    # balloon size / distance from shoulder, %
    dwell=250,                              # ms the hand must stay on a balloon
    timeout=15,                             # s before a balloon counts as missed
    gate=10,                                # deg tolerance below the target angle
    lean=15,                                # trunk lean limit, deg (0 = off)
    man_abd=0, man_flex=0,                  # manual max angle, deg (0 = calibrate)
)
_RANGES = dict(lvl_low=(20, 95), lvl_mid=(25, 98), lvl_high=(30, 100), order=(0, 2),
               size=(60, 160), reach=(70, 130), dwell=(100, 1000), timeout=(5, 40),
               gate=(0, 30), lean=(5, 30), man_abd=(40, 180), man_flex=(40, 180))
_ZERO_OK = ("lean", "man_abd", "man_flex")


_CHOICES = dict(
    arm=[("left", "Left arm"), ("right", "Right arm"), ("both_alt", "Both - alternate")],
    mode=[("abd", "Abduction (side)"), ("flex", "Flexion (forward)"), ("mixed", "Mixed")],
)


def config_schema() -> list:
    """Field descriptions the session page builds its options UI from.
    Only the arm and the movement are chosen on the page; every other value
    keeps DEFAULT_CFG."""
    return [
        dict(key=key, label=label, type="choice", group="main", default=DEFAULT_CFG[key],
             options=[dict(value=v, label=l) for v, l in _CHOICES[key]])
        for key, label in (("arm", "Arm"), ("mode", "Movement"))
    ]

# MediaPipe pose indices (two groups; which one is "left" is decided per frame)
_GRP_A = dict(sh=11, el=13, wr=15, pk=17, ix=19)
_GRP_B = dict(sh=12, el=14, wr=16, pk=18, ix=20)
_HIP_L, _HIP_R = 23, 24


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _deg(x):
    return math.degrees(x)


def _r4(v):
    return [round(float(x), 4) for x in v]


class _EMA:
    def __init__(self, alpha):
        self.alpha, self.v = alpha, None

    def __call__(self, x):
        x = np.asarray(x, float)
        self.v = x if self.v is None else self.alpha * x + (1 - self.alpha) * self.v
        return self.v


class _Hold:
    """Detects a steady hold (value stays within a tolerance) for `duration` seconds."""

    def __init__(self, duration, tol):
        self.duration, self.tol, self.buf = duration, tol, []

    def reset(self):
        self.buf = []

    def update(self, t, v, cond):
        if not cond:
            self.buf = []
            return 0.0, None
        self.buf.append((t, v))
        vals = [b[1] for b in self.buf]
        if max(vals) - min(vals) > 2 * self.tol:
            self.buf = [(t, v)]
            return 0.0, None
        span = self.buf[-1][0] - self.buf[0][0]
        if span >= self.duration:
            return 1.0, float(np.median(vals))
        return span / self.duration, None


@dataclass
class _Arm:
    sh: np.ndarray      # scene units
    el: np.ndarray
    wr: np.ndarray
    hand: np.ndarray
    osh: np.ndarray     # other shoulder
    sh_w: np.ndarray    # MediaPipe world coords (metres, hip-centred)
    el_w: np.ndarray
    wr_w: np.ndarray
    osh_w: np.ndarray
    hip_w: np.ndarray


@dataclass
class _Metrics:
    elev: float         # shoulder elevation from the torso "down" axis (0 = arm hanging)
    lean: float         # trunk lean vs. calibrated upright (deg)
    up: np.ndarray      # torso up vector (world)
    arm_len: float      # shoulder->wrist length in scene units


def _arm_metrics(a: _Arm, ref_up=None) -> Optional[_Metrics]:
    up = _unit((a.sh_w + a.osh_w) / 2 - a.hip_w)
    v = a.el_w - a.sh_w
    if np.linalg.norm(v) < 1e-6:
        return None
    elev = _deg(math.acos(float(np.clip(np.dot(_unit(v), -up), -1, 1))))
    ref = up if ref_up is None else ref_up
    lean = _deg(math.acos(float(np.clip(np.dot(up, ref), -1, 1))))
    sw = float(np.linalg.norm(a.sh_w - a.osh_w))
    per_m = float(np.linalg.norm(a.sh - a.osh)) / sw if sw > 1e-6 else 0.0
    arm_m = float(np.linalg.norm(a.el_w - a.sh_w) + np.linalg.norm(a.wr_w - a.el_w))
    return _Metrics(elev, lean, up, arm_m * per_m)


@dataclass
class _Balloon:
    side: str
    plane: str
    level: int
    target: float
    pos: np.ndarray
    r: float
    color: int
    phase: float
    age: float = 0.0
    dwell: float = 0.0
    lean_events: int = 0
    leaning: bool = False
    gone: float = 0.0           # > 0 while floating away after a miss
    peaks: dict = field(default_factory=dict)
    popper: str = ""


# ----------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------
class BalloonPopEngine:
    """One instance per active game session. Not a thread: the caller (the
    camera loop) calls process_frame() once per frame."""

    def __init__(self):
        self.lock = threading.Lock()        # guards self.status (read from request threads)
        self._mutex = threading.RLock()     # guards engine state (frame thread vs. routes)
        self.cfg = dict(DEFAULT_CFG)
        self.aspect = 4 / 3
        self.status: dict = {}
        self._new_session()
        self._publish("none")

    # ---- public API (same shape as game/rehab_runner/worker.py) ------------
    @staticmethod
    def config_schema() -> list:
        return config_schema()

    def configure(self, cfg: dict):
        """Apply therapist choices (mode, arm, optional tuning). Unknown or
        invalid values are ignored. Takes effect on the next calibration."""
        with self._mutex:
            c = self.cfg
            if cfg.get("mode") in MODE_PLANES:
                c["mode"] = cfg["mode"]
            if cfg.get("arm") in ARM_SIDES:
                c["arm"] = cfg["arm"]
            for k, (lo, hi) in _RANGES.items():
                if k not in cfg:
                    continue
                try:
                    v = int(cfg[k])
                except (TypeError, ValueError):
                    continue
                if v == 0 and k in _ZERO_OK:
                    c[k] = 0
                elif lo <= v <= hi:
                    c[k] = v
            c["lvl_low"] = min(c["lvl_low"], c["lvl_mid"] - 5)
            c["lvl_mid"] = min(max(c["lvl_mid"], c["lvl_low"] + 5), c["lvl_high"] - 5)
            c["lvl_high"] = max(c["lvl_high"], c["lvl_mid"] + 5)
            self._publish("none")

    def reset(self):
        """Back to idle (not calibrated). Keeps the configuration."""
        with self._mutex:
            self._new_session()
            self._publish("none")

    def request_calibration(self):
        """Start a fresh session: rest posture -> max angle per arm/plane -> countdown."""
        with self._mutex:
            self._new_session()
            for s in self.sides:
                for p in MODE_PLANES[self.cfg["mode"]]:
                    man = self.cfg["man_" + p]
                    if man:
                        self.cal_max[(s, p)] = float(man)
                    else:
                        self.steps.append((s, p))
            self.state = "CAL_REST"
            self._publish("none")

    def get_status(self) -> dict:
        with self.lock:
            return dict(self.status)

    def get_events_since(self, since: int):
        with self.lock:
            return [e for e in self.rep_log if e["id"] > since]

    def get_summary(self) -> dict:
        """Totals for the session save payload (see session.js / routers/sessions.py)."""
        with self._mutex:
            reps = list(self.rep_log)
            by_plane, by_arm, by_arm_plane = {}, {}, {}
            for p in ("abd", "flex"):
                pr = [r for r in reps if r["plane"] == p]
                by_plane[p] = dict(reps=len(pr), peak=max([r["peak"] for r in pr], default=0.0))
            for s in ("left", "right"):
                by_arm[s] = len([r for r in reps if r["side"] == s])
            for (s, p), mx in self.cal_max.items():
                pr = [r for r in reps if r["side"] == s and r["plane"] == p]
                by_arm_plane[f"{s}_{p}"] = dict(
                    reps=len(pr), peak=round(max([r["peak"] for r in pr], default=0.0), 1),
                    calibrated_max=round(mx, 1))
            return {
                "reps": len(reps),
                "misses": self.misses,
                "lean_events": self.lean_events,
                "avg_time_to_pop": round(float(np.mean([r["time_to_pop"] for r in reps])), 2) if reps else 0.0,
                "max_elevation": round(max([r["peak"] for r in reps], default=0.0), 1),
                "left_reps": by_arm["left"],
                "right_reps": by_arm["right"],
                "by_plane": by_plane,
                "by_arm_plane": by_arm_plane,
                "calibration": {f"{k[0]}_{k[1]}": round(v, 1) for k, v in self.cal_max.items()},
                "settings": dict(self.cfg),
                "calibrated": self.state == "PLAY",
                "events": reps,         # per-rep log, one dict per popped balloon
            }

    # ---- per-frame entry point -------------------------------------------
    def process_frame(self, landmarks, world_landmarks, now: float, frame_shape=None):
        with self._mutex:
            if frame_shape:
                fh, fw = frame_shape[:2]
                if fh > 0:
                    self.aspect = fw / float(fh)
            dt = 0.0 if self._prev_t is None else min(max(now - self._prev_t, 0.0), MAX_DT)
            self._prev_t = now
            self.t += dt

            arms = {}
            if landmarks is not None and world_landmarks is not None:
                try:
                    arms = self._read_arms(landmarks, world_landmarks)
                except Exception as exc:        # a bad frame must never kill the loop
                    print(f"[BalloonPopEngine] frame skipped: {exc}")
                    arms = {}
            person = "none" if landmarks is None else ("ok" if arms and len(arms) == len(self.sides) else "partial")

            self.arms, self.ms, self.hands = {}, {}, {}
            if self.state != "IDLE":
                for s, a in arms.items():
                    m = _arm_metrics(a, self.ref_up)
                    if m is None:
                        continue
                    f = self.filters[s]
                    m.elev = float(f["elev"](m.elev))
                    m.arm_len = float(f["len"](m.arm_len))
                    self.arms[s], self.ms[s] = a, m
                    self.hands[s] = f["hand"](a.hand)

            if self.state == "CAL_REST":
                self._cal_rest()
            elif self.state == "CAL_PLANE":
                self._cal_plane()
            elif self.state == "COUNTDOWN":
                self.countdown -= dt
                self.msg = "Get ready"
                if self.countdown <= 0:
                    self.state = "PLAY"
                    self.msg = ""
            elif self.state == "PLAY":
                self._play(dt)
            self._publish(person)

    # ---- state ------------------------------------------------------------
    @property
    def sides(self) -> List[str]:
        return ARM_SIDES[self.cfg["arm"]]

    @property
    def planes(self) -> List[str]:
        return MODE_PLANES[self.cfg["mode"]]

    def _new_session(self):
        self.state = "IDLE"     # IDLE | CAL_REST | CAL_PLANE | COUNTDOWN | PLAY
        self.t = 0.0
        self._prev_t = None
        self.msg = ""
        self.arms, self.ms, self.hands = {}, {}, {}
        self.filters = {s: dict(len=_EMA(0.1), hand=_EMA(0.6), elev=_EMA(0.5)) for s in ("left", "right")}
        self.cal_max: Dict[tuple, float] = {}
        self.ref_up = None
        self.ups: list = []
        self.steps: list = []
        self.step_i, self.cal_done_wait = 0, False
        self.hold = _Hold(1.5, 4.0)
        self.hold_progress = 0.0
        self.countdown = COUNTDOWN_S
        self.flash = 0.0
        # play
        self.balloons: Dict[str, _Balloon] = {}
        self.rep_log: list = []
        self.misses = 0
        self.lean_events = 0
        self.rest_t = 0.0
        self.seq = 0
        self.alt_i = 0
        self.last_key = None
        self.pop_seq = 0
        self.last_pop = None

    # ---- pose -> arms -----------------------------------------------------
    def _read_arms(self, lm, world) -> Dict[str, _Arm]:
        wl = getattr(world, "landmark", world)
        a_is_left = lm[_GRP_A["sh"]].x <= lm[_GRP_B["sh"]].x      # screen-left shoulder = patient's left
        grp = {"left": _GRP_A if a_is_left else _GRP_B, "right": _GRP_B if a_is_left else _GRP_A}
        asp = self.aspect

        def px(i):
            return np.array([lm[i].x * asp, lm[i].y])

        def wd(i):
            return np.array([wl[i].x, wl[i].y, wl[i].z])

        hip_w = (wd(_HIP_L) + wd(_HIP_R)) / 2
        out = {}
        for side in self.sides:
            me, other = grp[side], grp["right" if side == "left" else "left"]
            if min(lm[me[k]].visibility for k in ("sh", "el", "wr")) < VIS_MIN or lm[other["sh"]].visibility < VIS_MIN:
                continue
            pts = [px(me["wr"])] + [px(me[k]) for k in ("ix", "pk") if lm[me[k]].visibility > 0.3]
            out[side] = _Arm(px(me["sh"]), px(me["el"]), px(me["wr"]), np.mean(pts, axis=0), px(other["sh"]),
                             wd(me["sh"]), wd(me["el"]), wd(me["wr"]), wd(other["sh"]), hip_w)
        return out

    # ---- calibration ------------------------------------------------------
    def _need_view(self):
        return "Stand ~2 m from the camera - shoulders, arms and hips visible"

    def _cal_rest(self):
        self.hold.duration, self.hold.tol = 1.5, 4.0
        if len(self.ms) < len(self.sides):
            self.msg = self._need_view()
            self.hold.reset()
            self.ups = []
            self.hold_progress = 0.0
            return
        self.msg = "Stand straight, arms relaxed by your sides"
        ok = all(m.elev < REST_ANGLE for m in self.ms.values())
        e = max(m.elev for m in self.ms.values())
        if ok:
            self.ups.append(next(iter(self.ms.values())).up)
        else:
            self.ups.clear()
        self.hold_progress, val = self.hold.update(self.t, e, ok)
        if val is not None:
            self.ref_up = _unit(np.mean(self.ups[-30:], axis=0))
            self.step_i, self.cal_done_wait = 0, False
            self.hold = _Hold(2.0, 5.0)
            self.hold_progress = 0.0
            if self.steps:
                self.state = "CAL_PLANE"
            else:
                self.countdown, self.state = COUNTDOWN_S, "COUNTDOWN"

    def _cal_plane(self):
        side, plane = self.steps[self.step_i]
        m = self.ms.get(side)
        if m is None:
            self.msg = self._need_view()
            self.hold.reset()
            self.hold_progress = 0.0
            return
        if self.cal_done_wait:
            self.msg = "Saved! Now lower your arm"
            if m.elev < REST_ANGLE:
                self.cal_done_wait = False
                self.step_i += 1
                self.hold.reset()
                self.hold_progress = 0.0
                if self.step_i >= len(self.steps):
                    self.countdown, self.state = COUNTDOWN_S, "COUNTDOWN"
            return
        word = "to the SIDE" if plane == "abd" else "FORWARD"
        self.msg = f"Raise your {side.upper()} arm {word} as high as comfortable and hold"
        lim = self.cfg["lean"]
        leaning = lim > 0 and m.lean > lim
        if leaning:
            self.msg = "Keep your back straight"
        ok = m.elev >= 30 and not leaning
        self.hold_progress, val = self.hold.update(self.t, m.elev, ok)
        if val is not None:
            self.cal_max[(side, plane)] = val
            self.cal_done_wait = True
            self.hold.reset()
            self.flash = 0.35

    # ---- gameplay ---------------------------------------------------------
    def _choose(self):
        plane = random.choice(self.planes)
        order = self.cfg["order"]
        if order == 0:
            lvl = random.choice([i for i in range(3) if (plane, i) != self.last_key])
        else:
            lvl = self.seq % 3 if order == 1 else 2 - self.seq % 3
        self.seq += 1
        self.last_key = (plane, lvl)
        return plane, lvl

    def _make_balloon(self, side, plane, lvl) -> _Balloon:
        m, arm, c = self.ms[side], self.arms[side], self.cfg
        mx = self.cal_max[(side, plane)]
        frac = (c["lvl_low"], c["lvl_mid"], c["lvl_high"])[lvl] / 100.0
        target = float(np.clip(frac * mx, min(REST_ANGLE + 20, mx), mx))
        L = m.arm_len * c["reach"] / 100.0
        sh = arm.sh
        out = 1.0 if arm.sh[0] > arm.osh[0] else -1.0
        th = math.radians(target)
        x = sh[0] + (out * L * math.sin(th) if plane == "abd" else 0.0)
        y = sh[1] + L * math.cos(th)
        r = float(np.clip(0.2 * L * c["size"] / 100.0, R_MIN, R_MAX))
        x_lo, x_hi = M_SIDE + r, max(M_SIDE + r, self.aspect - M_SIDE - r)
        y_lo, y_hi = M_TOP + r, max(M_TOP + r, 1.0 - r - M_BOTTOM)
        cx, cy = float(np.clip(x, x_lo, x_hi)), float(np.clip(y, y_lo, y_hi))
        # if the balloon had to be clamped into the frame, lower the target to match its real position
        dy = cy - sh[1]
        if plane == "abd":
            th2 = math.atan2(abs(cx - sh[0]), dy)
        else:
            th2 = math.acos(float(np.clip(dy / max(L, 1e-6), -1, 1)))
        target = max(min(target, _deg(th2)), REST_ANGLE + 15)
        return _Balloon(side, plane, lvl, target, np.array([cx, cy]), r,
                        random.randrange(N_COLORS), random.uniform(0, 6.28))

    def _spawn(self):
        plane, lvl = self._choose()
        if self.cfg["arm"] == "both_alt":
            side = self.sides[self.alt_i % 2]
            self.alt_i += 1
            targets = [side]
        else:
            targets = list(self.sides)
        for s in targets:
            self.balloons[s] = self._make_balloon(s, plane, lvl)

    def _play(self, dt):
        if len(self.ms) < len(self.sides):
            self.msg = "Paused - step back into view"
            return
        self.msg = ""
        lim = self.cfg["lean"]
        for side, b in list(self.balloons.items()):
            if b.gone > 0:
                b.gone -= dt
                b.pos = b.pos + np.array([0.0, -FLOAT_AWAY_SPEED]) * dt
                if b.gone <= 0:
                    del self.balloons[side]
                continue
            b.age += dt
            for s, m in self.ms.items():
                b.peaks[s] = max(b.peaks.get(s, 0.0), m.elev)
            leaning = lim > 0 and any(m.lean > lim for m in self.ms.values())
            if leaning and not b.leaning:
                b.lean_events += 1
                self.lean_events += 1
            b.leaning = leaning
            if leaning:
                self.msg = "Keep your back straight"
            toucher, touching = None, False
            for s, m in self.ms.items():
                hr = float(np.clip(0.07 * m.arm_len, HAND_R_MIN, HAND_R_MAX))
                if np.linalg.norm(self.hands[s] - b.pos) > b.r + hr:
                    continue
                touching = True
                tgt = min(b.target, self.cal_max.get((s, b.plane), b.target))
                if m.elev >= tgt - self.cfg["gate"]:
                    toucher = s
                    break
            if toucher and not leaning:
                b.dwell += dt
                b.popper = toucher
            else:
                b.dwell = max(0.0, b.dwell - 2 * dt)
                if touching and not toucher and not leaning:
                    self.msg = "Raise your arm higher"
            if b.dwell >= self.cfg["dwell"] / 1000.0:
                self._pop(b)
                del self.balloons[side]
            elif b.age >= self.cfg["timeout"]:
                self.misses += 1
                b.gone = FLOAT_AWAY_S
        if not self.balloons:
            ok_rest = all(m.elev < REST_ANGLE + 5 for m in self.ms.values())
            self.rest_t = self.rest_t + dt if ok_rest else 0.0
            if not ok_rest:
                self.msg = "Lower your arm" if len(self.sides) == 1 else "Lower your arms"
            if self.rest_t >= REST_HOLD:
                self._spawn()
                self.rest_t = 0.0

    def _pop(self, b: _Balloon):
        side = b.popper or b.side
        rep = dict(id=len(self.rep_log) + 1, side=side, plane=b.plane, level=b.level,
                   target=round(b.target, 1), peak=round(b.peaks.get(side, 0.0), 1),
                   time_to_pop=round(b.age, 2), lean_events=b.lean_events)
        self.rep_log.append(rep)
        self.pop_seq += 1
        self.last_pop = dict(seq=self.pop_seq, x=round(float(b.pos[0]), 4), y=round(float(b.pos[1]), 4),
                             r=round(b.r, 4), color=b.color)
        self.flash = 0.08

    # ---- status snapshot for the browser ----------------------------------
    def _progress(self) -> float:
        n = len(self.steps) + 1                         # rest + one hold per arm/plane
        if self.state == "CAL_REST":
            return self.hold_progress / n
        if self.state == "CAL_PLANE":
            return (1 + self.step_i + (1.0 if self.cal_done_wait else self.hold_progress)) / n
        if self.state in ("COUNTDOWN", "PLAY"):
            return 1.0
        return 0.0

    def _chips(self):
        chips = [dict(label="Rest", state="done" if self.state != "CAL_REST" else "now")]
        for i, (s, p) in enumerate(self.steps):
            st = "pending"
            if self.state in ("COUNTDOWN", "PLAY"):
                st = "done"
            elif self.state == "CAL_PLANE":
                if i < self.step_i or (i == self.step_i and self.cal_done_wait):
                    st = "done"
                elif i == self.step_i:
                    st = "now"
            chips.append(dict(label=f"{s[0].upper()} {PLANE_NAME[p]}", state=st))
        return chips

    def _publish(self, person: str):
        st = self.state
        playing = st == "PLAY"
        calibrating = st in ("CAL_REST", "CAL_PLANE", "COUNTDOWN")
        if st == "IDLE":
            msg, level = "Press Start Session to begin", "warn"
        else:
            msg = self.msg
            level = "ok" if (playing and not msg) else "warn"
        if person == "none" and st != "IDLE":
            msg = "No person detected. Please stand in front of the camera."
            level = "warn"

        arms = {}
        for side, a in self.arms.items():
            m = self.ms[side]
            b = self.balloons.get(side) or next(iter(self.balloons.values()), None)
            tgt = b.target if (b is not None and b.gone <= 0) else None
            arms[side] = dict(sh=_r4(a.sh), el=_r4(a.el), wr=_r4(a.wr), hand=_r4(self.hands[side]),
                              osh=_r4(a.osh), elev=round(m.elev, 1), lean=round(m.lean, 1),
                              target=round(tgt, 1) if tgt is not None else None,
                              hand_r=round(float(np.clip(0.07 * m.arm_len, HAND_R_MIN, HAND_R_MAX)), 4),
                              arm_len=round(m.arm_len, 4))
        dwell_s = max(self.cfg["dwell"], 1) / 1000.0
        balloons = [dict(side=b.side, plane=b.plane, level=b.level, target=round(b.target, 1),
                         x=round(float(b.pos[0]), 4), y=round(float(b.pos[1]), 4), r=round(b.r, 4),
                         color=b.color, progress=round(min(b.dwell / dwell_s, 1.0), 3),
                         age=round(b.age, 2), gone=b.gone > 0, phase=round(b.phase, 2))
                    for b in self.balloons.values()]
        cur = next((b for b in self.balloons.values() if b.gone <= 0), None)
        snap = dict(
            person=person, message=msg, guidance=msg, level=level,
            phase=st, calibrated=playing, calibrating=calibrating,
            calib_progress=round(self._progress(), 3),
            cal_chips=self._chips() if st != "IDLE" else [],
            countdown=round(max(self.countdown, 0.0), 2) if st == "COUNTDOWN" else None,
            hold_progress=round(self.hold_progress, 3),
            mode=self.cfg["mode"], arm=self.cfg["arm"], sides=list(self.sides), aspect=round(self.aspect, 4),
            arms=arms, balloons=balloons,
            reps=len(self.rep_log), misses=self.misses, lean_events=self.lean_events,
            left_reps=sum(1 for r in self.rep_log if r["side"] == "left"),
            right_reps=sum(1 for r in self.rep_log if r["side"] == "right"),
            current_target=round(cur.target, 1) if cur else None,
            current_plane=cur.plane if cur else None,
            pop_seq=self.pop_seq, last_pop=self.last_pop,
            gate=self.cfg["gate"],
        )
        with self.lock:
            self.status = snap
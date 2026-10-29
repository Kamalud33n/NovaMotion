"""Rule-based label suggestions and automatic quality scores for the repetitions of one recording.

Nothing here is a trained model. A suggestion is a first guess the clinician confirms or
changes; it never creates a label by itself. 'compensating' is never suggested: it needs a
judgement about the rest of the body that these rules cannot make.

Every repetition also gets an `auto_score` (good / ok / bad / unknown) and a list of `flags`.
The score is computed from the keypoints only, never from the clinician's session rating, so
the two stay independent; Stage 2 compares them.

Everything is compared with the same recording's own typical repetition, because no clinical
target is stored. This catches repetitions that differ from the patient's usual ones; it does
not catch a patient who is weak or compensating on every repetition. All thresholds below are
first guesses from test recordings and must be checked on real patient data.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MIN_FRAMES = 5
MIN_DETECTED = 0.6
MIN_ROM_RATIO = 0.7
MIN_REPS_FOR_REFERENCE = 3
MIN_ANGLE_SAMPLES = 3

# speed: repetition duration against the recording's median duration
FAST_RATIO = 0.4
SLOW_RATIO = 2.5
# smoothness: angle path length / (2 x ROM). 1.0 is one clean there-and-back.
JERKY_REL = 2.0
JERKY_ABS = 2.0
MIN_ROM_FOR_SMOOTHNESS = 10.0
# compensation (pose recordings only)
MIN_VISIBILITY = 0.5
LEAN_DEG = 12.0
TILT_DEG = 8.0
HIKE_DROP = 0.15

_L_EAR, _R_EAR, _L_SHO, _R_SHO, _L_HIP, _R_HIP = 7, 8, 11, 12, 23, 24
SOFT_FLAGS = ("very_fast", "paused_or_slow", "jerky")
COMPENSATION_FLAGS = ("trunk_lean", "shoulder_tilt", "shoulder_hike")


def rep_segments(reps: np.ndarray) -> List[Dict[str, Any]]:
    """Runs of frames in which the rep counter holds one value (end exclusive).

    A run is 'counted' when the counter then moved up. The run left open at the end, or ended
    by a reset, was never counted. Same rule as the timeline in static/labeling/timeline.js.
    """
    out: List[Dict[str, Any]] = []
    value: Optional[int] = None
    start = 0

    def close(end: int, counted: bool) -> None:
        nonlocal value
        if value is not None:
            out.append({"rep": value + 1, "start": start, "end": end, "counted": counted})
        value = None

    for i, raw in enumerate(reps):
        v = int(raw)
        if v < 0:
            close(i, False)
            continue
        if v == value:
            continue
        if value is not None:
            close(i, v > value)
        value = v
        start = i
    close(len(reps), False)
    return out


def _posture_series(pose: Optional[np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    """Per-frame posture signals from the pose landmarks, NaN where the needed landmarks are not
    clearly visible. Left/right names do not matter: only midpoints and the shoulder line are used.

    shoulder_tilt  angle of the line between the shoulders (degrees)
    ear_ratio      ear-to-shoulder distance as a share of the shoulder width (drops when a
                   shoulder is hiked towards the ear)
    lean           trunk lean from vertical (degrees); needs both hips in view, which a camera
                   framed on the upper body often does not give
    """
    if pose is None or pose.ndim != 3 or pose.shape[1] < 25 or pose.shape[2] < 4:
        return None
    p = pose.astype(float)
    upper = np.all(p[:, (_L_EAR, _R_EAR, _L_SHO, _R_SHO), 3] >= MIN_VISIBILITY, axis=1)
    hips = np.all(p[:, (_L_HIP, _R_HIP), 3] >= MIN_VISIBILITY, axis=1)
    sho = (p[:, _L_SHO, :2] + p[:, _R_SHO, :2]) / 2
    ear = (p[:, _L_EAR, :2] + p[:, _R_EAR, :2]) / 2
    hip = (p[:, _L_HIP, :2] + p[:, _R_HIP, :2]) / 2
    line = p[:, _R_SHO, :2] - p[:, _L_SHO, :2]
    width = np.linalg.norm(line, axis=1)
    ok = upper & (width > 1e-6)
    n = len(p)
    tilt, ear_ratio, lean = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    tilt[ok] = np.degrees(np.arctan2(line[ok, 1], np.abs(line[ok, 0]) + 1e-9))
    ear_ratio[ok] = (sho[ok, 1] - ear[ok, 1]) / width[ok]
    trunk = sho - hip                                   # image y grows downwards
    lean_ok = ok & hips & (np.linalg.norm(trunk, axis=1) > 1e-6)
    lean[lean_ok] = np.degrees(np.arctan2(trunk[lean_ok, 0], -trunk[lean_ok, 1]))
    return {"lean": lean, "tilt": tilt, "ear_ratio": ear_ratio}


def _baseline(posture: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, float]]:
    """Median of each posture signal over the whole recording (the patient's own usual posture)."""
    if posture is None:
        return None
    out: Dict[str, float] = {}
    for key, series in posture.items():
        good = series[np.isfinite(series)]
        if len(good) >= MIN_ANGLE_SAMPLES * 5:
            out[key] = float(np.median(good))
    return out or None


def _measure(seg: Dict[str, Any], detected: np.ndarray, angle: np.ndarray,
             t: Optional[np.ndarray] = None,
             posture: Optional[Dict[str, np.ndarray]] = None,
             baseline: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    s, e = seg["start"], seg["end"]
    frames = e - s
    ratio = float(np.mean(detected[s:e] > 0)) if frames else 0.0
    keep = (detected[s:e] > 0) & np.isfinite(angle[s:e])
    values = angle[s:e][keep]
    rom = float(values.max() - values.min()) if len(values) >= MIN_ANGLE_SAMPLES else None

    smooth = None
    if rom is not None and rom >= MIN_ROM_FOR_SMOOTHNESS:
        smooth = float(np.abs(np.diff(values)).sum() / (2 * rom))
    duration = float(t[e - 1] - t[s]) if t is not None and frames > 1 else None

    lean_dev = tilt_dev = hike_drop = None
    if posture is not None and baseline:
        def seg_values(key):
            v = posture[key][s:e]
            return v[np.isfinite(v)]
        if "lean" in baseline and len(seg_values("lean")) >= MIN_ANGLE_SAMPLES:
            lean_dev = float(np.percentile(np.abs(seg_values("lean") - baseline["lean"]), 90))
        if "tilt" in baseline and len(seg_values("tilt")) >= MIN_ANGLE_SAMPLES:
            tilt_dev = float(np.percentile(np.abs(seg_values("tilt") - baseline["tilt"]), 90))
        base_ear = baseline.get("ear_ratio")
        if base_ear and len(seg_values("ear_ratio")) >= MIN_ANGLE_SAMPLES:
            hike_drop = float((base_ear - np.percentile(seg_values("ear_ratio"), 10)) / base_ear)

    def r(x, n):
        return None if x is None else round(x, n)

    return {"frames": frames, "detected_ratio": round(ratio, 3), "rom": r(rom, 1),
            "duration_s": r(duration, 2), "smooth_ratio": r(smooth, 2),
            "lean_dev_deg": r(lean_dev, 1), "tilt_dev_deg": r(tilt_dev, 1),
            "hike_drop": r(hike_drop, 3)}


def _usable(seg: Dict[str, Any], m: Dict[str, Any]) -> bool:
    return seg["counted"] and m["frames"] >= MIN_FRAMES and m["detected_ratio"] >= MIN_DETECTED


def _reference_rom(measured: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Optional[float]:
    roms = [m["rom"] for seg, m in measured
            if seg["counted"] and m["rom"] is not None and m["detected_ratio"] >= MIN_DETECTED]
    return float(np.median(roms)) if len(roms) >= MIN_REPS_FOR_REFERENCE else None


def _reference(measured: List[Tuple[Dict[str, Any], Dict[str, Any]]], key: str) -> Optional[float]:
    vals = [m[key] for seg, m in measured if _usable(seg, m) and m[key] is not None]
    return float(np.median(vals)) if len(vals) >= MIN_REPS_FOR_REFERENCE else None


def _decide(seg: Dict[str, Any], m: Dict[str, Any], ref: Optional[float]) -> Dict[str, Any]:
    def result(label, confidence, reason):
        return {"label": label, "confidence": confidence, "reason": reason}

    if m["frames"] < MIN_FRAMES:
        return result("unusable", "medium", f"Only {m['frames']} frames, too short for a repetition.")
    if m["detected_ratio"] < MIN_DETECTED:
        return result("unusable", "high",
                      f"Tracking lost in {round((1 - m['detected_ratio']) * 100)}% of the frames.")

    rom, weak = m["rom"], None
    if rom is not None and ref:
        weak = rom < MIN_ROM_RATIO * ref
    ref_text = f"{rom:.0f}° against a typical {ref:.0f}°" if rom is not None and ref else ""

    if not seg["counted"]:
        if weak:
            return result("incomplete", "medium", f"Counter did not advance; range {ref_text}.")
        return result(None, None, "Counter did not advance. Check whether the patient stopped or "
                                  "the repetition was missed.")
    if weak:
        return result("incomplete", "high", f"Range {ref_text}.")
    if ref is None:
        return result("good", "low", "Counted, but too few repetitions here to compare the range.")
    return result("good", "high", f"Counted, range {ref_text}." if ref_text else "Counted.")


def _flags(m: Dict[str, Any], ref_duration: Optional[float], ref_smooth: Optional[float]) -> List[str]:
    flags: List[str] = []
    if ref_duration and m["duration_s"] is not None:
        if m["duration_s"] < FAST_RATIO * ref_duration:
            flags.append("very_fast")
        elif m["duration_s"] > SLOW_RATIO * ref_duration:
            flags.append("paused_or_slow")
    if ref_smooth and m["smooth_ratio"] is not None:
        if m["smooth_ratio"] > max(JERKY_ABS, JERKY_REL * ref_smooth):
            flags.append("jerky")
    if m["lean_dev_deg"] is not None and m["lean_dev_deg"] > LEAN_DEG:
        flags.append("trunk_lean")
    if m["tilt_dev_deg"] is not None and m["tilt_dev_deg"] > TILT_DEG:
        flags.append("shoulder_tilt")
    if m["hike_drop"] is not None and m["hike_drop"] > HIKE_DROP:
        flags.append("shoulder_hike")
    return flags


def _auto_score(decision: Dict[str, Any], flags: List[str]) -> str:
    """good / ok / bad, or unknown when the repetition cannot be judged from its own keypoints
    (tracking lost, counter did not advance, or too few repetitions to compare with)."""
    label = decision["label"]
    if label in (None, "unusable"):
        return "unknown"
    if label == "incomplete":
        return "bad"
    if len(flags) >= 2:
        return "bad"
    if flags:
        return "ok"
    return "good" if decision["confidence"] == "high" else "unknown"


def suggest(reps: np.ndarray, detected: np.ndarray, angle: np.ndarray,
            t: Optional[np.ndarray] = None, pose: Optional[np.ndarray] = None) -> Dict[str, Any]:
    segments = rep_segments(reps)
    posture = _posture_series(pose)
    baseline = _baseline(posture)

    measured = [(seg, _measure(seg, detected, angle, t, posture, baseline)) for seg in segments]
    ref = _reference_rom(measured)
    ref_duration = _reference(measured, "duration_s")
    ref_smooth = _reference(measured, "smooth_ratio")

    items = []
    for seg, m in measured:
        decision = _decide(seg, m, ref)
        flags = _flags(m, ref_duration, ref_smooth) if _usable(seg, m) or seg["counted"] else []
        items.append({
            "rep": seg["rep"], "rep_index": seg["rep"] - 1,
            "start_frame": seg["start"], "end_frame": seg["end"], "counted": seg["counted"],
            "suggestion": decision["label"], "confidence": decision["confidence"],
            "reason": decision["reason"], "metrics": m,
            "auto_score": _auto_score(decision, flags), "flags": flags,
        })
    return {"reference_rom": None if ref is None else round(ref, 1),
            "compensation_checked": baseline is not None,
            "compensation_signals": sorted(baseline) if baseline else [], "items": items}
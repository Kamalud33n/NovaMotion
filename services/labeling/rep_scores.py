"""Automatic per-repetition quality scores for a saved session's recording.

Computed from the recording's keypoints only (see suggest.py), never from the clinician's
session rating. The compact result is stored in session_data["rep_auto_scores"] when the
session is saved. Nothing here may break saving a session: failures are logged and None is
returned.
"""
from typing import Any, Dict, Optional

import numpy as np

from services.keypoints.settings import KeypointConfig
from services.keypoints.storage import recording_path
from services.labeling import suggest

SCORE_VERSION = 1
SCORES = ("good", "ok", "bad", "unknown")


def load_and_suggest(patient_id: str, recording_id: str) -> Dict[str, Any]:
    """Full suggestion result (with metrics) for one recording on disk. May raise."""
    path = recording_path(KeypointConfig.from_env(), patient_id, recording_id)
    with np.load(path) as data:
        reps = data["reps"]
        detected = data["detected"]
        angle = data["angle"].astype(float)
        t = data["t"].astype(float) if "t" in data.files else None
        pose = np.array(data["pose"]) if "pose" in data.files else None
    return suggest.suggest(reps, detected, angle, t=t, pose=pose)


def compact(result: Dict[str, Any]) -> Dict[str, Any]:
    items = result.get("items", [])
    counts = {name: 0 for name in SCORES}
    for item in items:
        counts[item["auto_score"]] = counts.get(item["auto_score"], 0) + 1
    return {
        "version": SCORE_VERSION,
        "reference_rom": result.get("reference_rom"),
        "compensation_checked": bool(result.get("compensation_checked")),
        "summary": {"reps": len(items), **counts},
        "items": [{"rep": i["rep"], "start_frame": i["start_frame"], "end_frame": i["end_frame"],
                   "auto_score": i["auto_score"], "flags": i["flags"]} for i in items],
    }


def compute_for_recording(patient_id: str, recording_id: str) -> Optional[Dict[str, Any]]:
    """Compact scores for session_data, or None when they cannot be computed. Never raises."""
    try:
        return compact(load_and_suggest(patient_id, recording_id))
    except Exception as exc:
        print(f"Rep scores: could not score {recording_id}: {exc}")
        return None
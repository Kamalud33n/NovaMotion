"""Training samples taken straight from TRUSTED reps - no manual labelling needed.
"""
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from database import get_db
from models import SessionModel
from services.keypoints.settings import KeypointConfig
from services.keypoints.storage import recording_path
from services.labeling import agreement, dataset_store

SOURCE = "auto_trusted"
DEFS_VERSION = "auto-trusted-1"
LABEL_FOR_RATING = {"good": "good", "bad": "bad", "worst": "bad"}


def _load_sessions() -> List[Dict[str, Any]]:
    with get_db() as db:
        return [{"id": sid, "patient_id": pid, "session_data": data if isinstance(data, dict) else {}}
                for sid, pid, data in db.query(SessionModel.id, SessionModel.patient_id,
                                                SessionModel.session_data).all()]


def collect() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(candidate trusted reps, stats). A candidate has everything needed to build its sample.
    Stats count what was left out and why, so an empty result can be explained."""
    cfg = KeypointConfig.from_env()
    stats: Counter = Counter()
    candidates: List[Dict[str, Any]] = []
    for sess in _load_sessions():
        stats["sessions"] += 1
        data = sess["session_data"]
        rec = data.get("recording")
        rid = rec.get("recording_id") if isinstance(rec, dict) else None
        if not rid:
            stats["sessions_no_recording"] += 1
            continue
        rating = agreement.session_rating(data)
        if rating not in LABEL_FOR_RATING:
            stats["sessions_not_rated"] += 1
            continue
        if not recording_path(cfg, sess["patient_id"], str(rid)).is_file():
            stats["sessions_recording_missing"] += 1
            continue
        stats["sessions_usable"] += 1
        frames = {i.get("rep"): i for i in
                  ((data.get("rep_auto_scores") or {}).get("items") or []) if isinstance(i, dict)}
        for row in agreement.rep_agreements(data):
            stats["reps_" + row["agreement"]] += 1
            if row["agreement"] != agreement.TRUSTED:
                continue
            item = frames.get(row["rep"]) or {}
            start, end = item.get("start_frame"), item.get("end_frame")
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                stats["reps_bad_frames"] += 1
                continue
            candidates.append({
                "session_id": sess["id"], "patient_id": sess["patient_id"], "recording_id": str(rid),
                "rep": row["rep"], "rep_index": row["rep_index"], "start_frame": start,
                "end_frame": end, "label": LABEL_FOR_RATING[rating], "rating": rating,
                "auto_score": row["auto_score"], "flags": list(item.get("flags") or []),
            })
    return candidates, dict(stats)


def build(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Sample columns + npz bytes for one candidate (same arrays as the manual dataset), or None."""
    sample = dataset_store.build_sample({
        "id": 0, "patient_id": candidate["patient_id"], "recording_id": candidate["recording_id"],
        "start_frame": candidate["start_frame"], "end_frame": candidate["end_frame"],
        "labeler_id": None, "label": candidate["label"], "rep_index": candidate["rep_index"],
        "tags": [], "note": None, "defs_version": DEFS_VERSION,
    })
    if sample is not None:
        sample["source"] = SOURCE
    return sample
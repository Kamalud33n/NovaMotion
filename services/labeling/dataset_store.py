"""Automatic ML dataset: every saved label (except `unusable`) is mirrored into dataset_samples.

label_store calls sync_label() after a label is created or changed and delete_for_label() when it
is removed; consent withdrawal and recording purge go through delete_for_patient() and
delete_for_recording(). Nothing here may break saving a label: failures are logged and
scripts/rebuild_dataset.py can redo them.

A sample holds a compressed .npz of the labelled frames: frame_index, t (seconds from the start
of the segment, so no real clock time is kept), detected, focus, reps, angle and `pose` (33x4) or
`hands` (2x21x3). Pose left/right are the patient's own sides. The patient is stored only as a
salted hash (DATASET_SALT in .env); without the salt nothing is stored.
"""
import hashlib
import hmac
import io
import os
from typing import Any, Dict, Optional

import numpy as np

from database import get_db
from models import DatasetSample
from services.keypoints.laterality import to_patient_sides
from services.keypoints.settings import KeypointConfig
from services.keypoints.storage import read_meta, recording_path

EXCLUDED_LABELS = frozenset({"unusable"})
_ARRAYS = ("t", "detected", "focus", "reps", "angle")
_warned_salt = False


def _salt() -> Optional[bytes]:
    global _warned_salt
    value = (os.getenv("DATASET_SALT") or "").strip()
    if value:
        return value.encode("utf-8")
    if not _warned_salt:
        print("Dataset: DATASET_SALT is not set in .env; labelled segments are not being stored")
        _warned_salt = True
    return None


def subject_hash(patient_id: str) -> Optional[str]:
    salt = _salt()
    if salt is None or not patient_id:
        return None
    return hmac.new(salt, str(patient_id).encode("utf-8"), hashlib.sha256).hexdigest()


def build_sample(label: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Column values for one labelled segment, or None when it cannot be built."""
    subject = subject_hash(label["patient_id"])
    if subject is None:
        return None
    cfg = KeypointConfig.from_env()
    patient_id, recording_id = label["patient_id"], label["recording_id"]
    meta = read_meta(cfg, patient_id, recording_id)
    if meta is None:
        print(f"Dataset: recording {recording_id} is not available for label {label['id']}")
        return None
    start, end = int(label["start_frame"]), int(label["end_frame"])

    with np.load(recording_path(cfg, patient_id, recording_id)) as data:
        total = int(len(data["t"]))
        if start < 0 or end <= start or end > total:
            print(f"Dataset: label {label['id']} ({start}-{end}) is outside {recording_id} ({total} frames)")
            return None
        window = slice(start, end)
        arrays: Dict[str, np.ndarray] = {"frame_index": np.arange(start, end, dtype=np.int32)}
        for name in _ARRAYS:
            if name in data.files:
                arrays[name] = np.array(data[name][window])
        landmarks = "pose" if "pose" in data.files else "hands"
        points = np.array(data[landmarks][window], dtype=np.float32)

    arrays["t"] = (arrays["t"] - arrays["t"][0]).astype(np.float32)
    if "angle" in arrays:
        arrays["angle"] = arrays["angle"].astype(np.float32)
    arrays[landmarks] = to_patient_sides(points, meta) if landmarks == "pose" else points

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    detected = arrays.get("detected")
    extra = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
    avg_fps = meta.get("avg_fps")
    return {
        "label_id": int(label["id"]),
        "recording_id": recording_id,
        "subject": subject,
        "labeler_id": label["labeler_id"],
        "kind": landmarks if landmarks == "hands" else "pose",
        "exercise": (str(meta.get("label"))[:100] if meta.get("label") else None),
        "mode": (str(extra.get("mode"))[:20] if extra.get("mode") else None),
        "side": (str(meta.get("side"))[:10] if meta.get("side") else None),
        "fps": float(avg_fps) if avg_fps else None,
        "start_frame": start,
        "end_frame": end,
        "n_frames": end - start,
        "rep_index": label.get("rep_index"),
        "label": label["label"],
        "tags": list(label.get("tags") or []),
        "note": label.get("note"),
        "defs_version": label["defs_version"],
        "detected_ratio": float(np.mean(detected > 0)) if detected is not None and len(detected) else None,
        "data": buffer.getvalue(),
    }


def _upsert(sample: Dict[str, Any]) -> None:
    with get_db() as db:
        row = db.query(DatasetSample).filter(DatasetSample.label_id == sample["label_id"]).first()
        if row is None:
            db.add(DatasetSample(**sample))
        else:
            for key, value in sample.items():
                setattr(row, key, value)
        db.commit()


def sync_label(label: Dict[str, Any]) -> bool:
    """Store (or refresh) the sample of a saved label; excluded labels are removed instead.
    Never raises. True when the dataset now matches the label."""
    try:
        if label.get("label") in EXCLUDED_LABELS:
            delete_for_label(int(label["id"]))
            return True
        sample = build_sample(label)
        if sample is None:
            return False
        _upsert(sample)
        return True
    except Exception as exc:
        print(f"Dataset: could not store label {label.get('id')}: {exc}")
        return False


def _delete(*conditions) -> int:
    with get_db() as db:
        n = db.query(DatasetSample).filter(*conditions).delete(synchronize_session=False)
        db.commit()
        return int(n or 0)


def delete_for_label(label_id: int) -> int:
    try:
        return _delete(DatasetSample.label_id == int(label_id))
    except Exception as exc:
        print(f"Dataset: could not remove label {label_id}: {exc}")
        return 0


def delete_for_recording(recording_id: str) -> int:
    try:
        return _delete(DatasetSample.recording_id == recording_id)
    except Exception as exc:
        print(f"Dataset: could not remove recording {recording_id}: {exc}")
        return 0


def delete_for_patient(patient_id: str) -> int:
    subject = subject_hash(patient_id)
    if subject is None:
        return 0
    try:
        return _delete(DatasetSample.subject == subject)
    except Exception as exc:
        print(f"Dataset: could not remove samples of patient {patient_id}: {exc}")
        return 0
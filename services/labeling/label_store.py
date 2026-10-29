"""Database access for recording labels. All rules about a valid label live here; the
router only does auth, ownership and HTTP.
"""
from typing import Any, Dict, List, Optional

from database import get_db
from models import RecordingLabel

from . import dataset_store
from .definitions import check_label, get_definitions

MAX_NOTE = 1000


class LabelError(Exception):
    """A request the user can fix. `status` is the HTTP status the router should return."""
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def to_dict(r: RecordingLabel) -> Dict[str, Any]:
    return {
        "id": r.id, "recording_id": r.recording_id, "patient_id": r.patient_id,
        "labeler_id": r.labeler_id, "start_frame": r.start_frame, "end_frame": r.end_frame,
        "rep_index": r.rep_index, "label": r.label, "tags": list(r.tags or []),
        "note": r.note, "defs_version": r.defs_version,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }


def _clean(start_frame: int, end_frame: int, label: str, tags: Optional[List[str]],
           note: Optional[str], rep_index: Optional[int], frames_total: Optional[int]):
    if not isinstance(start_frame, int) or not isinstance(end_frame, int) or isinstance(start_frame, bool) \
            or isinstance(end_frame, bool):
        raise LabelError(400, "start_frame and end_frame must be whole numbers")
    if start_frame < 0 or end_frame <= start_frame:
        raise LabelError(400, "Need 0 <= start_frame < end_frame")
    if frames_total is not None and end_frame > frames_total:
        raise LabelError(400, f"end_frame {end_frame} is past the end of the recording ({frames_total} frames)")
    if rep_index is not None and (isinstance(rep_index, bool) or not isinstance(rep_index, int) or rep_index < 0):
        raise LabelError(400, "rep_index must be a whole number >= 0")
    note = (note or "").strip() or None
    if note and len(note) > MAX_NOTE:
        raise LabelError(400, f"note is too long (max {MAX_NOTE} characters)")
    try:
        clean_tags = check_label(label, tags)
    except ValueError as exc:
        raise LabelError(400, str(exc))
    return note, clean_tags


def _overlapping(db, recording_id: str, labeler_id: str, start: int, end: int,
                 exclude_id: Optional[int]) -> Optional[RecordingLabel]:
    q = db.query(RecordingLabel).filter(
        RecordingLabel.recording_id == recording_id,
        RecordingLabel.labeler_id == labeler_id,
        RecordingLabel.start_frame < end,
        RecordingLabel.end_frame > start,
    )
    if exclude_id is not None:
        q = q.filter(RecordingLabel.id != exclude_id)
    return q.first()


def list_labels(patient_id: str, recording_id: str, labeler_id: str) -> List[Dict[str, Any]]:
    with get_db() as db:
        rows = (db.query(RecordingLabel)
                .filter(RecordingLabel.patient_id == patient_id,
                        RecordingLabel.recording_id == recording_id,
                        RecordingLabel.labeler_id == labeler_id)
                .order_by(RecordingLabel.start_frame, RecordingLabel.id).all())
        return [to_dict(r) for r in rows]


def create_label(patient_id: str, recording_id: str, labeler_id: str, *, start_frame: int,
                 end_frame: int, label: str, tags: Optional[List[str]] = None,
                 note: Optional[str] = None, rep_index: Optional[int] = None,
                 frames_total: Optional[int] = None) -> Dict[str, Any]:
    note, clean_tags = _clean(start_frame, end_frame, label, tags, note, rep_index, frames_total)
    with get_db() as db:
        if _overlapping(db, recording_id, labeler_id, start_frame, end_frame, None):
            raise LabelError(409, "This overlaps a segment you already labelled; edit or delete that one")
        row = RecordingLabel(
            recording_id=recording_id, patient_id=patient_id, labeler_id=labeler_id,
            start_frame=start_frame, end_frame=end_frame, rep_index=rep_index, label=label,
            tags=clean_tags, note=note, defs_version=get_definitions()["version"],
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        saved = to_dict(row)
    dataset_store.sync_label(saved)       # mirror into the ML dataset; never raises
    return saved


def update_label(label_id: int, patient_id: str, recording_id: str, labeler_id: str, *,
                 start_frame: int, end_frame: int, label: str, tags: Optional[List[str]] = None,
                 note: Optional[str] = None, rep_index: Optional[int] = None,
                 frames_total: Optional[int] = None) -> Dict[str, Any]:
    note, clean_tags = _clean(start_frame, end_frame, label, tags, note, rep_index, frames_total)
    with get_db() as db:
        row = _own_row(db, label_id, patient_id, recording_id, labeler_id)
        if _overlapping(db, recording_id, labeler_id, start_frame, end_frame, row.id):
            raise LabelError(409, "This overlaps another segment you already labelled")
        row.start_frame, row.end_frame, row.rep_index = start_frame, end_frame, rep_index
        row.label, row.tags, row.note = label, clean_tags, note
        row.defs_version = get_definitions()["version"]
        db.commit()
        db.refresh(row)
        saved = to_dict(row)
    dataset_store.sync_label(saved)       # refresh (or drop, if now `unusable`) the dataset sample
    return saved


def delete_label(label_id: int, patient_id: str, recording_id: str, labeler_id: str) -> None:
    with get_db() as db:
        db.delete(_own_row(db, label_id, patient_id, recording_id, labeler_id))
        db.commit()
    dataset_store.delete_for_label(label_id)


def _own_row(db, label_id: int, patient_id: str, recording_id: str, labeler_id: str) -> RecordingLabel:
    row = (db.query(RecordingLabel)
           .filter(RecordingLabel.id == label_id,
                   RecordingLabel.patient_id == patient_id,
                   RecordingLabel.recording_id == recording_id,
                   RecordingLabel.labeler_id == labeler_id).first())
    if row is None:                       # missing and not-yours look identical
        raise LabelError(404, "Label not found")
    return row


def delete_for_patient(patient_id: str) -> int:
    """Consent withdrawn: every label of this patient's recordings goes with the recordings."""
    with get_db() as db:
        recording_ids = {r[0] for r in db.query(RecordingLabel.recording_id)
                         .filter(RecordingLabel.patient_id == patient_id).distinct().all()}
        n = db.query(RecordingLabel).filter(RecordingLabel.patient_id == patient_id).delete()
        db.commit()
    for rid in recording_ids:             # by recording first: works even if DATASET_SALT changed
        dataset_store.delete_for_recording(rid)
    dataset_store.delete_for_patient(patient_id)
    return int(n or 0)


def delete_for_recording(patient_id: str, recording_id: str) -> int:
    """A recording was deleted from disk (abandoned-session purge): drop its labels too."""
    with get_db() as db:
        n = (db.query(RecordingLabel)
             .filter(RecordingLabel.patient_id == patient_id,
                     RecordingLabel.recording_id == recording_id).delete())
        db.commit()
    dataset_store.delete_for_recording(recording_id)
    return int(n or 0)
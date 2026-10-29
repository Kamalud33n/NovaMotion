"""Which dataset samples are `trusted`, for `scripts/export_dataset.py --trusted-only`.

A dataset sample is trusted only when it is a single rep (rep_index set) of a recording that
belongs to a saved session, and that rep's auto-score agrees with the session's clinician
rating (see agreement.py). Samples that cannot be checked - no rep_index, recording not linked
to a session, session not rated or not scored - are NOT trusted.
"""
from typing import Dict, Set, Tuple

from database import get_db
from models import DatasetSample, SessionModel
from services.labeling import agreement


def _trusted_reps_by_recording() -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {}
    with get_db() as db:
        for (data,) in db.query(SessionModel.session_data).all():
            rec = (data or {}).get("recording") if isinstance(data, dict) else None
            rid = rec.get("recording_id") if isinstance(rec, dict) else None
            if rid:
                out.setdefault(str(rid), set()).update(agreement.trusted_rep_indexes(data))
    return out


def trusted_sample_ids() -> Tuple[Set[int], int]:
    """(ids of trusted dataset_samples rows, number of samples left out)."""
    by_rec = _trusted_reps_by_recording()
    keep: Set[int] = set()
    total = 0
    with get_db() as db:
        for sid, rid, rep in db.query(DatasetSample.id, DatasetSample.recording_id,
                                      DatasetSample.rep_index).all():
            total += 1
            if rep is not None and rep in by_rec.get(rid, ()):
                keep.add(sid)
    return keep, total - len(keep)
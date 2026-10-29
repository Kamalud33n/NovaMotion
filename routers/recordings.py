"""Read-only API over the saved keypoint recordings (input for the labelling screen).

Every route is limited to patients owned by the logged-in user; a recording of
someone else's patient looks exactly like a missing one (404).
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from database import get_db
from services.auth.deps import CurrentUser, require_clinical
from services.keypoints.laterality import pose_names_swapped, swap_pose_sides
from services.keypoints.settings import KeypointConfig
from services.keypoints.storage import is_valid_id, list_recordings, read_meta, recording_path
from services.ownership import get_owned_patient, owned_patients

router = APIRouter()

_MAX_LIST = 500
_MAX_ROWS = 5000           # hard cap on frames returned by one /series call
_DEFAULT_ROWS = 2000       # target when the caller gives no step


def _clean(value: float, digits: int) -> Optional[float]:
    """JSON has no NaN: missing values become null."""
    v = float(value)
    return None if math.isnan(v) or math.isinf(v) else round(v, digits)


def _owned_patient_ids(me: CurrentUser, patient_id: Optional[str]) -> List[str]:
    with get_db() as db:
        if patient_id:
            get_owned_patient(db, patient_id, me)           # 404 when not this user's
            return [patient_id]
        return [p.id for p in owned_patients(db, me).all()]


def _require_recording(me: CurrentUser, patient_id: str, recording_id: str) -> Dict[str, Any]:
    if not (is_valid_id(patient_id) and is_valid_id(recording_id)):
        raise HTTPException(404, "Recording not found")
    with get_db() as db:
        get_owned_patient(db, patient_id, me)
    meta = read_meta(KeypointConfig.from_env(), patient_id, recording_id)
    if meta is None or meta.get("patient_id") not in (None, patient_id):
        raise HTTPException(404, "Recording not found")
    return meta


@router.get("/api/recordings")
async def list_my_recordings(
    patient_id: Optional[str] = None,
    exercise: Optional[str] = Query(None, description="case-insensitive substring of the exercise / game label"),
    kind: Optional[str] = Query(None, description="pose | hands"),
    mode: Optional[str] = Query(None, description="exercise | game"),
    linked: Optional[bool] = Query(None, description="true = belongs to a saved session"),
    limit: int = Query(100, ge=1, le=_MAX_LIST),
    me: CurrentUser = Depends(require_clinical),
):
    """Newest first. Summaries only; use the detail / series routes for content."""
    if kind not in (None, "pose", "hands") or mode not in (None, "exercise", "game"):
        raise HTTPException(400, "kind must be pose|hands and mode must be exercise|game")
    cfg = KeypointConfig.from_env()
    pids = await run_in_threadpool(_owned_patient_ids, me, patient_id)

    def _collect() -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for pid in pids:
            rows.extend(list_recordings(cfg, pid))
        return rows

    rows = await run_in_threadpool(_collect)
    needle = (exercise or "").strip().lower()
    if needle:
        rows = [r for r in rows if needle in str(r.get("label") or "").lower()]
    if kind:
        rows = [r for r in rows if r.get("kind") == kind]
    if mode:
        rows = [r for r in rows if r.get("mode") == mode]
    if linked is not None:
        rows = [r for r in rows if bool(r.get("linked")) == linked]
    rows.sort(key=lambda r: str(r.get("ended_at") or ""), reverse=True)
    return JSONResponse({"total": len(rows), "recordings": rows[:limit]})


@router.get("/api/recordings/{patient_id}/{recording_id}")
async def recording_detail(patient_id: str, recording_id: str,
                           me: CurrentUser = Depends(require_clinical)):
    """Full meta of one recording: capture settings, timing, game/exercise events."""
    meta = await run_in_threadpool(_require_recording, me, patient_id, recording_id)
    return JSONResponse(meta)


def _read_series(patient_id: str, recording_id: str, start: int, end: Optional[int],
                 step: Optional[int], landmarks: bool, patient_sides: bool = False,
                 meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = KeypointConfig.from_env()
    path = recording_path(cfg, patient_id, recording_id)
    with np.load(path) as data:                       # arrays are read lazily, only what is asked for
        total = int(len(data["t"]))
        lo = max(0, min(start, total))
        hi = total if end is None else max(lo, min(end, total))
        span = hi - lo
        limit = _MAX_ROWS if not landmarks else 1500
        st = step if step else max(1, math.ceil(span / (_DEFAULT_ROWS if not landmarks else 800)))
        if math.ceil(span / st) > limit:
            st = math.ceil(span / limit)               # never return more than the cap
        idx = slice(lo, hi, st)
        t_all = data["t"]
        t0 = float(t_all[0]) if total else 0.0
        out: Dict[str, Any] = {
            "frames_total": total, "start": lo, "end": hi, "step": st,
            "t0": _clean(t0, 3),
            "index": list(range(lo, hi, st)),
            "t": [_clean(v - t0, 3) for v in t_all[idx]],
            "detected": [int(v) for v in data["detected"][idx]],
            "focus": [int(v) for v in data["focus"][idx]],
            "reps": [int(v) for v in data["reps"][idx]],
            "angle": [_clean(v, 2) for v in data["angle"][idx]],
        }
        if landmarks:
            name = "pose" if "pose" in data.files else "hands"
            arr = data[name][idx].astype(np.float32)
            swapped = name == "pose" and pose_names_swapped(meta or {})
            if patient_sides and swapped:
                arr = swap_pose_sides(arr)
            out["landmarks_name"] = name
            out["sides"] = "patient" if (patient_sides or not swapped) else "raw_mediapipe"
            out["landmarks"] = np.where(np.isfinite(arr), np.round(arr, 4), None).tolist()
    return out


@router.get("/api/recordings/{patient_id}/{recording_id}/series")
async def recording_series(
    patient_id: str, recording_id: str,
    start: int = Query(0, ge=0),
    end: Optional[int] = Query(None, ge=0),
    step: Optional[int] = Query(None, ge=1, description="every Nth frame; omitted = auto"),
    landmarks: bool = Query(False, description="also return pose (33x4) or hands (2x21x3) per frame"),
    sides: str = Query("raw", description="raw = MediaPipe order as stored (pose left_* = patient's right); "
                                          "patient = pose left/right swapped so 'left' is the patient's left"),
    me: CurrentUser = Depends(require_clinical),
):
    """Time series for the labelling timeline: t (seconds from the first frame), t0 (absolute
    time of the first frame, the same clock as the `t` of events in the recording meta), detected,
    focus code, rep counter, joint angle. `index` holds the real frame number of each row, so a
    label made on a down-sampled view still points at an exact frame. Capped at 5000 rows
    (1500 with landmarks)."""
    if sides not in ("raw", "patient"):
        raise HTTPException(400, "sides must be raw or patient")
    meta = await run_in_threadpool(_require_recording, me, patient_id, recording_id)
    try:
        body = await run_in_threadpool(_read_series, patient_id, recording_id, start, end, step,
                                       landmarks, sides == "patient", meta)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Recordings: could not read {recording_id}: {exc}")
        raise HTTPException(500, "Recording could not be read")
    return JSONResponse(body)
"""Label API for saved keypoint recordings (input for the labelling screen).

Same access rules as routers/recordings.py: doctor/therapist only, and only recordings of
patients they own (anything else is a 404). A label can only be changed by the user who made it.
"""
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.auth.deps import CurrentUser, require_clinical
from services.keypoints.settings import KeypointConfig
from services.keypoints.storage import recording_path
from services.labeling import label_store, rep_scores
from services.labeling.definitions import get_definitions

from routers.recordings import _require_recording

router = APIRouter()


class LabelIn(BaseModel):
    start_frame: int = Field(..., description="first frame of the segment (inclusive)")
    end_frame: int = Field(..., description="end of the segment (exclusive)")
    label: str
    tags: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    rep_index: Optional[int] = None


def _frames_total(meta: Dict[str, Any], patient_id: str, recording_id: str) -> int:
    n = meta.get("frame_count")
    if isinstance(n, int) and n > 0:
        return n
    with np.load(recording_path(KeypointConfig.from_env(), patient_id, recording_id)) as data:
        return int(len(data["t"]))


async def _context(me: CurrentUser, patient_id: str, recording_id: str) -> int:
    meta = await run_in_threadpool(_require_recording, me, patient_id, recording_id)
    try:
        return await run_in_threadpool(_frames_total, meta, patient_id, recording_id)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Labels: could not read {recording_id}: {exc}")
        raise HTTPException(500, "Recording could not be read")


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except label_store.LabelError as exc:
        raise HTTPException(exc.status, exc.message)


@router.get("/api/label-definitions")
async def label_definitions(me: CurrentUser = Depends(require_clinical)):
    """The labels and compensation tags the UI should offer (from label_definitions.json)."""
    try:
        return JSONResponse(get_definitions())
    except RuntimeError as exc:
        print(f"Labels: {exc}")
        raise HTTPException(500, "Label definitions are not available")


@router.get("/api/recordings/{patient_id}/{recording_id}/labels")
async def list_labels(patient_id: str, recording_id: str, me: CurrentUser = Depends(require_clinical)):
    frames_total = await _context(me, patient_id, recording_id)
    rows = await run_in_threadpool(label_store.list_labels, patient_id, recording_id, me.id)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return JSONResponse({"recording_id": recording_id, "frames_total": frames_total,
                         "counts": counts, "labels": rows})


def _read_suggestions(patient_id: str, recording_id: str) -> Dict[str, Any]:
    return rep_scores.load_and_suggest(patient_id, recording_id)


@router.get("/api/recordings/{patient_id}/{recording_id}/label-suggestions")
async def label_suggestions(patient_id: str, recording_id: str,
                            me: CurrentUser = Depends(require_clinical)):
    """A first-guess label per repetition. Suggestions are never saved; the clinician accepts them."""
    await _context(me, patient_id, recording_id)
    try:
        result = await run_in_threadpool(_read_suggestions, patient_id, recording_id)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Labels: could not build suggestions for {recording_id}: {exc}")
        raise HTTPException(500, "Recording could not be read")
    mine = await run_in_threadpool(label_store.list_labels, patient_id, recording_id, me.id)
    for item in result["items"]:
        item["labeled"] = any(l["start_frame"] < item["end_frame"] and item["start_frame"] < l["end_frame"]
                              for l in mine)
    result["recording_id"] = recording_id
    return JSONResponse(result)


@router.post("/api/recordings/{patient_id}/{recording_id}/labels", status_code=201)
async def create_label(patient_id: str, recording_id: str, body: LabelIn,
                       me: CurrentUser = Depends(require_clinical)):
    frames_total = await _context(me, patient_id, recording_id)
    row = await run_in_threadpool(_run, label_store.create_label, patient_id, recording_id, me.id,
                                  frames_total=frames_total, **body.dict())
    return JSONResponse(row, status_code=201)


@router.put("/api/recordings/{patient_id}/{recording_id}/labels/{label_id}")
async def update_label(patient_id: str, recording_id: str, label_id: int, body: LabelIn,
                       me: CurrentUser = Depends(require_clinical)):
    frames_total = await _context(me, patient_id, recording_id)
    row = await run_in_threadpool(_run, label_store.update_label, label_id, patient_id, recording_id,
                                  me.id, frames_total=frames_total, **body.dict())
    return JSONResponse(row)


@router.delete("/api/recordings/{patient_id}/{recording_id}/labels/{label_id}")
async def delete_label(patient_id: str, recording_id: str, label_id: int,
                       me: CurrentUser = Depends(require_clinical)):
    await _context(me, patient_id, recording_id)
    await run_in_threadpool(_run, label_store.delete_label, label_id, patient_id, recording_id, me.id)
    return JSONResponse({"success": True})
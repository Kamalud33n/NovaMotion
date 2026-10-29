"""Ties recording start/finish to the session routes."""
from typing import Any, Dict, Optional

from fastapi import HTTPException

from database import get_db
from services.auth.deps import CurrentUser
from services.ownership import get_owned_patient
from services.user_context import UserContext

from .capture_meta import build_capture_meta
from .recorder import KIND_HANDS, KIND_POSE
from .settings import KeypointConfig
from .storage import (delete_patient_recordings as _delete_patient_files, link_recording,
                      purge_unlinked, recording_info, recover_partials)


def _capture(hand_mode: bool) -> Dict[str, Any]:
    try:
        return build_capture_meta(hand_mode)
    except Exception as exc:     # metadata must never block a recording
        print(f"Keypoints: capture meta failed: {exc}")
        return {"capture_meta_error": str(exc)}


def block_reason(ctx: UserContext, me: CurrentUser, patient_id: Optional[str]) -> Optional[str]:
    """Why recording must NOT start for this patient, or None when it may.
    Fails closed: any error counts as "no consent"."""
    if not ctx.recorder.enabled:
        return "disabled"
    if not patient_id:
        return "no_patient"
    try:
        with get_db() as db:
            patient = get_owned_patient(db, patient_id, me)
            consented = patient.keypoint_consent is True
    except HTTPException:
        return "not_found"
    except Exception as exc:
        print(f"Keypoints: patient check failed: {exc}")
        return "check_failed"
    return None if consented else "no_consent"


def _can_record(ctx: UserContext, me: CurrentUser, patient_id: Optional[str]) -> bool:
    return block_reason(ctx, me, patient_id) is None


def recover_after_restart() -> list:
    """Server start: turn checkpoint files left by a crash/kill into normal recordings.
    A partial is only kept when the patient still has consent on record."""
    from models import Patient
    cfg = KeypointConfig.from_env()

    def _still_consented(patient_id) -> bool:
        try:
            with get_db() as db:
                p = db.get(Patient, patient_id)
                return bool(p is not None and p.keypoint_consent is True)
        except Exception as exc:
            print(f"Keypoints: consent lookup failed during recovery: {exc}")
            return False          # fail closed: do not promote

    results = recover_partials(cfg, allow=_still_consented)
    for r in results:
        print(f"Keypoints recovery: {r}")
    return results


def purge_abandoned() -> list:
    """Delete finished recordings that never got a saved session (older than
    KEYPOINT_UNLINKED_TTL_HOURS). Before deleting, the DB is checked: a recording that a
    saved session still points to is re-linked, never deleted. Never raises."""
    from models import SessionModel
    cfg = KeypointConfig.from_env()
    if cfg.unlinked_ttl_hours <= 0:
        return []
    cache: Dict[str, Dict[str, str]] = {}

    def _saved_session(patient_id: str, recording_id: str) -> Optional[str]:
        if patient_id not in cache:        # one query per patient; an error propagates -> file is kept
            with get_db() as db:
                rows = db.query(SessionModel.id, SessionModel.session_data)\
                         .filter(SessionModel.patient_id == patient_id).all()
            found: Dict[str, str] = {}
            for sid, data in rows:
                if not isinstance(data, dict):
                    continue
                rec = data.get("recording")
                for rid in (rec.get("recording_id") if isinstance(rec, dict) else None,
                            data.get("recording_id")):
                    if rid:
                        found[str(rid)] = str(sid)
            cache[patient_id] = found
        return cache[patient_id].get(recording_id)

    try:
        results = purge_unlinked(cfg, cfg.unlinked_ttl_hours, _saved_session)
    except Exception as exc:
        print(f"Keypoints: purge failed: {exc}")
        return []
    for r in results:
        print(f"Keypoints purge: {r}")
    gone = [r for r in results if r.get("action") == "deleted_abandoned"]
    if gone:
        from services.labeling import label_store
        for r in gone:               # labels never outlive their recording
            try:
                label_store.delete_for_recording(r["patient_id"], r["recording_id"])
            except Exception as exc:
                print(f"Keypoints: could not delete labels of purged {r['recording_id']}: {exc}")
    return results


def delete_patient_recordings(patient_id: str) -> Dict[str, Any]:
    """Consent withdrawn: delete every saved recording of this patient from disk (and the labels made on them), then mark
    the sessions that pointed to them (session_data["recording"] -> deleted) so no report
    refers to a file that no longer exists. The session rows themselves stay.
    A DB problem while marking is logged and reported, never raised: the files are already gone."""
    import datetime
    from models import SessionModel
    result = _delete_patient_files(KeypointConfig.from_env(), patient_id)
    ids = {r["recording_id"] for r in result["recordings"]}
    marked, mark_error = 0, None
    if ids:
        try:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            with get_db() as db:
                for sess in db.query(SessionModel).filter(SessionModel.patient_id == patient_id).all():
                    data = sess.session_data
                    if not isinstance(data, dict):
                        continue
                    rec = data.get("recording")
                    rid = (rec.get("recording_id") if isinstance(rec, dict) else None) or data.get("recording_id")
                    if rid and str(rid) in ids:
                        new = dict(data)         # new object: SQLAlchemy does not see in-place JSON edits
                        new["recording"] = {"recording_id": str(rid), "deleted": True,
                                            "deleted_at": now, "reason": "consent_withdrawn"}
                        sess.session_data = new
                        marked += 1
                db.commit()
        except Exception as exc:
            mark_error = str(exc)
            print(f"Keypoints: could not mark sessions after deleting recordings of {patient_id}: {exc}")
    labels_deleted = 0
    try:                                  # labels of deleted recordings go too (retry-safe)
        from services.labeling import label_store
        labels_deleted = label_store.delete_for_patient(patient_id)
    except Exception as exc:
        mark_error = mark_error or f"labels: {exc}"
        print(f"Keypoints: could not delete labels of {patient_id}: {exc}")
    return {
        "recordings_deleted": len(ids),
        "files_deleted": result["files_deleted"],
        "failed_files": result["failed"],
        "sessions_marked": marked,
        "labels_deleted": labels_deleted,
        "mark_error": mark_error,      # any DB cleanup problem (session marks or labels)
    }


def withdraw_recording(ctx: UserContext, patient_id: str) -> bool:
    """Consent was withdrawn: throw away anything still in memory for this patient."""
    return ctx.recorder.discard_for_patient(patient_id)


def start_exercise_recording(ctx: UserContext, me: CurrentUser, patient_id: Optional[str], *,
                             hand_mode: bool, exercise: str, side: Optional[str],
                             target_rom: Optional[float]) -> Optional[str]:
    if not _can_record(ctx, me, patient_id):
        ctx.recorder.stop("superseded")
        return None
    return ctx.recorder.start(
        KIND_HANDS if hand_mode else KIND_POSE,
        patient_id=patient_id, owner_id=me.id, label=exercise, side=side,
        extra={"mode": "exercise", "target_rom": target_rom},
        capture=_capture(hand_mode),
    )


def start_game_recording(ctx: UserContext, me: CurrentUser, patient_id: Optional[str]) -> Optional[str]:
    if not _can_record(ctx, me, patient_id):
        ctx.recorder.stop("superseded")
        return None
    game_id = ctx.game.get_active_game_id()
    ctx.game_tap.reset()
    extra: Dict[str, Any] = {"mode": "game", "game_id": game_id}
    settings = getattr(ctx.game.get_engine(), "cfg", None)
    if isinstance(settings, dict):
        extra["settings"] = dict(settings)
    return ctx.recorder.start(
        KIND_POSE, patient_id=patient_id, owner_id=me.id,
        label=f"Game: {game_id}", extra=extra, capture=_capture(False),
    )


def attach_recording(ctx: UserContext, session_data: Dict[str, Any],
                     patient_id: Optional[str] = None) -> Dict[str, Any]:
    """Finish the recording named in session_data and note its file there.

    The recording is flushed if it is still running, then its info is read from the
    file on disk (patient_id + recording_id), so it still works for a retried save after
    a reload, a server restart, or when a newer recording replaced the in-memory copy.
    A recording of another patient, or one already linked to another session, is never
    attached."""
    session_data.pop("recording", None)      # only ever set by the server, never trusted from the client
    recording_id = session_data.get("recording_id")
    if not recording_id:
        return session_data
    recording_id = str(recording_id)

    mem = ctx.recorder.finish(recording_id)      # also flushes a recording that is still running
    info = None
    if patient_id:
        try:
            info = recording_info(KeypointConfig.from_env(), patient_id, recording_id)
        except Exception as exc:
            print(f"Keypoints: disk lookup failed for {recording_id}: {exc}")
    if info is None and mem is not None:
        info = {**mem, "session_id": None}       # disk unreadable / no patient_id: use the in-memory info

    if not info:
        return session_data
    if patient_id and info.get("patient_id") != patient_id:
        return session_data
    if info.pop("session_id", None):
        return session_data                  # already belongs to another saved session
    session_data["recording"] = info
    return session_data


def link_session(patient_id: str, session_data: Dict[str, Any], session_id) -> bool:
    """After the session row is committed: write its id into the recording's meta.
    Never raises -- a failed link must not fail the session save."""
    recording = session_data.get("recording")
    if not isinstance(recording, dict) or not recording.get("recording_id"):
        return False
    try:
        return link_recording(KeypointConfig.from_env(), patient_id, str(recording["recording_id"]), session_id)
    except Exception as exc:
        print(f"Keypoints: could not link recording to session {session_id}: {exc}")
        return False
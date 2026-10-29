import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from database import get_db
from models import SessionModel, JointAngle, ExerciseResult, History
from services.auth.deps import CurrentUser, require_clinical
from services.ownership import get_owned_patient
from services.helpers import game_result_line
from services.keypoints import lifecycle
from services.labeling import rep_scores
from services.user_context import get_context

router = APIRouter()

RATINGS = {"good", "bad", "worst"}


@router.get("/api/sessions/{patient_id}")
async def get_sessions(patient_id: str, me: CurrentUser = Depends(require_clinical)):
    with get_db() as db:
        get_owned_patient(db, patient_id, me)
        sessions = (
            db.query(SessionModel)
            .filter(SessionModel.patient_id == patient_id)
            .order_by(SessionModel.start_time.desc())
            .all()
        )
        out = []
        for s in sessions:
            sd = s.session_data or {}
            out.append({
                "session_id":          s.id,
                "patient_id":          s.patient_id,
                "exercise_type":       s.exercise_type,
                "start_time":          s.start_time.isoformat(),
                "end_time":            s.end_time.isoformat() if s.end_time else None,
                "duration_seconds":    s.duration_seconds,
                "total_reps":          s.total_reps,
                "completed_reps":      s.completed_reps,
                "accuracy_percentage": s.accuracy_percentage,
                "average_rom":         s.average_rom,
                "incorrect_movements": s.incorrect_movements,
                "stability_score":     s.stability_score,
                "balance_score":       s.balance_score,
                "movement_smoothness": s.movement_smoothness,
                "fatigue_estimation":  s.fatigue_estimation,
                "recovery_score":      s.recovery_score,
                # Game-mode sessions (services/game_state.py) stash their
                # own metrics under session_data — surface the useful bits
                # at the top level so patients.html / dashboard.html /
                # reports don't need to know the session_data shape.
                "is_game_session":     sd.get("mode") == "game",
                "game_id":             sd.get("game_id"),
                "target_steps":        sd.get("target_steps"),
                "left_steps":          sd.get("left_steps"),
                "right_steps":         sd.get("right_steps"),
                "final_score":         sd.get("final_score"),
                "lives_remaining":     sd.get("lives_remaining"),
                "finish_reason":       sd.get("finish_reason"),
                "game_result":         game_result_line(s),   # per-game one-liner for the reports list
                "joint_angles": [
                    {"joint_name": ja.joint_name, "angle": ja.angle_value,
                     "target": ja.target_angle, "is_correct": ja.is_correct}
                    for ja in s.joint_angles[:20]
                ],
            })
        return JSONResponse(out)


@router.post("/api/sessions")
async def save_session(payload: Dict[str, Any], me: CurrentUser = Depends(require_clinical)):
    with get_db() as db:
        pid = payload.get("patient_id")
        get_owned_patient(db, pid, me)

        session_data = payload.get("session_data")
        if not isinstance(session_data, dict):
            session_data = {}
        session_data.pop("recording", None)      # server-owned key; never trust one sent by the client
        session_data.pop("rep_auto_scores", None)  # server-owned key too
        if session_data.get("recording_id"):
            await run_in_threadpool(lifecycle.attach_recording, get_context(me.id), session_data, pid)
        recording = session_data.get("recording")
        if isinstance(recording, dict) and recording.get("recording_id"):
            # Automatic per-rep scores from the keypoints (never from the clinician's rating).
            scores = await run_in_threadpool(rep_scores.compute_for_recording, pid,
                                             str(recording["recording_id"]))
            if scores:
                session_data["rep_auto_scores"] = scores

        def _dt(key):
            v = payload.get(key)
            if not v:
                return None
            # JS toISOString() emits a trailing 'Z', which Python 3.10's
            # fromisoformat() can't parse directly (only 3.11+ supports it).
            if v.endswith("Z"):
                v = v[:-1] + "+00:00"
            return datetime.datetime.fromisoformat(v)

        sess = SessionModel(
            patient_id          = pid,
            exercise_type       = payload.get("exercise_type", "General Exercise"),
            start_time          = _dt("start_time") or datetime.datetime.now(),
            end_time            = _dt("end_time"),
            duration_seconds    = payload.get("duration_seconds", 0),
            total_reps          = payload.get("total_reps", 0),
            completed_reps      = payload.get("completed_reps", 0),
            accuracy_percentage = payload.get("accuracy_percentage", 0.0),
            average_rom         = payload.get("average_rom", 0.0),
            incorrect_movements = payload.get("incorrect_movements", 0),
            stability_score     = payload.get("stability_score", 0.0),
            balance_score       = payload.get("balance_score", 0.0),
            movement_smoothness = payload.get("movement_smoothness", 0.0),
            fatigue_estimation  = payload.get("fatigue_estimation", 0.0),
            recovery_score      = payload.get("recovery_score", 0.0),
            session_data        = session_data,
        )
        db.add(sess)
        db.flush()

        for ja in payload.get("joint_angles", []):
            db.add(JointAngle(
                session_id   = sess.id,
                joint_name   = ja.get("joint_name", "Unknown"),
                angle_value  = ja.get("angle_value", 0.0),
                target_angle = ja.get("target_angle"),
                deviation    = ja.get("deviation"),
                is_correct   = ja.get("is_correct", True),
            ))

        for er in payload.get("exercise_results", []):
            db.add(ExerciseResult(
                session_id         = sess.id,
                exercise_name      = er.get("exercise_name", "Unknown"),
                repetition_number  = er.get("repetition_number", 0),
                accuracy           = er.get("accuracy", 0.0),
                rom_achieved       = er.get("rom_achieved", 0.0),
                speed              = er.get("speed", 0.0),
                hold_duration      = er.get("hold_duration", 0.0),
                compensation_score = er.get("compensation_score", 0.0),
                is_completed       = er.get("is_completed", False),
                feedback           = er.get("feedback", ""),
            ))

        db.add(History(
            patient_id = pid,
            action     = "Session Saved",
            details    = f"Session {sess.id} — {sess.completed_reps} reps",
        ))
        session_id = sess.id
        db.commit()
        if session_data.get("recording"):
            # Session is saved: write its id into the recording's meta (never fails the save).
            await run_in_threadpool(lifecycle.link_session, pid, session_data, session_id)
        return JSONResponse({"success": True, "message": "Session saved", "session_id": session_id})


@router.patch("/api/sessions/{session_id}/rating")
async def rate_session(session_id: str, payload: Dict[str, Any],
                       me: CurrentUser = Depends(require_clinical)):
    """The clinician's overall rating of one saved session: good, bad or worst."""
    rating = payload.get("rating")
    if rating not in RATINGS:
        raise HTTPException(400, "rating must be good, bad or worst")
    with get_db() as db:
        sess = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if sess is None:
            raise HTTPException(404, "Session not found")
        get_owned_patient(db, sess.patient_id, me)
        data = dict(sess.session_data or {})
        data["clinician_rating"] = {
            "value": rating,
            "by": me.id,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        sess.session_data = data
        db.commit()
    return JSONResponse({"success": True, "rating": rating})
"""
Session report export — implements the MedNova draft contract (v0.1) from
MedNova_Integration_Overview.md, section 4.2.

This endpoint is the "USB-C port": it doesn't expose our internal DB shape,
it just emits a stable, versioned JSON report for any external system
(MedNova or otherwise) to consume.
"""
from collections import Counter

from fastapi import APIRouter, HTTPException

from database import get_db
from models import SessionModel, JointAngle, Patient

router = APIRouter(prefix="/api/sessions", tags=["export"])

SCHEMA_VERSION = "0.1"


@router.get("/{session_id}/export")
async def export_session_report(session_id: str):
    with get_db() as db:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        patient = db.query(Patient).filter(Patient.id == session.patient_id).first()

        joint_angles = (
            db.query(JointAngle)
            .filter(JointAngle.session_id == session_id)
            .all()
        )

        # Pick the joint tracked most often in this session as the
        # representative joint for the rom.min/max summary. Individual
        # per-joint angles are still available internally if a richer
        # contract is agreed on later.
        rom_joint = None
        min_angle = 0
        max_angle = 0
        if joint_angles:
            joint_counts = Counter(ja.joint_name for ja in joint_angles)
            rom_joint = joint_counts.most_common(1)[0][0]
            angles_for_joint = [
                ja.angle_value for ja in joint_angles if ja.joint_name == rom_joint
            ]
            min_angle = round(min(angles_for_joint), 1)
            max_angle = round(max(angles_for_joint), 1)

        total_reps = session.total_reps or 0
        avg_time_per_rep = (
            round(session.duration_seconds / total_reps, 2)
            if session.duration_seconds and total_reps
            else 0.0
        )
        completion_pct = (
            round((session.completed_reps / total_reps) * 100, 1)
            if total_reps
            else 0.0
        )

        pdf_url = None
        if session.session_data and isinstance(session.session_data, dict):
            pdf_url = session.session_data.get("pdf_report_url")

        report = {
            "schema_version": SCHEMA_VERSION,
            "report_type": "rehab_session",
            "source_system": "rehab-ai",
            "session": {
                "session_id": session.id,
                "patient_external_id": patient.external_id if patient else None,
                "started_at": session.start_time.isoformat() if session.start_time else None,
                "ended_at": session.end_time.isoformat() if session.end_time else None,
                "exercise_type": session.exercise_type,
            },
            "metrics": {
                "rom": {
                    "joint": rom_joint,
                    "min_angle_deg": min_angle,
                    "max_angle_deg": max_angle,
                },
                "repetitions": {
                    "completed": session.completed_reps or 0,
                    "target": total_reps,
                },
                "stability_score": round(session.stability_score or 0.0, 1),
                "balance_score": round(session.balance_score or 0.0, 1),
                "accuracy_pct": round(session.accuracy_percentage or 0.0, 1),
                "recovery_score": round(session.recovery_score or 0.0, 1),
            },
            "quality": {
                "avg_time_per_rep_sec": avg_time_per_rep,
                "completion_pct": completion_pct,
            },
            "artifacts": {
                "pdf_report_url": pdf_url,
            },
        }

        return report
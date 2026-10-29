import base64
import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from database import get_db
from models import Patient, History
from services.auth.deps import CurrentUser, require_clinical
from services.helpers import compress_photo, session_summary
from services.keypoints import lifecycle
from services.ownership import get_owned_patient, owned_patients
from services.user_context import get_context

router = APIRouter()


@router.get("/api/patients")
async def list_patients(request: Request, me: CurrentUser = Depends(require_clinical)):
    search = request.query_params.get("search", "").strip()
    include_inactive = request.query_params.get("include_inactive", "false").lower() == "true"
    with get_db() as db:
        q = owned_patients(db, me)
        if not include_inactive:
            q = q.filter(Patient.is_active == True)
        if search:
            q = q.filter(
                Patient.name.contains(search)
                | Patient.id.contains(search)
                | Patient.phone.contains(search)
            )
        patients = q.order_by(Patient.date_created.desc()).all()
        out = []
        for p in patients:
            n = len(p.sessions)
            avg_acc = (sum(s.accuracy_percentage for s in p.sessions) / n) if n else 0.0
            out.append({
                "id":                p.id,
                "name":              p.name,
                "age":               p.age,
                "gender":            p.gender,
                "diagnosis":         p.diagnosis,
                "affected_body_part": p.affected_body_part,
                "therapist_name":    p.therapist_name,
                "date_created":      p.date_created.isoformat(),
                "is_active":         p.is_active,
                "sessions_count":    n,
                "avg_accuracy":      round(avg_acc, 1),
                "photo":             base64.b64encode(p.photo).decode() if p.photo else None,
                "email":             p.email,
                "phone":             p.phone,
                "keypoint_consent":  p.keypoint_consent is True,
            })
        return JSONResponse(out)


@router.post("/api/patients")
async def create_patient(
    name: str = Form(...), age: int = Form(...), gender: str = Form(...),
    weight: float = Form(None), height: float = Form(None),
    diagnosis: str = Form(None), affected_body_part: str = Form(None),
    doctor_name: str = Form(None), therapist_name: str = Form(None),
    phone: str = Form(None), email: str = Form(None),
    medical_history: str = Form(None), previous_injury: str = Form(None),
    current_treatment: str = Form(None), exercise_plan: str = Form(None),
    photo: UploadFile = File(None),
    keypoint_consent: bool = Form(False),
    me: CurrentUser = Depends(require_clinical),
):
    with get_db() as db:
        # Duplicate checks are per-user: they must not reveal another user's patients.
        if email and owned_patients(db, me).filter(Patient.email == email).first():
            raise HTTPException(400, "Email already registered")
        if phone and owned_patients(db, me).filter(Patient.phone == phone).first():
            raise HTTPException(400, "Phone number already registered")
        photo_data = compress_photo(await photo.read()) if photo and photo.filename else None
        p = Patient(
            name=name, age=age, gender=gender, weight=weight, height=height,
            diagnosis=diagnosis, affected_body_part=affected_body_part,
            doctor_name=doctor_name, therapist_name=therapist_name,
            phone=phone, email=email, medical_history=medical_history,
            previous_injury=previous_injury, current_treatment=current_treatment,
            exercise_plan=exercise_plan, photo=photo_data,
            owner_id=me.id,
        )
        if keypoint_consent:
            p.keypoint_consent = True
            p.keypoint_consent_at = datetime.datetime.now()
            p.keypoint_consent_by = me.id
        db.add(p)
        db.flush()
        db.add(History(patient_id=p.id, action="Patient Created", details=f"{p.name} registered"))
        if keypoint_consent:
            db.add(History(patient_id=p.id, action="Keypoint Consent Granted",
                           details=f"Recorded by user {me.id}"))
        db.commit()
        return JSONResponse({"success": True, "message": "Patient created", "patient_id": p.id})


@router.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str, me: CurrentUser = Depends(require_clinical)):
    with get_db() as db:
        p = get_owned_patient(db, patient_id, me)
        return JSONResponse({
            "id": p.id, "name": p.name, "age": p.age, "gender": p.gender,
            "weight": p.weight, "height": p.height, "diagnosis": p.diagnosis,
            "affected_body_part": p.affected_body_part, "doctor_name": p.doctor_name,
            "therapist_name": p.therapist_name, "phone": p.phone, "email": p.email,
            "medical_history": p.medical_history, "previous_injury": p.previous_injury,
            "current_treatment": p.current_treatment, "exercise_plan": p.exercise_plan,
            "photo": base64.b64encode(p.photo).decode() if p.photo else None,
            "date_created": p.date_created.isoformat(), "is_active": p.is_active,
            "keypoint_consent": p.keypoint_consent is True,
            "keypoint_consent_at": p.keypoint_consent_at.isoformat() if p.keypoint_consent_at else None,
            "sessions": [session_summary(s) for s in p.sessions],
        })


@router.post("/api/patients/{patient_id}/keypoint-consent")
async def set_keypoint_consent(patient_id: str, payload: Dict[str, Any],
                               me: CurrentUser = Depends(require_clinical)):
    """Record or withdraw the patient's consent to keypoint recording.
    Body: {"granted": true|false}. Withdrawing drops any recording of this patient that is
    still in memory AND permanently deletes every saved recording file of this patient
    (sessions stay; their recording reference is marked deleted). The delete also runs when
    consent was already off, so a failed earlier attempt is cleaned up by retrying."""
    granted = payload.get("granted")
    if not isinstance(granted, bool):
        raise HTTPException(400, "'granted' must be true or false")
    with get_db() as db:
        p = get_owned_patient(db, patient_id, me)
        if (p.keypoint_consent is True) != granted:
            p.keypoint_consent = granted
            p.keypoint_consent_at = datetime.datetime.now()
            p.keypoint_consent_by = me.id
            db.add(History(
                patient_id=patient_id,
                action="Keypoint Consent Granted" if granted else "Keypoint Consent Withdrawn",
                details=f"Recorded by user {me.id}",
            ))
            db.commit()
    dropped = False
    deletion = None
    if not granted:
        dropped = lifecycle.withdraw_recording(get_context(me.id), patient_id)
        try:
            deletion = await run_in_threadpool(lifecycle.delete_patient_recordings, patient_id)
        except Exception as exc:        # consent is already withdrawn; report that the delete needs a retry
            print(f"Keypoints: deleting recordings of {patient_id} failed: {exc}")
            deletion = {"error": "deletion_failed"}
        if deletion.get("recordings_deleted"):
            with get_db() as db:
                db.add(History(patient_id=patient_id, action="Keypoint Recordings Deleted",
                               details=f"{deletion['recordings_deleted']} recording(s) deleted after consent withdrawal (user {me.id})"))
                db.commit()
    return JSONResponse({
        "success": True, "keypoint_consent": granted, "recording_dropped": dropped,
        "recordings_deleted": (deletion or {}).get("recordings_deleted", 0),
        "deletion_incomplete": bool(deletion and (deletion.get("error") or deletion.get("failed_files")
                                                  or deletion.get("mark_error"))),
    })


@router.put("/api/patients/{patient_id}")
async def update_patient(
    patient_id: str,
    name: str = Form(...), age: int = Form(...), gender: str = Form(...),
    weight: float = Form(None), height: float = Form(None),
    diagnosis: str = Form(None), affected_body_part: str = Form(None),
    doctor_name: str = Form(None), therapist_name: str = Form(None),
    phone: str = Form(None), email: str = Form(None),
    medical_history: str = Form(None), previous_injury: str = Form(None),
    current_treatment: str = Form(None), exercise_plan: str = Form(None),
    is_active: bool = Form(True), photo: UploadFile = File(None),
    me: CurrentUser = Depends(require_clinical),
):
    with get_db() as db:
        p = get_owned_patient(db, patient_id, me)
        for field, val in [
            ("name", name), ("age", age), ("gender", gender), ("weight", weight),
            ("height", height), ("diagnosis", diagnosis), ("affected_body_part", affected_body_part),
            ("doctor_name", doctor_name), ("therapist_name", therapist_name),
            ("phone", phone), ("email", email), ("medical_history", medical_history),
            ("previous_injury", previous_injury), ("current_treatment", current_treatment),
            ("exercise_plan", exercise_plan), ("is_active", is_active),
        ]:
            setattr(p, field, val)
        if photo and photo.filename:
            p.photo = compress_photo(await photo.read())
        db.add(History(patient_id=patient_id, action="Patient Updated", details=f"{p.name} updated"))
        db.commit()
        return JSONResponse({"success": True, "message": "Patient updated"})


@router.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: str, me: CurrentUser = Depends(require_clinical)):
    with get_db() as db:
        p = get_owned_patient(db, patient_id, me)
        p.is_active = False
        db.add(History(patient_id=patient_id, action="Patient Deactivated", details=f"{p.name} deactivated"))
        db.commit()
        return JSONResponse({"success": True, "message": "Patient deactivated"})
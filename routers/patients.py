import base64

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

from database import get_db
from models import Patient, History
from services.auth.deps import CurrentUser, require_clinical
from services.helpers import compress_photo, session_summary
from services.ownership import get_owned_patient, owned_patients

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
        db.add(p)
        db.flush()
        db.add(History(patient_id=p.id, action="Patient Created", details=f"{p.name} registered"))
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
            "sessions": [session_summary(s) for s in p.sessions],
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

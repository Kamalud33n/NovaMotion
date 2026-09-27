"""
Per-user data scoping for the clinical side.

A doctor/therapist only ever sees patients they registered (Patient.owner_id ==
their user id) and the sessions / reports / analytics that belong to those
patients. Every clinical router goes through these helpers instead of querying
Patient / SessionModel directly, so there is exactly one place that decides
"whose data is this".

A patient that isn't yours (or doesn't exist) is always a plain 404 - the API
never confirms that another user's record exists.
"""
from fastapi import HTTPException

from models import Patient, SessionModel
from services.auth.deps import CurrentUser


def owned_patients(db, me: CurrentUser):
    """Query of the patients registered by this user."""
    return db.query(Patient).filter(Patient.owner_id == me.id)


def owned_sessions(db, me: CurrentUser):
    """Query of every session that belongs to one of this user's patients."""
    return (
        db.query(SessionModel)
        .join(Patient, SessionModel.patient_id == Patient.id)
        .filter(Patient.owner_id == me.id)
    )


def get_owned_patient(db, patient_id, me: CurrentUser) -> Patient:
    """This user's patient, or a 404 (missing and not-yours look identical)."""
    p = owned_patients(db, me).filter(Patient.id == patient_id).first()
    if p is None:
        raise HTTPException(404, "Patient not found")
    return p

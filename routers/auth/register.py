"""Self-registration. New accounts start as 'pending' until an admin approves."""
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from database import get_db
from models import User
from routers.auth._common import clean_profile_fields, get_str
from services.auth.roles import SELF_REGISTER_ROLES, STATUS_PENDING
from services.auth.security import (
    hash_password, is_valid_email, normalize_email, validate_password_strength,
)

router = APIRouter()


@router.post("/api/auth/register")
def api_register(payload: Dict[str, Any]):
    fields = clean_profile_fields(payload, name_required=True)

    email = normalize_email(get_str(payload, "email"))
    if not is_valid_email(email):
        raise HTTPException(400, "Please enter a valid email address.")

    role = get_str(payload, "role").lower()
    if role not in SELF_REGISTER_ROLES:
        raise HTTPException(400, "Please choose Doctor or Therapist.")

    password = get_str(payload, "password", strip=False)
    err = validate_password_strength(password)
    if err:
        raise HTTPException(400, err)
    if password != get_str(payload, "confirm_password", strip=False):
        raise HTTPException(400, "Passwords do not match.")

    with get_db() as db:
        if db.query(User).filter(User.email == email).first() is not None:
            raise HTTPException(409, "An account with this email already exists.")
        db.add(User(
            full_name      = fields["full_name"],
            email          = email,
            phone          = fields.get("phone"),
            specialization = fields.get("specialization"),
            password_hash  = hash_password(password),
            role           = role,
            status         = STATUS_PENDING,
        ))
        try:
            db.commit()
        except IntegrityError:            # two submits racing on the same email
            db.rollback()
            raise HTTPException(409, "An account with this email already exists.")

    return JSONResponse(
        {"success": True,
         "message": "Registration submitted. An administrator needs to approve your account before you can sign in."},
        status_code=201,
    )

"""OTP-gated registration endpoints.

    POST /api/auth/register             form -> OTP emailed, no User row yet
    POST /api/auth/register/verify-otp  correct code -> real User row (status=pending)
    POST /api/auth/register/resend-otp  reissue a code for a signup in flight

This replaces the old routers/auth/register.py. app.py mounts this router
directly (routers/auth/__init__.py no longer includes a register module).
"""
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from database import get_db
from registration_otp import service
from registration_otp.mailer import send_otp_email, send_registration_received_email
from registration_otp.service import OTP_LENGTH, OTP_TTL_MINUTES, SignupError
from routers.auth._common import clean_profile_fields, get_str
from services.auth.roles import SELF_REGISTER_ROLES
from services.auth.security import (
    hash_password, is_valid_email, normalize_email, validate_password_strength,
)

router = APIRouter()


@router.post("/api/auth/register")
def api_register(payload: Dict[str, Any], background_tasks: BackgroundTasks):
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
        try:
            raw_otp = service.start_signup(
                db, full_name=fields["full_name"], email=email,
                phone=fields.get("phone"), specialization=fields.get("specialization"),
                password_hash=hash_password(password), role=role,
            )
        except SignupError as e:
            raise HTTPException(e.status_code, e.message)

    background_tasks.add_task(send_otp_email, email, fields["full_name"], raw_otp, OTP_TTL_MINUTES)
    return JSONResponse(
        {"success": True, "email": email,
         "message": f"We've emailed a {OTP_LENGTH}-digit code to {email}. Enter it to continue."},
    )


@router.post("/api/auth/register/verify-otp")
def api_verify_otp(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    email = normalize_email(get_str(payload, "email"))
    code = get_str(payload, "otp")
    if not email or not code:
        raise HTTPException(400, "Email and code are required.")

    with get_db() as db:
        try:
            user = service.verify_otp(db, email, code)
        except SignupError as e:
            raise HTTPException(e.status_code, e.message)
        full_name = user.full_name

    background_tasks.add_task(send_registration_received_email, email, full_name)
    return JSONResponse(
        {"success": True,
         "message": "Thank you! Your email is verified and your registration has been submitted. Please wait while an admin reviews your account — once approved, you'll be able to log in."},
        status_code=201,
    )


@router.post("/api/auth/register/resend-otp")
def api_resend_otp(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    email = normalize_email(get_str(payload, "email"))
    if not email:
        raise HTTPException(400, "Email is required.")

    with get_db() as db:
        try:
            raw_otp, full_name = service.resend_otp(db, email)
        except SignupError as e:
            raise HTTPException(e.status_code, e.message)

    background_tasks.add_task(send_otp_email, email, full_name, raw_otp, OTP_TTL_MINUTES)
    return JSONResponse({"success": True, "message": "A new code has been sent."})
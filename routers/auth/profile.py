"""The signed-in user's own details (used by the Settings page)."""
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from database import get_db
from models import User
from routers.auth._common import clean_profile_fields, get_str, note_failed_attempt, lock_message
from services.auth.deps import CurrentUser, get_current_user
from services.auth.security import (
    create_access_token, hash_password, set_auth_cookie, utcnow,
    validate_password_strength, verify_password,
)

router = APIRouter()


def _profile_dict(u: User) -> Dict[str, Any]:
    return {
        "id": u.id,
        "full_name": u.full_name,
        "email": u.email,
        "phone": u.phone,
        "specialization": u.specialization,
        "role": u.role,
        "status": u.status,
        "date_created": u.date_created.isoformat() if u.date_created else None,
        "last_login": u.last_login.isoformat() if u.last_login else None,
    }


@router.get("/api/auth/profile")
def api_get_profile(me: CurrentUser = Depends(get_current_user)):
    with get_db() as db:
        u = db.query(User).filter(User.id == me.id).first()
        return _profile_dict(u)


@router.put("/api/auth/profile")
def api_update_profile(payload: Dict[str, Any], me: CurrentUser = Depends(get_current_user)):
    """Editable by the user: name, phone, specialization.
    Email and role are NOT editable here — only an admin changes those."""
    fields = clean_profile_fields(payload, name_required=False)
    if not fields:
        raise HTTPException(400, "Nothing to update.")
    with get_db() as db:
        u = db.query(User).filter(User.id == me.id).first()
        for key, value in fields.items():
            setattr(u, key, value)
        db.commit()
        return {"success": True, "profile": _profile_dict(u)}


@router.post("/api/auth/change-password")
def api_change_password(payload: Dict[str, Any], me: CurrentUser = Depends(get_current_user)):
    current = get_str(payload, "current_password", strip=False)
    new = get_str(payload, "new_password", strip=False)
    now = utcnow()

    with get_db() as db:
        u = db.query(User).filter(User.id == me.id).first()

        # Wrong "current password" counts toward the same lockout as login, so a
        # stolen session can't be used to brute-force the password.
        if u.locked_until and u.locked_until > now:
            raise HTTPException(429, lock_message(u, now))
        if not verify_password(current, u.password_hash):
            locked = note_failed_attempt(u, now)
            db.commit()
            raise HTTPException(429 if locked else 400,
                                lock_message(u, now) if locked else "Current password is incorrect.")

        err = validate_password_strength(new)
        if err:
            raise HTTPException(400, err)
        if new != get_str(payload, "confirm_password", strip=False):
            raise HTTPException(400, "New passwords do not match.")
        if new == current:
            raise HTTPException(400, "New password must be different from the current one.")

        u.password_hash = hash_password(new)
        u.failed_login_attempts = 0
        u.locked_until = None
        db.commit()
        # The old cookie just became invalid (it's bound to the old password) —
        # hand this browser a fresh one so the user stays signed in.
        token = create_access_token(u.id, u.role, u.password_hash)

    resp = JSONResponse({"success": True, "message": "Password updated."})
    set_auth_cookie(resp, token)
    return resp

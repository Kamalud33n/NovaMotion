"""Validation + lockout helpers shared by the auth routes."""
import datetime
import math
import re
from typing import Any, Dict, Optional

from fastapi import HTTPException

from models import User
from services.auth.config import settings

PHONE_RE = re.compile(r"^[0-9+()\-\s]{6,20}$")


def get_str(payload: Dict[str, Any], key: str, *, strip: bool = True) -> str:
    """String value for `key` ('' if missing/null). A non-string value is a 400."""
    v = payload.get(key)
    if v is None:
        return ""
    if not isinstance(v, str):
        raise HTTPException(400, f"Invalid value for {key}.")
    return v.strip() if strip else v


def clean_profile_fields(payload: Dict[str, Any], *, name_required: bool) -> Dict[str, Optional[str]]:
    """Validate full_name / phone / specialization. Only keys present in the
    payload are returned (so a PUT can update just one field). Optional fields
    come back as None when blank."""
    out: Dict[str, Optional[str]] = {}

    if "full_name" in payload or name_required:
        name = get_str(payload, "full_name")
        if len(name) < 2:
            raise HTTPException(400, "Please enter your full name.")
        if len(name) > 100:
            raise HTTPException(400, "Name is too long (max 100 characters).")
        out["full_name"] = name

    if "phone" in payload:
        phone = get_str(payload, "phone")
        if phone and not PHONE_RE.match(phone):
            raise HTTPException(400, "Phone number can only contain digits, +, -, spaces and brackets (6-20 characters).")
        out["phone"] = phone or None

    if "specialization" in payload:
        spec = get_str(payload, "specialization")
        if len(spec) > 100:
            raise HTTPException(400, "Specialization is too long (max 100 characters).")
        out["specialization"] = spec or None

    return out


def note_failed_attempt(user: User, now) -> bool:
    """Count one wrong password. Returns True if this attempt locked the
    account (the caller must commit)."""
    attempts = (user.failed_login_attempts or 0) + 1
    if attempts >= settings.max_login_attempts:
        user.locked_until = now + datetime.timedelta(minutes=settings.lockout_minutes)
        user.failed_login_attempts = 0
        return True
    user.failed_login_attempts = attempts
    return False


def lock_message(user: User, now) -> str:
    mins = max(1, math.ceil((user.locked_until - now).total_seconds() / 60))
    return f"Too many failed attempts. Try again in {mins} minute{'s' if mins != 1 else ''}."

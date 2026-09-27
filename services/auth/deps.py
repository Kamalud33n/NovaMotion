"""
FastAPI dependencies for authentication and role checks.

  API routes   : Depends(require_clinical) / Depends(require_admin)
                 -> 401 JSON if not logged in, 403 JSON if wrong role.
  Page routes  : Depends(require_clinical_page) / Depends(require_admin_page)
                 -> 303 redirect to /login if not logged in, and to the
                    user's own home page if their role can't open this one.

The user is re-read from the DB on every request, so suspending/rejecting an
account takes effect immediately, and a password change invalidates older
cookies (the cookie alone is never trusted for status or role). Dependencies return a plain CurrentUser dataclass, not an ORM
object, so nothing is left attached to a closed DB session.
"""
import hmac
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, Request

from database import get_db
from models import User
from services.auth.config import settings
from services.auth.roles import (
    CLINICAL_ROLES, ROLE_ADMIN, STATUS_APPROVED, home_for_role,
)
from services.auth.security import decode_access_token, password_fingerprint

LOGIN_PATH = "/login"


@dataclass(frozen=True)
class CurrentUser:
    id: str
    full_name: str
    email: str
    role: str
    status: str


def get_optional_user(request: Request) -> Optional[CurrentUser]:
    """The logged-in, approved user for this request — or None."""
    token = request.cookies.get(settings.cookie_name)
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload or not payload.get("sub"):
        return None
    with get_db() as db:
        u = db.query(User).filter(User.id == payload["sub"]).first()
        if u is None or u.status != STATUS_APPROVED:
            return None
        # Token must match the *current* password — a password change/reset
        # logs out every older session.
        if not hmac.compare_digest(str(payload.get("pv", "")), password_fingerprint(u.password_hash)):
            return None
        return CurrentUser(id=u.id, full_name=u.full_name, email=u.email,
                           role=u.role, status=u.status)


def get_current_user(request: Request) -> CurrentUser:
    """Any logged-in approved user (API: 401 otherwise)."""
    user = get_optional_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_roles(*roles: str):
    """API guard: 401 if not logged in, 403 if the role isn't allowed."""
    def dependency(request: Request) -> CurrentUser:
        user = get_optional_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="You do not have access to this resource")
        return user
    return dependency


def require_roles_page(*roles: str):
    """Page guard: redirects instead of returning JSON errors."""
    def dependency(request: Request) -> CurrentUser:
        user = get_optional_user(request)
        if user is None:
            raise HTTPException(status_code=303, headers={"Location": LOGIN_PATH})
        if user.role not in roles:
            raise HTTPException(status_code=303, headers={"Location": home_for_role(user.role)})
        return user
    return dependency


# Ready-made guards (doctor + therapist are the same "clinical" tier)
require_clinical      = require_roles(*CLINICAL_ROLES)
require_admin         = require_roles(ROLE_ADMIN)
require_clinical_page = require_roles_page(*CLINICAL_ROLES)
require_admin_page    = require_roles_page(ROLE_ADMIN)

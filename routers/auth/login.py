"""Login page, login/logout API, and "who am I"."""
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from database import get_db
from models import User
from routers.auth._common import get_str, lock_message, note_failed_attempt
from services.auth.deps import CurrentUser, get_current_user, get_optional_user
from services.auth.roles import (
    STATUS_APPROVED, STATUS_PENDING, STATUS_REJECTED, STATUS_SUSPENDED, home_for_role,
)
from services.auth.security import (
    burn_password_check, clear_auth_cookie, create_access_token, normalize_email,
    set_auth_cookie, utcnow, verify_password,
)
from services.auth.web import render

router = APIRouter()

INVALID_LOGIN = "Invalid email or password."

_STATUS_MESSAGES = {
    STATUS_PENDING:   "Thank you for registering. Your account is currently awaiting admin approval — please wait, and you'll be able to sign in as soon as it's approved.",
    STATUS_REJECTED:  "Your registration was not approved. Please contact the clinic administrator.",
    STATUS_SUSPENDED: "Your account has been suspended. Please contact the clinic administrator.",
}


@router.get("/login", response_class=HTMLResponse)
async def page_login(request: Request):
    user = get_optional_user(request)
    if user is not None:
        return RedirectResponse(url=home_for_role(user.role), status_code=303)
    return render(request, "auth/login.html")


# Plain `def` (not async) on purpose: password hashing takes a few hundred ms,
# and FastAPI runs sync routes in a worker thread so the camera stream and
# every other request keep flowing meanwhile.
@router.post("/api/auth/login")
def api_login(payload: Dict[str, Any]):
    email = normalize_email(get_str(payload, "email"))
    password = get_str(payload, "password", strip=False)
    if not email or not password:
        raise HTTPException(400, "Email and password are required.")

    now = utcnow()
    with get_db() as db:
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            burn_password_check(password)          # same timing as a real check
            raise HTTPException(401, INVALID_LOGIN)

        if user.locked_until and user.locked_until > now:
            raise HTTPException(429, lock_message(user, now))

        if not verify_password(password, user.password_hash):
            locked = note_failed_attempt(user, now)
            db.commit()
            if locked:
                raise HTTPException(429, lock_message(user, now))
            raise HTTPException(401, INVALID_LOGIN)

        # Password is right — now tell them if the account can't sign in yet.
        if user.status != STATUS_APPROVED:
            raise HTTPException(403, _STATUS_MESSAGES.get(user.status, "This account cannot sign in."))

        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = now
        db.commit()
        role = user.role
        token = create_access_token(user.id, role, user.password_hash)

    resp = JSONResponse({"success": True, "role": role, "redirect": home_for_role(role)})
    set_auth_cookie(resp, token)
    return resp


@router.post("/api/auth/logout")
def api_logout():
    resp = JSONResponse({"success": True})
    clear_auth_cookie(resp)
    return resp


@router.get("/logout")
async def page_logout():
    resp = RedirectResponse(url="/login", status_code=303)
    clear_auth_cookie(resp)
    return resp


@router.get("/api/auth/me")
def api_me(user: CurrentUser = Depends(get_current_user)):
    return {
        "id": user.id, "full_name": user.full_name, "email": user.email,
        "role": user.role, "status": user.status,
    }

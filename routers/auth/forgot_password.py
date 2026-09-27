"""Forgot-password: request a reset link by email."""
import datetime
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse

from database import get_db
from models import PasswordResetToken, User
from routers.auth._common import get_str
from services.auth.config import settings
from services.auth.mailer import send_password_reset_email
from services.auth.roles import STATUS_APPROVED
from services.auth.security import generate_reset_token, is_valid_email, normalize_email, utcnow
from services.auth.web import render

router = APIRouter()

# Same answer whether or not the email is registered, approved, rate-limited,
# or SMTP is down — so this endpoint can't be used to discover accounts.
GENERIC_MESSAGE = "If an account with that email exists, a password reset link has been sent."


@router.get("/forgot-password", response_class=HTMLResponse)
async def page_forgot_password(request: Request):
    return render(request, "auth/forgot_password.html")


@router.post("/api/auth/forgot-password")
def api_forgot_password(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    email = normalize_email(get_str(payload, "email"))
    if not is_valid_email(email):
        raise HTTPException(400, "Please enter a valid email address.")

    now = utcnow()
    with get_db() as db:
        user = (db.query(User)
                  .filter(User.email == email, User.status == STATUS_APPROVED)
                  .first())
        if user is not None:
            last = (db.query(PasswordResetToken)
                      .filter(PasswordResetToken.user_id == user.id)
                      .order_by(PasswordResetToken.id.desc())
                      .first())
            cooling_down = (
                last is not None and last.created_at is not None
                and (now - last.created_at).total_seconds() < settings.reset_request_cooldown_seconds
            )
            if not cooling_down:
                # Only the newest link works: retire any earlier unused ones.
                (db.query(PasswordResetToken)
                   .filter(PasswordResetToken.user_id == user.id,
                           PasswordResetToken.used_at.is_(None))
                   .update({"used_at": now}, synchronize_session=False))
                raw, token_hash = generate_reset_token()
                db.add(PasswordResetToken(
                    user_id    = user.id,
                    token_hash = token_hash,
                    created_at = now,
                    expires_at = now + datetime.timedelta(minutes=settings.reset_token_expire_minutes),
                ))
                db.commit()
                # SMTP is slow/blocking — send after the response is returned.
                background_tasks.add_task(send_password_reset_email, user.email, user.full_name, raw)

    return {"success": True, "message": GENERIC_MESSAGE}

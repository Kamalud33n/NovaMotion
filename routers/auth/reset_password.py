"""Reset password using the emailed one-time link."""
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from database import get_db
from models import PasswordResetToken, User
from routers.auth._common import get_str
from services.auth.roles import STATUS_APPROVED
from services.auth.security import (
    hash_password, hash_reset_token, utcnow, validate_password_strength,
)
from services.auth.web import render

router = APIRouter()

INVALID_LINK = "This reset link is invalid or has expired. Please request a new one."


def _valid_token_row(db, raw_token: str, now) -> Optional[PasswordResetToken]:
    """The reset-token row if the token exists, is unused, unexpired, and its
    account is still approved — else None."""
    if not raw_token:
        return None
    row = (db.query(PasswordResetToken)
             .filter(PasswordResetToken.token_hash == hash_reset_token(raw_token))
             .first())
    if row is None or row.used_at is not None or row.expires_at <= now:
        return None
    user = db.query(User).filter(User.id == row.user_id).first()
    if user is None or user.status != STATUS_APPROVED:
        return None
    return row


@router.get("/reset-password", response_class=HTMLResponse)
async def page_reset_password(request: Request, token: str = ""):
    with get_db() as db:
        valid = _valid_token_row(db, token, utcnow()) is not None
    return render(request, "auth/reset_password.html", token=token, token_valid=valid)


@router.post("/api/auth/reset-password")
def api_reset_password(payload: Dict[str, Any]):
    raw = get_str(payload, "token")
    password = get_str(payload, "password", strip=False)
    now = utcnow()

    with get_db() as db:
        row = _valid_token_row(db, raw, now)
        if row is None:
            raise HTTPException(400, INVALID_LINK)

        err = validate_password_strength(password)
        if err:
            raise HTTPException(400, err)
        if password != get_str(payload, "confirm_password", strip=False):
            raise HTTPException(400, "Passwords do not match.")

        user = db.query(User).filter(User.id == row.user_id).first()
        user.password_hash = hash_password(password)     # also kills every older login cookie
        user.failed_login_attempts = 0
        user.locked_until = None
        (db.query(PasswordResetToken)
           .filter(PasswordResetToken.user_id == user.id,
                   PasswordResetToken.used_at.is_(None))
           .update({"used_at": now}, synchronize_session=False))
        db.commit()

    return {"success": True, "message": "Password updated. You can now sign in."}

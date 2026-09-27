"""OTP-gated registration — no DB access happens outside these functions.

Flow: start_signup() sends a code -> verify_otp() checks it and only then
creates the real `models.User` row (status=pending, same as before, so the
existing admin-approval flow needs no changes). resend_otp() reissues a
code for a signup that's already in flight.

A rejected account frees up its email again after REJECTED_RETRY_HOURS.
"""
import datetime
import hashlib
import hmac
import math
import secrets

from sqlalchemy.orm import Session

from models import PendingSignup, User
from services.auth.config import settings
from services.auth.roles import STATUS_APPROVED, STATUS_PENDING, STATUS_REJECTED
from services.auth.security import utcnow

OTP_LENGTH              = 6
OTP_TTL_MINUTES         = 10
MAX_OTP_ATTEMPTS        = 5
RESEND_COOLDOWN_SECONDS = 60
REJECTED_RETRY_HOURS    = 24


class SignupError(Exception):
    """Message + the HTTP status code the route should answer with."""
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


# ── OTP helpers ─────────────────────────────────────────────────────────
def _generate_otp() -> str:
    return f"{secrets.randbelow(10 ** OTP_LENGTH):0{OTP_LENGTH}d}"


def _hash_otp(otp: str, email: str) -> str:
    """Keyed + per-email salted, so a leaked pending_signups row alone isn't
    enough to brute-force a 6-digit code offline."""
    return hmac.new(
        settings.secret_key.encode("utf-8"),
        f"{email}:{otp}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _cooldown_remaining(last_sent_at: datetime.datetime, now: datetime.datetime) -> int:
    elapsed = (now - last_sent_at).total_seconds()
    return max(0, math.ceil(RESEND_COOLDOWN_SECONDS - elapsed))


def _rejected_retry_message(rejected_at: datetime.datetime, now: datetime.datetime) -> str:
    remaining = datetime.timedelta(hours=REJECTED_RETRY_HOURS) - (now - rejected_at)
    hours = max(1, math.ceil(remaining.total_seconds() / 3600))
    return (f"Your previous request for this email was rejected. "
            f"You can try again in {hours} hour{'s' if hours != 1 else ''}.")


# ── public API ──────────────────────────────────────────────────────────
def start_signup(db: Session, *, full_name: str, email: str, phone, specialization,
                  password_hash: str, role: str) -> str:
    """Validate the email is free to (re)register, upsert a PendingSignup
    row and return the raw OTP for the caller to email out."""
    now = utcnow()

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user is not None:
        if existing_user.status in (STATUS_APPROVED, STATUS_PENDING):
            raise SignupError(409, "An account with this email already exists.")
        if existing_user.status == STATUS_REJECTED:
            rejected_at = existing_user.approved_at or existing_user.date_created
            if now - rejected_at < datetime.timedelta(hours=REJECTED_RETRY_HOURS):
                raise SignupError(403, _rejected_retry_message(rejected_at, now))
            db.delete(existing_user)   # retry window passed — free the email
            db.flush()

    pending = db.query(PendingSignup).filter(PendingSignup.email == email).first()
    if pending is not None:
        wait = _cooldown_remaining(pending.last_sent_at, now)
        if wait > 0:
            raise SignupError(429, f"Please wait {wait}s before requesting another code.")

    raw_otp = _generate_otp()
    fields = dict(
        full_name=full_name, phone=phone, specialization=specialization,
        password_hash=password_hash, role=role,
        otp_hash=_hash_otp(raw_otp, email),
        otp_expires_at=now + datetime.timedelta(minutes=OTP_TTL_MINUTES),
        otp_attempts=0, last_sent_at=now,
    )
    if pending is None:
        db.add(PendingSignup(email=email, **fields))
    else:
        for key, value in fields.items():
            setattr(pending, key, value)
    db.commit()
    return raw_otp


def resend_otp(db: Session, email: str) -> tuple:
    """Reissue a fresh code for a signup that's already in flight.
    Returns (raw_otp, full_name) — the name is for the email greeting."""
    now = utcnow()
    pending = db.query(PendingSignup).filter(PendingSignup.email == email).first()
    if pending is None:
        raise SignupError(404, "No pending registration found for this email.")

    wait = _cooldown_remaining(pending.last_sent_at, now)
    if wait > 0:
        raise SignupError(429, f"Please wait {wait}s before requesting another code.")

    raw_otp = _generate_otp()
    pending.otp_hash = _hash_otp(raw_otp, email)
    pending.otp_expires_at = now + datetime.timedelta(minutes=OTP_TTL_MINUTES)
    pending.otp_attempts = 0
    pending.last_sent_at = now
    db.commit()
    return raw_otp, pending.full_name


def verify_otp(db: Session, email: str, code: str) -> User:
    """Check the code and, only on success, create the real (pending-status)
    User row and drop the PendingSignup row."""
    now = utcnow()
    pending = db.query(PendingSignup).filter(PendingSignup.email == email).first()
    if pending is None:
        raise SignupError(404, "No pending registration found for this email. Please register again.")

    if now > pending.otp_expires_at:
        db.delete(pending)
        db.commit()
        raise SignupError(400, "This code has expired. Please register again to get a new one.")

    if pending.otp_attempts >= MAX_OTP_ATTEMPTS:
        db.delete(pending)
        db.commit()
        raise SignupError(429, "Too many incorrect attempts. Please register again to get a new code.")

    if not hmac.compare_digest(_hash_otp(code, email), pending.otp_hash):
        pending.otp_attempts += 1
        db.commit()
        left = MAX_OTP_ATTEMPTS - pending.otp_attempts
        raise SignupError(400, f"Incorrect code. {left} attempt{'s' if left != 1 else ''} left.")

    user = User(
        full_name=pending.full_name, email=pending.email, phone=pending.phone,
        specialization=pending.specialization, password_hash=pending.password_hash,
        role=pending.role, status=STATUS_PENDING,
    )
    db.add(user)
    db.delete(pending)
    db.commit()
    db.refresh(user)
    return user
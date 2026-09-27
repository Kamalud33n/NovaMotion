"""
Crypto + token helpers. No DB access, no FastAPI routing here.

Passwords : PBKDF2-HMAC-SHA256 (stdlib), per-user random salt. The iteration
            count is stored inside the hash string, so it can be raised later
            without breaking existing accounts.
Sessions  : short JWT (HS256, signed with SECRET_KEY) kept in an HttpOnly
            cookie. The cookie is what lets <img src="/video_feed"> and normal
            page loads authenticate — they can't send an Authorization header.
Reset     : random URL-safe token; only its SHA-256 is stored in the DB.
"""
import base64
import datetime
import hashlib
import hmac
import os
import re
import secrets
from typing import Optional, Tuple

import jwt

from services.auth.config import settings

_ALGO       = "pbkdf2_sha256"
_ITERATIONS = 600_000
_JWT_ALG    = "HS256"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ── time ─────────────────────────────────────────────────────────────────
def utcnow() -> datetime.datetime:
    """Naive UTC 'now' — matches how auth timestamps are stored/compared."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


# ── input helpers ────────────────────────────────────────────────────────
def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(email) and len(email) <= 150 and bool(_EMAIL_RE.match(email))


def validate_password_strength(password: str) -> Optional[str]:
    """Return an error message, or None if the password is acceptable."""
    if not password or len(password) < settings.password_min_length:
        return f"Password must be at least {settings.password_min_length} characters."
    if len(password) > 128:
        return "Password must be at most 128 characters."
    if not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
        return "Password must contain at least one letter and one number."
    return None


# ── passwords ────────────────────────────────────────────────────────────
def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"{_ALGO}${_ITERATIONS}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algo, iters, salt_b64, dk_b64 = stored_hash.split("$")
        if algo != _ALGO:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iters))
    except Exception:
        return False
    return hmac.compare_digest(actual, expected)


_dummy_hash: Optional[str] = None


def burn_password_check(password: str) -> None:
    """Spend the same time as a real verify. Call when the email isn't found
    so response time doesn't reveal which emails are registered."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password("not-a-real-password")
    verify_password(password, _dummy_hash)


# ── access token (login session) ─────────────────────────────────────────
def password_fingerprint(password_hash: str) -> str:
    """Short keyed digest of the stored password hash. It rides inside the
    login token, so changing/resetting a password instantly invalidates every
    older cookie — without needing any extra DB column."""
    return hmac.new(settings.secret_key.encode("utf-8"),
                    password_hash.encode("utf-8"), hashlib.sha256).hexdigest()[:20]


def create_access_token(user_id: str, role: str, password_hash: str) -> str:
    now = utcnow()
    payload = {
        "sub":  user_id,
        "role": role,
        "pv":   password_fingerprint(password_hash),
        "iat":  now,
        "exp":  now + datetime.timedelta(hours=settings.session_hours),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=_JWT_ALG)


def decode_access_token(token: str) -> Optional[dict]:
    """Payload if the token is valid and unexpired, else None."""
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[_JWT_ALG])
    except jwt.PyJWTError:
        return None


# ── cookie ───────────────────────────────────────────────────────────────
def set_auth_cookie(response, token: str) -> None:
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        max_age=settings.session_hours * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


def clear_auth_cookie(response) -> None:
    response.delete_cookie(
        key=settings.cookie_name,
        path="/",
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )


# ── password-reset token ─────────────────────────────────────────────────
def hash_reset_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def generate_reset_token() -> Tuple[str, str]:
    """(raw_token, token_hash). Email the raw one; store only the hash."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_reset_token(raw)

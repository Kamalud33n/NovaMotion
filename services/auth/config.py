"""
Auth settings — every value comes from the environment (.env), nothing is
hardcoded here. database.py already calls load_dotenv(); we call it again so
this module also works when imported on its own (tests, scripts).

Groups
  required  : the app refuses to start if any of these is missing/invalid
              (all problems are reported together in one error)
  SMTP      : optional group. Leave SMTP_HOST empty and email is simply
              "not configured" (forgot-password then can't send mail).
              If SMTP_HOST is set, SMTP_PORT / SMTP_FROM_EMAIL /
              SMTP_TIMEOUT_SECONDS become required.
  admin seed: optional group. ADMIN_EMAIL / ADMIN_PASSWORD / ADMIN_NAME —
              set all three or none.

See .env.example for every variable with comments.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

_TRUE  = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

MIN_SECRET_KEY_LENGTH = 32


class AuthConfigError(RuntimeError):
    """Raised at startup when required auth settings are missing/invalid."""


def _raw(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


@dataclass(frozen=True)
class AuthSettings:
    # security
    secret_key: str
    cookie_name: str
    cookie_secure: bool
    session_hours: int
    password_min_length: int
    max_login_attempts: int
    lockout_minutes: int
    # app / links
    app_name: str
    app_base_url: str
    reset_token_expire_minutes: int
    reset_request_cooldown_seconds: int
    # smtp (optional group)
    smtp_host: Optional[str]
    smtp_port: Optional[int]
    smtp_username: Optional[str]
    smtp_password: Optional[str]
    smtp_use_tls: bool
    smtp_use_ssl: bool
    smtp_from_email: Optional[str]
    smtp_from_name: Optional[str]
    smtp_timeout_seconds: Optional[int]
    # first-admin seed (optional group)
    admin_name: Optional[str]
    admin_email: Optional[str]
    admin_password: Optional[str]


def _load() -> AuthSettings:
    errors: List[str] = []

    def req_str(name: str) -> str:
        v = _raw(name)
        if v is None:
            errors.append(f"{name} is required")
            return ""
        return v

    def req_int(name: str, minimum: int = 1) -> int:
        v = _raw(name)
        if v is None:
            errors.append(f"{name} is required")
            return 0
        try:
            n = int(v)
        except ValueError:
            errors.append(f"{name} must be a whole number (got {v!r})")
            return 0
        if n < minimum:
            errors.append(f"{name} must be >= {minimum} (got {n})")
        return n

    def req_bool(name: str) -> bool:
        v = _raw(name)
        if v is None:
            errors.append(f"{name} is required (true/false)")
            return False
        if v.lower() in _TRUE:
            return True
        if v.lower() in _FALSE:
            return False
        errors.append(f"{name} must be true or false (got {v!r})")
        return False

    def opt_bool(name: str) -> bool:
        v = _raw(name)
        if v is None:
            return False
        if v.lower() in _TRUE:
            return True
        if v.lower() in _FALSE:
            return False
        errors.append(f"{name} must be true or false (got {v!r})")
        return False

    # ── required ────────────────────────────────────────────────────────
    secret_key = req_str("SECRET_KEY")
    if secret_key and len(secret_key) < MIN_SECRET_KEY_LENGTH:
        errors.append(f"SECRET_KEY must be at least {MIN_SECRET_KEY_LENGTH} characters")

    cookie_name         = req_str("AUTH_COOKIE_NAME")
    cookie_secure       = req_bool("AUTH_COOKIE_SECURE")
    session_hours       = req_int("AUTH_SESSION_HOURS")
    password_min_length = req_int("PASSWORD_MIN_LENGTH", minimum=6)
    max_attempts        = req_int("MAX_LOGIN_ATTEMPTS")
    lockout_minutes     = req_int("LOCKOUT_MINUTES")
    app_name            = req_str("APP_NAME")
    app_base_url        = req_str("APP_BASE_URL").rstrip("/")
    reset_minutes       = req_int("RESET_TOKEN_EXPIRE_MINUTES")
    reset_cooldown      = req_int("RESET_REQUEST_COOLDOWN_SECONDS", minimum=0)

    if app_base_url and not app_base_url.lower().startswith(("http://", "https://")):
        errors.append("APP_BASE_URL must start with http:// or https://")

    # ── SMTP (optional group) ───────────────────────────────────────────
    smtp_host = _raw("SMTP_HOST")
    smtp_port = smtp_from_email = smtp_timeout = None
    smtp_use_tls = opt_bool("SMTP_USE_TLS")
    smtp_use_ssl = opt_bool("SMTP_USE_SSL")
    if smtp_use_tls and smtp_use_ssl:
        errors.append("SMTP_USE_TLS and SMTP_USE_SSL cannot both be true "
                      "(TLS = STARTTLS, usually port 587; SSL = implicit, usually port 465)")
    if smtp_host:
        smtp_port       = req_int("SMTP_PORT")
        smtp_from_email = req_str("SMTP_FROM_EMAIL")
        smtp_timeout    = req_int("SMTP_TIMEOUT_SECONDS")

    # ── first-admin seed (optional group, all-or-nothing) ───────────────
    admin_name, admin_email, admin_password = (
        _raw("ADMIN_NAME"), _raw("ADMIN_EMAIL"), _raw("ADMIN_PASSWORD"))
    seed_vals = (admin_name, admin_email, admin_password)
    if any(seed_vals) and not all(seed_vals):
        errors.append("ADMIN_NAME, ADMIN_EMAIL and ADMIN_PASSWORD must be set together "
                      "(or all left empty)")

    if errors:
        raise AuthConfigError(
            "Invalid auth configuration in .env:\n  - " + "\n  - ".join(errors) +
            "\nCopy .env.example to .env and fill in the values."
        )

    return AuthSettings(
        secret_key=secret_key, cookie_name=cookie_name, cookie_secure=cookie_secure,
        session_hours=session_hours, password_min_length=password_min_length,
        max_login_attempts=max_attempts, lockout_minutes=lockout_minutes,
        app_name=app_name, app_base_url=app_base_url,
        reset_token_expire_minutes=reset_minutes,
        reset_request_cooldown_seconds=reset_cooldown,
        smtp_host=smtp_host, smtp_port=smtp_port,
        smtp_username=_raw("SMTP_USERNAME"), smtp_password=_raw("SMTP_PASSWORD"),
        smtp_use_tls=smtp_use_tls, smtp_use_ssl=smtp_use_ssl,
        smtp_from_email=smtp_from_email, smtp_from_name=_raw("SMTP_FROM_NAME"),
        smtp_timeout_seconds=smtp_timeout,
        admin_name=admin_name, admin_email=admin_email, admin_password=admin_password,
    )


settings = _load()

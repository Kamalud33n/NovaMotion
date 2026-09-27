"""
First-admin seeding. Call seed_admin() once at startup (after init_db()).

Creates the admin from ADMIN_NAME / ADMIN_EMAIL / ADMIN_PASSWORD in .env, but
only if NO admin account exists yet — so it never overwrites or resets an
admin you've already changed, and removing the .env values later is safe.
"""
from database import get_db
from models import User
from services.auth.config import AuthConfigError, settings
from services.auth.roles import ROLE_ADMIN, STATUS_APPROVED
from services.auth.security import (
    hash_password, is_valid_email, normalize_email, utcnow, validate_password_strength,
)


def seed_admin() -> None:
    if not (settings.admin_email and settings.admin_password and settings.admin_name):
        return

    email = normalize_email(settings.admin_email)
    if not is_valid_email(email):
        raise AuthConfigError(f"ADMIN_EMAIL is not a valid email address: {settings.admin_email!r}")
    err = validate_password_strength(settings.admin_password)
    if err:
        raise AuthConfigError(f"ADMIN_PASSWORD rejected: {err}")

    with get_db() as db:
        if db.query(User).filter(User.role == ROLE_ADMIN).count() > 0:
            return
        if db.query(User).filter(User.email == email).first() is not None:
            print(f"Admin seed skipped: {email} already exists as a non-admin account.")
            return
        db.add(User(
            full_name     = settings.admin_name,
            email         = email,
            password_hash = hash_password(settings.admin_password),
            role          = ROLE_ADMIN,
            status        = STATUS_APPROVED,
            approved_at   = utcnow(),
        ))
        db.commit()
        print(f"Admin account seeded from .env: {email}")

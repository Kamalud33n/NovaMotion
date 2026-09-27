"""
services/auth — authentication / authorization building blocks.

    config.py     all settings, read once from .env (nothing hardcoded)
    roles.py      role + account-status constants
    security.py   password hashing, access tokens, reset tokens, cookies
    deps.py       FastAPI dependencies: current user, role guards
    mailer.py     SMTP email (password reset)
    bootstrap.py  first-admin seeding from .env

Import from the submodules directly (e.g. `from services.auth.deps import ...`).
This file is intentionally empty so importing the package has no side effects.
"""

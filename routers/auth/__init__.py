"""
/routers/auth — everything about signing in.

    login.py            /login page, POST /api/auth/login, logout, /api/auth/me
    forgot_password.py  /forgot-password page + POST /api/auth/forgot-password
    reset_password.py   /reset-password page + POST /api/auth/reset-password
    profile.py          GET/PUT /api/auth/profile, POST /api/auth/change-password

Registration lives in registration_otp/routes.py (OTP-gated) and is mounted
separately in app.py — not part of this combiner.

app.py does `app.include_router(auth.router)` once; this file combines them.
"""
from fastapi import APIRouter

from routers.auth import forgot_password, login, profile, reset_password

router = APIRouter()
for _r in (login, forgot_password, reset_password, profile):
    router.include_router(_r.router)
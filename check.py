"""
Run this INSIDE your thero project folder (same place as .env, database.py):

    python check_admin.py

It connects to your MySQL DB using the same .env values the app uses,
lists every row in `users`, and — if you pass --reset — resets the
admin password to whatever you give it (or creates the admin row if
it's missing entirely).

Usage:
    python check_admin.py                     # just inspect
    python check_admin.py --reset "NewPass123"    # force-reset/create admin
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_db, init_db          # noqa: E402
from models import User                        # noqa: E402
from services.auth.security import hash_password, normalize_email, utcnow  # noqa: E402
from services.auth.roles import ROLE_ADMIN, STATUS_APPROVED                # noqa: E402
from services.auth.config import settings as auth_settings                 # noqa: E402

def main():
    init_db()  # make sure tables exist / DB is reachable

    with get_db() as db:
        users = db.query(User).all()
        if not users:
            print("No rows at all in `users` table. The admin was never seeded.")
        else:
            print(f"Found {len(users)} user(s):\n")
            for u in users:
                print(f"- email={u.email!r} role={u.role} status={u.status} "
                      f"failed_attempts={u.failed_login_attempts} "
                      f"locked_until={u.locked_until}")

        if "--reset" in sys.argv:
            idx = sys.argv.index("--reset")
            new_password = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
            if not new_password:
                print("\nUsage: python check_admin.py --reset \"NewPass123\"")
                return

            target_email = normalize_email(auth_settings.admin_email or "admin@novamotion.local")
            admin = db.query(User).filter(User.email == target_email).first()

            if admin is None:
                admin = User(
                    full_name=auth_settings.admin_name or "Admin",
                    email=target_email,
                    password_hash=hash_password(new_password),
                    role=ROLE_ADMIN,
                    status=STATUS_APPROVED,
                    approved_at=utcnow(),
                )
                db.add(admin)
                print(f"\nCreated new admin: {target_email}")
            else:
                admin.password_hash = hash_password(new_password)
                admin.role = ROLE_ADMIN
                admin.status = STATUS_APPROVED
                admin.failed_login_attempts = 0
                admin.locked_until = None
                print(f"\nReset password for existing user: {target_email} (role set to admin, unlocked)")

            db.commit()
            print(f"You can now log in with: {target_email} / {new_password}")

if __name__ == "__main__":
    main()
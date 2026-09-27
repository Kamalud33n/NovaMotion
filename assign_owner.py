"""
One-time helper for data that existed BEFORE per-user scoping was added.

Every patient now belongs to the doctor/therapist who registered them, and each
user only sees their own. Old patients have no owner yet, so nobody sees them
until you hand them to an account:

    python assign_owner.py --list                  # accounts + how many patients are unassigned
    python assign_owner.py you@clinic.com          # give ALL unassigned patients to that account
    python assign_owner.py you@clinic.com --yes    # same, without the confirmation prompt

Only patients with no owner (or whose owner account no longer exists) are touched.
A patient that already belongs to someone is never moved. Sessions and reports
follow their patient automatically, so nothing else needs migrating.
"""
import argparse
import sys

from sqlalchemy import func, select

from database import get_db, init_db
from models import Patient, User
from services.auth.roles import CLINICAL_ROLES


def _unassigned(db):
    """Patients with no owner, or an owner id that isn't a real user any more."""
    return db.query(Patient).filter(
        Patient.owner_id.is_(None) | ~Patient.owner_id.in_(select(User.id))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Assign legacy (owner-less) patients to a doctor/therapist account.")
    parser.add_argument("email", nargs="?", help="email of the doctor/therapist who should own the unassigned patients")
    parser.add_argument("--list", action="store_true", help="show accounts and the unassigned-patient count, then exit")
    parser.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    args = parser.parse_args()

    if not args.email and not args.list:
        parser.print_help()
        return 1

    init_db()  # makes sure the patients.owner_id column exists before we touch it

    with get_db() as db:
        unassigned = _unassigned(db).count()

        if args.list:
            print(f"Unassigned patients: {unassigned}\n")
            users = (db.query(User).filter(User.role.in_(CLINICAL_ROLES))
                     .order_by(User.date_created).all())
            if not users:
                print("No doctor/therapist accounts yet.")
            for u in users:
                owned = db.query(Patient).filter(Patient.owner_id == u.id).count()
                print(f"  {u.email:36} {u.role:10} {u.status:10} owns {owned} patient(s)")
            return 0

        user = (db.query(User)
                .filter(func.lower(User.email) == args.email.strip().lower())
                .first())
        if user is None:
            print(f"No account with email {args.email!r}. Run with --list to see the accounts.")
            return 1
        if user.role not in CLINICAL_ROLES:
            print(f"{user.email} is a '{user.role}' account - pick a doctor or therapist.")
            return 1
        if unassigned == 0:
            print("Nothing to do: every patient already has an owner.")
            return 0

        if not args.yes:
            answer = input(f"Assign {unassigned} unassigned patient(s) to {user.email} ({user.role})? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Cancelled.")
                return 1

        moved = _unassigned(db).update({Patient.owner_id: user.id}, synchronize_session=False)
        db.commit()
        print(f"Done: {moved} patient(s) now belong to {user.email}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

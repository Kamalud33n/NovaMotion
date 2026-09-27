"""Admin dashboard — user statistics + simple growth/usage numbers."""
import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from database import get_db
from models import Patient, SessionModel, Ticket, User
from services.auth.roles import (
    ALL_ROLES, STATUS_APPROVED, STATUS_PENDING, STATUS_REJECTED, STATUS_SUSPENDED,
)

router = APIRouter()


@router.get("/api/admin/dashboard")
async def admin_dashboard():
    with get_db() as db:
        users = db.query(User).all()
        total_users = len(users)

        by_status = {STATUS_PENDING: 0, STATUS_APPROVED: 0, STATUS_REJECTED: 0, STATUS_SUSPENDED: 0}
        by_role = {r: 0 for r in ALL_ROLES}
        for u in users:
            by_status[u.status] = by_status.get(u.status, 0) + 1
            by_role[u.role] = by_role.get(u.role, 0) + 1

        total_patients = db.query(Patient).filter(Patient.is_active == True).count()
        total_sessions = db.query(SessionModel).count()

        tickets = db.query(Ticket).all()
        open_tickets = sum(1 for t in tickets if t.status in ("open", "in_progress"))

        # New-registration trend for the last 14 days — the "growth" signal
        # for the dashboard chart (a simple, honest count, no fabricated data).
        today = datetime.date.today()
        days = [today - datetime.timedelta(days=i) for i in range(13, -1, -1)]
        counts = {d.isoformat(): 0 for d in days}
        cutoff = datetime.datetime.combine(days[0], datetime.time.min)
        for u in users:
            if u.date_created and u.date_created >= cutoff:
                key = u.date_created.date().isoformat()
                if key in counts:
                    counts[key] += 1
        registrations_trend = [{"date": d, "count": counts[d]} for d in counts]

        recent_users = sorted(users, key=lambda u: u.date_created or datetime.datetime.min, reverse=True)[:5]
        recent_tickets = sorted(tickets, key=lambda t: t.created_at or datetime.datetime.min, reverse=True)[:5]

        return JSONResponse({
            "statistics": {
                "total_users":       total_users,
                "pending_approvals": by_status[STATUS_PENDING],
                "active_users":      by_status[STATUS_APPROVED],
                "suspended_users":   by_status[STATUS_SUSPENDED],
                "rejected_users":    by_status[STATUS_REJECTED],
                "total_patients":    total_patients,
                "total_sessions":    total_sessions,
                "open_tickets":      open_tickets,
                "total_tickets":     len(tickets),
            },
            "users_by_role": by_role,
            "registrations_trend": registrations_trend,
            "recent_users": [
                {
                    "id": u.id, "full_name": u.full_name, "email": u.email,
                    "role": u.role, "status": u.status,
                    "date_created": u.date_created.isoformat() if u.date_created else None,
                }
                for u in recent_users
            ],
            "recent_tickets": [
                {
                    "id": t.id, "subject": t.subject, "status": t.status,
                    "user_name": t.user.full_name if t.user else "Unknown",
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
                for t in recent_tickets
            ],
        })

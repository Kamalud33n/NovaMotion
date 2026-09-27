"""
Support tickets — clinical side.

Any signed-in doctor/therapist can raise a ticket and see their own.
Admin-side viewing/management lives in routers/admin/tickets.py.
"""
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from database import get_db
from models import Ticket
from routers.auth._common import get_str
from services.auth.deps import CurrentUser, get_current_user

router = APIRouter()


def _ticket_dict(t: Ticket) -> Dict[str, Any]:
    return {
        "id": t.id,
        "subject": t.subject,
        "message": t.message,
        "status": t.status,
        "admin_response": t.admin_response,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
    }


@router.get("/api/tickets")
def api_list_my_tickets(me: CurrentUser = Depends(get_current_user)):
    with get_db() as db:
        rows = (
            db.query(Ticket)
            .filter(Ticket.user_id == me.id)
            .order_by(Ticket.created_at.desc())
            .all()
        )
        return JSONResponse([_ticket_dict(t) for t in rows])


@router.post("/api/tickets")
def api_create_ticket(payload: Dict[str, Any], me: CurrentUser = Depends(get_current_user)):
    subject = get_str(payload, "subject")
    message = get_str(payload, "message", strip=False)
    if len(subject) < 3:
        raise HTTPException(400, "Please enter a subject (at least 3 characters).")
    if len(subject) > 150:
        raise HTTPException(400, "Subject is too long (max 150 characters).")
    if len(message.strip()) < 5:
        raise HTTPException(400, "Please describe the issue (at least 5 characters).")
    if len(message) > 5000:
        raise HTTPException(400, "Description is too long (max 5000 characters).")

    with get_db() as db:
        t = Ticket(user_id=me.id, subject=subject, message=message, status="open")
        db.add(t)
        db.commit()
        db.refresh(t)
        return JSONResponse({"success": True, "message": "Ticket submitted.", "ticket": _ticket_dict(t)}, status_code=201)

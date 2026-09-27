"""Admin — view and manage support tickets raised by users."""
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from database import get_db
from models import TICKET_STATUSES, Ticket
from services.auth.deps import CurrentUser, require_admin
from services.auth.security import utcnow

router = APIRouter()


def _ticket_dict(t: Ticket) -> Dict[str, Any]:
    return {
        "id": t.id,
        "subject": t.subject,
        "message": t.message,
        "status": t.status,
        "admin_response": t.admin_response,
        "user_id": t.user_id,
        "user_name": t.user.full_name if t.user else "Unknown",
        "user_email": t.user.email if t.user else None,
        "user_role": t.user.role if t.user else None,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
    }


@router.get("/api/admin/tickets")
async def list_tickets(request: Request):
    search = request.query_params.get("search", "").strip()
    status = request.query_params.get("status", "").strip().lower()

    with get_db() as db:
        q = db.query(Ticket)
        if status in TICKET_STATUSES:
            q = q.filter(Ticket.status == status)
        rows = q.order_by(Ticket.created_at.desc()).all()
        if search:
            s = search.lower()
            rows = [
                t for t in rows
                if s in (t.subject or "").lower()
                or s in (t.user.full_name.lower() if t.user else "")
                or s in (t.user.email.lower() if t.user else "")
            ]
        return JSONResponse([_ticket_dict(t) for t in rows])


@router.get("/api/admin/tickets/{ticket_id}")
async def get_ticket(ticket_id: int):
    with get_db() as db:
        t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if not t:
            raise HTTPException(404, "Ticket not found")
        return JSONResponse(_ticket_dict(t))


@router.put("/api/admin/tickets/{ticket_id}")
async def update_ticket(ticket_id: int, payload: Dict[str, Any], me: CurrentUser = Depends(require_admin)):
    with get_db() as db:
        t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if not t:
            raise HTTPException(404, "Ticket not found")

        if "status" in payload:
            status = str(payload.get("status") or "").strip().lower()
            if status not in TICKET_STATUSES:
                raise HTTPException(400, "Invalid status.")
            t.status = status
            if status in ("resolved", "closed"):
                t.resolved_at = utcnow()
            else:
                t.resolved_at = None

        if "admin_response" in payload:
            response = str(payload.get("admin_response") or "").strip()
            t.admin_response = response or None

        t.handled_by = me.id
        db.commit()
        db.refresh(t)
        return JSONResponse({"success": True, "message": "Ticket updated", "ticket": _ticket_dict(t)})

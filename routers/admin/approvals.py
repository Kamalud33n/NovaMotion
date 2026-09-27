"""Admin — view and handle pending account-approval requests."""
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from database import get_db
from models import User
from registration_otp.mailer import send_account_approved_email, send_account_rejected_email
from services.auth.deps import CurrentUser, require_admin
from services.auth.roles import STATUS_APPROVED, STATUS_PENDING, STATUS_REJECTED
from services.auth.security import utcnow

router = APIRouter()


def _request_dict(u: User) -> Dict[str, Any]:
    return {
        "id": u.id,
        "full_name": u.full_name,
        "email": u.email,
        "phone": u.phone,
        "specialization": u.specialization,
        "role": u.role,
        "status": u.status,
        "date_created": u.date_created.isoformat() if u.date_created else None,
        "approved_at": u.approved_at.isoformat() if u.approved_at else None,
    }


@router.get("/api/admin/approvals")
async def list_approvals(request: Request):
    """Defaults to pending requests; pass ?status=approved|rejected for history."""
    status = request.query_params.get("status", STATUS_PENDING).strip().lower()
    if status not in (STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED):
        status = STATUS_PENDING
    with get_db() as db:
        rows = (
            db.query(User)
            .filter(User.status == status)
            .order_by(User.date_created.desc())
            .all()
        )
        return JSONResponse([_request_dict(u) for u in rows])


@router.post("/api/admin/approvals/{user_id}/approve")
async def approve_user(user_id: str, background_tasks: BackgroundTasks,
                        me: CurrentUser = Depends(require_admin)):
    with get_db() as db:
        u = db.query(User).filter(User.id == user_id).first()
        if not u:
            raise HTTPException(404, "Request not found")
        if u.status != STATUS_PENDING:
            raise HTTPException(400, "This request has already been handled.")
        u.status = STATUS_APPROVED
        u.approved_at = utcnow()
        u.approved_by = me.id
        db.commit()
        email, full_name = u.email, u.full_name
    background_tasks.add_task(send_account_approved_email, email, full_name)
    return JSONResponse({"success": True, "message": f"{full_name} approved."})


@router.post("/api/admin/approvals/{user_id}/reject")
async def reject_user(user_id: str, background_tasks: BackgroundTasks,
                       me: CurrentUser = Depends(require_admin)):
    with get_db() as db:
        u = db.query(User).filter(User.id == user_id).first()
        if not u:
            raise HTTPException(404, "Request not found")
        if u.status != STATUS_PENDING:
            raise HTTPException(400, "This request has already been handled.")
        u.status = STATUS_REJECTED
        u.approved_at = utcnow()
        u.approved_by = me.id
        db.commit()
        email, full_name = u.email, u.full_name
    background_tasks.add_task(send_account_rejected_email, email, full_name)
    return JSONResponse({"success": True, "message": f"{full_name}'s request was rejected."})
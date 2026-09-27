"""Admin — view, search and manage registered users."""
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from database import get_db
from models import User
from routers.auth._common import clean_profile_fields
from services.auth.deps import CurrentUser, require_admin
from services.auth.roles import ALL_ROLES, ALL_STATUSES, ROLE_ADMIN, STATUS_APPROVED
from services.auth.security import utcnow

router = APIRouter()


def _user_dict(u: User) -> Dict[str, Any]:
    return {
        "id": u.id,
        "full_name": u.full_name,
        "email": u.email,
        "phone": u.phone,
        "specialization": u.specialization,
        "role": u.role,
        "status": u.status,
        "date_created": u.date_created.isoformat() if u.date_created else None,
        "last_login": u.last_login.isoformat() if u.last_login else None,
        "approved_at": u.approved_at.isoformat() if u.approved_at else None,
    }


@router.get("/api/admin/users")
async def list_users(request: Request):
    search = request.query_params.get("search", "").strip()
    role = request.query_params.get("role", "").strip().lower()
    status = request.query_params.get("status", "").strip().lower()

    with get_db() as db:
        q = db.query(User)
        if role in ALL_ROLES:
            q = q.filter(User.role == role)
        if status in ALL_STATUSES:
            q = q.filter(User.status == status)
        if search:
            q = q.filter(
                User.full_name.contains(search)
                | User.email.contains(search)
                | User.phone.contains(search)
            )
        users = q.order_by(User.date_created.desc()).all()
        return JSONResponse([_user_dict(u) for u in users])


@router.get("/api/admin/users/{user_id}")
async def get_user(user_id: str):
    with get_db() as db:
        u = db.query(User).filter(User.id == user_id).first()
        if not u:
            raise HTTPException(404, "User not found")
        return JSONResponse(_user_dict(u))


@router.put("/api/admin/users/{user_id}")
async def update_user(user_id: str, payload: Dict[str, Any], me: CurrentUser = Depends(require_admin)):
    fields = clean_profile_fields(payload, name_required=False)

    with get_db() as db:
        u = db.query(User).filter(User.id == user_id).first()
        if not u:
            raise HTTPException(404, "User not found")

        if "role" in payload:
            role = str(payload.get("role") or "").strip().lower()
            if role not in ALL_ROLES:
                raise HTTPException(400, "Invalid role.")
            if u.id == me.id and role != ROLE_ADMIN:
                raise HTTPException(400, "You can't change your own role away from admin.")
            fields["role"] = role

        if "status" in payload:
            status = str(payload.get("status") or "").strip().lower()
            if status not in ALL_STATUSES:
                raise HTTPException(400, "Invalid status.")
            if u.id == me.id and status != STATUS_APPROVED:
                raise HTTPException(400, "You can't change your own account status.")
            fields["status"] = status
            if status == STATUS_APPROVED and u.status != STATUS_APPROVED:
                fields["approved_at"] = utcnow()
                fields["approved_by"] = me.id

        if not fields:
            raise HTTPException(400, "Nothing to update.")

        for key, value in fields.items():
            setattr(u, key, value)
        db.commit()
        db.refresh(u)
        return JSONResponse({"success": True, "message": "User updated", "user": _user_dict(u)})


@router.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: str, me: CurrentUser = Depends(require_admin)):
    with get_db() as db:
        u = db.query(User).filter(User.id == user_id).first()
        if not u:
            raise HTTPException(404, "User not found")
        if u.id == me.id:
            raise HTTPException(400, "You can't delete your own account.")
        if u.role == ROLE_ADMIN and db.query(User).filter(User.role == ROLE_ADMIN).count() <= 1:
            raise HTTPException(400, "Can't delete the last remaining admin account.")
        db.delete(u)
        db.commit()
        return JSONResponse({"success": True, "message": "User deleted"})

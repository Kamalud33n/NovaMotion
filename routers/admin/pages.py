"""
Admin pages — Dashboard, User Management, Approval Requests, Tickets, Settings.
Every route here is a page (HTML), guarded by require_admin_page (redirects
instead of returning JSON errors). The data each page needs comes from the
/api/admin/* endpoints in dashboard.py / users.py / approvals.py / tickets.py,
called client-side — this file only renders the shells.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from services.auth.deps import CurrentUser, require_admin_page
from services.auth.web import render

router = APIRouter()


@router.get("/admin", response_class=HTMLResponse)
async def page_admin_home(request: Request, user: CurrentUser = Depends(require_admin_page)):
    return render(request, "admin/dashboard.html", user=user, active_page="dashboard")


@router.get("/admin/users", response_class=HTMLResponse)
async def page_admin_users(request: Request, user: CurrentUser = Depends(require_admin_page)):
    return render(request, "admin/users.html", user=user, active_page="users")


@router.get("/admin/approvals", response_class=HTMLResponse)
async def page_admin_approvals(request: Request, user: CurrentUser = Depends(require_admin_page)):
    return render(request, "admin/approvals.html", user=user, active_page="approvals")


@router.get("/admin/tickets", response_class=HTMLResponse)
async def page_admin_tickets(request: Request, user: CurrentUser = Depends(require_admin_page)):
    return render(request, "admin/tickets.html", user=user, active_page="tickets")


@router.get("/admin/settings", response_class=HTMLResponse)
async def page_admin_settings(request: Request, user: CurrentUser = Depends(require_admin_page)):
    return render(request, "admin/settings.html", user=user, active_page="settings")

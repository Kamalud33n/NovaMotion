from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from config import templates
from services.auth.deps import CurrentUser, get_optional_user, require_clinical_page
from services.auth.roles import home_for_role
from services.auth.web import render

router = APIRouter()

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

clinical_page = [Depends(require_clinical_page)]


@router.get("/", response_class=HTMLResponse)
async def page_index(request: Request):
    user = get_optional_user(request)
    return render(request, "index.html", home_url=home_for_role(user.role) if user else None)


@router.get("/dashboard", response_class=HTMLResponse, dependencies=clinical_page)
async def page_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", headers=NO_CACHE_HEADERS)


@router.get("/patients", response_class=HTMLResponse, dependencies=clinical_page)
async def page_patients(request: Request):
    return templates.TemplateResponse(request, "patients.html", headers=NO_CACHE_HEADERS)


@router.get("/session", response_class=HTMLResponse, dependencies=clinical_page)
async def page_session(request: Request):
    return templates.TemplateResponse(request, "session.html", headers=NO_CACHE_HEADERS)


@router.get("/reports", response_class=HTMLResponse, dependencies=clinical_page)
async def page_reports(request: Request):
    return templates.TemplateResponse(request, "reports.html", headers=NO_CACHE_HEADERS)


@router.get("/analytics", response_class=HTMLResponse, dependencies=clinical_page)
async def page_analytics(request: Request):
    return templates.TemplateResponse(request, "analytics.html", headers=NO_CACHE_HEADERS)


@router.get("/settings", response_class=HTMLResponse)
async def page_settings(request: Request, user: CurrentUser = Depends(require_clinical_page)):
    """Doctor / therapist settings page — raise a support ticket (goes straight
    to the admin's /admin/tickets queue) and follow up on their own tickets."""
    return render(request, "settings.html", user=user)

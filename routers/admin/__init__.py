"""/routers/admin — admin-only area (dashboard, users, approvals, tickets, settings)."""
from fastapi import APIRouter, Depends

from routers.admin import approvals, dashboard, pages, tickets, users
from services.auth.deps import require_admin

router = APIRouter()

# Page routes redirect (not JSON 401/403) on their own via require_admin_page,
# so they're included with no extra guard here.
router.include_router(pages.router)

# API routes: guarded as a group so every /api/admin/* endpoint is admin-only,
# the same 401/403 JSON style as the other API routers in app.py.
admin_api = [Depends(require_admin)]
router.include_router(dashboard.router, dependencies=admin_api)
router.include_router(users.router,     dependencies=admin_api)
router.include_router(approvals.router, dependencies=admin_api)
router.include_router(tickets.router,   dependencies=admin_api)

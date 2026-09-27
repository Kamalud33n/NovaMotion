"""Small web helpers shared by the auth and admin page routes."""
from config import templates
from services.auth.config import settings

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def render(request, template: str, **context):
    """TemplateResponse with no-cache headers and the values every auth
    template needs (app name, password rule)."""
    ctx = {
        "app_name": settings.app_name,
        "password_min_length": settings.password_min_length,
    }
    ctx.update(context)
    return templates.TemplateResponse(request, template, ctx, headers=NO_CACHE_HEADERS)

import hmac

from fastapi import HTTPException, Request

from services.app_config import app_settings


def require_export_key(request: Request) -> None:
    key = app_settings.export_api_key
    if not key:
        raise HTTPException(status_code=503, detail="Export API is not configured")
    supplied = request.headers.get(app_settings.export_key_header, "")
    if not hmac.compare_digest(supplied.encode("utf-8"), key.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid API key")
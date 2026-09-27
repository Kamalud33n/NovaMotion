import os
from dataclasses import dataclass
from typing import Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

MIN_EXPORT_KEY_LENGTH = 24


class AppConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AppSettings:
    host: str
    port: int
    cors_origins: Tuple[str, ...]
    seed_demo_data: bool
    export_api_key: Optional[str]
    export_key_header: str


def _load() -> AppSettings:
    errors = []

    def req_str(name: str) -> str:
        v = (os.getenv(name) or "").strip()
        if not v:
            errors.append(f"{name} is required")
        return v

    def req_int(name: str) -> int:
        v = req_str(name)
        if not v:
            return 0
        try:
            n = int(v)
        except ValueError:
            errors.append(f"{name} must be a whole number (got {v!r})")
            return 0
        if not 1 <= n <= 65535:
            errors.append(f"{name} must be between 1 and 65535 (got {n})")
        return n

    def req_bool(name: str) -> bool:
        v = req_str(name).lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
        if v:
            errors.append(f"{name} must be true or false (got {v!r})")
        return False

    host = req_str("APP_HOST")
    port = req_int("APP_PORT")
    seed = req_bool("SEED_DEMO_DATA")
    header = req_str("EXPORT_API_KEY_HEADER")

    raw_origins = os.getenv("CORS_ALLOW_ORIGINS")
    if raw_origins is None:
        errors.append("CORS_ALLOW_ORIGINS must be present (comma separated; leave empty for same-origin only)")
        raw_origins = ""
    origins = tuple(o.strip().rstrip("/") for o in raw_origins.split(",") if o.strip())
    if "*" in origins:
        errors.append("CORS_ALLOW_ORIGINS cannot contain '*' because login uses cookies; list the exact origins")

    key = (os.getenv("EXPORT_API_KEY") or "").strip() or None
    if key and len(key) < MIN_EXPORT_KEY_LENGTH:
        errors.append(f"EXPORT_API_KEY must be at least {MIN_EXPORT_KEY_LENGTH} characters (or left empty to disable the export API)")

    if errors:
        raise AppConfigError(
            "Invalid app configuration in .env:\n  - " + "\n  - ".join(errors) +
            "\nCopy .env.example to .env and fill in the values."
        )

    return AppSettings(
        host=host, port=port, cors_origins=origins, seed_demo_data=seed,
        export_api_key=key, export_key_header=header,
    )


app_settings = _load()
"""Label definitions: which labels / compensation tags exist and which combinations are valid.

They live in label_definitions.json (or the file named by LABEL_DEFINITIONS_FILE), not in code,
so the clinical wording can be changed without touching Python. The file is re-read whenever
its modification time changes; if a new version is broken the last good one stays in use.
Each saved label stores the definitions `version` it was made under.
"""
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_PATH = Path(__file__).with_name("label_definitions.json")
_lock = threading.Lock()
_cache: Dict[str, Any] = {"path": None, "mtime": None, "data": None}


def _path() -> Path:
    override = (os.getenv("LABEL_DEFINITIONS_FILE") or "").strip()
    return Path(override) if override else _DEFAULT_PATH


def _validate(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("definitions must be a JSON object")
    version = str(data.get("version") or "").strip()
    if not version or len(version) > 30:
        raise ValueError("'version' is required (max 30 characters)")
    labels, tags = data.get("labels"), data.get("tags")
    if not isinstance(labels, list) or not labels or not isinstance(tags, list):
        raise ValueError("'labels' (non-empty) and 'tags' must be lists")
    for group in (labels, tags):
        seen = set()
        for item in group:
            if not isinstance(item, dict) or not str(item.get("id") or "").strip():
                raise ValueError("every label and tag needs an 'id'")
            if item["id"] in seen:
                raise ValueError(f"duplicate id {item['id']!r}")
            seen.add(item["id"])
    for lab in labels:
        lab["allow_tags"] = bool(lab.get("allow_tags"))
        lab["require_tags"] = bool(lab.get("require_tags"))
        if lab["require_tags"] and not lab["allow_tags"]:
            raise ValueError(f"label {lab['id']!r}: require_tags needs allow_tags")
    return data


def get_definitions() -> Dict[str, Any]:
    """Current definitions (cached by file modification time)."""
    path = _path()
    with _lock:
        try:
            mtime = path.stat().st_mtime
            if _cache["path"] == str(path) and _cache["mtime"] == mtime and _cache["data"]:
                return _cache["data"]
            with open(path, "r", encoding="utf-8") as fh:
                data = _validate(json.load(fh))
            _cache.update(path=str(path), mtime=mtime, data=data)
            return data
        except (OSError, ValueError) as exc:
            if _cache["data"] is not None and _cache["path"] == str(path):
                print(f"Labeling: could not reload {path.name} ({exc}); keeping the previous definitions")
                return _cache["data"]
            raise RuntimeError(f"Label definitions unusable ({path}): {exc}") from exc


def check_label(label: str, tags: Optional[List[str]]) -> List[str]:
    """Validate a label + tags pair against the current definitions.
    Returns the cleaned tag list (de-duplicated, original order). Raises ValueError with a
    message that is safe to show to the user."""
    defs = get_definitions()
    spec = next((l for l in defs["labels"] if l["id"] == label), None)
    if spec is None:
        raise ValueError("Unknown label. Allowed: " + ", ".join(l["id"] for l in defs["labels"]))
    tags = [str(t) for t in (tags or [])]
    allowed = {t["id"] for t in defs["tags"]}
    bad = [t for t in tags if t not in allowed]
    if bad:
        raise ValueError("Unknown tag(s): " + ", ".join(bad))
    clean = list(dict.fromkeys(tags))
    if clean and not spec["allow_tags"]:
        raise ValueError(f"Label '{label}' does not take compensation tags")
    if spec["require_tags"] and not clean:
        raise ValueError(f"Label '{label}' needs at least one compensation tag")
    return clean
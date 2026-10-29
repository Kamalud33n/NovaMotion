import datetime
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .settings import KeypointConfig

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_LINK_LOCK = threading.Lock()      # serialises read-modify-write of a recording's meta file


def is_valid_id(name) -> bool:
    """True when `name` is usable as a folder / file stem (no path tricks)."""
    return bool(_SAFE_NAME.match(str(name or "")))


def _safe(name) -> str:
    text = str(name or "")
    return text if _SAFE_NAME.match(text) else "unassigned"


def recording_path(cfg: KeypointConfig, patient_id, recording_id: str) -> Path:
    return cfg.base_path / _safe(patient_id) / f"{_safe(recording_id)}.npz"


def _atomic_write(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        writer(fh)
    os.replace(tmp, path)


def save_recording(cfg: KeypointConfig, patient_id, recording_id: str,
                   arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> str:
    """Returns the file path relative to the recordings directory."""
    path = recording_path(cfg, patient_id, recording_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, lambda fh: np.savez_compressed(fh, **arrays))
    meta_text = json.dumps(meta, default=str, indent=2).encode("utf-8")
    _atomic_write(path.with_suffix(".json"), lambda fh: fh.write(meta_text))
    return path.relative_to(cfg.base_path).as_posix()


def _meta_path(cfg: KeypointConfig, patient_id, recording_id: str) -> Path:
    return recording_path(cfg, patient_id, recording_id).with_suffix(".json")


def read_meta(cfg: KeypointConfig, patient_id, recording_id: str) -> Optional[Dict[str, Any]]:
    """Meta of a FINISHED recording, or None when it is not on disk, is still a
    partial, or cannot be read."""
    if not recording_path(cfg, patient_id, recording_id).is_file():
        return None
    try:
        with open(_meta_path(cfg, patient_id, recording_id), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"Keypoints: could not read meta for {recording_id}: {exc}")
        return None
    return None if meta.get("partial") else meta


def recording_info(cfg: KeypointConfig, patient_id, recording_id: str) -> Optional[Dict[str, Any]]:
    """File info for a finished recording read from disk (same shape the recorder
    returns when it saves), plus `session_id` = the DB session it is linked to,
    or None while it is unlinked. Used when the in-memory copy of that info is gone
    (server restart, context closed, a newer recording replaced it)."""
    meta = read_meta(cfg, patient_id, recording_id)
    if meta is None:
        return None
    return {
        "recording_id": meta.get("recording_id", recording_id),
        "patient_id": meta.get("patient_id"),
        "file": recording_path(cfg, patient_id, recording_id).relative_to(cfg.base_path).as_posix(),
        "frames": meta.get("frame_count"),
        "duration_seconds": meta.get("duration_seconds"),
        "truncated": bool(meta.get("truncated")),
        "session_id": meta.get("session_id"),
    }


def link_recording(cfg: KeypointConfig, patient_id, recording_id: str, session_id) -> bool:
    """Write the DB session id into a finished recording's meta so the file can be
    traced back to its session. Returns False when the recording is not on disk or is
    already linked to a DIFFERENT session; linking the same session twice is a no-op."""
    session_id = str(session_id)
    with _LINK_LOCK:
        meta = read_meta(cfg, patient_id, recording_id)
        if meta is None:
            return False
        existing = meta.get("session_id")
        if existing not in (None, session_id):
            return False
        if existing == session_id:
            return True
        meta["session_id"] = session_id
        meta["linked_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        text = json.dumps(meta, default=str, indent=2).encode("utf-8")
        _atomic_write(_meta_path(cfg, patient_id, recording_id), lambda fh: fh.write(text))
    return True


def list_recordings(cfg: KeypointConfig, patient_id=None) -> List[Dict[str, Any]]:
    """Light summary of every finished recording (optionally one patient's).
    `linked` is False for a recording that belongs to no saved session -- e.g. the tab
    was closed mid-session. Those files are kept; they are only marked."""
    base = cfg.base_path
    if not base.is_dir():
        return []
    pattern = f"{_safe(patient_id)}/*.json" if patient_id else "*/*.json"
    out: List[Dict[str, Any]] = []
    for path in sorted(base.glob(pattern)):
        if path.name.endswith(".partial.json"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        if meta.get("partial") or not path.with_suffix(".npz").is_file():
            continue
        out.append({
            "recording_id": meta.get("recording_id"),
            "patient_id": meta.get("patient_id"),
            "kind": meta.get("kind"),
            "label": meta.get("label"),
            "side": meta.get("side"),
            "mode": (meta.get("extra") or {}).get("mode") if isinstance(meta.get("extra"), dict) else None,
            "started_at": meta.get("started_at"),
            "ended_at": meta.get("ended_at"),
            "frames": meta.get("frame_count"),
            "duration_seconds": meta.get("duration_seconds"),
            "end_reason": meta.get("end_reason"),
            "session_id": meta.get("session_id"),
            "linked": bool(meta.get("session_id")),
        })
    return out


def _ended_epoch(meta: Dict[str, Any], json_path: Path) -> float:
    """When the recording ended (UTC epoch seconds); file mtime when meta has no usable ended_at."""
    ended = meta.get("ended_at")
    if ended:
        try:
            dt = datetime.datetime.fromisoformat(str(ended).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass
    return json_path.stat().st_mtime


def purge_unlinked(cfg: KeypointConfig, ttl_hours: int,
                   saved_session: Callable[[str, str], Optional[str]],
                   now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Housekeeping for recordings of sessions that were never saved (tab closed,
    page reloaded, crash...). A FINISHED recording whose meta has no session_id and
    which ended more than `ttl_hours` ago is handled like this:

      - `saved_session(patient_id, recording_id)` returns a session id  -> the session WAS
        saved but its link step failed: link it now, keep the files
      - it returns None                                                 -> truly abandoned: delete
      - it raises                                                       -> keep (fail safe, retried next run)

    ttl_hours <= 0 disables the purge. Partial (checkpoint) files are never touched here;
    server-start recovery owns those. Returns one dict per action taken.
    """
    results: List[Dict[str, Any]] = []
    if ttl_hours <= 0:
        return results
    base = cfg.base_path
    if not base.is_dir():
        return results
    cutoff = (time.time() if now is None else now) - ttl_hours * 3600
    for jpath in sorted(base.glob("*/*.json")):
        if jpath.name.endswith(".partial.json"):
            continue
        patient_id = jpath.parent.name
        recording_id = jpath.stem
        info: Dict[str, Any] = {"recording_id": recording_id, "patient_id": patient_id}
        try:
            with open(jpath, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("partial") or meta.get("session_id"):
                continue
            if not jpath.with_suffix(".npz").is_file():
                continue
            if _ended_epoch(meta, jpath) > cutoff:
                continue
            session_id = saved_session(patient_id, recording_id)
            if session_id:
                linked = link_recording(cfg, patient_id, recording_id, session_id)
                results.append({**info, "action": "linked_late" if linked else "link_failed",
                                "session_id": str(session_id)})
                continue
            with _LINK_LOCK:
                fresh = read_meta(cfg, patient_id, recording_id)    # re-check: may have been linked meanwhile
                if fresh is None or fresh.get("session_id"):
                    continue
                for p in (recording_path(cfg, patient_id, recording_id), jpath):
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
            results.append({**info, "action": "deleted_abandoned"})
        except Exception as exc:
            print(f"Keypoints: purge skipped {recording_id}: {exc}")
            results.append({**info, "action": "skipped_error", "error": str(exc)})
    return results


def delete_patient_recordings(cfg: KeypointConfig, patient_id) -> Dict[str, Any]:
    """Consent withdrawn: remove EVERYTHING this patient has on disk -- finished recordings
    (linked or not), their meta files, checkpoint partials and leftover .tmp files.
    Returns {"recordings": [{"recording_id", "session_id"}...], "files_deleted": n,
    "failed": [file names that could not be removed]}.
    Raises ValueError for an id that is not a plain folder name, so a bad id can never
    point the delete at another folder (_safe() would map it to "unassigned")."""
    pid = str(patient_id or "")
    if not _SAFE_NAME.match(pid):
        raise ValueError("unsafe patient id")
    out: Dict[str, Any] = {"recordings": [], "files_deleted": 0, "failed": []}
    folder = cfg.base_path / pid
    if not folder.is_dir():
        return out
    with _LINK_LOCK:
        for jpath in sorted(folder.glob("*.json")):
            if jpath.name.endswith(".partial.json"):
                continue
            try:
                with open(jpath, "r", encoding="utf-8") as fh:
                    session_id = json.load(fh).get("session_id")
            except (OSError, ValueError):
                session_id = None
            out["recordings"].append({"recording_id": jpath.stem, "session_id": session_id})
        for f in sorted(folder.iterdir()):
            if not f.is_file() or f.suffix not in (".npz", ".json", ".tmp"):
                continue
            try:
                f.unlink()
                out["files_deleted"] += 1
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"Keypoints: could not delete {f.name}: {exc}")
                out["failed"].append(f.name)
        try:
            folder.rmdir()           # only succeeds when empty
        except OSError:
            pass
    return out


def partial_path(cfg: KeypointConfig, patient_id, recording_id: str) -> Path:
    return cfg.base_path / _safe(patient_id) / f"{_safe(recording_id)}.partial.npz"


def save_partial(cfg: KeypointConfig, patient_id, recording_id: str,
                 arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    """Crash-safety checkpoint of a recording that is still running."""
    path = partial_path(cfg, patient_id, recording_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, lambda fh: np.savez_compressed(fh, **arrays))
    meta_text = json.dumps(meta, default=str, indent=2).encode("utf-8")
    _atomic_write(path.with_suffix(".json"), lambda fh: fh.write(meta_text))


def delete_partial(cfg: KeypointConfig, patient_id, recording_id: str) -> None:
    path = partial_path(cfg, patient_id, recording_id)
    for p in (path, path.with_suffix(".json"), path.with_name(path.name + ".tmp"),
              path.with_suffix(".json").with_name(path.with_suffix(".json").name + ".tmp")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"Keypoints: could not delete {p.name}: {exc}")


def recover_partials(cfg: KeypointConfig,
                     allow: Optional[Callable[[Optional[str]], bool]] = None) -> List[Dict[str, Any]]:
    """Run once at server start. Every *.partial.npz left behind belongs to a recording
    whose server died (or was killed) before it could save normally:

      - the final file already exists  -> the partial is just a leftover, delete it
      - `allow(patient_id)` is False   -> (e.g. consent no longer on record) delete it, never promote
      - otherwise                      -> promote it to a normal recording (end_reason
                                          "recovered_after_crash")

    Unreadable partials are left alone and reported, never deleted. Returns one dict per partial.
    """
    results: List[Dict[str, Any]] = []
    base = cfg.base_path
    if not base.is_dir():
        return results
    for npz in sorted(base.glob("*/*.partial.npz")):
        patient_id = npz.parent.name
        recording_id = npz.name[: -len(".partial.npz")]
        info: Dict[str, Any] = {"recording_id": recording_id, "patient_id": patient_id}
        try:
            final = recording_path(cfg, patient_id, recording_id)
            if final.exists():
                delete_partial(cfg, patient_id, recording_id)
                results.append({**info, "action": "deleted_leftover"})
                continue
            if allow is not None and not allow(patient_id):
                delete_partial(cfg, patient_id, recording_id)
                results.append({**info, "action": "deleted_no_consent"})
                continue
            arrays, meta = load_recording(npz)
            frames = int(len(arrays["t"]))
            if frames < cfg.min_frames:
                delete_partial(cfg, patient_id, recording_id)
                results.append({**info, "action": "deleted_too_short", "frames": frames})
                continue
            meta.update({
                "partial": False,
                "end_reason": "recovered_after_crash",
                "recovered": True,
                "recovered_at_unix": round(time.time(), 3),
                "checkpoint_end_reason": meta.get("end_reason"),
                "frame_count": frames,
            })
            save_recording(cfg, patient_id, recording_id, arrays, meta)
            delete_partial(cfg, patient_id, recording_id)
            results.append({**info, "action": "recovered", "frames": frames})
        except Exception as exc:
            print(f"Keypoints: could not recover {npz.name}: {exc}")
            results.append({**info, "action": "unreadable", "error": str(exc)})
    return results


def load_recording(path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    path = Path(path)
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    with open(path.with_suffix(".json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return arrays, meta
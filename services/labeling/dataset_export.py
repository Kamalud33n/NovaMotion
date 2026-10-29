"""Export the labelled ML dataset (dataset_samples) to one zip file, on the server.

Used by scripts/export_dataset.py (and usable from a router). The zip holds:

    manifest.json            counts, filters, split rule, warnings, definitions version
    label_definitions.json   the label definitions the labels were made under
    samples.csv              one row per sample (metadata, split, file name)
    data/<split>/<name>.npz  the keypoint arrays of each sample

Patients appear only as the salted `subject` hash. Train / val / test are split BY SUBJECT
(never inside one patient) with a fixed rule, so re-exporting gives the same split.
`ExportFilters.sample_ids` restricts the export to those dataset_samples.id values
(used by --trusted-only); None means no restriction, an empty set means nothing.
"""
import csv
import datetime
import hashlib
import io
import json
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import defer

from database import get_db
from models import DatasetSample
from services.labeling.definitions import get_definitions

FORMAT_VERSION = 1
MIN_SUBJECTS_FOR_SPLIT = 3
_SPOOL_BYTES = 64 * 1024 * 1024


class ExportError(Exception):
    """A request the caller can fix. `status` follows HTTP (503 = service not usable)."""
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class ExportFilters:
    labels: List[str] = field(default_factory=list)
    kind: Optional[str] = None
    exercise: Optional[str] = None
    defs_version: Optional[str] = None
    min_detected: Optional[float] = None
    sample_ids: Optional[Set[int]] = None


def definitions_info() -> Dict[str, Any]:
    """Current definitions version and whether they are still an unsigned draft.
    Raises RuntimeError when the definitions file is unusable."""
    defs = get_definitions()
    version = str(defs["version"])
    status = str(defs.get("status") or "")
    draft = "draft" in version.lower() or status.strip().upper().startswith("DRAFT")
    return {"version": version, "draft": draft, "status": status}


def _apply(q, f: ExportFilters):
    if f.labels:
        q = q.filter(DatasetSample.label.in_(list(f.labels)))
    if f.kind:
        q = q.filter(DatasetSample.kind == f.kind)
    if f.exercise:
        q = q.filter(DatasetSample.exercise == f.exercise)
    if f.defs_version:
        q = q.filter(DatasetSample.defs_version == f.defs_version)
    if f.min_detected is not None:
        q = q.filter(DatasetSample.detected_ratio >= f.min_detected)
    if f.sample_ids is not None:
        q = q.filter(DatasetSample.id.in_(list(f.sample_ids))) if f.sample_ids else q.filter(False)
    return q


def _rows(f: ExportFilters):
    """Sample rows without the data blob, oldest first."""
    with get_db() as db:
        rows = (_apply(db.query(DatasetSample).options(defer(DatasetSample.data)), f)
                .order_by(DatasetSample.id).all())
        db.expunge_all()
        return rows


def summary(f: ExportFilters) -> Dict[str, Any]:
    rows = _rows(f)

    def count(attr):
        return dict(sorted(Counter(str(getattr(r, attr) or "-") for r in rows).items()))

    return {"samples": len(rows), "subjects": len({r.subject for r in rows}),
            "frames_total": int(sum(r.n_frames or 0 for r in rows)),
            "by_label": count("label"), "by_kind": count("kind"),
            "by_exercise": count("exercise"), "by_defs_version": count("defs_version")}


def _split_subjects(subjects: List[str], val_pct: int, test_pct: int) -> Dict[str, str]:
    """subject -> train/val/test. Fixed order (hash of the subject), so it is repeatable."""
    ordered = sorted(subjects, key=lambda s: hashlib.sha256(("split:" + s).encode()).hexdigest())
    n = len(ordered)
    if n < MIN_SUBJECTS_FOR_SPLIT:
        return {s: "train" for s in ordered}
    n_test = max(1, round(n * test_pct / 100)) if test_pct > 0 else 0
    n_val = max(1, round(n * val_pct / 100)) if val_pct > 0 else 0
    while n_val + n_test >= n:                 # always keep at least one training subject
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break
    out = {}
    for i, s in enumerate(ordered):
        out[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def _fetch_data(sample_id: int) -> bytes:
    with get_db() as db:
        row = db.query(DatasetSample.data).filter(DatasetSample.id == sample_id).first()
        return bytes(row[0]) if row and row[0] is not None else b""


def build_export(f: ExportFilters, val_pct: int = 15, test_pct: int = 15,
                 include_notes: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """Build the zip. Returns (open file positioned at 0 - the caller closes it, manifest)."""
    if not (0 <= val_pct <= 50 and 0 <= test_pct <= 50 and val_pct + test_pct <= 60):
        raise ExportError(400, "--val-pct and --test-pct must be 0-50 and sum to at most 60")
    try:
        info = definitions_info()
        defs = get_definitions()
    except RuntimeError as exc:
        raise ExportError(503, f"Label definitions are not available: {exc}")

    rows = _rows(f)
    if not rows:
        raise ExportError(404, "No samples match these filters")
    subjects = sorted({r.subject for r in rows})
    split_of = _split_subjects(subjects, val_pct, test_pct)

    warnings: List[str] = []
    if info["draft"]:
        warnings.append(f"Label definitions {info['version']!r} are a draft, not signed off by a physiotherapist")
    if len(subjects) < MIN_SUBJECTS_FOR_SPLIT:
        warnings.append(f"Only {len(subjects)} subject(s): everything is in 'train', no val/test split")
    mixed = {r.defs_version for r in rows}
    if len(mixed) > 1:
        warnings.append("Samples come from several definitions versions: " + ", ".join(sorted(mixed)))

    columns = ["name", "split", "subject", "label", "tags", "kind", "exercise", "mode", "side", "fps",
               "rep_index", "start_frame", "end_frame", "n_frames", "detected_ratio", "defs_version"]
    if include_notes:
        columns.append("note")
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=columns)
    writer.writeheader()

    spool = tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES)
    by_split: Dict[str, Dict[str, set]] = {}
    try:
        with zipfile.ZipFile(spool, "w") as zf:
            for r in rows:
                split = split_of[r.subject]
                name = f"S{r.id:07d}"
                blob = _fetch_data(r.id)
                if not blob:
                    warnings.append(f"sample {r.id} has no data and was skipped")
                    continue
                zf.writestr(f"data/{split}/{name}.npz", blob, compress_type=zipfile.ZIP_STORED)
                part = by_split.setdefault(split, {"samples": 0, "subjects": set()})
                part["samples"] += 1
                part["subjects"].add(r.subject)
                row = {"name": name, "split": split, "subject": r.subject, "label": r.label,
                       "tags": "|".join(r.tags or []), "kind": r.kind, "exercise": r.exercise or "",
                       "mode": r.mode or "", "side": r.side or "",
                       "fps": "" if r.fps is None else round(r.fps, 2), "rep_index": "" if r.rep_index is None else r.rep_index,
                       "start_frame": r.start_frame, "end_frame": r.end_frame, "n_frames": r.n_frames,
                       "detected_ratio": "" if r.detected_ratio is None else round(r.detected_ratio, 4),
                       "defs_version": r.defs_version}
                if include_notes:
                    row["note"] = r.note or ""
                writer.writerow(row)

            n_samples = sum(p["samples"] for p in by_split.values())
            if n_samples == 0:
                raise ExportError(404, "No samples with data to export")
            for want in ("val", "test"):
                if (val_pct if want == "val" else test_pct) > 0 and want not in by_split \
                        and len(subjects) >= MIN_SUBJECTS_FOR_SPLIT:
                    warnings.append(f"'{want}' split is empty")
            manifest = {
                "format_version": FORMAT_VERSION,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "definitions": {"version": info["version"], "draft": info["draft"]},
                "filters": {"labels": list(f.labels), "kind": f.kind, "exercise": f.exercise,
                            "defs_version": f.defs_version, "min_detected": f.min_detected,
                            "restricted_to_sample_ids": None if f.sample_ids is None else len(f.sample_ids)},
                "split_rule": {"by": "subject", "val_pct": val_pct, "test_pct": test_pct,
                               "order": "sha256('split:'+subject)"},
                "counts": {"samples": n_samples,
                           "subjects": len({s for p in by_split.values() for s in p["subjects"]}),
                           "by_split": {k: {"samples": v["samples"], "subjects": len(v["subjects"])}
                                        for k, v in sorted(by_split.items())}},
                "warnings": warnings,
            }
            zf.writestr("samples.csv", table.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr("label_definitions.json", json.dumps(defs, indent=2, ensure_ascii=False),
                        compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2), compress_type=zipfile.ZIP_DEFLATED)
    except BaseException:
        spool.close()
        raise
    spool.seek(0)
    return spool, manifest
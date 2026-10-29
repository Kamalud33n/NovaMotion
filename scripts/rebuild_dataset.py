"""Rebuild the ML dataset (dataset_samples) from the saved labels (recording_labels).

Run from the project root, with the same .env as the app (DATASET_SALT must be set):

    python scripts/rebuild_dataset.py                # add missing samples, refresh stale ones
    python scripts/rebuild_dataset.py --dry-run      # only report what would change
    python scripts/rebuild_dataset.py --all          # rebuild every sample, even up-to-date ones
    python scripts/rebuild_dataset.py --prune        # also delete samples whose label no longer exists

When to use it: labels saved before DATASET_SALT was set (or before the dataset hooks were
wired) are not in the dataset; a label whose sample failed to build is retried; after a
DATASET_SALT change `--all` re-hashes every sample. Safe to run any number of times: each
sample is rebuilt from the recording on disk and replaces the old one (one row per label).

Labels marked `unusable` never get a sample (an existing one is removed). A label whose
recording is gone or whose frames are out of range is reported as FAILED and left alone.
Exit code 1 if any label failed, 2 if the salt is missing, 0 otherwise.
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import get_db                                   # noqa: E402  (loads .env)
from models import DatasetSample, RecordingLabel              # noqa: E402
from services.labeling import dataset_store, label_store      # noqa: E402

_SAMPLE_COLUMNS = (DatasetSample.label_id, DatasetSample.subject, DatasetSample.labeler_id,
                   DatasetSample.start_frame, DatasetSample.end_frame, DatasetSample.rep_index,
                   DatasetSample.label, DatasetSample.tags, DatasetSample.note,
                   DatasetSample.defs_version)


def _load_labels() -> List[Dict[str, Any]]:
    with get_db() as db:
        rows = db.query(RecordingLabel).order_by(RecordingLabel.id).all()
        return [label_store.to_dict(r) for r in rows]


def _load_samples() -> Dict[int, Dict[str, Any]]:
    """label_id -> the sample's columns (the .npz blob is deliberately not loaded)."""
    with get_db() as db:
        rows = db.query(*_SAMPLE_COLUMNS).all()
    return {r.label_id: {"subject": r.subject, "labeler_id": r.labeler_id,
                         "start_frame": r.start_frame, "end_frame": r.end_frame,
                         "rep_index": r.rep_index, "label": r.label,
                         "tags": list(r.tags or []), "note": r.note,
                         "defs_version": r.defs_version} for r in rows}


def _is_current(label: Dict[str, Any], sample: Dict[str, Any], subject: str) -> bool:
    """True when the stored sample already matches the label (and the current salt)."""
    return (sample["subject"] == subject
            and sample["labeler_id"] == label["labeler_id"]
            and sample["start_frame"] == label["start_frame"]
            and sample["end_frame"] == label["end_frame"]
            and sample["rep_index"] == label["rep_index"]
            and sample["label"] == label["label"]
            and sample["tags"] == list(label["tags"] or [])
            and sample["note"] == label["note"]
            and sample["defs_version"] == label["defs_version"])


def run(rebuild_all: bool = False, dry_run: bool = False, prune: bool = False) -> Dict[str, Any]:
    labels = _load_labels()
    samples = _load_samples()
    stats = {"labels": len(labels), "created": 0, "refreshed": 0, "unchanged": 0,
             "removed_unusable": 0, "failed": [], "orphans": 0, "orphans_deleted": 0}
    label_ids = set()

    for label in labels:
        lid = int(label["id"])
        label_ids.add(lid)
        existing = samples.get(lid)

        if label["label"] in dataset_store.EXCLUDED_LABELS:
            if existing is not None:
                stats["removed_unusable"] += 1
                if not dry_run:
                    dataset_store.delete_for_label(lid)
            continue

        subject = dataset_store.subject_hash(label["patient_id"])
        if (existing is not None and not rebuild_all and subject is not None
                and _is_current(label, existing, subject)):
            stats["unchanged"] += 1
            continue

        if dry_run:
            stats["refreshed" if existing is not None else "created"] += 1
            continue
        if dataset_store.sync_label(label):
            stats["refreshed" if existing is not None else "created"] += 1
        else:
            stats["failed"].append(lid)

    orphans = sorted(set(samples) - label_ids)
    stats["orphans"] = len(orphans)
    if prune and not dry_run:
        for lid in orphans:
            stats["orphans_deleted"] += dataset_store.delete_for_label(lid)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild dataset_samples from recording_labels.")
    ap.add_argument("--all", action="store_true", help="rebuild every sample, even up-to-date ones")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    ap.add_argument("--prune", action="store_true", help="delete samples whose label no longer exists")
    args = ap.parse_args(argv)

    if dataset_store.subject_hash("probe") is None:
        print("DATASET_SALT is not set in .env: nothing can be stored. Set it and run again.")
        return 2

    s = run(rebuild_all=args.all, dry_run=args.dry_run, prune=args.prune)
    tag = "  (dry run, nothing changed)" if args.dry_run else ""
    print(f"Labels: {s['labels']}{tag}")
    print(f"  created:           {s['created']}")
    print(f"  refreshed:         {s['refreshed']}")
    print(f"  unchanged:         {s['unchanged']}")
    print(f"  removed (unusable): {s['removed_unusable']}")
    if s["orphans"]:
        action = f"deleted {s['orphans_deleted']}" if args.prune and not args.dry_run else "use --prune to delete"
        print(f"  orphan samples:    {s['orphans']} ({action})")
    if s["failed"]:
        print(f"  FAILED:            {len(s['failed'])} label ids {s['failed']} "
              f"(see the 'Dataset:' lines above for the reason)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
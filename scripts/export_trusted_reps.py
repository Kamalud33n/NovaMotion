"""Export TRUSTED reps as a good / bad training set, on the server - no viewer labelling needed.

A rep is exported only when the auto-score from its keypoints agrees with the doctor's overall
session rating (good+good -> good, bad/worst+bad -> bad). Uncertain, ok, unknown and unrated
reps are left out. See services/labeling/trusted_reps.py.

    python scripts/export_trusted_reps.py --summary        # counts only, no file
    python scripts/export_trusted_reps.py                  # exports/thero_trusted_<time>.zip
    python scripts/export_trusted_reps.py --exercise "Hand Grip Exercise" --force -o my.zip

Needs DATASET_SALT in .env (the patient only appears as a salted hash). Train / val / test are
split BY SUBJECT with the same fixed rule as scripts/export_dataset.py.
Exit codes: 0 ok, 2 DATASET_SALT missing, 4 bad option, 5 output exists, 6 nothing to export.
"""
import argparse
import csv
import datetime
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.labeling import dataset_store, trusted_reps          # noqa: E402  (loads .env)
from services.labeling.dataset_export import (                      # noqa: E402
    MIN_SUBJECTS_FOR_SPLIT, _split_subjects)

DEFAULT_DIR = "exports"
FORMAT_VERSION = 1
MIN_PER_CLASS_WARN = 30
_SPOOL_BYTES = 64 * 1024 * 1024
COLUMNS = ["name", "split", "subject", "label", "source", "session_id", "rating", "auto_score",
           "flags", "kind", "exercise", "mode", "side", "fps", "rep_index", "start_frame",
           "end_frame", "n_frames", "detected_ratio"]


def _print_stats(stats: dict, candidates: list) -> None:
    print(f"Sessions: {stats.get('sessions', 0)}  (no recording: {stats.get('sessions_no_recording', 0)}, "
          f"not rated: {stats.get('sessions_not_rated', 0)}, recording file missing: "
          f"{stats.get('sessions_recording_missing', 0)}, usable: {stats.get('sessions_usable', 0)})")
    print(f"Reps in usable sessions: trusted {stats.get('reps_trusted', 0)}, "
          f"uncertain {stats.get('reps_uncertain', 0)}, unlabeled {stats.get('reps_unlabeled', 0)}")
    labels = Counter(c["label"] for c in candidates)
    print("Trusted by label: " + (", ".join(f"{k}: {v}" for k, v in sorted(labels.items())) or "-"))


def _write(fh, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(fh, out, 1 << 20)
        os.replace(part, target)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    return target.stat().st_size


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export trusted reps (good / bad) to a zip file.")
    ap.add_argument("--summary", action="store_true", help="print counts only, write no file")
    ap.add_argument("-o", "--output", help=f"output zip path (default {DEFAULT_DIR}/thero_trusted_<time>.zip)")
    ap.add_argument("--force", action="store_true", help="overwrite the output file if it exists")
    ap.add_argument("--exercise", help="only this exercise name")
    ap.add_argument("--min-detected", type=float, default=0.6,
                    help="only reps with detected_ratio >= this, 0-1 (default 0.6)")
    ap.add_argument("--val-pct", type=int, default=15, help="validation share of subjects (default 15)")
    ap.add_argument("--test-pct", type=int, default=15, help="test share of subjects (default 15)")
    args = ap.parse_args(argv)

    if not 0 <= args.min_detected <= 1:
        print("--min-detected must be between 0 and 1")
        return 4
    if not (0 <= args.val_pct <= 50 and 0 <= args.test_pct <= 50 and args.val_pct + args.test_pct <= 60):
        print("--val-pct and --test-pct must be 0-50 and sum to at most 60")
        return 4
    if dataset_store.subject_hash("probe") is None:
        print("DATASET_SALT is not set in .env. Set it first: patients must only appear as a salted hash.")
        return 2

    candidates, stats = trusted_reps.collect()
    _print_stats(stats, candidates)
    subject_of = {c["patient_id"]: dataset_store.subject_hash(c["patient_id"]) for c in candidates}
    print(f"Subjects with trusted reps: {len(set(subject_of.values()))}")
    if args.summary:
        return 0
    if not candidates:
        print("Nothing to export: no rep has a rating that its auto-score agrees with yet.")
        return 6

    target = Path(args.output) if args.output else Path(DEFAULT_DIR) / (
        "thero_trusted_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S") + ".zip")
    if target.exists() and not args.force:
        print(f"{target} already exists (use --force to overwrite)")
        return 5

    split_of = _split_subjects(sorted(set(subject_of.values())), args.val_pct, args.test_pct)
    warnings: List[str] = []
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=COLUMNS)
    writer.writeheader()
    by_split: Dict[str, Dict[str, set]] = {}
    skipped = Counter()
    labels = Counter()

    spool = tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES)
    try:
        with zipfile.ZipFile(spool, "w") as zf:
            n = 0
            for c in candidates:
                sample = trusted_reps.build(c)
                if sample is None:
                    skipped["could not be built"] += 1
                    continue
                if args.exercise and sample["exercise"] != args.exercise:
                    skipped["other exercise"] += 1
                    continue
                if sample["detected_ratio"] is not None and sample["detected_ratio"] < args.min_detected:
                    skipped["tracking too low"] += 1
                    continue
                n += 1
                name = f"T{n:07d}"
                split = split_of[sample["subject"]]
                zf.writestr(f"data/{split}/{name}.npz", sample["data"], compress_type=zipfile.ZIP_STORED)
                part = by_split.setdefault(split, {"samples": 0, "subjects": set()})
                part["samples"] += 1
                part["subjects"].add(sample["subject"])
                labels[sample["label"]] += 1
                writer.writerow({
                    "name": name, "split": split, "subject": sample["subject"], "label": sample["label"],
                    "source": trusted_reps.SOURCE, "session_id": c["session_id"], "rating": c["rating"],
                    "auto_score": c["auto_score"], "flags": "|".join(c["flags"]), "kind": sample["kind"],
                    "exercise": sample["exercise"] or "", "mode": sample["mode"] or "",
                    "side": sample["side"] or "",
                    "fps": "" if sample["fps"] is None else round(sample["fps"], 2),
                    "rep_index": c["rep_index"], "start_frame": sample["start_frame"],
                    "end_frame": sample["end_frame"], "n_frames": sample["n_frames"],
                    "detected_ratio": "" if sample["detected_ratio"] is None else round(sample["detected_ratio"], 4)})
            if n == 0:
                print("Nothing to export after the filters: " + (
                    ", ".join(f"{v} {k}" for k, v in skipped.items()) or "no samples"))
                return 6

            n_subjects = len({s for p in by_split.values() for s in p["subjects"]})
            warnings.append("Labels are coarse good/bad from a rule-based auto-score that agrees with "
                            "the doctor's overall session rating; not physiotherapist-reviewed labels")
            if n_subjects < MIN_SUBJECTS_FOR_SPLIT:
                warnings.append(f"Only {n_subjects} subject(s): everything is in 'train', no val/test split")
            for label in ("good", "bad"):
                if labels[label] < MIN_PER_CLASS_WARN:
                    warnings.append(f"Only {labels[label]} '{label}' sample(s); too few to train on")
            for k, v in skipped.items():
                warnings.append(f"{v} trusted rep(s) skipped: {k}")
            manifest = {
                "format_version": FORMAT_VERSION, "source": trusted_reps.SOURCE,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "rule": {"good": "session rating good + rep auto-score good",
                         "bad": "session rating bad or worst + rep auto-score bad",
                         "excluded": "uncertain, ok, unknown, unrated"},
                "filters": {"exercise": args.exercise, "min_detected": args.min_detected},
                "split_rule": {"by": "subject", "val_pct": args.val_pct, "test_pct": args.test_pct,
                               "order": "sha256('split:'+subject)"},
                "collect_stats": stats,
                "counts": {"samples": n, "subjects": n_subjects, "by_label": dict(labels),
                           "by_split": {k: {"samples": v["samples"], "subjects": len(v["subjects"])}
                                        for k, v in sorted(by_split.items())}},
                "warnings": warnings}
            zf.writestr("samples.csv", table.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2), compress_type=zipfile.ZIP_DEFLATED)
    except BaseException:
        spool.close()
        raise
    spool.seek(0)
    try:
        size = _write(spool, target)
    finally:
        spool.close()

    print(f"Exported {n} trusted rep(s) from {n_subjects} subject(s) to {target} ({size / 1024:.0f} KB)")
    print("  by label: " + ", ".join(f"{k}: {v}" for k, v in sorted(labels.items())))
    for name, part in manifest["counts"]["by_split"].items():
        print(f"  {name}: {part['samples']} samples, {part['subjects']} subject(s)")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
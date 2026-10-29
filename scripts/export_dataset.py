"""Export the labelled ML dataset (dataset_samples) to one zip file, on the server.
"""
import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.labeling import dataset_export                          # noqa: E402  (loads .env)
from services.labeling.dataset_export import ExportError, ExportFilters  # noqa: E402

DEFAULT_DIR = "exports"


def _filters(args) -> ExportFilters:
    extra = {}
    if not args.include_untrusted:
        # Default: only samples that are one rep whose auto-score agrees with the session
        # rating, so a single overall rating can never label reps it does not describe.
        # dataset_export applies `sample_ids` as DatasetSample.id IN (...).
        from services.labeling import trusted                 # noqa: E402
        ids, left_out = trusted.trusted_sample_ids()
        print(f"Trusted samples only: {len(ids)} kept, {left_out} left out "
              "(use --include-untrusted to keep them)")
        extra["sample_ids"] = ids
    else:
        print("WARNING: --include-untrusted: samples whose auto-score disagrees with, or cannot "
              "be checked against, the doctor's session rating are INCLUDED. Not for training.")
    return ExportFilters(labels=args.label or [], kind=args.kind, exercise=args.exercise,
                         defs_version=args.defs_version, min_detected=args.min_detected, **extra)


def _print_summary(summary: dict, defs: dict) -> None:
    state = "DRAFT" if defs["draft"] else "signed off"
    print(f"Label definitions: {defs['version']} ({state})")
    print(f"Samples: {summary['samples']}   Subjects: {summary['subjects']}   "
          f"Frames: {summary['frames_total']}")
    for title, key in (("By label", "by_label"), ("By kind", "by_kind"),
                       ("By exercise", "by_exercise"), ("By definitions version", "by_defs_version")):
        parts = ", ".join(f"{k}: {v}" for k, v in summary[key].items()) or "-"
        print(f"  {title}: {parts}")


def _output_path(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(DEFAULT_DIR) / f"thero_dataset_{stamp}.zip"


def _write(fh, target: Path) -> int:
    """Copy the spooled zip to `target` (owner-only), via a .part file so a failed run leaves
    no half-written zip behind. Returns the size in bytes."""
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
    ap = argparse.ArgumentParser(description="Export the labelled dataset to a zip file.")
    ap.add_argument("--summary", action="store_true", help="print counts only, write no file")
    ap.add_argument("-o", "--output", help=f"output zip path (default {DEFAULT_DIR}/thero_dataset_<time>.zip)")
    ap.add_argument("--force", action="store_true", help="overwrite the output file if it exists")
    ap.add_argument("--label", action="append", help="only this label (repeat for several)")
    ap.add_argument("--kind", choices=["pose", "hands"], help="only pose or only hands samples")
    ap.add_argument("--exercise", help="only this exercise name")
    ap.add_argument("--defs-version", help="only samples labelled under this definitions version")
    ap.add_argument("--min-detected", type=float, help="only samples with detected_ratio >= this (0-1)")
    ap.add_argument("--val-pct", type=int, default=15, help="validation share of subjects (default 15)")
    ap.add_argument("--test-pct", type=int, default=15, help="test share of subjects (default 15)")
    ap.add_argument("--trusted-only", action="store_true",
                    help="kept for old commands; this is now the default")
    ap.add_argument("--include-untrusted", action="store_true",
                    help="also export samples that are uncertain, unlabeled or cannot be checked "
                         "against the session rating (default: only rep samples whose auto-score "
                         "agrees with the clinician's rating)")
    ap.add_argument("--include-notes", action="store_true", help="add the free-text notes to samples.csv")
    ap.add_argument("--allow-draft", action="store_true",
                    help="export even though the label definitions are still a draft")
    args = ap.parse_args(argv)

    if args.trusted_only and args.include_untrusted:
        print("--trusted-only and --include-untrusted cannot be used together")
        return 4
    if args.min_detected is not None and not 0 <= args.min_detected <= 1:
        print("--min-detected must be between 0 and 1")
        return 4
    filters = _filters(args)

    try:
        defs = dataset_export.definitions_info()
    except RuntimeError as exc:
        print(f"Label definitions are not available: {exc}")
        return 1

    if args.summary:
        _print_summary(dataset_export.summary(filters), defs)
        return 0

    if defs["draft"] and not args.allow_draft:
        print(f"Label definitions {defs['version']!r} are still a draft and have not been signed "
              "off by a physiotherapist.\nAdd --allow-draft to export anyway.")
        return 3

    target = _output_path(args.output)
    if target.exists() and not args.force:
        print(f"{target} already exists (use --force to overwrite)")
        return 5

    try:
        fh, manifest = dataset_export.build_export(filters, args.val_pct, args.test_pct,
                                                   args.include_notes)
    except ExportError as exc:
        print(exc.message)
        return 2 if exc.status == 503 else 4
    try:
        size = _write(fh, target)
    finally:
        fh.close()

    counts = manifest["counts"]
    print(f"Exported {counts['samples']} samples from {counts['subjects']} subject(s) "
          f"to {target} ({size / 1024:.0f} KB)")
    for name, part in counts["by_split"].items():
        print(f"  {name}: {part['samples']} samples, {part['subjects']} subject(s)")
    for warning in manifest["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
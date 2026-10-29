"""Session-level export: clinician rating + auto-score aggregate + rep agreement, one CSV row
per session. Patients appear only as the salted dataset hash (DATASET_SALT must be set).

    python scripts/export_sessions.py                       # exports/thero_sessions_<time>.csv
    python scripts/export_sessions.py -o out.csv --force
    python scripts/export_sessions.py --rated-only          # skip sessions with no rating

pct_bad = bad reps / all scored reps. Sessions with no auto-scores (saved before scoring,
not yet backfilled - see backfill_rep_scores.py) are listed with scored=0 and empty counts.
"""
import argparse
import csv
import datetime
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import get_db                                  # noqa: E402  (loads .env)
from models import SessionModel                              # noqa: E402
from services.labeling import agreement, dataset_store       # noqa: E402

COLUMNS = ["session_id", "subject", "exercise", "recording_id", "rating", "scored", "reps",
           "good", "ok", "bad", "unknown", "pct_bad", "trusted", "uncertain", "unlabeled"]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export session ratings with rep auto-score aggregates.")
    ap.add_argument("-o", "--output", help="output csv (default exports/thero_sessions_<time>.csv)")
    ap.add_argument("--force", action="store_true", help="overwrite the output file if it exists")
    ap.add_argument("--rated-only", action="store_true", help="only sessions with a clinician rating")
    args = ap.parse_args(argv)

    if dataset_store.subject_hash("probe") is None:
        print("DATASET_SALT is not set in .env; patients cannot be anonymised, nothing exported.")
        return 2
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = Path(args.output) if args.output else Path("exports") / f"thero_sessions_{stamp}.csv"
    if target.exists() and not args.force:
        print(f"{target} already exists (use --force to overwrite)")
        return 5

    rows = []
    with get_db() as db:
        for s in db.query(SessionModel).order_by(SessionModel.start_time).all():
            data = s.session_data if isinstance(s.session_data, dict) else {}
            summ = agreement.session_summary(data)
            if args.rated_only and not summ["rating"]:
                continue
            rec = data.get("recording") if isinstance(data.get("recording"), dict) else {}
            row = {"session_id": s.id, "subject": dataset_store.subject_hash(s.patient_id),
                   "exercise": s.exercise_type, "recording_id": rec.get("recording_id") or "",
                   **summ, "scored": int(summ["scored"])}
            rows.append({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(part, target)
    print(f"Exported {len(rows)} session(s) to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
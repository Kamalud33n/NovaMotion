"""Score the repetitions of sessions saved before automatic scoring existed.

Run from the project root, with the same .env as the app:

    python scripts/backfill_rep_scores.py --dry-run          # only report
    python scripts/backfill_rep_scores.py                    # fill sessions with no rep_auto_scores
    python scripts/backfill_rep_scores.py --session SES-778D0E5B
    python scripts/backfill_rep_scores.py --force            # re-score even when scores exist
                                                             # (e.g. after thresholds change)

Scores come from the recording on disk only (never from the clinician's rating). Sessions
with no recording, or whose recording is gone, are reported and left untouched; nothing else in
session_data (including clinician_rating) is changed. Safe to run any number of times.
Exit code 1 if any session failed to score, otherwise 0.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import get_db                         # noqa: E402  (loads .env)
from models import SessionModel                     # noqa: E402
from services.labeling import rep_scores            # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Backfill rep_auto_scores for older sessions.")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    ap.add_argument("--session", action="append", help="only this session id (repeatable)")
    ap.add_argument("--force", action="store_true", help="re-score sessions that already have scores")
    args = ap.parse_args(argv)

    done = skipped = no_rec = failed = 0
    with get_db() as db:
        q = db.query(SessionModel).order_by(SessionModel.start_time)
        if args.session:
            q = q.filter(SessionModel.id.in_(args.session))
        for sess in q.all():
            data = dict(sess.session_data or {})
            if data.get("rep_auto_scores") and not args.force:
                skipped += 1
                continue
            rec = data.get("recording")
            rid = rec.get("recording_id") if isinstance(rec, dict) else None
            if not rid:
                print(f"{sess.id}: no recording, skipped")
                no_rec += 1
                continue
            scores = rep_scores.compute_for_recording(sess.patient_id, str(rid))
            if not scores:
                print(f"{sess.id}: FAILED to score {rid} (recording missing or unreadable)")
                failed += 1
                continue
            s = scores["summary"]
            print(f"{sess.id}: {s['reps']} reps  good {s['good']}  ok {s['ok']}  "
                  f"bad {s['bad']}  unknown {s['unknown']}" + ("   [dry run]" if args.dry_run else ""))
            if not args.dry_run:
                data["rep_auto_scores"] = scores
                sess.session_data = data          # new dict, so the JSON change is saved
            done += 1
        if not args.dry_run:
            db.commit()

    verb = "would score" if args.dry_run else "scored"
    print(f"{verb} {done}, already scored {skipped}, no recording {no_rec}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
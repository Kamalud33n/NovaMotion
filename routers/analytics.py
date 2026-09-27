import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from database import get_db
from models import SessionModel
from services.auth.deps import CurrentUser, require_clinical
from services.helpers import calculate_recovery_score, session_summary, split_sessions, game_steps
from services.ownership import get_owned_patient, owned_sessions

router = APIRouter()


@router.get("/api/analytics")
async def analytics(patient_id: Optional[str] = None, me: CurrentUser = Depends(require_clinical)):
    with get_db() as db:
        q = owned_sessions(db, me)
        if patient_id:
            get_owned_patient(db, patient_id, me)
            q = q.filter(SessionModel.patient_id == patient_id)
        sessions = q.order_by(SessionModel.start_time).all()

        today      = datetime.date.today()
        yesterday  = today - datetime.timedelta(days=1)

        def _day_range(d):
            return (
                datetime.datetime.combine(d, datetime.time.min),
                datetime.datetime.combine(d, datetime.time.max),
            )

        def _bucket(d):
            lo, hi = _day_range(d)
            return [s for s in sessions if lo <= s.start_time <= hi]

        def _summary(bucket):
            # Exercise-quality metrics (accuracy/ROM/stability/balance) are
            # averaged over exercise sessions only — game sessions store 0
            # for those columns by design and would otherwise skew them.
            ex_bucket, gm_bucket = split_sessions(bucket)
            n = len(ex_bucket)
            avg = lambda attr: round(sum(getattr(s, attr) or 0 for s in ex_bucket) / n, 1) if n else 0.0
            return {
                "sessions":       len(bucket),
                "total_reps":     sum(s.completed_reps or 0 for s in ex_bucket),
                "avg_accuracy":   avg("accuracy_percentage"),
                "avg_rom":        avg("average_rom"),
                "avg_stability":  avg("stability_score"),
                "avg_balance":    avg("balance_score"),
                "avg_smoothness": avg("movement_smoothness"),
                "recovery_score": round(calculate_recovery_score(bucket), 1),
                "incorrect_movements": sum(s.incorrect_movements or 0 for s in ex_bucket),
                "game_sessions":  len(gm_bucket),
                "total_steps":    sum(game_steps(s) for s in gm_bucket),
            }

        today_bucket     = _bucket(today)
        yesterday_bucket = _bucket(yesterday)
        today_sum        = _summary(today_bucket)
        yesterday_sum    = _summary(yesterday_bucket)

        def _delta(key):
            t, y = today_sum[key], yesterday_sum[key]
            if y == 0:
                return None if t == 0 else 100.0
            return round(((t - y) / y) * 100, 1)

        deltas = {
            key: _delta(key)
            for key in ("sessions", "total_reps", "avg_accuracy", "avg_rom",
                        "avg_stability", "avg_balance", "recovery_score")
        }

        # Last 7 days trend, for context below the today-vs-yesterday cards
        trend = []
        for i in range(6, -1, -1):
            d = today - datetime.timedelta(days=i)
            b = _bucket(d)
            s = _summary(b)
            trend.append({"date": d.isoformat(), **s})

        return JSONResponse({
            "today":     {"date": today.isoformat(),     **today_sum},
            "yesterday": {"date": yesterday.isoformat(), **yesterday_sum},
            "deltas":    deltas,
            "trend":     trend,
            "today_sessions":     [session_summary(s) for s in today_bucket],
            "yesterday_sessions": [session_summary(s) for s in yesterday_bucket],
        })

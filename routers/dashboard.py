import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from database import get_db
from models import Patient, SessionModel
from services.auth.deps import CurrentUser, require_clinical
from services.helpers import calculate_recovery_score, split_sessions, game_steps
from services.ownership import owned_patients, owned_sessions

router = APIRouter()


@router.get("/api/dashboard")
async def dashboard(me: CurrentUser = Depends(require_clinical)):
    with get_db() as db:
        total_patients = owned_patients(db, me).filter(Patient.is_active == True).count()
        total_sessions = owned_sessions(db, me).count()

        today     = datetime.date.today()
        today_min = datetime.datetime.combine(today, datetime.time.min)
        today_max = datetime.datetime.combine(today, datetime.time.max)
        today_sessions = (
            owned_sessions(db, me)
            .filter(SessionModel.start_time.between(today_min, today_max))
            .count()
        )

        all_sessions = owned_sessions(db, me).all()
        # Game sessions (Rehab Runner etc.) don't carry ROM/stability/balance
        # values — exclude them from those averages so they don't silently
        # drag exercise-quality metrics toward 0. They're still counted in
        # total_sessions/today_sessions and reported separately below.
        exercise_sessions, game_sessions = split_sessions(all_sessions)
        n = len(exercise_sessions)

        def _avg(attr):
            return sum(getattr(s, attr) or 0 for s in exercise_sessions) / n if n else 0

        patients       = owned_patients(db, me).filter(Patient.is_active == True).all()
        patient_scores = [calculate_recovery_score(p.sessions) for p in patients]
        recovery_score = sum(patient_scores) / len(patient_scores) if patient_scores else 0

        recent = owned_sessions(db, me).order_by(SessionModel.start_time.desc()).limit(10).all()
        progress = [
            {
                "date":      s.start_time.strftime("%Y-%m-%d"),
                "accuracy":  s.accuracy_percentage,
                "rom":       s.average_rom,
                "stability": s.stability_score or 0,
                "balance":   s.balance_score or 0,
                "exercise":  s.exercise_type,
            }
            for s in recent
        ]

        def _period(days):
            cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
            all_ss = [s for s in all_sessions if s.start_time >= cutoff]
            ex_ss, gm_ss = split_sessions(all_ss)
            m  = len(ex_ss)
            return {
                "sessions":      len(all_ss),
                "accuracy":      round(sum(s.accuracy_percentage for s in ex_ss) / m if m else 0, 1),
                "rom":           round(sum(s.average_rom for s in ex_ss) / m if m else 0, 1),
                "game_sessions": len(gm_ss),
                "total_steps":   sum(game_steps(s) for s in gm_ss),
            }

        return JSONResponse({
            "statistics": {
                "total_patients":      total_patients,
                "total_sessions":      total_sessions,
                "today_sessions":      today_sessions,
                "completed_exercises": sum(s.completed_reps for s in exercise_sessions),
                "avg_accuracy":        round(_avg("accuracy_percentage"), 1),
                "avg_rom":             round(_avg("average_rom"), 1),
                "incorrect_movements": sum(s.incorrect_movements for s in exercise_sessions),
                "recovery_score":      round(recovery_score, 1),
                "avg_stability":       round(_avg("stability_score"), 1),
                "avg_balance":         round(_avg("balance_score"), 1),
                "avg_smoothness":      round(_avg("movement_smoothness"), 1),
                "game_sessions":       len(game_sessions),
                "total_steps":         sum(game_steps(s) for s in game_sessions),
            },
            "progress": progress,
            "weekly":   _period(7),
            "monthly":  _period(30),
        })

"""Rep-level agreement between the clinician's session rating and the automatic rep scores.

Pure functions, no database. The rating (good / bad / worst) belongs to the whole session; the
auto-score (good / ok / bad / unknown) belongs to one repetition. A rep is only `trusted` when
the two clearly agree, so the rating is never copied onto reps.

Policy (strict, decided for Stage 2; thresholds are not yet validated on real patient data):
    rating good       + rep good  -> trusted
    rating bad/worst  + rep bad   -> trusted
    rating good       + rep bad   -> uncertain   (excluded from training export)
    rating bad/worst  + rep good  -> uncertain
    rep ok                         -> unlabeled  (a flag only; never decided by the rating)
    rep unknown / no auto-score    -> unlabeled  (never decided by the rating)
    session not rated              -> unlabeled

To make `ok` count as uncertain later, change only UNCERTAIN_SCORES / the ok branch below.
"""
from typing import Any, Dict, Iterable, List, Optional

TRUSTED = "trusted"
UNCERTAIN = "uncertain"
UNLABELED = "unlabeled"

GOOD_RATINGS = frozenset({"good"})
BAD_RATINGS = frozenset({"bad", "worst"})
# What the rating predicts for a rep, and which rep score contradicts it.
_EXPECTS = {**{r: "good" for r in GOOD_RATINGS}, **{r: "bad" for r in BAD_RATINGS}}
_CONTRADICTS = {"good": "bad", "bad": "good"}


def rep_agreement(rating: Optional[str], auto_score: Optional[str]) -> str:
    """trusted / uncertain / unlabeled for one rep."""
    expected = _EXPECTS.get(rating or "")
    if expected is None or auto_score not in ("good", "bad"):   # no rating, ok, unknown, missing
        return UNLABELED
    return TRUSTED if auto_score == expected else UNCERTAIN


def session_rating(session_data: Optional[Dict[str, Any]]) -> Optional[str]:
    value = ((session_data or {}).get("clinician_rating") or {}).get("value")
    return value if isinstance(value, str) else None


def rep_agreements(session_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per auto-scored rep: rep (1-based), rep_index (0-based), auto_score, agreement."""
    rating = session_rating(session_data)
    items = ((session_data or {}).get("rep_auto_scores") or {}).get("items") or []
    rows = []
    for item in items:
        rep = item.get("rep")
        if not isinstance(rep, int) or isinstance(rep, bool):
            continue
        score = item.get("auto_score")
        rows.append({"rep": rep, "rep_index": rep - 1, "auto_score": score,
                     "agreement": rep_agreement(rating, score)})
    return rows


def session_summary(session_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Session-level aggregate: rating, auto-score counts, % bad reps, agreement counts.

    pct_bad is bad reps / all scored reps (unknown reps stay in the denominator); None when the
    session has no auto-scores (e.g. saved before scoring existed and not backfilled yet).
    """
    rows = rep_agreements(session_data)
    n = len(rows)
    scores = {k: sum(r["auto_score"] == k for r in rows) for k in ("good", "ok", "bad", "unknown")}
    agree = {k: sum(r["agreement"] == k for r in rows) for k in (TRUSTED, UNCERTAIN, UNLABELED)}
    return {
        "rating": session_rating(session_data),
        "scored": n > 0,
        "reps": n,
        "good": scores["good"], "ok": scores["ok"], "bad": scores["bad"], "unknown": scores["unknown"],
        "pct_bad": round(100.0 * scores["bad"] / n, 1) if n else None,
        "trusted": agree[TRUSTED], "uncertain": agree[UNCERTAIN], "unlabeled": agree[UNLABELED],
    }


def trusted_rep_indexes(session_data: Optional[Dict[str, Any]]) -> Iterable[int]:
    return [r["rep_index"] for r in rep_agreements(session_data) if r["agreement"] == TRUSTED]
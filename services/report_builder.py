"""
PDF report generation (ReportLab + matplotlib chart), isolated from the WS/
MJPEG camera pipelines. build_report_sync() and build_session_report_sync()
are pure sync functions — both are called via loop.run_in_executor() from
routers/reports.py so neither ever blocks the asyncio event loop used by
streaming routes.
"""
import os
import io
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import HTTPException

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable,
)

from config import (
    PDF_NAVY, PDF_GREY_BORDER, PDF_GREY_BG, PDF_GREY_TEXT,
    PDF_ROW_ALT, PDF_BODY_TEXT,
)
from database import get_db
from models import Patient, SessionModel, Report
from game.registry import get_game
from services.helpers import (
    calculate_recovery_score, calculate_improvement, split_sessions,
    is_game_session, game_steps, game_left_steps, game_right_steps, game_final_score,
    game_id_of, is_balloon_session, balloon_stats,
)


def game_label(game_id: str | None) -> str:
    """Friendly display name for a game_id (game/registry.py), falling back
    to the raw id if it's ever missing from the registry."""
    if not game_id:
        return "Game"
    entry = get_game(game_id)
    return entry["label"] if entry else game_id


def _make_progress_chart(sessions: list) -> io.BytesIO:
    plot_sessions = sessions[-10:]
    idx   = list(range(1, len(plot_sessions) + 1))
    acc   = [s.accuracy_percentage for s in plot_sessions]
    rom   = [s.average_rom for s in plot_sessions]
    dates = [s.start_time.strftime("%m/%d") for s in plot_sessions]

    fig, ax1 = plt.subplots(figsize=(6.5, 2.3), dpi=150)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    ax1.plot(idx, acc, color="#1B2A4A", linewidth=2, marker="o",
              markersize=4, label="Accuracy %")
    ax2 = ax1.twinx()
    ax2.plot(idx, rom, color="#8B93A1", linewidth=1.6, linestyle="--",
              marker="s", markersize=3.5, label="ROM °")

    ax1.set_xticks(idx)
    ax1.set_xticklabels(dates, fontsize=7, color="#5A6472")
    ax1.tick_params(axis="y", labelsize=7, colors="#5A6472")
    ax2.tick_params(axis="y", labelsize=7, colors="#5A6472")
    ax1.set_ylim(0, 100)

    for spine in ("top",):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax1.spines["left"].set_color("#B7BEC9")
    ax1.spines["bottom"].set_color("#B7BEC9")
    ax2.spines["right"].set_color("#B7BEC9")
    ax1.grid(axis="y", color="#EEF1F5", linewidth=1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7,
               frameon=False, ncol=2, bbox_to_anchor=(0, 1.22))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _build_recommendations(sessions, avg_acc, avg_rom, rec_score, improvement) -> list:
    """
    Rule-based (non-AI) recommendation engine. Looks at every metric already
    stored per session — not just accuracy/ROM — and returns a short,
    priority-ordered list: real problems first, general coaching next,
    positive reinforcement last. Capped so the PDF section stays readable
    instead of dumping every possible sentence.
    """
    n = len(sessions)

    def _avg(attr):
        return sum((getattr(s, attr) or 0) for s in sessions) / n

    avg_stability = _avg("stability_score")
    avg_balance   = _avg("balance_score")
    avg_smooth    = _avg("movement_smoothness")
    avg_fatigue   = _avg("fatigue_estimation")
    avg_incorrect = _avg("incorrect_movements")

    total_reps_target = sum(s.total_reps for s in sessions)
    total_reps_done   = sum(s.completed_reps for s in sessions)
    rep_completion = (total_reps_done / total_reps_target * 100) if total_reps_target else 100

    # `sessions` is already ordered oldest -> newest by the caller's query,
    # so consecutive differences give real day-gaps between visits.
    gaps = [
        (sessions[i + 1].start_time - sessions[i].start_time).days
        for i in range(len(sessions) - 1)
    ]
    avg_gap_days = (sum(gaps) / len(gaps)) if gaps else None

    priority, general, positive = [], [], []

    # --- Priority: needs attention now ---
    if avg_acc < 60:
        priority.append("Accuracy is below target — slow down repetitions and prioritize correct form over speed.")
    if avg_rom < 50:
        priority.append("Range of motion is limited — add gentle stretching/mobility work before each session.")
    if avg_fatigue > 60:
        priority.append("Fatigue levels are running high during sessions — consider shorter sets with more rest between reps.")
    if avg_incorrect > 3:
        priority.append(f"An average of {avg_incorrect:.1f} incorrect movements per session were recorded — review technique with the therapist.")
    if avg_balance < 50:
        priority.append("Balance scores are low — incorporate dedicated balance/stability drills.")
    if improvement < -5:
        priority.append(f"Performance has dipped ({improvement:+.1f}%) compared to earlier sessions — worth discussing with the therapist.")

    # --- General: moderate areas still worth coaching ---
    if 60 <= avg_acc < 75:
        general.append("Accuracy is moderate — continued focus on controlled movement should improve this further.")
    if 50 <= avg_rom < 65:
        general.append("Range of motion is improving but still limited — keep up mobility exercises.")
    if avg_smooth < 60:
        general.append("Movement smoothness could improve — practicing at a slower, steadier pace may help.")
    if 50 <= avg_stability < 65:
        general.append("Stability is improving but still developing — continue balance-focused exercises.")
    if rep_completion < 80:
        general.append(f"Only {rep_completion:.0f}% of prescribed reps were completed on average — encourage finishing full sets where possible.")
    if avg_gap_days is not None and avg_gap_days > 5:
        general.append(f"Sessions are averaging {avg_gap_days:.1f} days apart — more frequent sessions (2–3x/week) would support faster recovery.")
    if n < 5:
        general.append("Still early in the program — consistent practice over at least 10 sessions is recommended before re-evaluation.")

    # --- Positive reinforcement ---
    if avg_acc >= 85 and avg_rom >= 80:
        positive.append("Excellent progress on both accuracy and range of motion — consider introducing more advanced functional exercises.")
    elif avg_acc >= 75 and avg_rom >= 70:
        positive.append("Good overall progress — maintain the current routine and gradually increase intensity.")
    if improvement > 15:
        positive.append(f"Strong improvement trend ({improvement:+.1f}%) since the first session — keep up the momentum.")
    if rec_score >= 75:
        positive.append("Recovery score indicates strong progress toward full functional recovery.")

    # Priority issues surface first, then coaching notes, then encouragement.
    # Capped at 6 so the section stays focused rather than an unfocused wall
    # of every possible sentence.
    recs = priority + general + positive
    if not recs:
        recs = ["Continue current therapy plan.", "Regular monitoring is recommended."]
    return recs[:6]


def _pdf_style_kit(doc_width: float) -> dict:
    """Paragraph styles + layout helpers (section headers, metric boxes,
    striped data tables) shared by every report flavour — period report and
    single-session report alike — so the look stays identical no matter
    which build_*_sync function produced the PDF. Helvetica family only,
    navy/grey/white palette (see config.py PDF_* constants)."""
    styles = {
        "title": ParagraphStyle(
            "PDFTitle", fontName="Helvetica-Bold", fontSize=19,
            textColor=PDF_NAVY, alignment=TA_CENTER, leading=22,
        ),
        "subtitle": ParagraphStyle(
            "PDFSubtitle", fontName="Helvetica", fontSize=9,
            textColor=PDF_GREY_TEXT, alignment=TA_CENTER, spaceAfter=4,
        ),
        "section_title": ParagraphStyle(
            "SectionTitle", fontName="Helvetica-Bold", fontSize=11,
            textColor=PDF_NAVY, leading=14,
        ),
        "info_label": ParagraphStyle(
            "InfoLabel", fontName="Helvetica-Bold", fontSize=9,
            textColor=PDF_NAVY, leading=13,
        ),
        "info_value": ParagraphStyle(
            "InfoValue", fontName="Helvetica", fontSize=9,
            textColor=PDF_BODY_TEXT, leading=13,
        ),
        "body": ParagraphStyle(
            "Body", fontName="Helvetica", fontSize=9.5,
            textColor=PDF_BODY_TEXT, leading=15,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel", fontName="Helvetica", fontSize=7.5,
            textColor=PDF_GREY_TEXT, leading=10,
        ),
        "metric_value": ParagraphStyle(
            "MetricValue", fontName="Helvetica-Bold", fontSize=16,
            textColor=PDF_NAVY, leading=19, spaceBefore=2,
        ),
        "table_header": ParagraphStyle(
            "TableHeader", fontName="Helvetica-Bold", fontSize=8.5,
            textColor=colors.white, alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell", fontName="Helvetica", fontSize=8.5,
            textColor=PDF_BODY_TEXT, alignment=TA_CENTER,
        ),
    }

    def section_header(text):
        """Light-grey banded section header — the only place grey fill is used."""
        tbl = Table([[Paragraph(text, styles["section_title"])]], colWidths=[doc_width])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PDF_GREY_BG),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, -1), 0.75, PDF_GREY_BORDER),
        ]))
        return tbl

    def metric_box(label, value, width):
        """Bordered white box for a single summary metric."""
        inner = Table(
            [[Paragraph(label.upper(), styles["metric_label"])],
             [Paragraph(str(value), styles["metric_value"])]],
            colWidths=[width],
        )
        inner.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.75, PDF_GREY_BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        return inner

    def metric_row(items, gap=8):
        """Lay N metric boxes side by side with an even gap between them."""
        n_items = len(items)
        box_w   = (doc_width - gap * (n_items - 1)) / n_items
        boxes   = [metric_box(l, v, box_w) for l, v in items]
        row = Table([boxes], colWidths=[box_w] * n_items)
        row.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-2, -1), gap),
            ("RIGHTPADDING", (-1, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return row

    def data_table(header, rows, col_weights):
        """Striped table (navy header, alternating row fill) — used for
        session history, rep breakdowns, and joint-angle summaries."""
        col_w = [doc_width * w for w in col_weights]
        data = [[Paragraph(h, styles["table_header"]) for h in header]] + rows
        tbl = Table(data, colWidths=col_w, repeatRows=1)
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), PDF_NAVY),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, PDF_GREY_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        for i in range(1, len(data)):
            bg = colors.white if i % 2 == 1 else PDF_ROW_ALT
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
        tbl.setStyle(TableStyle(style_cmds))
        return tbl

    styles["section_header"] = section_header
    styles["metric_row"]     = metric_row
    styles["data_table"]     = data_table
    return styles


def _patient_info_block(p: Patient, st: dict, doc_width: float) -> list:
    """Two-column patient info grid + rule, identical across every report."""
    label_w = 72
    info_rows = [
        [Paragraph("Patient", st["info_label"]),     Paragraph(p.name, st["info_value"]),
         Paragraph("Patient ID", st["info_label"]),   Paragraph(p.id, st["info_value"])],
        [Paragraph("Age / Gender", st["info_label"]), Paragraph(f"{p.age} yrs · {p.gender}", st["info_value"]),
         Paragraph("Therapist", st["info_label"]),     Paragraph(p.therapist_name or "—", st["info_value"])],
        [Paragraph("Diagnosis", st["info_label"]),     Paragraph(p.diagnosis or "—", st["info_value"]),
         Paragraph("Affected Area", st["info_label"]), Paragraph(p.affected_body_part or "—", st["info_value"])],
    ]
    info_tbl = Table(
        info_rows,
        colWidths=[label_w, doc_width / 2 - label_w, label_w + 10, doc_width / 2 - label_w - 10],
    )
    info_tbl.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return [
        info_tbl,
        HRFlowable(width="100%", thickness=0.75, color=PDF_GREY_BORDER, spaceBefore=6, spaceAfter=22),
    ]


def _footer_block(st: dict) -> list:
    return [
        Spacer(1, 22),
        HRFlowable(width="100%", thickness=0.75, color=PDF_GREY_BORDER, spaceAfter=14),
        Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", st["body"]),
        Spacer(1, 26),
        Paragraph("Therapist Signature: _________________________", st["body"]),
        Spacer(1, 10),
        Paragraph("Date: _________________________", st["body"]),
    ]


# ── Balloon Pop sections ─────────────────────────────────────────────────
# Balloon Pop is a shoulder-rehab game (reps / angles), so it gets its own
# report sections instead of the Rehab Runner "steps / score / lives" ones.
_FINISH_LABEL = {"target": "Target reached", "time": "Time up", "manual": "Stopped early"}
_B_ARM     = {"left": "Left", "right": "Right"}
_B_PLANE   = {"abd": "Abduction", "flex": "Flexion"}
def _make_balloon_chart(sessions: list):
    """Pop rate (%) and average pop time (s) over the last 10 Balloon Pop
    sessions. Returns None when there is nothing worth plotting (fewer than 2
    sessions, or no session with miss data so pop rate is unknown)."""
    plot = sessions[-10:]
    if len(plot) < 2:
        return None
    nan = float("nan")
    rates, times = [], []
    for s in plot:
        b = balloon_stats(s)
        judged = b["reps"] + (b["misses"] or 0)
        rates.append(b["reps"] / judged * 100 if b["misses"] is not None and judged else nan)
        times.append(b["avg_time_to_pop"] if b["reps"] else nan)
    if all(r != r for r in rates):             # NaN != NaN -> no pop rate anywhere
        return None
    idx   = list(range(1, len(plot) + 1))
    dates = [s.start_time.strftime("%m/%d") for s in plot]

    fig, ax1 = plt.subplots(figsize=(6.5, 2.3), dpi=150)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax1.plot(idx, rates, color="#1B2A4A", linewidth=2, marker="o", markersize=4, label="Pop rate %")
    ax2 = ax1.twinx()
    ax2.plot(idx, times, color="#8B93A1", linewidth=1.6, linestyle="--", marker="s",
             markersize=3.5, label="Avg pop time (s)")

    ax1.set_xticks(idx)
    ax1.set_xticklabels(dates, fontsize=7, color="#5A6472")
    ax1.tick_params(axis="y", labelsize=7, colors="#5A6472")
    ax2.tick_params(axis="y", labelsize=7, colors="#5A6472")
    ax1.set_ylim(0, 100)
    ax2.set_ylim(bottom=0)
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
    ax1.spines["left"].set_color("#B7BEC9")
    ax1.spines["bottom"].set_color("#B7BEC9")
    ax2.spines["right"].set_color("#B7BEC9")
    ax1.grid(axis="y", color="#EEF1F5", linewidth=1)

    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, loc="upper left", fontsize=7, frameon=False, ncol=2,
               bbox_to_anchor=(0, 1.22))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


_B_LEVEL   = {0: "Low", 1: "Mid", 2: "High"}
_B_ARMCFG  = {"left": "Left arm", "right": "Right arm", "both_alt": "Both (alternate)"}
_B_MODECFG = {"abd": "Abduction (side)", "flex": "Flexion (forward)", "mixed": "Mixed"}
_B_ORDER   = {0: "Random", 1: "Low to high", 2: "High to low"}
_B_EVENT_ROWS_MAX = 40


def _fmt_deg(v) -> str:
    return f"{float(v):.1f}°" if v not in (None, "") else "—"


def _balloon_period_blocks(sessions: list, st: dict, doc_width: float = None) -> list:
    """Summary + progress chart + history table for every Balloon Pop session
    in a period report."""
    if not sessions:
        return []
    pairs = [(s, balloon_stats(s)) for s in sessions]
    reps = sum(b["reps"] for _, b in pairs)
    known = [(s, b) for s, b in pairs if b["misses"] is not None]
    misses = sum(b["misses"] for _, b in known)
    judged = sum(b["reps"] + b["misses"] for _, b in known)
    pop_rate = f"{sum(b['reps'] for _, b in known) / judged * 100:.1f}%" if judged else "—"
    avg_pop = (sum(b["avg_time_to_pop"] * b["reps"] for _, b in pairs) / reps) if reps else 0.0
    max_elev = max((b["max_elevation"] for _, b in pairs), default=0.0)
    targets_met = sum(1 for s, _ in pairs if (s.session_data or {}).get("finish_reason") == "target")

    story = [
        st["section_header"]("Balloon Pop Summary"),
        Spacer(1, 12),
        st["metric_row"]([
            ("Balloon Sessions", str(len(pairs))),
            ("Balloons Popped",  str(reps)),
            ("Targets Met",      f"{targets_met}/{len(pairs)}"),
        ]),
        Spacer(1, 8),
        st["metric_row"]([
            ("Missed",        str(misses) if known else "—"),
            ("Pop Rate",      pop_rate),
            ("Avg Pop Time",  f"{avg_pop:.1f}s" if reps else "—"),
        ]),
        Spacer(1, 8),
        st["metric_row"]([
            ("Max Elevation",      _fmt_deg(max_elev) if reps else "—"),
            ("Left / Right Pops",  f"{sum(b['left_reps'] for _, b in pairs)} / {sum(b['right_reps'] for _, b in pairs)}"),
            ("Trunk Lean Events",  str(sum(b["lean_events"] for _, b in pairs))),
        ]),
        Spacer(1, 10),
    ]
    if len(known) < len(pairs):
        story += [
            Paragraph("Note: sessions saved before this update have no miss data; "
                      "missed / pop-rate figures cover the sessions that do.", st["body"]),
            Spacer(1, 6),
        ]
    story.append(Spacer(1, 12))

    chart_buf = _make_balloon_chart(sessions) if doc_width else None
    if chart_buf is not None:
        story += [
            st["section_header"]("Balloon Pop Progress"), Spacer(1, 10),
            RLImage(chart_buf, width=doc_width, height=doc_width * 0.34),
            Spacer(1, 18),
        ]

    story += [st["section_header"]("Balloon Pop History (Last 10)"), Spacer(1, 10)]
    rows = []
    for s, b in pairs[-10:]:
        sd = s.session_data or {}
        target = b["target_reps"]
        reason = sd.get("finish_reason") or "—"
        rows.append([
            Paragraph(s.id[:8], st["table_cell"]),
            Paragraph(s.start_time.strftime("%Y-%m-%d"), st["table_cell"]),
            Paragraph(f"{b['reps']}/{target}" if target else str(b["reps"]), st["table_cell"]),
            Paragraph(str(b["misses"]) if b["misses"] is not None else "—", st["table_cell"]),
            Paragraph(f"{b['avg_time_to_pop']:.1f}s" if b["reps"] else "—", st["table_cell"]),
            Paragraph(_fmt_deg(b["max_elevation"]) if b["reps"] else "—", st["table_cell"]),
            Paragraph(f"{b['left_reps']} / {b['right_reps']}", st["table_cell"]),
            Paragraph(str(b["lean_events"]), st["table_cell"]),
            Paragraph(f"{s.duration_seconds}s" if s.duration_seconds else "—", st["table_cell"]),
            Paragraph(_FINISH_LABEL.get(reason, reason), st["table_cell"]),
        ])
    story += [
        st["data_table"](
            ["Session", "Date", "Popped", "Missed", "Avg Pop", "Max Elev.", "L / R", "Lean", "Duration", "Result"],
            rows,
            [0.10, 0.11, 0.10, 0.08, 0.09, 0.11, 0.09, 0.07, 0.10, 0.15],
        ),
        Spacer(1, 24),
    ]
    return story


def _balloon_session_blocks(s: SessionModel, st: dict) -> list:
    """Full detail for one Balloon Pop round: overview, per-plane and per-arm
    results, calibration, the settings used, and every balloon popped."""
    sd = s.session_data or {}
    b = balloon_stats(s)
    cfg = b["settings"]
    reason = sd.get("finish_reason") or "—"
    target = b["target_reps"]
    judged = (b["reps"] + b["misses"]) if b["misses"] is not None else 0
    pop_rate = f"{b['reps'] / judged * 100:.1f}%" if judged else "—"

    story = [
        st["metric_row"]([
            ("Game",     "Balloon Pop"),
            ("Duration", f"{s.duration_seconds}s" if s.duration_seconds else "—"),
            ("Result",   _FINISH_LABEL.get(reason, reason)),
        ]),
        Spacer(1, 8),
        st["metric_row"]([
            ("Balloons Popped", f"{b['reps']}/{target}" if target else str(b["reps"])),
            ("Missed",          str(b["misses"]) if b["misses"] is not None else "—"),
            ("Pop Rate",        pop_rate),
        ]),
        Spacer(1, 8),
        st["metric_row"]([
            ("Avg Time to Pop",    f"{b['avg_time_to_pop']:.1f}s" if b["reps"] else "—"),
            ("Max Elevation",      _fmt_deg(b["max_elevation"]) if b["reps"] else "—"),
            ("Trunk Lean Events",  str(b["lean_events"])),
        ]),
        Spacer(1, 8),
        st["metric_row"]([
            ("Arm",        _B_ARMCFG.get(cfg.get("arm"), "—")),
            ("Movement",   _B_MODECFG.get(cfg.get("mode"), "—")),
            ("Left / Right Pops", f"{b['left_reps']} / {b['right_reps']}"),
        ]),
        Spacer(1, 22),
    ]
    if not b["complete"]:
        story += [
            Paragraph("Note: this session was saved before miss / calibration / settings data was recorded; "
                      "figures shown are rebuilt from the per-balloon log.", st["body"]),
            Spacer(1, 14),
        ]

    # Per movement plane
    rows = []
    for p in ("abd", "flex"):
        v = b["by_plane"].get(p) or {}
        if v.get("reps", 0) > 0:
            rows.append([
                Paragraph(_B_PLANE[p], st["table_cell"]),
                Paragraph(str(v.get("reps", 0)), st["table_cell"]),
                Paragraph(_fmt_deg(v.get("peak")), st["table_cell"]),
            ])
    if rows:
        story += [
            st["section_header"]("Results by Movement"), Spacer(1, 10),
            st["data_table"](["Movement", "Balloons Popped", "Peak Angle"], rows, [0.40, 0.30, 0.30]),
            Spacer(1, 22),
        ]

    # Per arm + plane, with the calibrated maximum
    keys = list(dict.fromkeys(list(b["by_arm_plane"].keys()) + list(b["calibration"].keys())))
    rows = []
    for k in keys:
        side, _, plane = k.partition("_")
        v = b["by_arm_plane"].get(k) or {}
        cal = v.get("calibrated_max") if v.get("calibrated_max") is not None else b["calibration"].get(k)
        peak = v.get("peak")
        pct = f"{peak / cal * 100:.0f}%" if (cal and peak) else "—"
        rows.append([
            Paragraph(f"{_B_ARM.get(side, side)} · {_B_PLANE.get(plane, plane)}", st["table_cell"]),
            Paragraph(str(v.get("reps", 0)), st["table_cell"]),
            Paragraph(_fmt_deg(peak) if peak else "—", st["table_cell"]),
            Paragraph(_fmt_deg(cal) if cal else "—", st["table_cell"]),
            Paragraph(pct, st["table_cell"]),
        ])
    if rows:
        story += [
            st["section_header"]("Arm & Calibration"), Spacer(1, 10),
            st["data_table"](["Arm · Movement", "Popped", "Peak Angle", "Calibrated Max", "% of Max Reached"],
                             rows, [0.28, 0.14, 0.18, 0.20, 0.20]),
            Spacer(1, 22),
        ]

    # Settings used for this round
    if cfg:
        def _row(label, value):
            return [Paragraph(label, st["table_cell"]), Paragraph(str(value), st["table_cell"])]
        lean = cfg.get("lean")
        rows = [
            _row("Balloon heights (low / mid / high, % of max)",
                 f"{cfg.get('lvl_low', '—')} / {cfg.get('lvl_mid', '—')} / {cfg.get('lvl_high', '—')}"),
            _row("Height order", _B_ORDER.get(cfg.get("order"), "—")),
            _row("Balloon size", f"{cfg.get('size', '—')}%"),
            _row("Distance from shoulder", f"{cfg.get('reach', '—')}%"),
            _row("Touch time needed to pop", f"{cfg.get('dwell', '—')} ms"),
            _row("Time before a balloon counts as missed", f"{cfg.get('timeout', '—')} s"),
            _row("Angle tolerance below target", f"{cfg.get('gate', '—')}°"),
            _row("Trunk lean limit", "Off" if lean == 0 else f"{lean}°" if lean is not None else "—"),
        ]
        story += [
            st["section_header"]("Settings Used"), Spacer(1, 10),
            st["data_table"](["Setting", "Value"], rows, [0.65, 0.35]),
            Spacer(1, 22),
        ]

    # Every balloon popped
    events = b["events"]
    if events:
        rows = []
        for e in events[:_B_EVENT_ROWS_MAX]:
            rows.append([
                Paragraph(str(e.get("id", "")), st["table_cell"]),
                Paragraph(_B_ARM.get(e.get("side"), "—"), st["table_cell"]),
                Paragraph(_B_PLANE.get(e.get("plane"), "—"), st["table_cell"]),
                Paragraph(_B_LEVEL.get(e.get("level"), "—"), st["table_cell"]),
                Paragraph(_fmt_deg(e.get("target")), st["table_cell"]),
                Paragraph(_fmt_deg(e.get("peak")), st["table_cell"]),
                Paragraph(f"{e.get('time_to_pop', 0):.1f}s", st["table_cell"]),
                Paragraph(str(e.get("lean_events", 0)), st["table_cell"]),
            ])
        story += [
            st["section_header"]("Balloon-by-Balloon Log"), Spacer(1, 10),
            st["data_table"](["#", "Arm", "Movement", "Height", "Target", "Peak", "Time to Pop", "Lean"],
                             rows, [0.07, 0.12, 0.17, 0.12, 0.14, 0.14, 0.14, 0.10]),
        ]
        if len(events) > _B_EVENT_ROWS_MAX:
            story += [Spacer(1, 6),
                      Paragraph(f"Showing the first {_B_EVENT_ROWS_MAX} of {len(events)} balloons.", st["body"])]
        story.append(Spacer(1, 22))
    return story


def build_report_sync(
    patient_id: str,
    report_type: str,
    range_start: datetime.datetime | None = None,
    range_end: datetime.datetime | None = None,
    exercise_type: str | None = None,
    game_id: str | None = None,
) -> str:
    """Pure sync function — safe to run in executor alongside async WS loop.

    report_type controls which sessions are included:
      - "weekly"  -> sessions from the last 7 days
      - "monthly" -> sessions from the last 30 days
      - "custom"  -> sessions between range_start and range_end (inclusive)
      - anything else (e.g. "history") -> all sessions

    exercise_type / game_id (optional, mutually exclusive — pass at most
    one) narrow that same period down to a single exercise or a single
    game, so a therapist can pull e.g. "last 30 days of Shoulder Flexion"
    or "full history of Rehab Runner" instead of every session mixed
    together.
    """
    with get_db() as db:
        p = db.query(Patient).filter(Patient.id == patient_id).first()
        if not p:
            raise HTTPException(404, "Patient not found")

        all_sessions = (
            db.query(SessionModel)
            .filter(SessionModel.patient_id == patient_id)
            .order_by(SessionModel.start_time)
            .all()
        )
        if not all_sessions:
            raise HTTPException(404, "No sessions found for this patient")

        now = datetime.datetime.now()
        if report_type == "weekly":
            cutoff = now - datetime.timedelta(days=7)
            sessions = [s for s in all_sessions if s.start_time >= cutoff]
            period_label = f"Weekly Report — last 7 days (as of {now.strftime('%Y-%m-%d')})"
        elif report_type == "monthly":
            cutoff = now - datetime.timedelta(days=30)
            sessions = [s for s in all_sessions if s.start_time >= cutoff]
            period_label = f"Monthly Report — last 30 days (as of {now.strftime('%Y-%m-%d')})"
        elif report_type == "custom":
            sessions = [
                s for s in all_sessions
                if range_start <= s.start_time <= range_end
            ]
            period_label = (
                f"Custom Report — {range_start.strftime('%Y-%m-%d')} "
                f"to {range_end.strftime('%Y-%m-%d')}"
            )
        else:
            sessions = all_sessions
            period_label = "Full History Report"

        # Narrow to a single exercise or a single game, on top of the period
        # filter above. Kept mutually exclusive: a report is either about
        # one named exercise or one named game, never both.
        if exercise_type:
            sessions = [s for s in sessions if not is_game_session(s) and s.exercise_type == exercise_type]
            period_label += f"  ·  Exercise: {exercise_type}"
        elif game_id:
            g_label = game_label(game_id)
            sessions = [s for s in sessions if is_game_session(s) and (s.session_data or {}).get("game_id") == game_id]
            period_label += f"  ·  Game: {g_label}"

        if not sessions:
            scope = f" for {exercise_type}" if exercise_type else f" for {game_label(game_id)}" if game_id else ""
            raise HTTPException(404, f"No sessions found in the selected date range{scope}")

        # Filter slug folded into the filename too (not just the download
        # name in routers/reports.py) — otherwise a "weekly" report and a
        # "weekly · Exercise: X" report generated in the same second would
        # collide on disk and silently overwrite one another.
        scope_slug = ""
        if exercise_type:
            scope_slug = "_" + "".join(c if c.isalnum() else "_" for c in exercise_type)
        elif game_id:
            scope_slug = "_" + "".join(c if c.isalnum() else "_" for c in game_id)

        report_dir = f"reports/{patient_id}"
        os.makedirs(report_dir, exist_ok=True)
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(report_dir, f"{patient_id}_{report_type}{scope_slug}_{ts}.pdf")

        # 28pt margins (~20-30px range) on every side, generous internal
        # whitespace is handled via Spacers between sections below.
        doc = SimpleDocTemplate(
            filepath, pagesize=A4,
            rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28,
        )
        st = _pdf_style_kit(doc.width)

        # Stats — exercise-quality averages exclude game sessions (Rehab
        # Runner etc. store 0 for ROM/stability by design; mixing them in
        # would understate real exercise performance). Game sessions get
        # their own summary block + table below instead.
        exercise_sessions, game_sessions = split_sessions(sessions)
        stat_sessions = exercise_sessions or sessions  # fall back if patient has only played games
        n           = len(stat_sessions)
        avg_acc     = sum(s.accuracy_percentage for s in stat_sessions) / n
        avg_rom     = sum(s.average_rom for s in stat_sessions) / n
        total_reps  = sum(s.completed_reps for s in stat_sessions)
        rec_score   = calculate_recovery_score(sessions)
        improvement = calculate_improvement(sessions)

        # Header
        story = [
            Paragraph("NovaMotion", st["title"]),
            Paragraph(period_label, st["subtitle"]),
            Spacer(1, 14),
        ]

        # Patient info — clean two-column key/value grid, no fills
        story += _patient_info_block(p, st, doc.width)

        # Summary metrics — bordered boxes. Exercise-quality numbers (accuracy / ROM /
        # recovery) only mean something for exercise sessions, so a games-only
        # report shows play totals instead of a wall of 0s.
        if exercise_sessions:
            story += [
                st["section_header"]("Summary Metrics"),
                Spacer(1, 12),
                st["metric_row"]([
                    ("Total Sessions", str(n)),
                    ("Avg Accuracy",   f"{avg_acc:.1f}%"),
                    ("Avg ROM",        f"{avg_rom:.1f}°"),
                ]),
                Spacer(1, 8),
                st["metric_row"]([
                    ("Total Reps",     str(total_reps)),
                    ("Recovery Score", f"{rec_score:.1f}%"),
                    ("Improvement",    f"{improvement:+.1f}%"),
                ]),
                Spacer(1, 22),
            ]

        else:
            play_secs = sum(x.duration_seconds or 0 for x in sessions)
            story += [
                st["section_header"]("Summary Metrics"),
                Spacer(1, 12),
                st["metric_row"]([
                    ("Total Sessions",  str(len(sessions))),
                    ("Games Played",    str(len({game_id_of(x) for x in sessions}))),
                    ("Total Play Time", f"{play_secs // 60}m {play_secs % 60:02d}s"),
                ]),
                Spacer(1, 22),
            ]

        # Game session summary — only shown if this patient has any (Rehab
        # Runner etc., services/game_state.py). Steps/target completion
        # instead of ROM/reps, since those don't apply to game sessions.
        # Each game reports its own metrics: Balloon Pop gets reps/angle sections,
        # every other (step-driven) game keeps the steps/score sections.
        balloon_sessions = [x for x in game_sessions if is_balloon_session(x)]
        step_sessions    = [x for x in game_sessions if not is_balloon_session(x)]

        if step_sessions:
            total_steps = sum(game_steps(s) for s in step_sessions)
            total_left_steps = sum(game_left_steps(s) for s in step_sessions)
            total_right_steps = sum(game_right_steps(s) for s in step_sessions)
            total_score = sum(game_final_score(s) for s in step_sessions)
            targets_met = sum(1 for s in step_sessions if (s.session_data or {}).get("finish_reason") == "target")

            story += [
                st["section_header"]("Game Session Summary"),
                Spacer(1, 12),
                st["metric_row"]([
                    ("Game Sessions",  str(len(step_sessions))),
                    ("Total Steps",    str(total_steps)),
                    ("Targets Met",    f"{targets_met}/{len(step_sessions)}"),
                ]),
                Spacer(1, 8),
                st["metric_row"]([
                    ("Left Steps",     str(total_left_steps)),
                    ("Right Steps",    str(total_right_steps)),
                    ("Total Score",    str(total_score)),
                ]),
                Spacer(1, 22),
            ]

        # Progress chart (matplotlib, embedded as PNG) — exercise sessions only,
        # same reasoning as the stats above (a 0° ROM game session would show
        # as a misleading dip in the trend line). Skipped when there isn't at
        # least one real exercise session to plot (e.g. a game-only report).
        if exercise_sessions:
            chart_buf = _make_progress_chart(stat_sessions)
            story += [
                st["section_header"]("Progress Trend"),
                Spacer(1, 10),
                RLImage(chart_buf, width=doc.width, height=doc.width * 0.34),
                Spacer(1, 22),
            ]

        # Session history — alternating row colors (exercise sessions only;
        # game sessions get their own table right after this one)
        if exercise_sessions:
            story += [st["section_header"]("Session History (Last 10)"), Spacer(1, 10)]

            rows = []
            for s in exercise_sessions[-10:]:
                rows.append([
                    Paragraph(s.id[:8], st["table_cell"]),
                    Paragraph(s.start_time.strftime("%Y-%m-%d"), st["table_cell"]),
                    Paragraph(s.exercise_type[:20], st["table_cell"]),
                    Paragraph(f"{s.accuracy_percentage:.1f}%", st["table_cell"]),
                    Paragraph(f"{s.average_rom:.1f}°", st["table_cell"]),
                    Paragraph(str(s.completed_reps), st["table_cell"]),
                    Paragraph(f"{s.stability_score:.1f}" if s.stability_score else "N/A", st["table_cell"]),
                ])

            story += [
                st["data_table"](
                    ["Session", "Date", "Exercise", "Accuracy", "ROM", "Reps", "Stability"],
                    rows,
                    [0.14, 0.16, 0.22, 0.13, 0.12, 0.10, 0.13],
                ),
                Spacer(1, 24),
            ]

        # Game session history — separate table (steps/target instead of ROM/reps)
        if step_sessions:
            story += [st["section_header"]("Game Session History (Last 10)"), Spacer(1, 10)]

            rows = []
            for s in step_sessions[-10:]:
                sd = s.session_data or {}
                target = sd.get("target_steps")
                reason = sd.get("finish_reason") or "—"
                result = {"target": "Target reached", "time": "Time up", "manual": "Stopped early"}.get(reason, reason)
                rows.append([
                    Paragraph(s.id[:8], st["table_cell"]),
                    Paragraph(s.start_time.strftime("%Y-%m-%d"), st["table_cell"]),
                    Paragraph(game_label(sd.get("game_id"))[:20], st["table_cell"]),
                    Paragraph(str(game_steps(s)), st["table_cell"]),
                    Paragraph(f"{game_left_steps(s)} / {game_right_steps(s)}", st["table_cell"]),
                    Paragraph(str(game_final_score(s)), st["table_cell"]),
                    Paragraph(str(target) if target else "—", st["table_cell"]),
                    Paragraph(f"{s.duration_seconds}s" if s.duration_seconds else "—", st["table_cell"]),
                    Paragraph(result, st["table_cell"]),
                ])

            story += [
                st["data_table"](
                    ["Session", "Date", "Game", "Steps", "L / R", "Score", "Target", "Duration", "Result"],
                    rows,
                    [0.10, 0.12, 0.14, 0.08, 0.10, 0.09, 0.09, 0.10, 0.18],
                ),
                Spacer(1, 24),
            ]

        # Balloon Pop — reps / angle based sections
        story += _balloon_period_blocks(balloon_sessions, st, doc.width)

        # AI Recommendations — rule-based, exercise metrics only; skipped
        # for a game-only report since accuracy/ROM/stability are 0 there
        # by design and would just produce noise.
        if exercise_sessions:
            story += [st["section_header"]("AI Recommendations"), Spacer(1, 10)]
            recs = _build_recommendations(stat_sessions, avg_acc, avg_rom, rec_score, improvement)
            for r in recs:
                story.append(Paragraph(f"■&nbsp;&nbsp;{r}", st["body"]))
                story.append(Spacer(1, 5))

        # Footer / signature
        story += _footer_block(st)

        doc.build(story)

        report_data = None
        if exercise_type:
            report_data = {"exercise_type": exercise_type}
        elif game_id:
            report_data = {"game_id": game_id}

        db.add(Report(patient_id=patient_id, report_type=report_type, file_path=filepath, report_data=report_data))
        db.commit()

    return filepath


def build_session_report_sync(patient_id: str, session_id: str) -> str:
    """Detailed report for a single session — one game round or one
    exercise session, shown in full rather than averaged into a period
    report. Pure sync function, same executor-safety note as build_report_sync."""
    with get_db() as db:
        p = db.query(Patient).filter(Patient.id == patient_id).first()
        if not p:
            raise HTTPException(404, "Patient not found")

        s = (
            db.query(SessionModel)
            .filter(SessionModel.id == session_id, SessionModel.patient_id == patient_id)
            .first()
        )
        if not s:
            raise HTTPException(404, "Session not found for this patient")

        is_game = is_game_session(s)
        sd = s.session_data or {}

        report_dir = f"reports/{patient_id}"
        os.makedirs(report_dir, exist_ok=True)
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(report_dir, f"{patient_id}_session_{s.id}_{ts}.pdf")

        doc = SimpleDocTemplate(
            filepath, pagesize=A4,
            rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28,
        )
        st = _pdf_style_kit(doc.width)

        title_label = game_label(sd.get("game_id")) if is_game else s.exercise_type
        story = [
            Paragraph("NovaMotion", st["title"]),
            Paragraph(
                f"Session Report — {title_label}  ·  {s.start_time.strftime('%Y-%m-%d %H:%M')}  ·  {s.id}",
                st["subtitle"],
            ),
            Spacer(1, 14),
        ]

        story += _patient_info_block(p, st, doc.width)

        story += [st["section_header"]("Session Overview"), Spacer(1, 12)]
        if is_game and is_balloon_session(s):
            story += _balloon_session_blocks(s, st)
        elif is_game:
            target = sd.get("target_steps")
            reason = sd.get("finish_reason") or "—"
            result = {"target": "Target reached", "time": "Time up", "manual": "Stopped early"}.get(reason, reason)
            lives  = sd.get("lives_remaining")
            story += [
                st["metric_row"]([
                    ("Game",     title_label),
                    ("Duration", f"{s.duration_seconds}s" if s.duration_seconds else "—"),
                    ("Result",   result),
                ]),
                Spacer(1, 8),
                st["metric_row"]([
                    ("Total Steps",  str(game_steps(s))),
                    ("Left / Right", f"{game_left_steps(s)} / {game_right_steps(s)}"),
                    ("Score",        str(game_final_score(s))),
                ]),
                Spacer(1, 8),
                st["metric_row"]([
                    ("Target Steps",     str(target) if target else "—"),
                    ("Lives Remaining",  str(lives) if lives is not None else "—"),
                    ("Recovery Score",   f"{s.recovery_score:.1f}%" if s.recovery_score else "—"),
                ]),
                Spacer(1, 22),
            ]
        else:
            story += [
                st["metric_row"]([
                    ("Exercise", s.exercise_type[:24]),
                    ("Duration", f"{s.duration_seconds}s" if s.duration_seconds else "—"),
                    ("Reps",     f"{s.completed_reps}/{s.total_reps}"),
                ]),
                Spacer(1, 8),
                st["metric_row"]([
                    ("Accuracy",       f"{s.accuracy_percentage:.1f}%"),
                    ("Avg ROM",        f"{s.average_rom:.1f}°"),
                    ("Recovery Score", f"{s.recovery_score:.1f}%"),
                ]),
                Spacer(1, 8),
                st["metric_row"]([
                    ("Stability",  f"{s.stability_score:.1f}" if s.stability_score else "N/A"),
                    ("Balance",    f"{s.balance_score:.1f}" if s.balance_score else "N/A"),
                    ("Smoothness", f"{s.movement_smoothness:.1f}" if s.movement_smoothness else "N/A"),
                ]),
                Spacer(1, 8),
                st["metric_row"]([
                    ("Fatigue",             f"{s.fatigue_estimation:.1f}" if s.fatigue_estimation else "N/A"),
                    ("Incorrect Movements", str(s.incorrect_movements)),
                    ("Session ID",          s.id[:8]),
                ]),
                Spacer(1, 22),
            ]

        # Per-rep breakdown — exercise sessions only (game rounds don't log
        # ExerciseResult rows, they log steps/score in session_data instead)
        if s.exercise_results:
            rows = []
            for er in sorted(s.exercise_results, key=lambda e: e.repetition_number)[:20]:
                rows.append([
                    Paragraph(str(er.repetition_number), st["table_cell"]),
                    Paragraph(f"{er.accuracy:.1f}%", st["table_cell"]),
                    Paragraph(f"{er.rom_achieved:.1f}°", st["table_cell"]),
                    Paragraph(f"{er.speed:.1f}", st["table_cell"]),
                    Paragraph("Yes" if er.is_completed else "No", st["table_cell"]),
                    Paragraph((er.feedback or "—")[:40], st["table_cell"]),
                ])
            story += [
                st["section_header"]("Repetition Breakdown"), Spacer(1, 10),
                st["data_table"](
                    ["Rep", "Accuracy", "ROM", "Speed", "Done", "Feedback"],
                    rows,
                    [0.08, 0.14, 0.12, 0.12, 0.10, 0.44],
                ),
                Spacer(1, 22),
            ]

        # Joint-level accuracy summary — aggregated per joint rather than a
        # raw per-frame log, which would run to hundreds of rows.
        if s.joint_angles:
            by_joint = {}
            for ja in s.joint_angles:
                bucket = by_joint.setdefault(ja.joint_name, {"angles": [], "targets": [], "correct": 0, "n": 0})
                bucket["angles"].append(ja.angle_value)
                if ja.target_angle is not None:
                    bucket["targets"].append(ja.target_angle)
                bucket["correct"] += 1 if ja.is_correct else 0
                bucket["n"] += 1

            rows = []
            for joint, b in sorted(by_joint.items()):
                avg_angle   = sum(b["angles"]) / b["n"]
                avg_target  = (sum(b["targets"]) / len(b["targets"])) if b["targets"] else None
                pct_correct = (b["correct"] / b["n"]) * 100
                rows.append([
                    Paragraph(joint, st["table_cell"]),
                    Paragraph(f"{avg_angle:.1f}°", st["table_cell"]),
                    Paragraph(f"{avg_target:.1f}°" if avg_target is not None else "—", st["table_cell"]),
                    Paragraph(f"{pct_correct:.0f}%", st["table_cell"]),
                ])
            story += [
                st["section_header"]("Joint Angle Summary"), Spacer(1, 10),
                st["data_table"](
                    ["Joint", "Avg Angle", "Avg Target", "% Correct"],
                    rows,
                    [0.32, 0.23, 0.23, 0.22],
                ),
                Spacer(1, 22),
            ]

        # Notes — rule-based recommendations, exercise sessions only (a
        # single game round has no ROM/stability data for the engine to
        # reason about; see helpers.py).
        if not is_game:
            story += [st["section_header"]("Notes"), Spacer(1, 10)]
            recs = _build_recommendations(
                [s], s.accuracy_percentage, s.average_rom,
                calculate_recovery_score([s]), 0.0,
            )
            for r in recs:
                story.append(Paragraph(f"■&nbsp;&nbsp;{r}", st["body"]))
                story.append(Spacer(1, 5))

        story += _footer_block(st)

        doc.build(story)

        db.add(Report(
            patient_id=patient_id, report_type="session", file_path=filepath,
            report_data={"session_id": s.id, "is_game_session": is_game},
        ))
        db.commit()

    return filepath
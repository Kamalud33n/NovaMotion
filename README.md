# NovaMotion — In-Clinic Rehabilitation AI System

FastAPI-based in-clinic platform for physiotherapy rehab tracking: webcam +
MediaPipe pose sessions, patient records, recovery analytics, PDF reports,
and a role-based portal for doctors, therapists and admins.

> `.env` setup (every variable explained) is in **`README-ENV-SETUP.md`**,
> not repeated here.

---

## 1. About

- **App name (code):** Rehabilitation AI System (In-Clinic) — `app.py`, version `2.3.0`
- **Runs on:** any server (local machine, VPS, Docker). The patient's camera is
  opened by the **doctor's browser** (`getUserMedia`), so the server needs no
  camera. Frames go to the server over a WebSocket (`/ws/camera`), MediaPipe runs
  server-side and the annotated frame comes back. Several doctors / clinics can
  use it at the same time, each with their own isolated session state.
- **Core idea:** a doctor/therapist starts a session with a patient in front
  of the webcam, MediaPipe tracks joint angles in real time, the app scores
  accuracy / ROM / stability / balance / smoothness / fatigue, and a
  recovery score + PDF report come out the other end.
- **Roles:**
  - `admin` — manages users (approve/reject/suspend), views all tickets,
    never registers through the public form (seeded from `.env` or created
    by another admin).
  - `doctor` and `therapist` — identical pages/permissions, just two labels.
    They register themselves (`/api/auth/register`), go through OTP email
    verification, then wait for admin approval before they can log in.
  - Every doctor/therapist only sees the patients *they* registered
    (`services/ownership.py` scopes every clinical query by `owner_id`).

## 2. Tech stack

| Layer      | Choice |
|------------|--------|
| Backend    | FastAPI + Uvicorn |
| DB         | MySQL via SQLAlchemy (`pymysql` driver) |
| Templates  | Jinja2 (server-rendered pages, JS calls JSON APIs for data) |
| Auth       | Cookie session + JWT (`pyjwt`), `SECRET_KEY`-signed |
| Vision     | OpenCV + MediaPipe (pose landmarks) |
| Reports    | ReportLab (PDF) |
| Mail       | stdlib `smtplib` via `services/auth/mailer.py` (OTP + forgot-password) |
| Game       | Pygame-based rehab mini-game, run out-of-process (`game/rehab_runner/worker.py`) |
| Containers | Docker + docker-compose (app + MySQL) |

## 3. Project structure

```
thero/
├── app.py                     # FastAPI app: middleware, router mounting, /api/health, demo seed
├── config.py                  # Jinja2 templates env, MediaPipe pose singleton
├── database.py                # SQLAlchemy engine/session, init_db(), MySQL env validation
├── models.py                  # Patient, SessionModel, JointAngle, ExerciseResult, Report,
│                               #   Setting, History, User, Ticket, PasswordResetToken, PendingSignup
├── requirements.txt
├── Dockerfile / docker-compose.yml
│
├── assign_owner.py            # one-off: assign legacy (owner-less) patients to a user
├── chaneg.py                  # one-off: copy an old SQLite DB into MySQL
├── check.py                   # one-off: inspect/reset the admin user directly in MySQL
│
├── routers/
│   ├── pages.py                # public/clinical HTML pages: /, /dashboard, /patients,
│   │                            #   /session, /reports, /analytics, /settings
│   ├── patients.py / sessions.py / reports.py / analytics.py / dashboard.py
│   ├── camera.py               # /ws/camera (browser camera) + /video_feed (server camera) + controls
│   ├── tickets.py              # clinical side: raise/view own tickets
│   ├── export.py               # MedNova integration: signed export API (EXPORT_API_KEY)
│   ├── auth/                   # login, forgot/reset password, profile
│   │   ├── login.py, forgot_password.py, reset_password.py, profile.py
│   │   ├── _common.py          # shared field-cleaning helpers
│   │   └── register.py         # ⚠ legacy/unused — replaced by registration_otp/ (see below)
│   └── admin/                  # everything admin-only
│       ├── pages.py            # /admin, /admin/users, /admin/approvals, /admin/tickets, /admin/settings
│       ├── dashboard.py, users.py, approvals.py, tickets.py
│       └── __init__.py         # combines the above; wraps API routes with require_admin
│
├── registration_otp/           # OTP-gated self-registration (kept separate from routers/auth)
│   ├── routes.py                # POST /api/auth/register, /register/verify-otp, /register/resend-otp
│   ├── service.py                # OTP generation/validation, PendingSignup lifecycle
│   └── mailer.py                 # OTP + "registration received" emails (uses services/auth/mailer.py)
│
├── services/
│   ├── app_config.py            # APP_HOST/PORT/CORS/EXPORT_API_KEY validation (fails fast on bad .env)
│   ├── auth/
│   │   ├── config.py             # all auth/SMTP/admin-seed env validation
│   │   ├── deps.py                # require_clinical / require_admin / require_*_page dependencies
│   │   ├── security.py            # password hashing, JWT, email/password validators
│   │   ├── roles.py                # role + status constants, home_for_role()
│   │   ├── bootstrap.py            # seed_admin() — creates the first admin from .env on startup
│   │   ├── mailer.py               # SMTP sending used by both forgot-password and OTP registration
│   │   └── web.py                  # render() helper for Jinja2 pages
│   ├── ownership.py              # per-user patient/session scoping
│   ├── helpers.py, report_builder.py
│   ├── user_context.py           # per-user live state (metrics, game, pose data, MediaPipe) + idle cleanup
│   ├── metrics.py                # MetricsState class: reps/stability/smoothness/balance/fatigue per user
│   ├── game_state.py             # GameState class: game engine/calibration/timer per user
│   ├── mjpeg_camera.py           # process_frame(ctx, frame) pose/game pipeline + optional server-camera stream
│   └── export_auth.py            # require_export_key dependency for routers/export.py
│
├── game/
│   ├── registry.py               # tracks running game-worker processes per session
│   └── rehab_runner/worker.py    # Pygame rehab mini-game, launched as a subprocess
│
├── templates/                   # Jinja2 HTML (dashboard, patients, session, reports, analytics, settings…)
│   ├── auth/                     # login.html (login + register tabs), forgot/reset password
│   └── admin/                    # admin dashboard/users/approvals/tickets/settings + shared sidebar
│
├── static/                      # app.js / style.css (shared), admin/ and auth/ (page-specific), chart.umd.min.js
├── assets/                       # logo + marketing/screenshot images
├── uploads/ / reports/ / data/   # runtime-written: patient photos, generated PDFs, rehab.db (dev SQLite)
└── .env                          # not committed — see README-ENV-SETUP.md
```

### A few things worth knowing before you touch the code

- **`routers/auth/register.py` is dead code.** Registration was moved to
  `registration_otp/` (OTP-gated) and `routers/auth/__init__.py` no longer
  includes it. Safe to delete once you've confirmed nothing else imports it.
- **`assign_owner.py`, `chaneg.py`, `check.py`** are one-off ops scripts, not
  part of the running app — run manually from the project root when needed
  (each has a docstring with usage at the top).
- **`data/rehab.db`** is a leftover/dev SQLite file — the app itself talks to
  MySQL only (`database.py`). `chaneg.py` exists specifically to migrate an
  old SQLite file like this into MySQL.
- **Camera runs in the browser.** `static/session.js` opens the camera with
  `getUserMedia` (external USB webcam preferred, built-in laptop camera as
  fallback), sends JPEG frames to `/ws/camera` and shows the annotated frame
  that comes back. `/video_feed` (server-attached camera via OpenCV) still
  exists for a purely local, in-clinic setup but is no longer used by the page.
- **Multi-user.** Live state is per logged-in user (`services/user_context.py`),
  so doctors at different clinics never share reps, exercise settings or game
  state. State is in process memory: run **one** uvicorn process (no
  `--workers N`). Idle sessions are freed after 30 minutes.
- **HTTPS is required in production** for browser camera access (`localhost`
  is the only exception). If you put a reverse proxy (nginx etc.) in front,
  it must forward WebSocket upgrades for `/ws/camera`
  (`proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";`).
- **Pin `mediapipe`** in `requirements.txt` — newer releases dropped `mp.solutions`.
- **Secrets & patient data are not in git.** Copy `.env.example` to `.env`;
  `data/`, `reports/` and `uploads/` are git-ignored (kept as empty folders).

## 4. Roles, approval & tickets — the flow

1. Doctor/therapist fills the register tab on `/` (login page) → OTP emailed
   → correct OTP creates a `User` row with `status=pending`.
2. Admin approves/rejects from `/admin/approvals`. Approve/reject each send
   the user an email; a rejected user can retry after 24 hours.
3. Once `status=approved`, the user can log in and lands on `/dashboard`.
4. Any doctor/therapist can raise a support ticket from `/settings` → shows
   up in the admin's `/admin/tickets` queue; admin's reply shows back under
   "My Tickets".
5. Admin's own `/admin/settings` page: profile + password only (no tickets —
   tickets are one-directional, clinical → admin).

## 5. Setup & run

### Option A — Docker (recommended)

```bash
cp .env.example .env      # see README-ENV-SETUP.md for every value
# edit .env

docker compose up --build
```

This starts the app **and** a MySQL container together. Read the notes at
the bottom of `docker-compose.yml` — in particular:
- `DB_USER` must be `root` for the compose file as written (it only creates
  a root MySQL user), unless you add `MYSQL_USER`/`MYSQL_PASSWORD` yourself.
- Already have a MySQL server elsewhere? Delete the `db:` service and point
  `DB_HOST` in `.env` at it directly.
- Webcam not detected in the container? Check `ls -l /dev/video*` on the
  host — `privileged: true` + `/dev:/dev` should make it visible, but if you
  can't run privileged containers, swap in an explicit `devices:` list.

### Option B — Local (no Docker)

Requirements: Python 3.10, a running MySQL server, and (for the webcam
session feature) system libs for OpenCV — on Debian/Ubuntu:
`libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg`.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # see README-ENV-SETUP.md
# edit .env

python app.py                   # or: uvicorn app:app --host 0.0.0.0 --port 8000
```

On startup the app:
1. Validates every required `.env` value (`services/app_config.py`,
   `services/auth/config.py`, `database.py`) — it refuses to start and
   prints every problem at once if anything's missing/invalid.
2. Creates the MySQL database + tables if they don't exist (`init_db()`).
3. Seeds the first admin from `ADMIN_NAME` / `ADMIN_EMAIL` / `ADMIN_PASSWORD`
   in `.env`, **only if no admin exists yet**.
4. Optionally seeds 3 demo patients + sessions if `SEED_DEMO_DATA=true`.

Then open `http://localhost:8000` (or whatever `APP_HOST`/`APP_PORT` you set).

## 6. Everyday admin ops

```bash
python assign_owner.py --list                 # see unassigned legacy patients
python assign_owner.py you@clinic.com --yes   # hand them all to one account

python check.py                               # inspect users table
python check.py --reset "NewPass123"          # force-reset/create the admin

python chaneg.py old_rehab.db                 # migrate an old SQLite DB into MySQL
```

## 7. API notes

- All clinical API routes (`patients`, `sessions`, `dashboard`, `analytics`,
  `reports`, `camera`, `tickets`) require a logged-in doctor/therapist/admin
  session (`require_clinical`).
- All `/api/admin/*` routes require an admin session (`require_admin`).
- `GET /api/health` is public — used for camera/mediapipe status checks.
- `POST /api/sessions/{id}/export` (in `routers/export.py`) is a separate,
  header-key-authenticated endpoint for the MedNova integration — it does
  **not** use cookie auth, it uses `EXPORT_API_KEY` /
  `EXPORT_API_KEY_HEADER` from `.env`. Leave `EXPORT_API_KEY` empty to
  disable this endpoint entirely.

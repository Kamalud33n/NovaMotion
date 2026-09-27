# .env Setup Guide

The app reads **every** setting from `.env` — nothing is hardcoded. On
startup, `services/app_config.py`, `services/auth/config.py` and
`database.py` each validate their own group of variables and, if anything
is missing or invalid, the app refuses to start and prints **all** the
problems at once (not just the first one).

There is no `.env.example` file in this build — create `.env` yourself in
the project root (same folder as `app.py`) using the template below, then
fill in real values.

---

## 1. Create the file

```bash
touch .env
```

Add the sections below, in any order. Comments (`#`) are optional but
recommended so future-you remembers why a value is what it is.

## 2. Database (MySQL) — required

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=thero
DB_CHARSET=utf8mb4
DB_COLLATION=utf8mb4_unicode_ci
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_RECYCLE_SECONDS=1800
DB_CONNECT_TIMEOUT_SECONDS=10
```

- `DB_PASSWORD` must be **present** in the file even if empty
  (`DB_PASSWORD=`) — only truly missing the key is rejected.
- `DB_NAME` is created automatically on first run if it doesn't exist
  (`database.py` connects without a DB name first, runs
  `CREATE DATABASE IF NOT EXISTS`, then connects to it).
- Running via `docker-compose.yml`: leave `DB_HOST` as whatever you like in
  `.env` — compose overrides it to `db` (the MySQL service name) for you.
  `DB_USER` must be `root` as the compose file is written (see notes at the
  bottom of `docker-compose.yml`).

## 3. App server / CORS — required

```env
APP_HOST=0.0.0.0
APP_PORT=8000
# Exact browser origins allowed to call the API with cookies. Comma
# separated, no trailing slash, and NEVER "*" (cookies + "*" don't mix).
# Leave empty for same-origin only (typical in-clinic single-machine setup).
CORS_ALLOW_ORIGINS=
SEED_DEMO_DATA=false
```

- `CORS_ALLOW_ORIGINS` must be **present** (the key itself, even if the
  value is empty) — omitting the line entirely is an error.
- Set `SEED_DEMO_DATA=true` only when you want the 3 sample patients + demo
  sessions inserted on next startup (only runs if the `patients` table is
  currently empty).

## 4. Auth — required

```env
SECRET_KEY=replace_with_a_long_random_string_at_least_32_chars
AUTH_COOKIE_NAME=thero_session
# true only when served over HTTPS — otherwise the browser drops the cookie
AUTH_COOKIE_SECURE=false
AUTH_SESSION_HOURS=12
PASSWORD_MIN_LENGTH=8
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_MINUTES=15
APP_NAME=Thero
# The address users open in their browser — used to build the
# password-reset link that gets emailed to them
APP_BASE_URL=http://localhost:8000
RESET_TOKEN_EXPIRE_MINUTES=30
RESET_REQUEST_COOLDOWN_SECONDS=60
```

- `SECRET_KEY` — minimum 32 characters, used to sign session JWTs. Generate
  one with:
  ```bash
  python -c "import secrets; print(secrets.token_hex(32))"
  ```
- `AUTH_COOKIE_SECURE` — set to `true` once you're behind HTTPS (production);
  `false` is fine for local `http://localhost` development.
- `APP_BASE_URL` **must** start with `http://` or `https://`, no trailing
  slash needed (it's stripped automatically).
- `PASSWORD_MIN_LENGTH` must be at least 6.

## 5. First admin — optional, but set all three or none

```env
ADMIN_NAME=Admin
ADMIN_EMAIL=admin@yourclinic.com
ADMIN_PASSWORD=set_a_strong_password_here
```

- Only used **once**, on first startup, and only if no admin user already
  exists in the database. Safe to leave in `.env` permanently — it's a
  no-op after the first admin is created.
- Leave all three empty if you'd rather create the admin manually with
  `python check.py --reset "SomePassword"`.

## 6. SMTP — optional (needed for forgot-password + registration OTP emails)

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_USE_TLS=true
SMTP_USE_SSL=false
SMTP_FROM_EMAIL=your_email@gmail.com
SMTP_FROM_NAME=Thero Support
SMTP_TIMEOUT_SECONDS=10
```

- Leave `SMTP_HOST` **empty** to disable email entirely — forgot-password
  and OTP-registration emails just won't send (registration OTP emails are
  required for the register flow to work, so email is effectively required
  for self-registration to function).
- If `SMTP_HOST` is set, `SMTP_PORT`, `SMTP_FROM_EMAIL` and
  `SMTP_TIMEOUT_SECONDS` become required too.
- `SMTP_USE_TLS` and `SMTP_USE_SSL` are mutually exclusive — pick one:
  - **TLS** (STARTTLS) → usually port `587`
  - **SSL** (implicit) → usually port `465`
- Using Gmail: you need an **App Password**, not your normal Google
  password (Google account → Security → 2-Step Verification → App
  passwords).

## 7. MedNova export API — optional

```env
# MedNova sends this key in the header below when calling
# /api/sessions/{id}/export. Leave empty to disable the endpoint.
EXPORT_API_KEY=
EXPORT_API_KEY_HEADER=X-MedNova-Key
```

- Leave `EXPORT_API_KEY` empty to disable `/api/sessions/{id}/export`
  entirely (it returns 401/403 for every request).
- If set, it must be at least 24 characters. Generate one with:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

## 8. Full template (copy-paste, then fill in)

```env
# ── Database (MySQL) ─────────────────────────────────────────────────────
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=thero
DB_CHARSET=utf8mb4
DB_COLLATION=utf8mb4_unicode_ci
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_RECYCLE_SECONDS=1800
DB_CONNECT_TIMEOUT_SECONDS=10

# ── App server / CORS ────────────────────────────────────────────────────
APP_HOST=0.0.0.0
APP_PORT=8000
CORS_ALLOW_ORIGINS=
SEED_DEMO_DATA=false

# ── Auth ─────────────────────────────────────────────────────────────────
SECRET_KEY=
AUTH_COOKIE_NAME=thero_session
AUTH_COOKIE_SECURE=false
AUTH_SESSION_HOURS=12
PASSWORD_MIN_LENGTH=8
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_MINUTES=15
APP_NAME=Thero
APP_BASE_URL=http://localhost:8000
RESET_TOKEN_EXPIRE_MINUTES=30
RESET_REQUEST_COOLDOWN_SECONDS=60

# ── Auth: first admin (created on startup only if no admin exists yet) ────
ADMIN_NAME=
ADMIN_EMAIL=
ADMIN_PASSWORD=

# ── SMTP (forgot-password + registration OTP mail) ───────────────────────
SMTP_HOST=
SMTP_PORT=
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_USE_TLS=true
SMTP_USE_SSL=false
SMTP_FROM_EMAIL=
SMTP_FROM_NAME=
SMTP_TIMEOUT_SECONDS=10

# ── MedNova export API ────────────────────────────────────────────────────
EXPORT_API_KEY=
EXPORT_API_KEY_HEADER=X-MedNova-Key
```

## 9. Quick troubleshooting

| Symptom on startup | Likely cause |
|---|---|
| `Invalid app configuration in .env: ...` | one of the App/CORS keys (section 3) is missing/invalid — the error message lists every problem |
| `Invalid auth configuration in .env: ...` | one of the Auth/SMTP/admin-seed keys (sections 4–6) is missing/invalid |
| Can't connect to MySQL on startup | check `DB_HOST`/`DB_PORT`/`DB_USER`/`DB_PASSWORD`, that MySQL is actually running, and that `DB_USER` can `CREATE DATABASE` |
| Forgot-password / OTP emails never arrive | `SMTP_HOST` is empty, or SMTP creds are wrong — check app logs for the SMTP error |
| `CORS_ALLOW_ORIGINS cannot contain '*'` | login uses cookies, so you must list exact origins (e.g. `http://localhost:3000`), never `*` |
| Admin not created | `ADMIN_NAME`/`ADMIN_EMAIL`/`ADMIN_PASSWORD` weren't all set together, or an admin already exists — check with `python check.py` |

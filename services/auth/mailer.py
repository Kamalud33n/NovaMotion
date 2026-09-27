"""
SMTP email. All connection details come from .env via services/auth/config.py.

send_email() is blocking (smtplib) and never raises — it returns True/False and
logs the real error server-side. Call it from a FastAPI BackgroundTask (or a
plain `def` route) so a slow mail server can't stall the event loop, and show
the user the same generic message whether or not the mail went out.
"""
import html
import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr
from typing import Optional
from urllib.parse import quote

from services.auth.config import settings

logger = logging.getLogger("thero.auth.mailer")


def smtp_configured() -> bool:
    return bool(settings.smtp_host)


def send_email(to_email: str, subject: str, text_body: str,
               html_body: Optional[str] = None) -> bool:
    if not smtp_configured():
        logger.error("Email not sent to %s: SMTP is not configured (SMTP_HOST empty in .env)", to_email)
        return False
    if any(ch in to_email for ch in "\r\n") or any(ch in subject for ch in "\r\n"):
        logger.error("Email not sent: header injection attempt blocked")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((settings.smtp_from_name or settings.app_name, settings.smtp_from_email))
    msg["To"] = to_email
    msg.set_content(text_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    try:
        ctx = ssl.create_default_context()
        if settings.smtp_use_ssl:
            server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port,
                                      timeout=settings.smtp_timeout_seconds, context=ctx)
        else:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port,
                                  timeout=settings.smtp_timeout_seconds)
        with server:
            if settings.smtp_use_tls and not settings.smtp_use_ssl:
                server.starttls(context=ctx)
            if settings.smtp_username:
                server.login(settings.smtp_username, settings.smtp_password or "")
            server.send_message(msg)
        return True
    except Exception:
        logger.exception("Failed to send email to %s via %s:%s",
                         to_email, settings.smtp_host, settings.smtp_port)
        return False


def build_reset_url(raw_token: str) -> str:
    return f"{settings.app_base_url}/reset-password?token={quote(raw_token, safe='')}"


def send_password_reset_email(to_email: str, full_name: str, raw_token: str) -> bool:
    url     = build_reset_url(raw_token)
    minutes = settings.reset_token_expire_minutes
    app     = settings.app_name
    name    = full_name or "there"

    text = (
        f"Hi {name},\n\n"
        f"We received a request to reset your {app} password.\n"
        f"Open this link to choose a new one (valid for {minutes} minutes, single use):\n\n"
        f"{url}\n\n"
        f"If you didn't ask for this, you can ignore this email — your password won't change.\n"
    )
    safe_url, safe_name, safe_app = html.escape(url, quote=True), html.escape(name), html.escape(app)
    html_body = (
        f"<p>Hi {safe_name},</p>"
        f"<p>We received a request to reset your {safe_app} password.</p>"
        f"<p><a href=\"{safe_url}\">Choose a new password</a><br>"
        f"<small>This link is valid for {minutes} minutes and can be used once.</small></p>"
        f"<p>If you didn't ask for this, you can ignore this email — your password won't change.</p>"
    )
    return send_email(to_email, f"{app} — reset your password", text, html_body)

"""Emails for the OTP-gated registration flow.

All SMTP plumbing already lives in services/auth/mailer.py (send_email) —
this module only builds the message bodies, so the connection handling,
.env config and header-injection guard aren't duplicated.
"""
import html

from services.auth.config import settings
from services.auth.mailer import send_email


def send_otp_email(to_email: str, full_name: str, otp: str, ttl_minutes: int) -> bool:
    name, app = full_name or "there", settings.app_name
    text = (
        f"Hi {name},\n\n"
        f"Thanks for signing up for {app}. Your verification code is:\n\n"
        f"    {otp}\n\n"
        f"This code will expire in {ttl_minutes} minutes, so please enter it soon.\n"
        f"If you didn't request this, you can safely ignore this email.\n\n"
        f"Warm regards,\n"
        f"The {app} Team\n"
    )
    safe_name, safe_app, safe_otp = html.escape(name), html.escape(app), html.escape(otp)
    html_body = (
        f"<p>Hi {safe_name},</p>"
        f"<p>Thanks for signing up for {safe_app}. Please use the verification code below to continue:</p>"
        f"<p style=\"font-size:24px;font-weight:bold;letter-spacing:4px;\">{safe_otp}</p>"
        f"<p><small>This code will expire in {ttl_minutes} minutes. If you didn't request this, please feel free to ignore this email.</small></p>"
        f"<p>Warm regards,<br>The {safe_app} Team</p>"
    )
    return send_email(to_email, f"{app} — your verification code", text, html_body)


def send_registration_received_email(to_email: str, full_name: str) -> bool:
    name, app = full_name or "there", settings.app_name
    text = (
        f"Hi {name},\n\n"
        f"Thank you for registering with {app}. Your email has been verified and your "
        f"registration has been submitted successfully.\n\n"
        f"Your account is now waiting for approval from our administrator. This usually "
        f"doesn't take long — as soon as it is approved, you will receive another email "
        f"and will be able to sign in right away.\n\n"
        f"We appreciate your patience, and thank you for choosing {app}.\n\n"
        f"Warm regards,\n"
        f"The {app} Team\n"
    )
    safe_name, safe_app = html.escape(name), html.escape(app)
    html_body = (
        f"<p>Hi {safe_name},</p>"
        f"<p>Thank you for registering with {safe_app}. Your email has been verified and your "
        f"registration has been submitted successfully.</p>"
        f"<p>Your account is now waiting for approval from our administrator. This usually doesn't "
        f"take long — as soon as it is approved, you will receive another email and will be able to "
        f"sign in right away.</p>"
        f"<p>We appreciate your patience, and thank you for choosing {safe_app}.</p>"
        f"<p>Warm regards,<br>The {safe_app} Team</p>"
    )
    return send_email(to_email, f"{app} — registration submitted, pending approval", text, html_body)


def send_account_approved_email(to_email: str, full_name: str) -> bool:
    name, app = full_name or "there", settings.app_name
    url = f"{settings.app_base_url}/login"
    text = (
        f"Hi {name},\n\n"
        f"We're happy to let you know that your {app} account has been reviewed and approved "
        f"by our administrator.\n\n"
        f"You're all set to sign in and get started:\n"
        f"{url}\n\n"
        f"Thank you for your patience during the review, and welcome aboard!\n\n"
        f"Warm regards,\n"
        f"The {app} Team\n"
    )
    safe_name, safe_app, safe_url = html.escape(name), html.escape(app), html.escape(url, quote=True)
    html_body = (
        f"<p>Hi {safe_name},</p>"
        f"<p>We're happy to let you know that your {safe_app} account has been reviewed and "
        f"approved by our administrator.</p>"
        f"<p>You're all set to sign in and get started:</p>"
        f"<p><a href=\"{safe_url}\">Sign in to {safe_app}</a></p>"
        f"<p>Thank you for your patience during the review, and welcome aboard!</p>"
        f"<p>Warm regards,<br>The {safe_app} Team</p>"
    )
    return send_email(to_email, f"{app} — your account has been approved", text, html_body)


def send_account_rejected_email(to_email: str, full_name: str) -> bool:
    name, app = full_name or "there", settings.app_name
    text = (
        f"Hi {name},\n\n"
        f"Thank you for your interest in {app} and for taking the time to register.\n\n"
        f"After review, our administrator was unable to approve your registration request "
        f"at this time. If you believe this may be a mistake, or if you have any questions, "
        f"please don't hesitate to reach out to your clinic administrator.\n\n"
        f"We're sorry for any inconvenience, and thank you for your understanding.\n\n"
        f"Warm regards,\n"
        f"The {app} Team\n"
    )
    safe_name, safe_app = html.escape(name), html.escape(app)
    html_body = (
        f"<p>Hi {safe_name},</p>"
        f"<p>Thank you for your interest in {safe_app} and for taking the time to register.</p>"
        f"<p>After review, our administrator was unable to approve your registration request at "
        f"this time. If you believe this may be a mistake, or if you have any questions, please "
        f"don't hesitate to reach out to your clinic administrator.</p>"
        f"<p>We're sorry for any inconvenience, and thank you for your understanding.</p>"
        f"<p>Warm regards,<br>The {safe_app} Team</p>"
    )
    return send_email(to_email, f"{app} — update on your registration", text, html_body)

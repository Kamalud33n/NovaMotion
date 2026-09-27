import uuid

from sqlalchemy import (
    Column, String, Integer, Float,
    DateTime, Text, Boolean, ForeignKey, JSON, LargeBinary
)
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


# ID generators 
def new_patient_id() -> str:
    return f"PAT-{uuid.uuid4().hex[:8].upper()}"


def new_session_id() -> str:
    return f"SES-{uuid.uuid4().hex[:8].upper()}"


def new_user_id() -> str:
    return f"USR-{uuid.uuid4().hex[:8].upper()}"


# Models 
class Patient(Base):
    __tablename__ = "patients"
    id                 = Column(String(50), primary_key=True, default=new_patient_id)
    name               = Column(String(100), nullable=False)
    age                = Column(Integer, nullable=False)
    gender             = Column(String(10), nullable=False)
    weight             = Column(Float, nullable=True)
    height             = Column(Float, nullable=True)
    diagnosis          = Column(String(200), nullable=True)
    affected_body_part = Column(String(100), nullable=True)
    doctor_name        = Column(String(100), nullable=True)
    therapist_name     = Column(String(100), nullable=True)
    phone              = Column(String(20), nullable=True)
    email              = Column(String(100), nullable=True)
    external_id        = Column(String(100), nullable=True, index=True)  # ID from an external system (e.g. MedNova); nullable since not every patient will have one
    # users.id of the doctor/therapist who registered this patient. Every clinical
    # query is scoped to this (services/ownership.py), so each user only ever sees
    # their own patients + sessions. Plain indexed column (no FK) so the startup
    # auto-ALTER in database.py can add it to an existing table. NULL = legacy
    # row from before ownership existed -> hidden until run: python assign_owner.py <email>
    owner_id           = Column(String(50), nullable=True, index=True)
    medical_history    = Column(Text, nullable=True)
    previous_injury    = Column(Text, nullable=True)
    current_treatment  = Column(Text, nullable=True)
    exercise_plan      = Column(Text, nullable=True)
    photo              = Column(LargeBinary().with_variant(LONGBLOB, "mysql"), nullable=True)
    date_created       = Column(DateTime, default=func.now())
    is_active          = Column(Boolean, default=True)
    sessions = relationship("SessionModel", back_populates="patient", cascade="all, delete-orphan")


class SessionModel(Base):
    __tablename__ = "sessions"
    id                  = Column(String(50), primary_key=True, default=new_session_id)
    patient_id          = Column(String(50), ForeignKey("patients.id"), nullable=False)
    exercise_type       = Column(String(100), nullable=False)
    start_time          = Column(DateTime, default=func.now())
    end_time            = Column(DateTime, nullable=True)
    duration_seconds    = Column(Integer, nullable=True)
    total_reps          = Column(Integer, default=0)
    completed_reps      = Column(Integer, default=0)
    accuracy_percentage = Column(Float, default=0.0)
    average_rom         = Column(Float, default=0.0)
    incorrect_movements = Column(Integer, default=0)
    stability_score     = Column(Float, default=0.0)
    balance_score       = Column(Float, default=0.0)
    movement_smoothness = Column(Float, default=0.0)
    fatigue_estimation  = Column(Float, default=0.0)
    recovery_score      = Column(Float, default=0.0)
    session_data        = Column(JSON, nullable=True)
    patient          = relationship("Patient", back_populates="sessions")
    joint_angles     = relationship("JointAngle", back_populates="session", cascade="all, delete-orphan")
    exercise_results = relationship("ExerciseResult", back_populates="session", cascade="all, delete-orphan")


class JointAngle(Base):
    __tablename__ = "joint_angles"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(String(50), ForeignKey("sessions.id"), nullable=False)
    timestamp    = Column(DateTime, default=func.now())
    joint_name   = Column(String(50), nullable=False)
    angle_value  = Column(Float, nullable=False)
    target_angle = Column(Float, nullable=True)
    deviation    = Column(Float, nullable=True)
    is_correct   = Column(Boolean, default=True)
    session = relationship("SessionModel", back_populates="joint_angles")


class ExerciseResult(Base):
    __tablename__ = "exercise_results"
    id                 = Column(Integer, primary_key=True, autoincrement=True)
    session_id         = Column(String(50), ForeignKey("sessions.id"), nullable=False)
    exercise_name      = Column(String(100), nullable=False)
    repetition_number  = Column(Integer, nullable=False)
    accuracy           = Column(Float, default=0.0)
    rom_achieved       = Column(Float, default=0.0)
    speed              = Column(Float, default=0.0)
    hold_duration      = Column(Float, default=0.0)
    compensation_score = Column(Float, default=0.0)
    is_completed       = Column(Boolean, default=False)
    feedback           = Column(Text, nullable=True)
    timestamp          = Column(DateTime, default=func.now())
    session = relationship("SessionModel", back_populates="exercise_results")


class Report(Base):
    __tablename__ = "reports"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    patient_id     = Column(String(50), ForeignKey("patients.id"), nullable=False)
    report_type    = Column(String(50), nullable=False)
    generated_date = Column(DateTime, default=func.now())
    file_path      = Column(String(200), nullable=True)
    report_data    = Column(JSON, nullable=True)


class Setting(Base):
    __tablename__ = "settings"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    key         = Column(String(50), unique=True, nullable=False)
    value       = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    updated_at  = Column(DateTime, default=func.now(), onupdate=func.now())


class History(Base):
    __tablename__ = "history"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), ForeignKey("patients.id"), nullable=False)
    action     = Column(String(100), nullable=False)
    details    = Column(Text, nullable=True)
    timestamp  = Column(DateTime, default=func.now())


# ── Auth ─────────────────────────────────────────────────────────────────
# role   : "admin" | "doctor" | "therapist"          (services/auth/roles.py)
# status : "pending" | "approved" | "rejected" | "suspended"
# All auth timestamps that get compared in Python (locked_until, expires_at,
# used_at) are naive UTC, written explicitly by services/auth/security.utcnow().
class User(Base):
    __tablename__ = "users"
    id                    = Column(String(50), primary_key=True, default=new_user_id)
    full_name             = Column(String(100), nullable=False)
    email                 = Column(String(150), nullable=False, unique=True, index=True)
    phone                 = Column(String(20), nullable=True)
    specialization        = Column(String(100), nullable=True)
    password_hash         = Column(String(255), nullable=False)
    role                  = Column(String(20), nullable=False)
    status                = Column(String(20), nullable=False, default="pending")
    failed_login_attempts = Column(Integer, default=0)
    locked_until          = Column(DateTime, nullable=True)
    last_login            = Column(DateTime, nullable=True)
    date_created          = Column(DateTime, default=func.now())
    approved_at           = Column(DateTime, nullable=True)
    approved_by           = Column(String(50), nullable=True)   # admin's users.id


# ── Support tickets ──────────────────────────────────────────────────────
# Raised by any clinical user (doctor/therapist), handled by an admin.
TICKET_STATUSES = ("open", "in_progress", "resolved", "closed")


class Ticket(Base):
    __tablename__ = "tickets"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    user_id        = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)
    subject        = Column(String(150), nullable=False)
    message        = Column(Text, nullable=False)
    status         = Column(String(20), nullable=False, default="open")
    admin_response = Column(Text, nullable=True)
    handled_by     = Column(String(50), nullable=True)   # admin's users.id
    created_at     = Column(DateTime, default=func.now())
    updated_at     = Column(DateTime, default=func.now(), onupdate=func.now())
    resolved_at    = Column(DateTime, nullable=True)
    user = relationship("User")


class PasswordResetToken(Base):
    """One-time password reset link. Only the SHA-256 hash of the token is
    stored — the raw token exists only in the emailed link."""
    __tablename__ = "password_reset_tokens"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(64), nullable=False, unique=True)
    expires_at = Column(DateTime, nullable=False)
    used_at    = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())


class PendingSignup(Base):
    """A registration in flight, waiting on its OTP (registration_otp/service.py).
    Nothing here becomes a real User until the code is verified. One row per
    email — re-registering the same email while a row exists refreshes it."""
    __tablename__ = "pending_signups"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    email           = Column(String(150), nullable=False, unique=True, index=True)
    full_name       = Column(String(100), nullable=False)
    phone           = Column(String(20), nullable=True)
    specialization  = Column(String(100), nullable=True)
    password_hash   = Column(String(255), nullable=False)
    role            = Column(String(20), nullable=False)
    otp_hash        = Column(String(64), nullable=False)
    otp_expires_at  = Column(DateTime, nullable=False)
    otp_attempts    = Column(Integer, nullable=False, default=0)
    last_sent_at    = Column(DateTime, nullable=False)
    created_at      = Column(DateTime, default=func.now())
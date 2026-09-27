import datetime

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from database import get_db, init_db
from models import Patient, SessionModel, JointAngle, Setting

from config import pose as _pose  # for /api/health mediapipe_ready flag
from services import mjpeg_camera

from routers import admin, analytics, auth, camera, dashboard, export, pages, patients, reports, sessions, tickets
from registration_otp.routes import router as registration_otp_router
from services.app_config import app_settings
from services.auth.bootstrap import seed_admin
from services.auth.deps import require_clinical
from services.export_auth import require_export_key

init_db()  # creates all tables (and the MySQL database itself, if missing)
seed_admin()  # first admin from .env, only if no admin exists yet

# ─── FastAPI app 
app = FastAPI(title="Rehabilitation AI System (In-Clinic)", version="2.3.0")

if app_settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Public on purpose: login page assets only
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

clinical = [Depends(require_clinical)]

app.include_router(auth.router)
app.include_router(registration_otp_router)
app.include_router(admin.router)
app.include_router(pages.router)
app.include_router(patients.router,  dependencies=clinical)
app.include_router(sessions.router,  dependencies=clinical)
app.include_router(dashboard.router, dependencies=clinical)
app.include_router(analytics.router, dependencies=clinical)
app.include_router(reports.router,   dependencies=clinical)
app.include_router(camera.router,    dependencies=clinical)
app.include_router(tickets.router,   dependencies=clinical)
app.include_router(export.router,    dependencies=[Depends(require_export_key)])


#Startup seed
# Off by default — set SEED_DEMO_DATA=true in .env if you ever want
# the 3 sample patients + random demo sessions back (e.g. for a fresh demo).
@app.on_event("startup")
async def seed():
    if not app_settings.seed_demo_data:
        return
    with get_db() as db:
        if db.query(Patient).count() > 0:
            return

        sample_patients = [
            Patient(name="John Smith",      age=45, gender="Male",   weight=82.5, height=178.0,
                    diagnosis="Rotator Cuff Tear",       affected_body_part="Right Shoulder",
                    doctor_name="Dr. Sarah Johnson",     therapist_name="Michael Brown",
                    phone="+1 (555) 123-4567",           email="john.smith@email.com"),
            Patient(name="Maria Garcia",    age=62, gender="Female", weight=68.0, height=165.0,
                    diagnosis="Knee Osteoarthritis",     affected_body_part="Left Knee",
                    doctor_name="Dr. Robert Chen",       therapist_name="Lisa Wong",
                    phone="+1 (555) 234-5678",           email="maria.garcia@email.com"),
            Patient(name="Robert Williams", age=38, gender="Male",   weight=90.0, height=183.0,
                    diagnosis="Lumbar Disc Herniation",  affected_body_part="Lower Back",
                    doctor_name="Dr. Emily Davis",       therapist_name="James Wilson",
                    phone="+1 (555) 345-6789",           email="robert.williams@email.com"),
        ]

        for p in sample_patients:
            db.add(p)
        db.flush()

        exercises = ["Shoulder Rehab", "Knee Flexion", "Arm Raise", "Balance Exercise"]
        joints    = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_knee", "right_knee"]

        # 5 sessions per patient, oldest first, with a fixed (not random)
        # mild-improvement progression — realistic demo data without
        # fabricating numbers via random.uniform/random.choice.
        for p in sample_patients:
            for i in range(5):
                days_ago  = 25 - (i * 5)          # 25, 20, 15, 10, 5 days ago
                accuracy  = 62.0 + (i * 6.0)       # 62 -> 86
                rom       = 45.0 + (i * 8.0)       # 45 -> 77
                stability = 55.0 + (i * 7.0)       # 55 -> 83
                balance   = 50.0 + (i * 7.5)       # 50 -> 80
                smoothness= 55.0 + (i * 7.0)       # 55 -> 83
                fatigue   = max(10.0, 35.0 - (i * 5.0))  # 35 -> 15
                total_reps     = 12 + i
                completed_reps = 8 + i

                sess = SessionModel(
                    patient_id          = p.id,
                    exercise_type       = exercises[i % len(exercises)],
                    start_time          = datetime.datetime.now() - datetime.timedelta(days=days_ago),
                    duration_seconds    = 240 + (i * 30),
                    total_reps          = total_reps,
                    completed_reps      = completed_reps,
                    accuracy_percentage = accuracy,
                    average_rom         = rom,
                    incorrect_movements = max(0, total_reps - completed_reps),
                    stability_score     = stability,
                    balance_score       = balance,
                    movement_smoothness = smoothness,
                    fatigue_estimation  = fatigue,
                    recovery_score      = round((accuracy * 0.30 + rom * 0.20 + stability * 0.25 + balance * 0.25), 1),
                )
                db.add(sess)
                db.flush()

                for j, joint in enumerate(joints):
                    target_angle = 90.0 + (j * 5.0)
                    angle_value  = target_angle - 10.0 + (i * 2.0)  # steadily closer to target
                    db.add(JointAngle(
                        session_id   = sess.id,
                        joint_name   = joint,
                        angle_value  = angle_value,
                        target_angle = target_angle,
                        deviation    = round(angle_value - target_angle, 1),
                        is_correct   = angle_value >= target_angle * 0.85,
                    ))

        for key, val, desc in [
            ("fps_target",           "10",      "Target FPS for pose streaming"),
            ("camera_resolution",    "1280x720", "Camera resolution"),
            ("confidence_threshold", "0.5",     "Min confidence threshold"),
            ("rom_warning",          "30",      "Min ROM warning threshold"),
        ]:
            db.add(Setting(key=key, value=val, description=desc))

        db.commit()
        print("Seed data inserted.")


# Health check
@app.get("/api/health")
async def health():
    # In-clinic only: one server-attached webcam, one MJPEG pipeline.
    # "camera_running" is what dashboard.html / index.html read for the
    # Camera Connected/Disconnected badge.
    running = mjpeg_camera.is_active()
    return {
        "status":          "healthy",
        "timestamp":       datetime.datetime.now().isoformat(),
        "camera_running":  running,
        "mjpeg_running":   running,
        "mediapipe_ready": _pose is not None,
        "resolution":      "1280x720",
    }


# Entry point 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=app_settings.host, port=app_settings.port, reload=False)
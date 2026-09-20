"""
SGBIT Attendance System - Central Configuration
All settings loaded from .env file
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── MongoDB ────────────────────────────────────────────────────────────────
MONGO_URI       = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME         = os.getenv("DB_NAME", "sgbit_attendance")

# ─── Flask ──────────────────────────────────────────────────────────────────
SECRET_KEY      = os.getenv("FLASK_SECRET_KEY", "sgbit_change_this_secret_2026")
DEBUG           = os.getenv("FLASK_DEBUG", "True").lower() == "true"
PORT            = int(os.getenv("FLASK_PORT", 5000))

# ─── Auth (DB-backed — these are only used to seed the default admin) ─────────
ADMIN_DEFAULT_USERNAME = os.getenv("ADMIN_DEFAULT_USERNAME", "admin")
ADMIN_DEFAULT_PASSWORD = os.getenv("ADMIN_DEFAULT_PASSWORD", "sgbit@2026")

# ─── SMTP Email (Gmail) ──────────────────────────────────────────────────────
SMTP_HOST       = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.getenv("SMTP_PORT", 587))
EMAIL_SENDER    = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD", "")          # Gmail App Password
EMAIL_SENDER_NAME = "SGBIT CSE Department"

# Alert trigger: send email after N consecutive absences
ABSENCE_ALERT_THRESHOLD = int(os.getenv("ABSENCE_ALERT_THRESHOLD", 2))

# ─── Camera Source ───────────────────────────────────────────────────────────
# Set CAMERA_SOURCE in .env to:
#   "0"                           → default built-in / USB webcam (index 0)
#   "1", "2", ...                 → other local webcam index
#   "http://192.168.x.x:8080/video" → IP Webcam app on Android phone
_cam_src = os.getenv("CAMERA_SOURCE", "0")
try:
    CAMERA_SOURCE = int(_cam_src)          # local webcam index
except ValueError:
    CAMERA_SOURCE = _cam_src               # IP Webcam URL string

# ─── Face Recognition ────────────────────────────────────────────────────────
RECOGNITION_THRESHOLD  = float(os.getenv("RECOGNITION_THRESHOLD", 0.60))
REGISTRATION_FRAMES    = int(os.getenv("REGISTRATION_FRAMES", 25))   # frames to capture
INSIGHTFACE_MODEL_PACK = os.getenv("INSIGHTFACE_MODEL_PACK", "buffalo_l")

# ─── Liveness ────────────────────────────────────────────────────────────────
LIVENESS_REQUIRED      = os.getenv("LIVENESS_REQUIRED", "True").lower() == "true"
EAR_BLINK_THRESHOLD    = float(os.getenv("EAR_BLINK_THRESHOLD", 0.25))
ANTI_SPOOF_THRESHOLD   = float(os.getenv("ANTI_SPOOF_THRESHOLD", 0.60))

# ─── Attendance / Eligibility ─────────────────────────────────────────────────
ATTENDANCE_THRESHOLD   = int(os.getenv("ATTENDANCE_THRESHOLD", 75))   # % for eligibility

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR      = os.path.join(BASE_DIR, "dataset")
EXPORTS_DIR      = os.path.join(BASE_DIR, "exports")
FACES_DIR        = os.path.join(BASE_DIR, "static", "uploads", "faces")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
ANTI_SPOOF_MODEL = os.path.join(MODELS_DIR, "minifasnet_v2.onnx")

# Ensure directories exist
for _dir in [DATASET_DIR, EXPORTS_DIR, FACES_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)

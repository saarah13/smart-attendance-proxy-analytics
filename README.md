# SGBIT Intelligent Attendance Management System
**Department of Computer Science & Engineering, SGBIT Belagavi**

A production-grade face recognition attendance system built with Python, Flask, MongoDB, InsightFace (ArcFace), FAISS, and MediaPipe.

---

## 🚀 Quick Start

### Prerequisites

| Software | Version | Download |
|---|---|---|
| Python | 3.10+ | https://python.org/downloads |
| MongoDB | 8.x Community | https://mongodb.com/try/download |
| Git (optional) | Any | https://git-scm.com |

> **Windows only**: If `insightface` fails to install, install **Visual C++ Build Tools** from:
> https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

### Step 1 — Run Setup (First Time Only)
```
Double-click: setup.bat
```
This will:
- Install all Python dependencies (`insightface`, `faiss-cpu`, `mediapipe`, etc.)
- Download ArcFace + SCRFD models (~700MB, one-time download)
- Create your `.env` config file

---

### Step 2 — Configure Environment
Open `.env` in Notepad and fill in:
```
FACULTY_USERNAME=admin
FACULTY_PASSWORD=your_password

EMAIL_SENDER=your_gmail@gmail.com
EMAIL_PASSWORD=xxxx_xxxx_xxxx_xxxx   # Gmail App Password (NOT your Gmail password)
```

**How to get Gmail App Password:**
1. Go to Google Account → Security
2. Enable 2-Step Verification
3. App passwords → Select app: Mail → Generate
4. Copy the 16-character password

---

### Step 3 — Start MongoDB
Make sure MongoDB is running:
```
net start MongoDB
```
Or launch `mongod.exe` / MongoDB Compass.

---

### Step 4 — Run the System
```
Double-click: run.bat
```
Then open your browser at: **http://localhost:5000**

---

## 📋 How to Use

### 1. Add Subjects First
- Go to **Subjects** → Add your subjects (name, code, semester, section)

### 2. Register Students
- Go to **Register Student**
- Fill in student details (Name, USN, Semester, Section, Parent Email)
- Click **Start Camera** → then **Start Capture**
- The system captures 25 frames and stores the ArcFace embedding

### 3. Take Attendance
- Go to **Take Attendance**
- Select the subject from dropdown
- Click **Scan Faces** — system will detect and identify students
- Click **Mark All Present** to save attendance to MongoDB

### 4. View Reports
- **Reports** → Filter by subject & date, download Excel
- **Eligibility** → See which students qualify for internals (≥75%)
- **IA Marks** → Enter/view internal assessment marks

### 5. Email Alerts
- Go to **Eligibility** → Click **Send Alerts**
- Emails are sent via SMTP to parents of at-risk and ineligible students

---

## 🔐 Faculty Login

Default credentials (change in `.env`):
```
Username: admin
Password: sgbit@2026
```

---

## 📁 Project Structure

```
project_sgbit/
├── app.py                 # Flask application (all routes)
├── config.py              # Configuration loader
├── requirements.txt       # Python dependencies
├── setup.bat              # First-time setup
├── run.bat                # Start server
├── make_zip.bat           # Package for sharing
│
├── camera/                # Webcam capture
├── detector/              # SCRFD face detection
├── aligner/               # 5-landmark face alignment
├── liveness/              # Blink + head pose + anti-spoof
├── embeddings/            # ArcFace 512-D embedding
├── matcher/               # FAISS cosine similarity search
├── modules/               # Business logic (attendance, email, marks...)
├── templates/             # Jinja2 HTML templates
├── static/                # CSS, JS
├── dataset/               # Stored .npy embeddings (auto-created)
├── exports/               # Excel report downloads (auto-created)
└── models/                # ONNX model files (auto-downloaded)
```

---

## 🧠 Face Recognition Pipeline

```
Webcam Frame
    → SCRFD Face Detection
    → 5-Landmark Affine Alignment (112×112)
    → MiniFASNet Anti-Spoof Check
    → ArcFace 512-D Embedding
    → L2 Normalization
    → FAISS Cosine Similarity Search
    → Identity (confidence > 0.60) or Unknown
```

---

## 📦 Sharing / Distribution

To create a ZIP for sharing:
```
Double-click: make_zip.bat
```
Output: `SGBIT_Attendance_System.zip`

> **.env** and **dataset/** (face data) are **NOT** included in the ZIP for privacy.

The recipient should:
1. Unzip the file
2. Run `setup.bat`
3. Edit `.env`
4. Run `run.bat`

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017/` | MongoDB connection string |
| `FACULTY_USERNAME` | `admin` | Faculty login username |
| `FACULTY_PASSWORD` | `sgbit@2026` | Faculty login password |
| `EMAIL_SENDER` | _(empty)_ | Gmail address for alerts |
| `EMAIL_PASSWORD` | _(empty)_ | Gmail App Password |
| `ATTENDANCE_THRESHOLD` | `75` | Eligibility cutoff % |
| `RECOGNITION_THRESHOLD` | `0.60` | Min cosine similarity to recognize |
| `LIVENESS_REQUIRED` | `True` | Enable anti-spoof check |
| `ABSENCE_ALERT_THRESHOLD` | `2` | Consecutive absences before email alert |

---

## 🛠️ Troubleshooting

**`insightface` install fails**
→ Install Visual C++ Build Tools, then rerun `setup.bat`

**Model download fails**
→ Check internet connection, rerun `setup.bat` (idempotent)

**Camera not detected**
→ Ensure no other app is using the webcam; check `src=0` in `config.py`

**MongoDB connection error**
→ Start MongoDB: `net start MongoDB` or launch mongod manually

**Emails not sending**
→ Verify Gmail App Password in `.env`; ensure 2-Step Verification is ON

---

## 👨‍💻 Technology Stack

| Component | Technology |
|---|---|
| Web Framework | Flask 3.0 |
| Database | MongoDB 8.x (PyMongo) |
| Face Detection | InsightFace SCRFD |
| Face Embedding | ArcFace (512-D) |
| Vector Search | FAISS |
| Liveness | MediaPipe + MiniFASNet |
| Email | Python smtplib (SMTP/TLS) |
| Excel Export | openpyxl |
| Frontend | HTML5 + Vanilla CSS + JS |

---

*SGBIT CSE Department · Belagavi · July 2026*

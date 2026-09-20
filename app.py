"""
app.py  -  SGBIT Intelligent Attendance Management System
Flask application with multi-role auth (Admin / Faculty / Student)
and entrance-camera auto-recognition pipeline.
"""
import os, json, base64, time, threading
from datetime import datetime, date
from functools import wraps

import cv2
import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, Response, send_file,
                   flash, stream_with_context)
from pymongo import MongoClient

import config
from modules.db          import get_db
from modules.auth        import authenticate, create_user, get_all_users, \
                                 set_active, reset_password, seed_admin

from camera.webcam       import get_stream
from detector.retinaface_detect import detect_faces, draw_detections
from aligner.align       import align_face, preprocess_for_arcface
from liveness.anti_spoof import is_real_face
from embeddings.generate import (get_embeddings_batch, average_embedding,
                                  save_embedding, get_embedding)
from matcher.faiss_matcher import search, build_index, total_registered

from modules.attendance  import (mark_attendance, mark_absent, get_daily_report,
                                  get_student_summary, get_class_summary,
                                  get_consecutive_absences, get_today_stats)
from modules.email_alerts import (send_absence_alert, send_low_attendance_alert,
                                   get_email_logs)
from modules.eligibility  import (get_eligible_students, generate_full_report,
                                   get_student_eligibility, check_low_attendance_alert)
from modules.marks        import upsert_marks, get_marks, get_student_marks
from modules.subjects     import (add_subject, get_subjects, delete_subject,
                                   get_subject_names, get_subject_threshold,
                                   update_subject_threshold)
from modules.excel_export import export_attendance, export_eligibility, export_marks

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.SECRET_KEY


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH DECORATORS
# ══════════════════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    """Restrict route to users whose session role is in `roles`."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("logged_in"):
                return redirect(url_for("login"))
            if session.get("role") not in roles:
                return render_template("403.html"), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# Convenience aliases
def admin_required(f):   return role_required("admin")(f)
def faculty_required(f): return role_required("admin", "faculty")(f)
def student_required(f): return role_required("student")(f)


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        role_tab = request.form.get("role", "faculty")   # tab selected by user

        user = authenticate(username, password)
        if user and user["role"] == role_tab and user["is_active"]:
            session["logged_in"] = True
            session["role"]      = user["role"]
            session["name"]      = user["name"]
            session["username"]  = user["username"]
            session["usn"]       = user.get("usn")
            if user["role"] == "student":
                return redirect(url_for("student_dashboard"))
            elif user["role"] == "admin":
                return redirect(url_for("admin_users"))
            return redirect(url_for("dashboard"))
        error = "Invalid credentials or role mismatch. Please try again."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD (Admin / Faculty)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
@faculty_required
def dashboard():
    db    = get_db()
    stats = {
        "total_students":  db.students.count_documents({}),
        "registered":      total_registered(),
        "subjects_count":  db.subjects.count_documents({}),
        "today":           get_today_stats(),
        "emails_sent":     db.email_logs.count_documents({"status": "sent"}),
    }
    recent = get_daily_report(date.today().isoformat())[:10]
    return render_template("dashboard.html", stats=stats, recent=recent)


# ══════════════════════════════════════════════════════════════════════════════
#  STUDENT PORTAL
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/student/dashboard")
@student_required
def student_dashboard():
    usn          = session.get("usn")
    attendance   = get_student_eligibility(usn) if usn else []
    marks        = get_student_marks(usn) if usn else []
    recent_log   = get_daily_report(subject=None)
    # Filter log to only this student
    recent_log   = [r for r in recent_log if r.get("usn") == usn][:30]
    return render_template("student_dashboard.html",
                           attendance=attendance, marks=marks,
                           recent_log=recent_log,
                           student_name=session.get("name", ""),
                           usn=usn,
                           now=date.today().strftime("%A, %d %B %Y"))


# ══════════════════════════════════════════════════════════════════════════════
#  STUDENTS (Admin only for register)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/students")
@faculty_required
def students_list():
    db   = get_db()
    rows = list(db.students.find({}, {"embedding": 0}).sort("name", 1))
    for r in rows:
        r["_id"] = str(r["_id"])
    return render_template("students.html", students=rows)


@app.route("/students/register", methods=["GET"])
def register_student():
    subjects = get_subject_names()
    return render_template("register_student.html", subjects=subjects)


@app.route("/api/register_student", methods=["POST"])
def api_register_student():
    """
    JSON endpoint called by the browser after capturing frames.
    Body: { name, usn, semester, section, parent_email, phone, password, frames: [base64, ...] }
    """
    data         = request.json or {}
    name         = data.get("name", "").strip()
    usn          = data.get("usn", "").strip().upper()
    semester     = int(data.get("semester", 1))
    section      = data.get("section", "A").upper()
    parent_email = data.get("parent_email", "").strip()
    phone        = data.get("phone", "").strip()
    password     = data.get("password", "").strip()
    b64_frames   = data.get("frames", [])

    if not name or not usn:
        return jsonify({"ok": False, "error": "Name and USN are required."}), 400
    if not password:
        return jsonify({"ok": False, "error": "Initial password is required."}), 400
    if len(b64_frames) < 5:
        return jsonify({"ok": False, "error": "Not enough frames captured."}), 400

    db = get_db()
    if db.students.find_one({"usn": usn}):
        return jsonify({"ok": False, "error": f"Student with USN {usn} already exists."}), 409

    # Decode base64 frames -> OpenCV images
    frames = []
    for b64 in b64_frames:
        try:
            header, encoded = b64.split(",", 1) if "," in b64 else ("", b64)
            arr   = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        except Exception:
            continue

    if len(frames) < 5:
        return jsonify({"ok": False, "error": "Could not decode enough valid frames."}), 400

    embeddings = get_embeddings_batch(frames)
    if len(embeddings) < 3:
        return jsonify({"ok": False, "error": "Not enough valid face frames. Ensure good lighting."}), 400

    avg_emb    = average_embedding(embeddings)
    save_embedding(usn, avg_emb)

    thumb_path = os.path.join(config.FACES_DIR, f"{usn}.jpg")
    cv2.imwrite(thumb_path, frames[len(frames) // 2])

    doc = {
        "name":           name,
        "usn":            usn,
        "semester":       semester,
        "section":        section,
        "parent_email":   parent_email,
        "phone":          phone,
        "embedding":      avg_emb.tolist(),
        "face_thumbnail": f"uploads/faces/{usn}.jpg",
        "registered_at":  datetime.utcnow(),
    }
    db.students.insert_one(doc)

    # Create student login account
    create_user(username=usn.lower(), password=password,
                role="student", name=name, usn=usn)

    build_index()
    return jsonify({"ok": True, "message": f"✅ {name} ({usn}) registered successfully!"})


@app.route("/api/delete_student/<usn>", methods=["DELETE"])
@faculty_required
def api_delete_student(usn):
    db = get_db()
    usn = usn.upper()
    student = db.students.find_one({"usn": usn})
    if not student:
        return jsonify({"ok": False, "error": "Student not found"}), 404

    # Delete thumbnail if exists
    if student.get("face_thumbnail"):
        thumb_path = os.path.join(config.BASE_DIR, "static", student["face_thumbnail"].replace('/', os.sep))
        if os.path.exists(thumb_path):
            try:
                os.remove(thumb_path)
            except Exception:
                pass

    # Delete from database
    db.students.delete_one({"usn": usn})
    db.users.delete_one({"usn": usn})
    # Optional: db.attendance.delete_many({"usn": usn})

    build_index()
    return jsonify({"ok": True, "message": f"Student {usn} deleted successfully"})


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRANCE CAMERA — Auto-Recognition Session (SSE)
# ══════════════════════════════════════════════════════════════════════════════

# Shared session state (single camera, single session at a time)
_active_session: dict = {
    "running":    False,
    "subject":    None,
    "session_id": None,
    "marked":     {},    # usn -> {name, time, pct}
    "events":     [],    # SSE event queue
    "detections": [],    # latest faces + labels for live overlay
}
_session_lock = threading.Lock()


def _recognition_loop():
    """
    Background thread: continuously grab frames, run the full pipeline,
    auto-mark attendance, push SSE events to the queue.
    """
    stream = get_stream(config.CAMERA_SOURCE)
    while True:
        with _session_lock:
            if not _active_session["running"]:
                break
            subject    = _active_session["subject"]
            session_id = _active_session["session_id"]
            marked     = _active_session["marked"]

        frame = stream.read()
        if frame is None:
            time.sleep(0.1)
            continue

        faces = detect_faces(frame)
        frame_labels = []       # labels for bounding-box overlay
        frame_faces  = []       # face dicts for overlay
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            # Liveness check
            is_real, spoof_score = is_real_face(crop)
            if config.LIVENESS_REQUIRED and not is_real:
                frame_faces.append(face)
                frame_labels.append("Spoof")
                continue

            emb = get_embedding(crop)
            if emb is None:
                frame_faces.append(face)
                frame_labels.append("Unknown")
                continue

            matched = search(emb)
            if not matched:
                frame_faces.append(face)
                frame_labels.append("Unknown")
                continue

            m   = matched[0]
            usn = m["usn"]
            label = f"{m['name']}  {m['similarity']*100:.0f}%"
            frame_faces.append(face)
            frame_labels.append(label)

            # Skip if already marked this session
            if usn in marked:
                continue

            # Auto-mark attendance
            mark_attendance(usn, m["name"], subject,
                                       m["similarity"], session_id)

            now_str = datetime.now().strftime("%H:%M:%S")
            with _session_lock:
                _active_session["marked"][usn] = {
                    "name": m["name"], "usn": usn,
                    "pct":  round(m["similarity"] * 100, 1),
                    "time": now_str,
                }
                _active_session["events"].append({
                    "type":  "marked",
                    "usn":   usn,
                    "name":  m["name"],
                    "pct":   round(m["similarity"] * 100, 1),
                    "time":  now_str,
                })

            # Check low attendance and send alert if needed
            _check_and_alert(usn, subject)

        # Store detections for the live video overlay
        with _session_lock:
            _active_session["detections"] = (frame_faces, frame_labels)
        if frame_faces:
            print(f"[detection] {len(frame_faces)} face(s) detected: {frame_labels}")

        time.sleep(0.5)   # scan every 0.5 seconds for responsive bounding boxes


def _check_and_alert(usn: str, subject: str):
    """After marking, check if attendance dropped below threshold and send email."""
    try:
        if check_low_attendance_alert(usn, subject):
            db      = get_db()
            student = db.students.find_one({"usn": usn})
            if student:
                summary   = get_student_summary(usn)
                threshold = get_subject_threshold(subject)
                for row in summary:
                    if row["subject"] == subject:
                        send_low_attendance_alert(student, subject,
                                                   float(row["pct"]), threshold)
                        break
    except Exception as e:
        print(f"[alert] Error checking low attendance for {usn}: {e}")


@app.route("/attendance/take")
@faculty_required
def take_attendance():
    subjects = get_subject_names()
    return render_template("take_attendance.html", subjects=subjects)


@app.route("/api/session/start", methods=["POST"])
@faculty_required
def session_start():
    data    = request.json or {}
    subject = data.get("subject", "").strip()
    if not subject:
        return jsonify({"ok": False, "error": "Subject required"}), 400

    # ── Camera source override (IP Webcam or local index) ─────────────────────
    camera_url = data.get("camera_url", "").strip()
    if camera_url:
        # User supplied an IP Webcam URL (e.g. http://192.168.1.5:8080/video)
        config.CAMERA_SOURCE = camera_url
    else:
        # Default back to local webcam index 0
        try:
            config.CAMERA_SOURCE = int(data.get("camera_index", 0))
        except (ValueError, TypeError):
            config.CAMERA_SOURCE = 0

    with _session_lock:
        if _active_session["running"]:
            return jsonify({"ok": False, "error": "A session is already running"}), 409
        session_id = f"{subject}_{date.today().isoformat()}_{int(time.time())}"
        _active_session.update({
            "running":    True,
            "subject":    subject,
            "session_id": session_id,
            "marked":     {},
            "events":     [{"type": "started", "subject": subject,
                             "time": datetime.now().strftime("%H:%M:%S")}],
        })

    t = threading.Thread(target=_recognition_loop, daemon=True)
    t.start()
    return jsonify({"ok": True, "session_id": session_id})


@app.route("/api/session/stop", methods=["POST"])
@faculty_required
def session_stop():
    with _session_lock:
        _active_session["running"] = False
        marked  = dict(_active_session["marked"])
        subject = _active_session["subject"]
        session_id = _active_session.get("session_id")
        _active_session["events"].append({
            "type":    "stopped",
            "total":   len(marked),
            "subject": subject,
            "time":    datetime.now().strftime("%H:%M:%S"),
        })
    # Release camera
    try:
        stream = get_stream(config.CAMERA_SOURCE)
        stream.stop()
    except Exception:
        pass

    # Auto-mark absent + fire email alerts (in background thread)
    def _post_session_tasks():
        try:
            db = get_db()
            subj_doc = db.subjects.find_one({"name": subject})
            if not subj_doc:
                return

            semester = subj_doc.get("semester")
            section  = subj_doc.get("section")
            from datetime import date as _date
            today = _date.today().isoformat()

            students = list(db.students.find({"semester": semester, "section": section}))
            for st in students:
                usn = st.get("usn")
                if not usn:
                    continue

                # ── 1. Mark absent if not recognized ─────────────────────────
                if usn not in marked:
                    mark_absent(usn, st.get("name"), subject, session_id)

                    # ── 2. Absence alert (consecutive absences) ───────────────
                    try:
                        consec = get_consecutive_absences(usn, subject,
                                                          config.ABSENCE_ALERT_THRESHOLD)
                        if consec >= config.ABSENCE_ALERT_THRESHOLD:
                            sent = send_absence_alert(st, subject, today, consec)
                            print(f"[email] Absence alert for {usn} → {'sent' if sent else 'skipped'}")
                    except Exception as ea:
                        print(f"[email] Absence alert error for {usn}: {ea}")

                # ── 3. Low-attendance warning (all students, present or not) ──
                try:
                    if check_low_attendance_alert(usn, subject):
                        from modules.attendance import get_student_summary
                        summary = get_student_summary(usn)
                        for row in summary:
                            if row["subject"] == subject:
                                threshold = get_subject_threshold(subject)
                                sent = send_low_attendance_alert(st, subject,
                                                                  float(row["pct"]), threshold)
                                print(f"[email] Low-att alert for {usn} → {'sent' if sent else 'skipped'}")
                                break
                except Exception as el:
                    print(f"[email] Low-att alert error for {usn}: {el}")

        except Exception as e:
            print(f"[session_stop] Post-session tasks error: {e}")

    import threading as _threading
    _threading.Thread(target=_post_session_tasks, daemon=True).start()

    return jsonify({"ok": True, "total_marked": len(marked)})


@app.route("/api/session/events")
@faculty_required
def session_events():
    """Server-Sent Events stream — pushes recognition events to the browser."""
    def generate():
        last_idx = 0
        while True:
            with _session_lock:
                events = _active_session["events"]
                new    = events[last_idx:]
                last_idx = len(events)
                running  = _active_session["running"]

            for evt in new:
                yield f"data: {json.dumps(evt)}\n\n"

            if not running and last_idx >= len(_active_session["events"]):
                yield "data: {\"type\": \"done\"}\n\n"
                break
            time.sleep(0.5)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/session/status")
@faculty_required
def session_status():
    with _session_lock:
        return jsonify({
            "running":  _active_session["running"],
            "subject":  _active_session["subject"],
            "marked":   len(_active_session["marked"]),
            "students": list(_active_session["marked"].values()),
        })


# Keep old /api/video_feed for live preview with bounding boxes
@app.route("/api/video_feed")
@faculty_required
def video_feed():
    stream = get_stream(config.CAMERA_SOURCE)

    def annotate(frame):
        if _active_session["running"]:
            # Use cached detections from _recognition_loop (no duplicate detection)
            with _session_lock:
                detections = _active_session.get("detections")
            if detections and isinstance(detections, tuple) and len(detections) == 2:
                faces, labels = detections
                if faces:
                    return draw_detections(frame, faces, labels)
            return frame

        # No session running — do live detection for preview
        faces  = detect_faces(frame)
        labels = []
        for face in faces:
            crop = frame[
                max(0, face["bbox"][1]):face["bbox"][3],
                max(0, face["bbox"][0]):face["bbox"][2]
            ]
            emb = get_embedding(crop)
            if emb is None:
                emb = np.zeros(512, dtype=np.float32)
            matched = search(emb)
            if matched:
                m = matched[0]
                labels.append(f"{m['name']}  {m['similarity']*100:.0f}%")
            else:
                labels.append("Unknown")
        return draw_detections(frame, faces, labels)

    return Response(
        stream.mjpeg_generator(annotate_fn=annotate),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ATTENDANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/attendance/report")
@faculty_required
def attendance_report():
    subjects    = get_subject_names()
    sel_subject = request.args.get("subject", "")
    sel_date    = request.args.get("date", date.today().isoformat())
    records     = get_daily_report(sel_date, sel_subject or None)
    return render_template("attendance_report.html",
                           records=records, subjects=subjects,
                           sel_subject=sel_subject, sel_date=sel_date)


@app.route("/attendance/export")
@faculty_required
def attendance_export():
    subject  = request.args.get("subject", "All")
    sel_date = request.args.get("date", "")
    records  = get_daily_report(sel_date or None, subject if subject != "All" else None)
    path     = export_attendance(records, subject, sel_date)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


# ══════════════════════════════════════════════════════════════════════════════
#  ELIGIBILITY
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/eligibility")
@faculty_required
def eligibility():
    subjects    = get_subject_names()
    sel_subject = request.args.get("subject", subjects[0] if subjects else "")
    report      = get_eligible_students(sel_subject) if sel_subject else {}
    return render_template("eligibility.html",
                           subjects=subjects, report=report,
                           sel_subject=sel_subject)


@app.route("/eligibility/export")
@faculty_required
def eligibility_export():
    subject = request.args.get("subject", "")
    report  = get_eligible_students(subject)
    path    = export_eligibility(report)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


@app.route("/eligibility/send_alerts", methods=["POST"])
@faculty_required
def send_alerts():
    subject = request.form.get("subject", "")
    report  = get_eligible_students(subject)
    db      = get_db()
    sent    = 0
    threshold = get_subject_threshold(subject)

    for row in report.get("ineligible", []) + report.get("at_risk", []):
        student = db.students.find_one({"usn": row["usn"]})
        if student:
            ok = send_low_attendance_alert(student, subject,
                                            float(row["pct"]), threshold)
            if ok:
                sent += 1
    flash(f"📧 Sent {sent} low-attendance email alerts.", "success")
    return redirect(url_for("eligibility", subject=subject))


# ══════════════════════════════════════════════════════════════════════════════
#  MARKS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/marks", methods=["GET", "POST"])
@faculty_required
def marks():
    subjects    = get_subject_names()
    sel_subject = request.args.get("subject", subjects[0] if subjects else "")

    if request.method == "POST":
        usn     = request.form.get("usn", "").upper()
        name    = request.form.get("name", "")
        subject = request.form.get("subject", "")
        sem     = int(request.form.get("semester", 1))
        ia1     = request.form.get("ia1") or None
        ia2     = request.form.get("ia2") or None
        ia3     = request.form.get("ia3") or None
        upsert_marks(usn, name, subject, sem,
                     float(ia1) if ia1 else None,
                     float(ia2) if ia2 else None,
                     float(ia3) if ia3 else None)
        flash("Marks saved.", "success")
        return redirect(url_for("marks", subject=subject))

    records = get_marks(subject=sel_subject) if sel_subject else []
    return render_template("marks.html",
                           subjects=subjects, records=records,
                           sel_subject=sel_subject)


@app.route("/marks/export")
@faculty_required
def marks_export():
    subject = request.args.get("subject", "All")
    records = get_marks(subject=subject if subject != "All" else None)
    path    = export_marks(records, subject)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


# ══════════════════════════════════════════════════════════════════════════════
#  SUBJECTS  (Admin only — includes threshold setting)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/subjects", methods=["GET", "POST"])
@faculty_required
def subjects():
    if request.method == "POST":
        action = request.form.get("action", "add")
        if action == "add":
            threshold = request.form.get("threshold")
            ok = add_subject(
                name      = request.form.get("name", ""),
                code      = request.form.get("code", ""),
                semester  = int(request.form.get("semester", 1)),
                section   = request.form.get("section", "A"),
                threshold = int(threshold) if threshold else None,
            )
            flash("Subject added." if ok else "Subject code already exists.",
                  "success" if ok else "warning")
        elif action == "delete":
            delete_subject(request.form.get("code", ""),
                           int(request.form.get("semester", 1)))
            flash("Subject deleted.", "info")
        elif action == "update_threshold":
            update_subject_threshold(
                code      = request.form.get("code", ""),
                semester  = int(request.form.get("semester", 1)),
                threshold = int(request.form.get("threshold", 75)),
            )
            flash("Attendance threshold updated.", "success")
        return redirect(url_for("subjects"))

    all_subjects = get_subjects()
    return render_template("subjects.html", subjects=all_subjects)


# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN — User Management
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/admin/users", methods=["GET", "POST"])
@admin_required
def admin_users():
    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "create":
            role     = request.form.get("role", "faculty")
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()
            name     = request.form.get("name", "").strip()
            result   = create_user(username, password, role, name)
            flash("User created." if result else "Username already exists.",
                  "success" if result else "warning")

        elif action == "deactivate":
            set_active(request.form.get("username", ""), False)
            flash("User deactivated.", "info")

        elif action == "activate":
            set_active(request.form.get("username", ""), True)
            flash("User activated.", "success")

        elif action == "reset_password":
            username = request.form.get("username", "")
            new_pwd  = request.form.get("new_password", "").strip()
            if new_pwd:
                reset_password(username, new_pwd)
                flash(f"Password reset for {username}.", "success")

        return redirect(url_for("admin_users"))

    faculty_users = get_all_users("faculty")
    admin_users   = get_all_users("admin")
    student_users = get_all_users("student")
    return render_template("admin_users.html",
                           faculty_users=faculty_users,
                           admin_users=admin_users,
                           student_users=student_users)


# ══════════════════════════════════════════════════════════════════════════════
#  EMAIL LOGS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/email-logs")
@faculty_required
def email_logs():
    logs = get_email_logs(limit=100)
    return render_template("email_logs.html", logs=logs)


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SGBIT Attendance Management System")
    print("  CSE Department, Belagavi")
    print("=" * 60)
    seed_admin()
    print(f"  Building FAISS index from MongoDB...")
    build_index()
    print(f"  Registered students in index: {total_registered()}")
    print(f"  Starting server at http://localhost:{config.PORT}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG,
            threaded=True, use_reloader=False)

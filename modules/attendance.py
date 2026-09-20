"""
modules/attendance.py
Attendance CRUD operations against MongoDB.
"""
from datetime import datetime, date
from pymongo import ASCENDING, DESCENDING
from modules.db import get_db
import config


# â”€â”€â”€ Mark Attendance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def mark_attendance(usn: str, name: str, subject: str,
                    confidence: float = 1.0,
                    session_id: str = None,
                    status: str = "Present") -> bool:
    """
    Insert or update attendance for usn+subject+date.
    Prevents duplicate marks within the same session.
    Returns True if newly marked, False if already recorded.
    """
    db  = get_db()
    today = date.today().isoformat()

    existing = db.attendance.find_one({
        "usn":     usn,
        "subject": subject,
        "date":    today,
        "status":  "Present",
    })
    if existing:
        return False    # already marked present today

    db.attendance.insert_one({
        "usn":           usn,
        "name":          name,
        "subject":       subject,
        "date":          today,
        "timestamp":     datetime.utcnow(),
        "status":        status,
        "confidence":    round(confidence, 4),
        "session_id":    session_id or f"{subject}_{today}",
        "marked_by":     "face_recognition",
    })
    return True


def mark_absent(usn: str, name: str, subject: str, session_id: str = None):
    """Mark a student absent for a subject on today's date (if not already marked)."""
    db    = get_db()
    today = date.today().isoformat()

    exists = db.attendance.find_one({"usn": usn, "subject": subject, "date": today})
    if exists:
        return  # already has a record (present or absent)

    db.attendance.insert_one({
        "usn":        usn,
        "name":       name,
        "subject":    subject,
        "date":       today,
        "timestamp":  datetime.utcnow(),
        "status":     "Absent",
        "confidence": 0.0,
        "session_id": session_id or f"{subject}_{today}",
        "marked_by":  "auto",
    })


# â”€â”€â”€ Queries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_daily_report(target_date: str = None, subject: str = None) -> list[dict]:
    """Fetch attendance records for a date (ISO string) and optional subject."""
    db    = get_db()
    query = {}
    if target_date:
        query["date"] = target_date
    if subject:
        query["subject"] = subject
    return list(db.attendance.find(query, {"_id": 0})
                              .sort("timestamp", DESCENDING))


def get_student_summary(usn: str) -> list[dict]:
    """
    Return attendance % per subject for a student.
    [{"subject": ..., "present": n, "total": n, "pct": float}]
    """
    db = get_db()
    pipeline = [
        {"$match": {"usn": usn}},
        {"$group": {
            "_id":    "$subject",
            "total":  {"$sum": 1},
            "present": {"$sum": {"$cond": [{"$eq": ["$status", "Present"]}, 1, 0]}},
        }},
        {"$project": {
            "subject": "$_id", "_id": 0,
            "total": 1, "present": 1,
            "pct": {"$multiply": [
                {"$divide": ["$present", "$total"]}, 100
            ]},
        }},
        {"$sort": {"subject": ASCENDING}},
    ]
    return list(db.attendance.aggregate(pipeline))


def get_class_summary(subject: str) -> list[dict]:
    """
    Return attendance % for every student in a subject.
    [{"usn": ..., "name": ..., "present": n, "total": n, "pct": float}]
    """
    db = get_db()
    pipeline = [
        {"$match": {"subject": subject}},
        {"$group": {
            "_id":    "$usn",
            "name":  {"$first": "$name"},
            "total": {"$sum": 1},
            "present": {"$sum": {"$cond": [{"$eq": ["$status", "Present"]}, 1, 0]}},
        }},
        {"$project": {
            "usn": "$_id", "_id": 0,
            "name": 1, "total": 1, "present": 1,
            "pct": {"$multiply": [
                {"$divide": ["$present", "$total"]}, 100
            ]},
        }},
        {"$sort": {"usn": ASCENDING}},
    ]
    return list(db.attendance.aggregate(pipeline))


def get_consecutive_absences(usn: str, subject: str, n: int = 2) -> int:
    """Return number of consecutive absences for usn in subject (most recent first)."""
    db = get_db()
    records = list(
        db.attendance.find(
            {"usn": usn, "subject": subject},
            {"status": 1, "_id": 0}
        ).sort("date", DESCENDING).limit(n)
    )
    count = 0
    for r in records:
        if r["status"] == "Absent":
            count += 1
        else:
            break
    return count


def get_today_stats() -> dict:
    """Dashboard stats: total present / absent today across all subjects."""
    db    = get_db()
    today = date.today().isoformat()
    total   = db.attendance.count_documents({"date": today})
    present = db.attendance.count_documents({"date": today, "status": "Present"})
    pct     = round((present / total * 100) if total > 0 else 0, 1)
    return {"total": total, "present": present,
            "absent": total - present, "pct": pct}


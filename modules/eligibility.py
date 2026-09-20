"""
modules/eligibility.py
Filters students by attendance percentage for internal exam eligibility.
Threshold is now per-subject (stored in subjects collection),
falling back to global config.ATTENDANCE_THRESHOLD.
"""
from pymongo import ASCENDING
from modules.db import get_db
from modules.attendance import get_class_summary
from modules.subjects import get_subject_threshold
import config


def get_eligible_students(subject: str, threshold: int = None) -> dict:
    """
    Classify all students in a subject as eligible / at-risk / ineligible.

    Uses per-subject threshold if set (via subjects collection),
    otherwise falls back to the passed threshold or global config.

    Returns:
    {
        "eligible":   [{usn, name, pct, present, total}],
        "at_risk":    [...],   # within 10 pct of threshold
        "ineligible": [...],   # below threshold - 10
        "threshold":  int,
        "subject":    subject,
    }
    """
    if threshold is None:
        threshold = get_subject_threshold(subject)

    at_risk_floor = max(0, threshold - 10)

    summary    = get_class_summary(subject)
    eligible   = []
    at_risk    = []
    ineligible = []

    for row in summary:
        pct   = round(float(row.get("pct", 0)), 2)
        entry = {**row, "pct": pct}
        if pct >= threshold:
            eligible.append(entry)
        elif pct >= at_risk_floor:
            at_risk.append(entry)
        else:
            ineligible.append(entry)

    eligible.sort(key=lambda x: x["pct"], reverse=True)
    at_risk.sort(key=lambda x: x["pct"], reverse=True)
    ineligible.sort(key=lambda x: x["pct"])

    return {
        "eligible":    eligible,
        "at_risk":     at_risk,
        "ineligible":  ineligible,
        "threshold":   threshold,
        "at_risk_floor": at_risk_floor,
        "subject":     subject,
    }


def generate_full_report(threshold: int = None) -> list[dict]:
    """Generate eligibility report across ALL subjects."""
    db       = get_db()
    subjects = db.subjects.distinct("name")
    return [get_eligible_students(subj, threshold) for subj in subjects]


def get_student_eligibility(usn: str, threshold: int = None) -> list[dict]:
    """
    Return eligibility status per subject for a single student.
    Uses per-subject threshold for each subject.
    """
    from pymongo import ASCENDING
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
            "pct": {"$multiply": [{"$divide": ["$present", "$total"]}, 100]},
        }},
    ]
    rows   = list(db.attendance.aggregate(pipeline))
    result = []
    for row in rows:
        pct  = round(float(row["pct"]), 2)
        subj_threshold = threshold if threshold else get_subject_threshold(row["subject"])
        at_risk_floor  = max(0, subj_threshold - 10)
        if pct >= subj_threshold:
            status = "Eligible"
        elif pct >= at_risk_floor:
            status = "At Risk"
        else:
            status = "Ineligible"
        result.append({**row, "pct": pct, "status": status, "threshold": subj_threshold})
    return result


def check_low_attendance_alert(usn: str, subject: str) -> bool:
    """
    Returns True if this student's attendance in the subject has dropped
    below the subject threshold (triggers an email alert).
    """
    from modules.attendance import get_student_summary
    summary = get_student_summary(usn)
    for row in summary:
        if row["subject"] == subject:
            threshold = get_subject_threshold(subject)
            return float(row["pct"]) < threshold
    return False

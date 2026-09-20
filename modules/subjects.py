"""
modules/subjects.py
Subject management - add, list, and delete subjects per semester/section.
Each subject stores a custom attendance threshold (default 75%).
"""
from datetime import datetime
from pymongo import ASCENDING
from modules.db import get_db
import config


def add_subject(name: str, code: str, semester: int,
                section: str = "A", threshold: int = None) -> bool:
    """
    Add a new subject. Returns False if code already exists for that semester.
    threshold: attendance % required for eligibility (defaults to config value).
    """
    db = get_db()
    exists = db.subjects.find_one({"code": code, "semester": semester})
    if exists:
        return False
    db.subjects.insert_one({
        "name":       name.strip(),
        "code":       code.strip().upper(),
        "semester":   semester,
        "section":    section.upper(),
        "threshold":  threshold if threshold is not None else config.ATTENDANCE_THRESHOLD,
        "created_at": datetime.utcnow(),
    })
    return True


def get_subjects(semester: int = None, section: str = None) -> list[dict]:
    """List subjects, optionally filtered by semester and/or section."""
    db    = get_db()
    query = {}
    if semester: query["semester"] = semester
    if section:  query["section"]  = section.upper()
    rows = list(db.subjects.find(query, {"_id": 0})
                            .sort([("semester", ASCENDING), ("name", ASCENDING)]))
    return rows


def get_subject(code: str, semester: int) -> dict | None:
    """Get a single subject document by code + semester."""
    db = get_db()
    return db.subjects.find_one({"code": code, "semester": semester}, {"_id": 0})


def update_subject_threshold(code: str, semester: int, threshold: int):
    """Update the attendance threshold for a specific subject."""
    db = get_db()
    db.subjects.update_one(
        {"code": code, "semester": semester},
        {"$set": {"threshold": threshold}}
    )


def delete_subject(code: str, semester: int):
    db = get_db()
    db.subjects.delete_one({"code": code, "semester": semester})


def get_subject_names(semester: int = None) -> list[str]:
    """Return just the names of subjects (for dropdowns)."""
    return [s["name"] for s in get_subjects(semester=semester)]


def get_subject_threshold(subject_name: str) -> int:
    """
    Return the attendance threshold for a subject by name.
    Falls back to global config threshold if not found.
    """
    db  = get_db()
    doc = db.subjects.find_one({"name": subject_name}, {"threshold": 1})
    if doc and "threshold" in doc:
        return int(doc["threshold"])
    return config.ATTENDANCE_THRESHOLD

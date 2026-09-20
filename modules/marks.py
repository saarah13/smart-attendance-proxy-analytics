"""
modules/marks.py
Internal Assessment (IA) marks CRUD for MongoDB.
Supports IA1, IA2, IA3 per student per subject.
"""
from datetime import datetime
from pymongo import ASCENDING
from modules.db import get_db
import config

# â”€â”€â”€ CRUD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def upsert_marks(usn: str, name: str, subject: str, semester: int,
                 ia1: float = None, ia2: float = None, ia3: float = None):
    """Insert or update marks for usn+subject. Only updates fields provided."""
    db  = get_db()
    update_fields = {"usn": usn, "name": name,
                     "subject": subject, "semester": semester,
                     "updated_at": datetime.utcnow()}
    if ia1 is not None: update_fields["ia1"] = float(ia1)
    if ia2 is not None: update_fields["ia2"] = float(ia2)
    if ia3 is not None: update_fields["ia3"] = float(ia3)

    db.marks.update_one(
        {"usn": usn, "subject": subject},
        {"$set": update_fields},
        upsert=True,
    )


def get_marks(usn: str = None, subject: str = None,
              semester: int = None) -> list[dict]:
    """
    Fetch marks records. Filter by any combination of usn / subject / semester.
    Each record includes computed total and average.
    """
    db    = get_db()
    query = {}
    if usn:      query["usn"]      = usn
    if subject:  query["subject"]  = subject
    if semester: query["semester"] = semester

    rows = list(db.marks.find(query, {"_id": 0}).sort("usn", ASCENDING))

    for r in rows:
        vals  = [r.get("ia1", 0), r.get("ia2", 0), r.get("ia3", 0)]
        filled = [v for v in vals if v is not None]
        r["total"] = round(sum(filled), 2)
        r["avg"]   = round(sum(filled) / len(filled), 2) if filled else 0
    return rows


def get_student_marks(usn: str) -> list[dict]:
    """All IA marks for a single student across all subjects."""
    return get_marks(usn=usn)


def delete_marks(usn: str, subject: str):
    db = get_db()
    db.marks.delete_one({"usn": usn, "subject": subject})


def bulk_import_from_list(records: list[dict]):
    """
    Bulk upsert marks from a list of dicts.
    Each dict must have: usn, name, subject, semester, ia1, ia2, ia3.
    """
    for r in records:
        upsert_marks(
            usn=r.get("usn", ""),
            name=r.get("name", ""),
            subject=r.get("subject", ""),
            semester=int(r.get("semester", 1)),
            ia1=r.get("ia1"),
            ia2=r.get("ia2"),
            ia3=r.get("ia3"),
        )


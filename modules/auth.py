"""
modules/auth.py
User authentication with bcrypt password hashing.
Manages Admin / Faculty / Student accounts in MongoDB users collection.
"""
import bcrypt
from datetime import datetime
from modules.db import get_db
import config


# ── Password helpers ───────────────────────────────────────────────────────────

def hash_password(plain: str) -> bytes:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt())


def verify_password(plain: str, hashed) -> bool:
    if isinstance(hashed, str):
        hashed = hashed.encode()
    return bcrypt.checkpw(plain.encode(), hashed)


# ── User CRUD ─────────────────────────────────────────────────────────────────

def create_user(username: str, password: str, role: str,
                name: str, usn: str = None) -> dict | None:
    """
    Create a new user. Returns user dict on success, None if username taken.
    role must be 'admin' | 'faculty' | 'student'
    """
    db = get_db()
    if db.users.find_one({"username": username.strip().lower()}):
        return None
    doc = {
        "username":      username.strip().lower(),
        "password_hash": hash_password(password).decode(),
        "role":          role,
        "name":          name.strip(),
        "usn":           usn.upper() if usn else None,
        "is_active":     True,
        "created_at":    datetime.utcnow(),
        "last_login":    None,
    }
    db.users.insert_one(doc)
    doc["_id"] = str(doc["_id"])
    return doc


def authenticate(username: str, password: str) -> dict | None:
    """
    Verify credentials. Returns sanitised user dict or None on failure.
    Also updates last_login timestamp.
    """
    db   = get_db()
    user = db.users.find_one({"username": username.strip().lower(), "is_active": True})
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    db.users.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    return {
        "username":   user["username"],
        "role":       user["role"],
        "name":       user["name"],
        "usn":        user.get("usn"),
        "is_active":  user["is_active"],
        "last_login": user.get("last_login"),
    }


def get_all_users(role: str = None) -> list[dict]:
    """Return all users, optionally filtered by role."""
    db    = get_db()
    query = {"role": role} if role else {}
    users = list(db.users.find(query, {"password_hash": 0}))
    for u in users:
        u["_id"] = str(u["_id"])
    return users


def set_active(username: str, active: bool):
    db = get_db()
    db.users.update_one({"username": username}, {"$set": {"is_active": active}})


def reset_password(username: str, new_password: str):
    db = get_db()
    db.users.update_one(
        {"username": username},
        {"$set": {"password_hash": hash_password(new_password).decode()}}
    )


# ── Seed admin ────────────────────────────────────────────────────────────────

def seed_admin():
    """
    Ensure at least one admin exists in the DB.
    Uses ADMIN_DEFAULT_USERNAME / ADMIN_DEFAULT_PASSWORD from config.
    Safe to call on every startup.
    """
    db = get_db()
    if db.users.find_one({"role": "admin"}):
        return   # admin already exists
    create_user(
        username=config.ADMIN_DEFAULT_USERNAME,
        password=config.ADMIN_DEFAULT_PASSWORD,
        role="admin",
        name="Administrator",
    )
    print(f"[auth] Default admin created: {config.ADMIN_DEFAULT_USERNAME}")

"""Authentication for Dr Stone: users + server-side sessions in SQLite.

Dependency-light by design — password hashing uses stdlib PBKDF2-HMAC-SHA256 and
sessions are server-side (a random opaque cookie token mapped to a user row), so
no itsdangerous/passlib/bcrypt is required. The auth DB lives alongside the
processed data tables (git-ignored, may contain credentials).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timezone

from drstone import config as C

DB_PATH = os.path.join(C.DATA_DIR, "drstone_auth.db")
SESSION_COOKIE = "drstone_session"
SESSION_MAX_AGE = 60 * 60 * 12          # 12 hours
_PBKDF2_ITER = 240_000

# Seed superuser (provisioned on startup if absent). Overridable via env so the
# credential need not live in source on shared/public deployments; the provided
# defaults keep a fresh clone working out of the box. ROTATE in production.
SUPERUSER_EMAIL = os.environ.get("DRSTONE_SUPERUSER_EMAIL", "rodriguezr32@uthscsa.edu")
SUPERUSER_PASSWORD = os.environ.get("DRSTONE_SUPERUSER_PASSWORD", "UroChair115!")
SUPERUSER_NAME = os.environ.get("DRSTONE_SUPERUSER_NAME", "Dr. Rodriguez")


# --------------------------------------------------------------------------
# Connection / schema
# --------------------------------------------------------------------------
def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS users(
                 id            INTEGER PRIMARY KEY AUTOINCREMENT,
                 email         TEXT UNIQUE NOT NULL,
                 full_name     TEXT,
                 password_hash TEXT NOT NULL,
                 role          TEXT NOT NULL DEFAULT 'user',
                 created_at    TEXT NOT NULL
               )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS sessions(
                 token      TEXT PRIMARY KEY,
                 user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                 created_at TEXT NOT NULL
               )"""
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256)
# --------------------------------------------------------------------------
def hash_password(password: str, *, iterations: int = _PBKDF2_ITER) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, dk_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                 bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, AttributeError):
        return False


# --------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------
def get_user_by_email(email: str):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email = ?",
                        (email.strip().lower(),)).fetchone()
    return dict(row) if row else None


def create_user(email: str, password: str, *, full_name: str = "",
                role: str = "user") -> dict:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("A valid email address is required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if get_user_by_email(email):
        raise ValueError("An account with that email already exists.")
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO users(email, full_name, password_hash, role, created_at) "
            "VALUES (?,?,?,?,?)",
            (email, full_name.strip(), hash_password(password), role, _now()),
        )
        uid = cur.lastrowid
    return {"id": uid, "email": email, "full_name": full_name, "role": role}


def authenticate(email: str, password: str):
    user = get_user_by_email(email)
    if user and verify_password(password, user["password_hash"]):
        user.pop("password_hash", None)
        return user
    return None


def seed_superuser() -> None:
    """Provision the superuser account on startup if it does not exist."""
    init_db()
    existing = get_user_by_email(SUPERUSER_EMAIL)
    if existing is None:
        create_user(SUPERUSER_EMAIL, SUPERUSER_PASSWORD,
                    full_name=SUPERUSER_NAME, role="superuser")
    elif existing["role"] != "superuser":
        with _conn() as c:
            c.execute("UPDATE users SET role='superuser' WHERE id=?",
                      (existing["id"],))


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------
def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _conn() as c:
        c.execute("INSERT INTO sessions(token, user_id, created_at) VALUES (?,?,?)",
                  (token, user_id, _now()))
    return token


def get_session_user(token: str):
    if not token:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT u.id, u.email, u.full_name, u.role, s.created_at AS s_created "
            "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?",
            (token,),
        ).fetchone()
    if not row:
        return None
    # Expire stale sessions.
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(row["s_created"])).total_seconds()
        if age > SESSION_MAX_AGE:
            delete_session(token)
            return None
    except (ValueError, TypeError):
        pass
    return {"id": row["id"], "email": row["email"],
            "full_name": row["full_name"], "role": row["role"]}


def delete_session(token: str) -> None:
    if not token:
        return
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (token,))


def current_user(request):
    """Return the logged-in user dict for a Starlette/FastAPI request, or None."""
    return get_session_user(request.cookies.get(SESSION_COOKIE))

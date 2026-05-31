# database.py — Shared MongoDB client and index initialisation
import os
import logging
from pymongo import MongoClient, ASCENDING, DESCENDING
from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)

_client: MongoClient | None = None


def get_db():
    """Lazily initialise and return the default MongoDB database."""
    global _client
    if _client is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/embedmindai")
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        logger.info("MongoDB client initialised")
    return _client.get_default_database()


def close_db():
    """Close the MongoDB connection (call on shutdown)."""
    global _client
    if _client:
        _client.close()
        _client = None


def init_indexes():
    """Create all collection indexes. Safe to call multiple times (idempotent)."""
    db = get_db()

    # ── sessions ──────────────────────────────────────────────────────
    db.sessions.create_index("expires_at", expireAfterSeconds=0)

    # ── users ─────────────────────────────────────────────────────────
    db.users.create_index("email", unique=True)

    # ── documents ─────────────────────────────────────────────────────
    db.documents.create_index([("user_email", ASCENDING), ("uploaded_at", DESCENDING)])

    # ── chat_sessions ─────────────────────────────────────────────────
    db.chat_sessions.create_index(
        [("user_email", ASCENDING), ("updated_at", DESCENDING)]
    )
    db.chat_sessions.create_index("session_id", unique=True)

    # ── messages ──────────────────────────────────────────────────────
    db.messages.create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)])

    # ── query_logs ────────────────────────────────────────────────────
    db.query_logs.create_index([("user_email", ASCENDING), ("created_at", DESCENDING)])

    logger.info("MongoDB indexes initialised")


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _get_sessions_col():
    return get_db().sessions


def get_current_user(request: Request) -> dict | None:
    """
    Read the session cookie and return the stored user dict, or None if
    the request is not authenticated / the session has expired.
    """
    token = request.cookies.get("session_token")
    if not token:
        return None
    doc = _get_sessions_col().find_one({"token": token})
    return doc["user"] if doc else None


def require_current_user(request: Request) -> dict:
    """
    Like get_current_user() but raises HTTP 401 when the user is not
    authenticated. Use as a FastAPI dependency or call directly inside a route.
    """
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

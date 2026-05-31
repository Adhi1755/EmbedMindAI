# chat_routes.py — Chat sessions and messages CRUD endpoints
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from database import get_db, require_current_user

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Pydantic request models ───────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    session_id: str
    title: str
    pdf_name: str | None = None
    document_id: str | None = None


class UpdateSessionTitleRequest(BaseModel):
    title: str


class AddMessageRequest(BaseModel):
    session_id: str
    sender: str        # "user" or "ai"
    text: str


# ── Helper ────────────────────────────────────────────────────────────────────

def _fmt(doc: dict) -> dict:
    """Convert ObjectId and datetime fields to JSON-serialisable types."""
    doc.pop("_id", None)
    for key, val in doc.items():
        if isinstance(val, datetime):
            doc[key] = val.isoformat()
    return doc


# ── Session endpoints ─────────────────────────────────────────────────────────

@router.post("/sessions", status_code=201)
async def create_session(body: CreateSessionRequest, request: Request):
    """
    Create or upsert a chat session.
    The frontend calls this right after creating a session in the local store.
    """
    user = require_current_user(request)
    db = get_db()
    now = datetime.now(timezone.utc)

    db.chat_sessions.update_one(
        {"session_id": body.session_id, "user_email": user["email"]},
        {
            "$setOnInsert": {
                "session_id":  body.session_id,
                "user_email":  user["email"],
                "title":       body.title,
                "pdf_name":    body.pdf_name,
                "document_id": body.document_id,
                "created_at":  now,
                "message_count": 0,
            },
            "$set": {"updated_at": now},
        },
        upsert=True,
    )
    return {"ok": True, "session_id": body.session_id}


@router.get("/sessions")
async def list_sessions(request: Request):
    """Return all sessions for the current user, newest first."""
    user = require_current_user(request)
    db = get_db()
    sessions = list(
        db.chat_sessions.find(
            {"user_email": user["email"]},
            {"_id": 0},
        ).sort("updated_at", -1).limit(100)
    )
    return [_fmt(s) for s in sessions]


@router.patch("/sessions/{session_id}")
async def update_session_title(
    session_id: str, body: UpdateSessionTitleRequest, request: Request
):
    """Rename a chat session."""
    user = require_current_user(request)
    db = get_db()
    result = db.chat_sessions.update_one(
        {"session_id": session_id, "user_email": user["email"]},
        {"$set": {"title": body.title, "updated_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Delete a session and all its messages."""
    user = require_current_user(request)
    db = get_db()
    db.chat_sessions.delete_one({"session_id": session_id, "user_email": user["email"]})
    db.messages.delete_many({"session_id": session_id, "user_email": user["email"]})
    return {"ok": True}


# ── Message endpoints ─────────────────────────────────────────────────────────

@router.post("/messages", status_code=201)
async def add_message(body: AddMessageRequest, request: Request):
    """Save a single message and bump the session's updated_at + message_count."""
    user = require_current_user(request)

    if body.sender not in ("user", "ai"):
        raise HTTPException(status_code=422, detail="sender must be 'user' or 'ai'")

    db = get_db()
    now = datetime.now(timezone.utc)

    db.messages.insert_one({
        "session_id": body.session_id,
        "user_email": user["email"],
        "sender":     body.sender,
        "text":       body.text,
        "timestamp":  now,
    })

    # Keep session metadata in sync
    db.chat_sessions.update_one(
        {"session_id": body.session_id, "user_email": user["email"]},
        {
            "$inc": {"message_count": 1},
            "$set": {"updated_at": now},
        },
    )
    return {"ok": True}


@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str, request: Request):
    """Return all messages for a session, in chronological order."""
    user = require_current_user(request)
    db = get_db()
    msgs = list(
        db.messages.find(
            {"session_id": session_id, "user_email": user["email"]},
            {"_id": 0, "session_id": 0, "user_email": 0},
        ).sort("timestamp", 1)
    )
    return [_fmt(m) for m in msgs]

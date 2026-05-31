# auth.py — Persistent sessions via MongoDB + configurable redirect URIs
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
import os, secrets
from dotenv import load_dotenv
import httpx
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
from database import get_db

router = APIRouter()
load_dotenv()

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# Configurable via env so it works in Docker, staging, and production
BACKEND_URL  = os.getenv("BACKEND_URL",  "http://localhost:8000")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
REDIRECT_URI = f"{BACKEND_URL}/auth/callback"

# ── MongoDB session store ─────────────────────────────────────────────────────

def _get_sessions_collection():
    """Return the sessions collection from the shared DB client."""
    return get_db().sessions

SESSION_TTL_DAYS = 7


# ── Password hashing (stdlib only – no bcrypt dependency) ────────────────────

def _hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 260_000)
    return salt.hex() + ':' + key.hex()

def _verify_password(password: str, stored: str) -> bool:
    """Verify a password against its stored hash."""
    try:
        salt_hex, key_hex = stored.split(':')
        salt = bytes.fromhex(salt_hex)
        key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 260_000)
        return hmac.compare_digest(key, new_key)
    except Exception:
        return False


# ── Pydantic models ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str


def _store_session(token: str, user_data: dict) -> None:
    sessions = _get_sessions_collection()
    sessions.replace_one(
        {"token": token},
        {
            "token": token,
            "user": user_data,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS),
        },
        upsert=True,
    )


def _get_session(token: str) -> dict | None:
    sessions = _get_sessions_collection()
    doc = sessions.find_one({"token": token})
    return doc["user"] if doc else None


def _delete_session(token: str) -> None:
    sessions = _get_sessions_collection()
    sessions.delete_one({"token": token})


# ── Helper: create session and set cookie ────────────────────────────────────

def _create_session_response(user_data: dict, redirect_url: str | None = None) -> JSONResponse:
    session_token = secrets.token_hex(32)
    _store_session(session_token, user_data)
    if redirect_url:
        response = RedirectResponse(url=redirect_url)
    else:
        response = JSONResponse(content={"ok": True, "user": user_data})
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_DAYS * 86400,
    )
    return response


# ── Auth routes ───────────────────────────────────────────────────────────────

@router.get("/auth/login")
def login():
    google_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope=openid%20email%20profile"
    )
    return RedirectResponse(google_url)


@router.get("/auth/callback")
async def auth_callback(request: Request, code: str):
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient() as client:
        token_response = await client.post(token_url, data=data)
        token_json = token_response.json()
        access_token = token_json.get("access_token")

        # Fetch user profile from Google
        user_info_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        user_response = await client.get(user_info_url, headers=headers)
        user_data = user_response.json()

    now = datetime.now(timezone.utc)
    user_email = user_data["email"]

    # ── Upsert full user profile in users collection ───────────────────────
    db = get_db()
    db.users.update_one(
        {"email": user_email},
        {
            "$set": {
                "name":          user_data["name"],
                "picture":       user_data.get("picture"),
                "provider":      "google",
                "last_login_at": now,
            },
            "$setOnInsert": {
                "email":      user_email,
                "created_at": now,
                "login_count": 0,
                "settings":   {"theme": "dark", "language": "en"},
            },
            "$inc": {"login_count": 1},
        },
        upsert=True,
    )

    user_payload = {
        "name":    user_data["name"],
        "email":   user_email,
        "picture": user_data.get("picture"),
    }
    return _create_session_response(user_payload, redirect_url=f"{FRONTEND_URL}/chat")


@router.get("/auth/me")
def get_logged_in_user(request: Request):
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = _get_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or not found")
    return user


@router.get("/auth/logout")
def logout(request: Request):
    token = request.cookies.get("session_token")
    if token:
        _delete_session(token)
    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie("session_token")
    return response


# ── Email / Password Auth ─────────────────────────────────────────────────────

@router.post("/auth/register")
async def register(body: RegisterRequest):
    """Create a new user with email + password."""
    db = get_db()
    users = db.users

    # Check if user already exists
    if users.find_one({"email": body.email}):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    now = datetime.now(timezone.utc)
    hashed = _hash_password(body.password)
    users.insert_one({
        "name":          body.name,
        "email":         body.email,
        "password_hash": hashed,
        "provider":      "email",
        "picture":       None,
        "created_at":    now,
        "last_login_at": now,
        "login_count":   1,
        "settings":      {"theme": "dark", "language": "en"},
    })

    user_data = {"name": body.name, "email": body.email, "picture": None}
    return _create_session_response(user_data)


@router.post("/auth/email-login")
async def email_login(body: LoginRequest):
    """Sign in with email + password."""
    db = get_db()
    users = db.users

    user_doc = users.find_one({"email": body.email})
    if not user_doc or not _verify_password(body.password, user_doc.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # Update login metadata
    users.update_one(
        {"email": body.email},
        {
            "$set": {"last_login_at": datetime.now(timezone.utc)},
            "$inc": {"login_count": 1},
        },
    )

    user_data = {
        "name":    user_doc["name"],
        "email":   user_doc["email"],
        "picture": user_doc.get("picture"),
    }
    return _create_session_response(user_data)


@router.get("/auth/profile")
def get_profile(request: Request):
    """Return the full user profile document (without password_hash)."""
    from database import require_current_user
    user = require_current_user(request)
    db = get_db()
    doc = db.users.find_one(
        {"email": user["email"]},
        {"_id": 0, "password_hash": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")
    # Convert datetime fields to ISO strings for JSON serialisation
    for field in ("created_at", "last_login_at"):
        if field in doc and doc[field]:
            doc[field] = doc[field].isoformat()
    return doc
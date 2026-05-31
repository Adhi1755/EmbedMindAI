# auth.py — Persistent sessions via MongoDB + configurable redirect URIs
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
import os, secrets
from dotenv import load_dotenv
import httpx
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone

router = APIRouter()
load_dotenv()

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# Configurable via env so it works in Docker, staging, and production
BACKEND_URL  = os.getenv("BACKEND_URL",  "http://localhost:8000")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
REDIRECT_URI = f"{BACKEND_URL}/auth/callback"

# ── MongoDB session store ─────────────────────────────────────────────────────
_mongo_client = None

def _get_sessions_collection():
    """Lazily initialize MongoDB connection and return sessions collection."""
    global _mongo_client
    if _mongo_client is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/embedmindai")
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # TTL index: sessions expire after 7 days automatically
        db = _mongo_client.get_default_database()
        db.sessions.create_index("expires_at", expireAfterSeconds=0)
    return _mongo_client.get_default_database().sessions

SESSION_TTL_DAYS = 7


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

    session_token = secrets.token_hex(32)
    _store_session(session_token, {
        "name":    user_data["name"],
        "email":   user_data["email"],
        "picture": user_data.get("picture"),
    })

    response = RedirectResponse(url=f"{FRONTEND_URL}/chat")
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_DAYS * 86400,
    )
    return response


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
# path.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
import os, requests, secrets
from dotenv import load_dotenv
import httpx


router = APIRouter()
load_dotenv()

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/auth/callback"

SESSION_STORE = {}

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
    # Step 1: Exchange code for token
    token_url = "https://oauth2.googleapis.com/token"
    data = {
    "code": code,
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "redirect_uri": REDIRECT_URI,
    "grant_type": "authorization_code"
}


    async with httpx.AsyncClient() as client:
        token_response = await client.post(token_url, data=data)
        token_json = token_response.json()

        access_token = token_json.get("access_token")

        # Step 2: Fetch user info
        user_info_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        user_response = await client.get(user_info_url, headers=headers)
        user_data = user_response.json()

        # Example user_data keys: sub, name, email, picture

        # Step 3: Store user in session
        session_token = secrets.token_hex(16)
        SESSION_STORE[session_token] = {
            "name": user_data["name"],
            "email": user_data["email"],
            "picture": user_data.get("picture")
        }

        # Set session cookie
        response = RedirectResponse(url="http://localhost:3000/chat")  # your frontend
        response.set_cookie(key="session_token", value=session_token, httponly=True)
        return response

@router.get("/auth/me")
def get_logged_in_user(request: Request):
    token = request.cookies.get("session_token")
    user = SESSION_STORE.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
@router.get("/auth/logout")
def logout(request: Request):
    token = request.cookies.get("session_token")
    if token and token in SESSION_STORE:
        del SESSION_STORE[token]

    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie("session_token")
    return response
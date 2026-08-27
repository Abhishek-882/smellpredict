"""
SmellPredict — GitHub OAuth2 Authentication
============================================
Implements:
  - GitHub OAuth App redirect flow
  - Authorization code → access token exchange
  - JWT issue and verify (python-jose + HS256)
  - Fernet-encrypted GitHub token embedded in JWT payload

Endpoints:
  GET /auth/github          → redirect to GitHub OAuth authorize URL
  GET /auth/github/callback → exchange code for token, issue JWT, redirect to IDE
  GET /auth/me              → return {login, avatar_url} from JWT
  GET /auth/refresh         → silently renew JWT if expiring within 30 min

Environment variables required:
  GITHUB_CLIENT_ID
  GITHUB_CLIENT_SECRET
  GITHUB_REDIRECT_URI  (default: http://localhost:8000/auth/github/callback)
  JWT_SECRET_KEY       (random 256-bit hex)
"""

from __future__ import annotations

import base64
import os
from datetime import datetime, timedelta, timezone

from dotenv import find_dotenv, load_dotenv

# Ensure .env is loaded
load_dotenv(find_dotenv())

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, HTTPException, Query, Request, status
from jose import JWTError, jwt
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("smellpredict.auth")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (from environment)
# ─────────────────────────────────────────────────────────────────────────────

def get_github_client_id() -> str:
    return os.getenv("GITHUB_CLIENT_ID", "")

def get_github_client_secret() -> str:
    return os.getenv("GITHUB_CLIENT_SECRET", "")

def get_github_redirect_uri() -> str:
    return os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback")

GITHUB_CLIENT_ID: str = get_github_client_id()
GITHUB_CLIENT_SECRET: str = get_github_client_secret()
GITHUB_REDIRECT_URI: str = get_github_redirect_uri()
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRE_HOURS: int = 8

# Derive a Fernet key from JWT_SECRET_KEY (Fernet requires a 32-byte URL-safe base64 key)
_raw_key = (JWT_SECRET_KEY.encode()[:32]).ljust(32, b"\x00")
_FERNET_KEY = base64.urlsafe_b64encode(_raw_key)
_fernet = Fernet(_FERNET_KEY)

# GitHub OAuth URLs
GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_OAUTH_SCOPE = "repo,user"

# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/auth", tags=["auth"])


# ─────────────────────────────────────────────────────────────────────────────
# JWT / Token helpers (also imported by github_api.py and collab.py)
# ─────────────────────────────────────────────────────────────────────────────


def _encrypt_token(github_token: str) -> str:
    """Fernet-encrypt a GitHub access token for safe embedding in JWT payload."""
    return _fernet.encrypt(github_token.encode()).decode()


def _decrypt_token(encrypted: str) -> str:
    """Decrypt a Fernet-encrypted GitHub token from the JWT payload."""
    try:
        return _fernet.decrypt(encrypted.encode()).decode()
    except InvalidToken as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token decryption failed — invalid or tampered JWT payload.",
        ) from exc


def issue_jwt(login: str, avatar_url: str, github_token: str) -> str:
    """
    Issue a signed JWT containing:
      sub          = GitHub login (username)
      avatar_url   = GitHub avatar URL (for UI display)
      gh_tok       = Fernet-encrypted GitHub OAuth access token
      iat, exp     = issued at / expires at
    """
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": login,
        "avatar_url": avatar_url,
        "gh_tok": _encrypt_token(github_token),
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_jwt(token: str) -> dict:
    """
    Verify and decode a JWT. Returns the decoded payload dict.
    Raises HTTP 401 if invalid or expired.
    """
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
        ) from exc


def extract_github_token(bearer_jwt: str) -> str:
    """
    Decode a JWT and decrypt the embedded GitHub access token.
    Used by github_api.py and collab.py to authenticate GitHub API calls.
    """
    payload = verify_jwt(bearer_jwt)
    return _decrypt_token(payload["gh_tok"])


def _bearer_token(request: Request) -> str:
    """Extract Bearer token from Authorization header or ?token= query param."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    token_param = request.query_params.get("token", "")
    if token_param:
        return token_param
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing Authorization header or ?token= query parameter.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/github", summary="Redirect to GitHub OAuth authorize page")
async def auth_github_redirect(request: Request):
    """
    Initiates the GitHub OAuth2 authorization flow.
    Redirects the browser to GitHub's authorization page.
    After the user grants access, GitHub calls back to /auth/github/callback.
    """
    client_id = get_github_client_id()
    
    # Auto-detect dynamic protocol and host if behind a tunnel (Localtunnel / Cloudflare)
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    dynamic_redirect = f"{proto}://{host}/auth/github/callback"
    
    redirect_uri = os.environ.get("GITHUB_REDIRECT_URI") or dynamic_redirect
    
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GITHUB_CLIENT_ID is not configured. Set it in your .env file.",
        )
    params = (
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&scope={GITHUB_OAUTH_SCOPE}"
        f"&allow_signup=true"
    )
    redirect_url = GITHUB_AUTHORIZE_URL + params
    logger.info(f"Redirecting to GitHub OAuth: {redirect_url}")
    return RedirectResponse(url=redirect_url, status_code=302)


@router.get("/github/callback", summary="GitHub OAuth callback — exchange code for JWT")
async def auth_github_callback(
    request: Request,
    code: str = Query(..., description="OAuth authorization code from GitHub"),
):
    """
    GitHub redirects here after the user grants access.
    Exchanges the one-time code for a GitHub access token, fetches user info,
    issues a SmellPredict JWT, and redirects the browser to the IDE with the
    token embedded in the URL hash (#token=<jwt>).
    """
    client_id = get_github_client_id()
    client_secret = get_github_client_secret()
    
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    dynamic_redirect = f"{proto}://{host}/auth/github/callback"
    
    redirect_uri = os.environ.get("GITHUB_REDIRECT_URI") or dynamic_redirect
    
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth credentials not configured.",
        )

    async with httpx.AsyncClient(timeout=15) as client:
        # Exchange authorization code for GitHub access token
        token_resp = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()

        if "error" in token_data:
            logger.error(f"GitHub token exchange error: {token_data}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"GitHub OAuth error: "
                    f"{token_data.get('error_description', token_data['error'])}"
                ),
            )

        github_token: str = token_data["access_token"]

        # Fetch authenticated user profile
        user_resp = await client.get(
            GITHUB_USER_URL,
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        user_resp.raise_for_status()
        user_data = user_resp.json()

    login: str = user_data["login"]
    avatar_url: str = user_data.get("avatar_url", "")
    logger.info(f"GitHub OAuth success for user: {login}")

    # Issue SmellPredict JWT (GitHub token encrypted inside payload)
    jwt_token = issue_jwt(login=login, avatar_url=avatar_url, github_token=github_token)

    # Redirect to IDE — token in URL hash (never transmitted to server on redirect)
    ide_url = f"/ui/ide.html#token={jwt_token}"
    return RedirectResponse(url=ide_url, status_code=302)


@router.get("/me", summary="Get authenticated user info from JWT")
async def auth_me(request: Request):
    """
    Returns the authenticated user's GitHub login and avatar URL from the JWT.
    Used by the IDE frontend to display the logged-in user in the header.

    Requires: Authorization: Bearer <jwt>  OR  ?token=<jwt>
    """
    token = _bearer_token(request)
    payload = verify_jwt(token)
    return JSONResponse({
        "login": payload["sub"],
        "avatar_url": payload.get("avatar_url", ""),
        "expires_at": payload["exp"],
    })


@router.get("/refresh", summary="Refresh JWT if expiring within 30 minutes")
async def auth_refresh(request: Request):
    """
    Silently refreshes the JWT if it will expire within 30 minutes.
    Returns the new token if refreshed, or {refreshed: false} if still fresh.
    """
    token = _bearer_token(request)
    payload = verify_jwt(token)

    now = datetime.now(tz=timezone.utc)
    exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    time_remaining = (exp - now).total_seconds()

    if time_remaining > 1800:  # > 30 minutes — still fresh
        return JSONResponse({"refreshed": False, "expires_at": payload["exp"]})

    # Decrypt the stored GitHub token and re-issue
    github_token = _decrypt_token(payload["gh_tok"])
    new_jwt = issue_jwt(
        login=payload["sub"],
        avatar_url=payload.get("avatar_url", ""),
        github_token=github_token,
    )
    logger.info(f"Refreshed JWT for user: {payload['sub']}")
    return JSONResponse({"refreshed": True, "token": new_jwt})

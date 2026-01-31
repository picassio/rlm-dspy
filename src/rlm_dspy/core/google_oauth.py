"""Google Gemini CLI OAuth flow.

Implements OAuth for Google Cloud Code Assist (Gemini CLI) which provides
access to Gemini models (gemini-2.0-flash, gemini-2.5-*).
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import logging
import os
import secrets
import socketserver
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)

# OAuth configuration
# These are public OAuth client credentials from Google's Gemini CLI
# They are NOT secrets - they identify the application, not the user
# User authentication happens via browser OAuth flow
_DEFAULT_CLIENT_ID = "681255809395-oo8ft2oprdrn" + "p9e3aqf6av3hmdib135j.apps" + ".googleusercontent.com"
_DEFAULT_CLIENT_SECRET = "GOCSPX-4uHgMPm-1o7Sk" + "-geV6Cu5clXFsxl"
CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", _DEFAULT_CLIENT_ID)
CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", _DEFAULT_CLIENT_SECRET)
REDIRECT_URI = "http://localhost:8085/oauth2callback"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform",
          "https://www.googleapis.com/auth/userinfo.email",
          "https://www.googleapis.com/auth/userinfo.profile"]
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
CODE_ASSIST = "https://cloudcode-pa.googleapis.com"


@dataclass
class GoogleOAuthCredentials:
    """Google OAuth credentials with project info."""
    provider: str = "google"
    access_token: str = ""
    refresh_token: str = ""
    project_id: str = ""
    email: str | None = None
    expires_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    @property
    def is_expired(self) -> bool:
        return time.time() >= (self.expires_at - 300)
    
    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ["provider", "access_token", "refresh_token",
                                                "project_id", "email", "expires_at", "created_at"]}
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoogleOAuthCredentials":
        return cls(**{k: data.get(k, getattr(cls, k, None)) for k in 
                      ["provider", "access_token", "refresh_token", "project_id", "email", "expires_at", "created_at"]})


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge."""
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """OAuth callback handler."""
    code: str | None = None
    state: str | None = None
    error: str | None = None
    
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        if urlparse(self.path).path != "/oauth2callback":
            self.send_response(404)
            self.end_headers()
            return
        
        if error := params.get("error", [None])[0]:
            _CallbackHandler.error = error
            self._html(400, f"<h1>Failed</h1><p>{error}</p>")
        elif (code := params.get("code", [None])[0]) and (state := params.get("state", [None])[0]):
            _CallbackHandler.code, _CallbackHandler.state = code, state
            self._html(200, "<h1>Success</h1><p>You can close this window.</p>")
        else:
            self._html(400, "<h1>Failed</h1><p>Missing parameters.</p>")
    
    def _html(self, status: int, body: str):
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(f"<html><body>{body}</body></html>".encode())
    
    def log_message(self, *args): pass


def _parse_redirect_url(url: str) -> tuple[str | None, str | None]:
    """Parse redirect URL to extract code and state."""
    try:
        parsed = urlparse(url.strip())
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        return code, state
    except Exception:
        return None, None


def _prompt_for_url() -> tuple[str | None, str | None]:
    """Prompt user to paste the redirect URL (for headless/SSH use)."""
    print("\n" + "-" * 60)
    print("MANUAL MODE: Paste the full redirect URL from your browser")
    print("(The URL starting with http://localhost:8085/oauth2callback?...)")
    print("-" * 60)
    try:
        url = input("\nRedirect URL: ").strip()
        if url:
            return _parse_redirect_url(url)
    except (KeyboardInterrupt, EOFError):
        pass
    return None, None


def _discover_project(token: str, on_progress: callable | None = None) -> str:
    """Discover or provision a Google Cloud project."""
    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    if on_progress:
        on_progress("Checking for Cloud Code Assist project...")
    
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{CODE_ASSIST}/v1internal:loadCodeAssist", headers=headers,
                          json={"cloudaicompanionProject": env_project,
                                "metadata": {"ideType": "IDE_UNSPECIFIED", "platform": "PLATFORM_UNSPECIFIED",
                                           "pluginType": "GEMINI", "duetProject": env_project}})
        if resp.status_code != 200:
            raise ValueError(f"loadCodeAssist failed: {resp.status_code} {resp.text}")
        
        data = resp.json()
        if data.get("currentTier"):
            return data.get("cloudaicompanionProject") or env_project or _require_project()
        
        # Need onboarding
        tier = next((t for t in data.get("allowedTiers", []) if t.get("isDefault")), {}).get("id", "free-tier")
        if tier != "free-tier" and not env_project:
            _require_project()
        
        if on_progress:
            on_progress("Provisioning project...")
        
        body = {"tierId": tier, "metadata": {"ideType": "IDE_UNSPECIFIED", "platform": "PLATFORM_UNSPECIFIED", "pluginType": "GEMINI"}}
        if tier != "free-tier" and env_project:
            body["cloudaicompanionProject"] = env_project
        
        onboard_resp = client.post(f"{CODE_ASSIST}/v1internal:onboardUser", headers=headers, json=body)
        if onboard_resp.status_code != 200:
            raise ValueError(f"onboardUser failed: {onboard_resp.status_code}")
        
        lro = onboard_resp.json()
        while not lro.get("done") and lro.get("name"):
            time.sleep(5)
            lro = client.get(f"{CODE_ASSIST}/v1internal/{lro['name']}", headers=headers).json()
        
        return lro.get("response", {}).get("cloudaicompanionProject", {}).get("id") or env_project or _require_project()


def _require_project():
    raise ValueError("Set GOOGLE_CLOUD_PROJECT env var. See https://goo.gle/gemini-cli-auth-docs")


def _get_email(token: str) -> str | None:
    try:
        resp = httpx.get("https://www.googleapis.com/oauth2/v1/userinfo", params={"alt": "json"},
                        headers={"Authorization": f"Bearer {token}"}, timeout=10.0)
        return resp.json().get("email") if resp.status_code == 200 else None
    except Exception:
        return None


def _save(creds: GoogleOAuthCredentials) -> None:
    from .oauth import OAUTH_DIR, CREDENTIALS_FILE
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    all_creds = json.loads(CREDENTIALS_FILE.read_text()) if CREDENTIALS_FILE.exists() else {}
    all_creds["google"] = creds.to_dict()
    tmp = CREDENTIALS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(all_creds, indent=2))
    tmp.replace(CREDENTIALS_FILE)
    os.chmod(CREDENTIALS_FILE, 0o600)


def _load() -> GoogleOAuthCredentials | None:
    from .oauth import CREDENTIALS_FILE
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        data = json.loads(CREDENTIALS_FILE.read_text())
        return GoogleOAuthCredentials.from_dict(data["google"]) if "google" in data else None
    except Exception:
        return None


def google_login(open_browser: bool = True, timeout: float = 300.0,
                 on_progress: callable | None = None, manual: bool = False) -> GoogleOAuthCredentials:
    """Login with Google Gemini CLI OAuth.
    
    Args:
        open_browser: Whether to try opening a browser
        timeout: Timeout for callback (ignored in manual mode)
        on_progress: Progress callback
        manual: If True, prompt user to paste redirect URL (for SSH/headless)
    """
    verifier, challenge = _generate_pkce()
    
    auth_url = f"{AUTH_URL}?{urlencode({'client_id': CLIENT_ID, 'response_type': 'code',
        'redirect_uri': REDIRECT_URI, 'scope': ' '.join(SCOPES), 'code_challenge': challenge,
        'code_challenge_method': 'S256', 'state': verifier, 'access_type': 'offline', 'prompt': 'consent'})}"
    
    print(f"\n{'='*60}\nGoogle Gemini CLI OAuth Login\n{'='*60}")
    print(f"\nVisit this URL to authenticate:\n\n{auth_url}\n")
    
    if manual or not open_browser:
        # Manual mode: user pastes the redirect URL
        print("After authenticating, copy the FULL URL from your browser")
        print("(it will show 'This site can't be reached' - that's OK)")
        code, state = _prompt_for_url()
    else:
        # Try browser + callback server
        if open_browser:
            webbrowser.open(auth_url)
        
        print("Waiting for callback on localhost:8085...")
        print("(If using SSH, press Ctrl+C and use: rlm-dspy auth login google --no-browser)")
        
        _CallbackHandler.code = _CallbackHandler.state = _CallbackHandler.error = None
        server = socketserver.TCPServer(("127.0.0.1", 8085), _CallbackHandler)
        server.timeout = 1.0
        
        try:
            start = time.time()
            while time.time() - start < timeout:
                server.handle_request()
                if _CallbackHandler.error:
                    raise ValueError(f"OAuth error: {_CallbackHandler.error}")
                if _CallbackHandler.code:
                    code, state = _CallbackHandler.code, _CallbackHandler.state
                    break
            else:
                code, state = None, None
        finally:
            server.server_close()
    
    if not code:
        raise ValueError("No authorization code received")
    if state != verifier:
        raise ValueError("OAuth state mismatch")
    
    if on_progress:
        on_progress("Exchanging code for tokens...")
    
    resp = httpx.post(TOKEN_URL, data={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
        "code": code, "grant_type": "authorization_code", "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier}, timeout=30.0)
    
    if resp.status_code != 200:
        raise ValueError(f"Token exchange failed: {resp.text}")
    
    tokens = resp.json()
    if not tokens.get("refresh_token"):
        raise ValueError("No refresh token received")
    
    email = _get_email(tokens["access_token"])
    project = _discover_project(tokens["access_token"], on_progress)
    expires_at = time.time() + tokens.get("expires_in", 3600) - 300
    
    creds = GoogleOAuthCredentials(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"],
                                    project_id=project, email=email, expires_at=expires_at)
    _save(creds)
    
    print(f"\n✓ Authenticated!{f' ({email})' if email else ''}\n  Project: {project}")
    print(f"  Expires: {datetime.fromtimestamp(expires_at, UTC).isoformat()}")
    return creds


def google_refresh_token(creds: GoogleOAuthCredentials) -> GoogleOAuthCredentials:
    """Refresh Google OAuth token."""
    resp = httpx.post(TOKEN_URL, data={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
        "refresh_token": creds.refresh_token, "grant_type": "refresh_token"}, timeout=30.0)
    
    if resp.status_code != 200:
        raise ValueError(f"Refresh failed: {resp.text}")
    
    data = resp.json()
    new_creds = GoogleOAuthCredentials(access_token=data["access_token"],
        refresh_token=data.get("refresh_token", creds.refresh_token),
        project_id=creds.project_id, email=creds.email,
        expires_at=time.time() + data.get("expires_in", 3600) - 300)
    _save(new_creds)
    return new_creds


def get_google_token() -> tuple[str, str] | None:
    """Get valid Google OAuth token and project ID. Auto-refreshes if expired."""
    creds = _load()
    if not creds:
        return None
    if creds.is_expired:
        try:
            creds = google_refresh_token(creds)
        except Exception:
            return None
    return creds.access_token, creds.project_id


def google_logout() -> bool:
    """Logout from Google OAuth."""
    from .oauth import _delete_credentials
    return _delete_credentials("google")


def is_google_authenticated() -> bool:
    """Check if authenticated with Google OAuth."""
    return _load() is not None

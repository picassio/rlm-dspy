"""OAuth authentication for LLM providers.

Implements OAuth flows for:
- Anthropic (Claude Pro/Max via claude.ai)
- Future: Google, GitHub Copilot, etc.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

# Storage paths
OAUTH_DIR = Path.home() / ".rlm" / "oauth"
CREDENTIALS_FILE = OAUTH_DIR / "credentials.json"

# Anthropic OAuth configuration (from Claude Code)
ANTHROPIC_CLIENT_ID = base64.b64decode(
    "OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl"
).decode()
ANTHROPIC_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
ANTHROPIC_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
ANTHROPIC_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
ANTHROPIC_SCOPES = "org:create_api_key user:profile user:inference"


@dataclass
class OAuthCredentials:
    """OAuth credentials for a provider."""
    
    provider: str
    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp
    created_at: float = field(default_factory=lambda: time.time())
    
    @property
    def is_expired(self) -> bool:
        """Check if the access token is expired (with 5 min buffer)."""
        return time.time() >= (self.expires_at - 300)
    
    @property
    def is_oauth_token(self) -> bool:
        """Check if this is an Anthropic OAuth token."""
        return self.access_token.startswith("sk-ant-oat")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "provider": self.provider,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthCredentials":
        """Create from dictionary."""
        return cls(
            provider=data["provider"],
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            created_at=data.get("created_at", time.time()),
        )


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge.
    
    Returns:
        Tuple of (verifier, challenge)
    """
    # Generate random verifier (43-128 characters, URL-safe)
    verifier = secrets.token_urlsafe(32)
    
    # Create S256 challenge
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    
    return verifier, challenge


def _save_credentials(credentials: OAuthCredentials) -> None:
    """Save credentials to disk."""
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing credentials
    all_creds: dict[str, dict] = {}
    if CREDENTIALS_FILE.exists():
        try:
            all_creds = json.loads(CREDENTIALS_FILE.read_text())
        except Exception:
            pass
    
    # Update with new credentials
    all_creds[credentials.provider] = credentials.to_dict()
    
    # Save atomically
    temp_file = CREDENTIALS_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(all_creds, indent=2))
    temp_file.replace(CREDENTIALS_FILE)
    
    # Restrict permissions (owner read/write only)
    os.chmod(CREDENTIALS_FILE, 0o600)
    
    logger.info("Saved OAuth credentials for %s", credentials.provider)


def _load_credentials(provider: str) -> OAuthCredentials | None:
    """Load credentials for a provider from disk."""
    if not CREDENTIALS_FILE.exists():
        return None
    
    try:
        all_creds = json.loads(CREDENTIALS_FILE.read_text())
        if provider in all_creds:
            return OAuthCredentials.from_dict(all_creds[provider])
    except Exception as e:
        logger.warning("Failed to load OAuth credentials: %s", e)
    
    return None


def _delete_credentials(provider: str) -> bool:
    """Delete credentials for a provider.
    
    Returns:
        True if credentials were deleted
    """
    if not CREDENTIALS_FILE.exists():
        return False
    
    try:
        all_creds = json.loads(CREDENTIALS_FILE.read_text())
        if provider in all_creds:
            del all_creds[provider]
            CREDENTIALS_FILE.write_text(json.dumps(all_creds, indent=2))
            logger.info("Deleted OAuth credentials for %s", provider)
            return True
    except Exception as e:
        logger.warning("Failed to delete OAuth credentials: %s", e)
    
    return False


# ============================================================================
# Anthropic OAuth Implementation
# ============================================================================

def anthropic_login(
    open_browser: bool = True,
    timeout: float = 300.0,
) -> OAuthCredentials:
    """Login with Anthropic OAuth (Claude Pro/Max).
    
    This opens a browser for the user to authenticate, then prompts
    for the authorization code.
    
    Args:
        open_browser: Whether to automatically open the browser
        timeout: Timeout for user to complete authentication
        
    Returns:
        OAuth credentials
        
    Raises:
        ValueError: If authentication fails
        TimeoutError: If user doesn't complete in time
    """
    # Generate PKCE
    verifier, challenge = _generate_pkce()
    
    # Build authorization URL
    params = {
        "code": "true",
        "client_id": ANTHROPIC_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": ANTHROPIC_REDIRECT_URI,
        "scope": ANTHROPIC_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    
    auth_url = f"{ANTHROPIC_AUTHORIZE_URL}?{urlencode(params)}"
    
    print("\n" + "=" * 60)
    print("Anthropic OAuth Login (Claude Pro/Max)")
    print("=" * 60)
    print("\nOpening browser for authentication...")
    print(f"\nIf browser doesn't open, visit:\n{auth_url}\n")
    
    if open_browser:
        webbrowser.open(auth_url)
    
    print("After authorizing, you'll see a page with an authorization code.")
    print("Copy the FULL code (format: CODE#STATE) and paste it below.\n")
    
    # Prompt for authorization code
    try:
        auth_code = input("Authorization code: ").strip()
    except (KeyboardInterrupt, EOFError):
        raise ValueError("Authentication cancelled by user")
    
    if not auth_code:
        raise ValueError("No authorization code provided")
    
    # Parse code and state
    if "#" in auth_code:
        code, state = auth_code.split("#", 1)
    else:
        code = auth_code
        state = verifier
    
    # Exchange code for tokens
    return _anthropic_exchange_code(code, state, verifier)


def _anthropic_exchange_code(
    code: str,
    state: str,
    verifier: str,
) -> OAuthCredentials:
    """Exchange authorization code for tokens."""
    
    payload = {
        "grant_type": "authorization_code",
        "client_id": ANTHROPIC_CLIENT_ID,
        "code": code,
        "state": state,
        "redirect_uri": ANTHROPIC_REDIRECT_URI,
        "code_verifier": verifier,
    }
    
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            ANTHROPIC_TOKEN_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
    
    if response.status_code != 200:
        error_text = response.text
        logger.error("Token exchange failed: %s", error_text)
        raise ValueError(f"Token exchange failed: {error_text}")
    
    data = response.json()
    
    # Calculate expiry (current time + expires_in - 5 min buffer)
    expires_in = data.get("expires_in", 3600)
    expires_at = time.time() + expires_in - 300
    
    credentials = OAuthCredentials(
        provider="anthropic",
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        expires_at=expires_at,
    )
    
    # Save credentials
    _save_credentials(credentials)
    
    print("\n✓ Successfully authenticated with Anthropic!")
    print(f"  Token expires: {datetime.fromtimestamp(expires_at, UTC).isoformat()}")
    
    return credentials


def anthropic_refresh_token(credentials: OAuthCredentials) -> OAuthCredentials:
    """Refresh Anthropic OAuth token.
    
    Args:
        credentials: Current credentials with refresh token
        
    Returns:
        New credentials with refreshed access token
    """
    payload = {
        "grant_type": "refresh_token",
        "client_id": ANTHROPIC_CLIENT_ID,
        "refresh_token": credentials.refresh_token,
    }
    
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            ANTHROPIC_TOKEN_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
    
    if response.status_code != 200:
        error_text = response.text
        logger.error("Token refresh failed: %s", error_text)
        raise ValueError(f"Token refresh failed: {error_text}")
    
    data = response.json()
    
    expires_in = data.get("expires_in", 3600)
    expires_at = time.time() + expires_in - 300
    
    new_credentials = OAuthCredentials(
        provider="anthropic",
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", credentials.refresh_token),
        expires_at=expires_at,
    )
    
    # Save updated credentials
    _save_credentials(new_credentials)
    
    logger.info("Refreshed Anthropic OAuth token")
    
    return new_credentials


def get_anthropic_token() -> str | None:
    """Get a valid Anthropic OAuth token.
    
    Automatically refreshes if expired.
    
    Returns:
        Access token or None if not authenticated
    """
    credentials = _load_credentials("anthropic")
    if not credentials:
        return None
    
    # Refresh if expired
    if credentials.is_expired:
        try:
            credentials = anthropic_refresh_token(credentials)
        except Exception as e:
            logger.warning("Failed to refresh token: %s", e)
            return None
    
    return credentials.access_token


def anthropic_logout() -> bool:
    """Logout from Anthropic OAuth.
    
    Returns:
        True if credentials were deleted
    """
    return _delete_credentials("anthropic")


def is_anthropic_authenticated() -> bool:
    """Check if authenticated with Anthropic OAuth."""
    credentials = _load_credentials("anthropic")
    return credentials is not None


# ============================================================================
# Generic OAuth API
# ============================================================================

def get_oauth_token(provider: str) -> str | None:
    """Get OAuth token for a provider.
    
    Args:
        provider: Provider name (e.g., "anthropic")
        
    Returns:
        Access token or None if not authenticated
    """
    if provider == "anthropic":
        return get_anthropic_token()
    
    # Add more providers here
    return None


def oauth_login(provider: str, **kwargs) -> OAuthCredentials:
    """Login with OAuth for a provider.
    
    Args:
        provider: Provider name
        **kwargs: Provider-specific options
        
    Returns:
        OAuth credentials
    """
    if provider == "anthropic":
        return anthropic_login(**kwargs)
    
    raise ValueError(f"Unknown OAuth provider: {provider}")


def oauth_logout(provider: str) -> bool:
    """Logout from OAuth for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        True if logged out
    """
    if provider == "anthropic":
        return anthropic_logout()
    
    return False


def oauth_status(provider: str) -> dict[str, Any]:
    """Get OAuth status for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        Status dictionary
    """
    credentials = _load_credentials(provider)
    
    if not credentials:
        return {
            "authenticated": False,
            "provider": provider,
        }
    
    return {
        "authenticated": True,
        "provider": provider,
        "expires_at": credentials.expires_at,
        "is_expired": credentials.is_expired,
        "created_at": credentials.created_at,
    }

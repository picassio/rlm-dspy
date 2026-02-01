"""Base classes for OAuth providers.

Provides abstract base class and common utilities for OAuth implementations.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import logging
import secrets
import socketserver
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# Storage paths
OAUTH_DIR = Path.home() / ".rlm" / "oauth"
CREDENTIALS_FILE = OAUTH_DIR / "credentials.json"


class OAuthError(Exception):
    """Base OAuth error."""
    pass


class AuthenticationError(OAuthError):
    """Authentication failed."""
    pass


class TokenRefreshError(OAuthError):
    """Token refresh failed."""
    pass


@dataclass
class OAuthCredentials:
    """OAuth credentials for a provider."""
    
    provider: str
    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp
    created_at: float = field(default_factory=time.time)
    # Optional provider-specific fields
    project_id: str = ""
    email: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if the access token is expired (with 5 min buffer)."""
        return time.time() >= (self.expires_at - 300)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "provider": self.provider,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
            "project_id": self.project_id,
            "email": self.email,
            "extra": self.extra,
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
            project_id=data.get("project_id", ""),
            email=data.get("email"),
            extra=data.get("extra", {}),
        )


@dataclass
class OAuthConfig:
    """OAuth provider configuration."""
    
    provider_name: str
    client_id: str
    client_secret: str
    auth_url: str
    token_url: str
    redirect_uri: str
    scopes: list[str]
    callback_path: str = "/oauth2callback"
    callback_port: int = 8085
    
    # Optional provider-specific settings
    extra_auth_params: dict[str, str] = field(default_factory=dict)
    extra_token_params: dict[str, str] = field(default_factory=dict)


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge.
    
    Returns:
        Tuple of (verifier, challenge)
    """
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def generate_state() -> str:
    """Generate a random state parameter."""
    return secrets.token_urlsafe(16)


class CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Generic OAuth callback handler.
    
    Note: Class-level variables are used here because the HTTP server
    creates new handler instances for each request, so we need shared
    state to communicate the OAuth result back to the waiting code.
    This is acceptable because OAuth flows are typically user-initiated
    and single-threaded per user session.
    """
    
    code: str | None = None
    state: str | None = None
    error: str | None = None
    callback_path: str = "/oauth2callback"
    
    @classmethod
    def reset(cls) -> None:
        """Reset callback state before starting a new OAuth flow."""
        cls.code = None
        cls.state = None
        cls.error = None
    
    def do_GET(self):
        """Handle OAuth callback GET request."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        if parsed.path != self.callback_path:
            self.send_response(404)
            self.end_headers()
            return
        
        if error := params.get("error", [None])[0]:
            CallbackHandler.error = error
            self._send_html(400, f"<h1>Authentication Failed</h1><p>{error}</p>")
        elif (code := params.get("code", [None])[0]) and (state := params.get("state", [None])[0]):
            CallbackHandler.code = code
            CallbackHandler.state = state
            self._send_html(200, "<h1>Success!</h1><p>You can close this window.</p>")
        else:
            self._send_html(400, "<h1>Failed</h1><p>Missing required parameters.</p>")
    
    def _send_html(self, status: int, body: str):
        """Send HTML response."""
        html = f"""<!DOCTYPE html>
<html><head><title>OAuth</title>
<style>body{{font-family:system-ui;display:flex;justify-content:center;align-items:center;
height:100vh;margin:0;background:#f5f5f5}}div{{text-align:center;padding:2em;
background:white;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.1)}}</style>
</head><body><div>{body}</div></body></html>"""
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html.encode())
    
    def log_message(self, format, *args):
        """Suppress logging."""
        pass


class OAuthProvider(ABC):
    """Abstract base class for OAuth providers."""
    
    def __init__(self, config: OAuthConfig):
        self.config = config
    
    @property
    def name(self) -> str:
        """Provider name."""
        return self.config.provider_name
    
    @abstractmethod
    def authenticate(self) -> OAuthCredentials:
        """Run OAuth flow and return credentials.
        
        Raises:
            AuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    def refresh(self, credentials: OAuthCredentials) -> OAuthCredentials:
        """Refresh expired credentials.
        
        Args:
            credentials: Existing credentials with refresh token
            
        Returns:
            Updated credentials with new access token
            
        Raises:
            TokenRefreshError: If refresh fails
        """
        pass
    
    def revoke(self, credentials: OAuthCredentials) -> bool:
        """Revoke credentials. Override if provider supports revocation.
        
        Returns:
            True if revoked, False if not supported
        """
        return False
    
    def _start_callback_server(self) -> socketserver.TCPServer:
        """Start local callback server."""
        # Create handler with correct callback path
        class Handler(CallbackHandler):
            callback_path = self.config.callback_path
        
        # Reset state
        Handler.code = None
        Handler.state = None
        Handler.error = None
        
        server = socketserver.TCPServer(
            ("localhost", self.config.callback_port),
            Handler,
            bind_and_activate=False
        )
        server.allow_reuse_address = True
        server.server_bind()
        server.server_activate()
        return server
    
    def _wait_for_callback(self, server: socketserver.TCPServer, timeout: float = 120) -> tuple[str, str]:
        """Wait for OAuth callback.
        
        Returns:
            Tuple of (code, state)
            
        Raises:
            AuthenticationError: If callback fails or times out
        """
        import select
        
        start = time.time()
        while time.time() - start < timeout:
            # Use select for timeout
            readable, _, _ = select.select([server.socket], [], [], 1.0)
            if readable:
                server.handle_request()
                
                handler_class = server.RequestHandlerClass
                if handler_class.error:
                    raise AuthenticationError(f"OAuth error: {handler_class.error}")
                if handler_class.code and handler_class.state:
                    return handler_class.code, handler_class.state
        
        raise AuthenticationError("OAuth callback timed out")


def save_credentials(credentials: OAuthCredentials) -> None:
    """Save credentials to disk."""
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing credentials
    all_creds: dict[str, dict] = {}
    if CREDENTIALS_FILE.exists():
        try:
            all_creds = json.loads(CREDENTIALS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    
    # Update with new credentials
    all_creds[credentials.provider] = credentials.to_dict()
    
    # Atomic write with secure permissions
    import tempfile
    import os
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', dir=OAUTH_DIR, suffix='.tmp', delete=False
        ) as f:
            json.dump(all_creds, f, indent=2)
            temp_path = Path(f.name)
        # Set secure permissions before moving to final location (owner read/write only)
        os.chmod(temp_path, 0o600)
        temp_path.replace(CREDENTIALS_FILE)
        # Ensure final file also has secure permissions
        os.chmod(CREDENTIALS_FILE, 0o600)
    except Exception:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def load_credentials(provider: str) -> OAuthCredentials | None:
    """Load credentials for a provider from disk."""
    if not CREDENTIALS_FILE.exists():
        return None
    
    try:
        all_creds = json.loads(CREDENTIALS_FILE.read_text())
        if provider in all_creds:
            return OAuthCredentials.from_dict(all_creds[provider])
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    
    return None


def delete_credentials(provider: str) -> bool:
    """Delete credentials for a provider.
    
    Returns:
        True if deleted, False if not found
    """
    if not CREDENTIALS_FILE.exists():
        return False
    
    try:
        all_creds = json.loads(CREDENTIALS_FILE.read_text())
        if provider not in all_creds:
            return False
        
        del all_creds[provider]
        
        # Atomic write
        import tempfile
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', dir=OAUTH_DIR, suffix='.tmp', delete=False
            ) as f:
                json.dump(all_creds, f, indent=2)
                temp_path = Path(f.name)
            temp_path.replace(CREDENTIALS_FILE)
            return True
        except Exception:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise
    except (json.JSONDecodeError, OSError):
        return False

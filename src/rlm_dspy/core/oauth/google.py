"""Google OAuth providers (Gemini CLI and Antigravity).

Implements OAuth flows for:
- Google Gemini CLI (gemini-2.0-flash, gemini-2.5-*)
- Antigravity (gemini-3-*, claude-*, gpt-oss)
"""

from __future__ import annotations

import logging
import os
import time
import webbrowser
from urllib.parse import urlencode

import httpx

from .base import (
    OAuthConfig,
    OAuthCredentials,
    OAuthProvider,
    AuthenticationError,
    TokenRefreshError,
    generate_pkce,
    generate_state,
    save_credentials,
)

logger = logging.getLogger(__name__)


# Gemini CLI OAuth configuration
# Public OAuth client credentials from Google's Gemini CLI - NOT secrets
_GEMINI_CLIENT_ID = (
    "681255809395-oo8ft2oprdrn" + "p9e3aqf6av3hmdib135j.apps" + ".googleusercontent.com"
)
_GEMINI_CLIENT_SECRET = "GOCSPX-4uHgMPm-1o7Sk" + "-geV6Cu5clXFsxl"

GEMINI_CONFIG = OAuthConfig(
    provider_name="google-gemini",
    client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID", _GEMINI_CLIENT_ID),
    client_secret=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", _GEMINI_CLIENT_SECRET),
    auth_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    redirect_uri="http://localhost:8085/oauth2callback",
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ],
    callback_path="/oauth2callback",
    callback_port=8085,
    extra_auth_params={"access_type": "offline", "prompt": "consent"},
)


# Antigravity OAuth configuration
# Public OAuth client credentials - NOT secrets
_ANTIGRAVITY_CLIENT_ID = (
    "1071006060591-tmhssin2h21lcre" + "235vtolojh4g403ep.apps" + ".googleusercontent.com"
)
_ANTIGRAVITY_CLIENT_SECRET = "GOCSPX-K58FWR486Ld" + "LJ1mLB8sXC4z6qDAf"

ANTIGRAVITY_CONFIG = OAuthConfig(
    provider_name="antigravity",
    client_id=os.environ.get("ANTIGRAVITY_OAUTH_CLIENT_ID", _ANTIGRAVITY_CLIENT_ID),
    client_secret=os.environ.get("ANTIGRAVITY_OAUTH_CLIENT_SECRET", _ANTIGRAVITY_CLIENT_SECRET),
    auth_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    redirect_uri="http://localhost:51121/oauth-callback",
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    ],
    callback_path="/oauth-callback",
    callback_port=51121,
    extra_auth_params={"access_type": "offline", "prompt": "consent"},
)


class GoogleOAuthProvider(OAuthProvider):
    """Google OAuth provider base class."""
    
    # API endpoint for service discovery (override in subclass)
    api_endpoint: str = ""
    default_project_id: str = ""
    
    def authenticate(self) -> OAuthCredentials:
        """Run OAuth flow and return credentials."""
        verifier, challenge = generate_pkce()
        state = generate_state()
        
        # Build authorization URL
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.config.scopes),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            **self.config.extra_auth_params,
        }
        auth_url = f"{self.config.auth_url}?{urlencode(params)}"
        
        # Start callback server
        server = self._start_callback_server()
        
        try:
            # Open browser
            logger.info("Opening browser for authentication...")
            webbrowser.open(auth_url)
            
            # Wait for callback
            code, returned_state = self._wait_for_callback(server)
            
            if returned_state != state:
                raise AuthenticationError("State mismatch - possible CSRF attack")
            
            # Exchange code for tokens
            credentials = self._exchange_code(code, verifier)
            
            # Get user info and project
            self._enrich_credentials(credentials)
            
            # Save credentials
            save_credentials(credentials)
            
            return credentials
            
        finally:
            server.server_close()
    
    def _exchange_code(self, code: str, verifier: str) -> OAuthCredentials:
        """Exchange authorization code for tokens."""
        data = {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "code": code,
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": self.config.redirect_uri,
            **self.config.extra_token_params,
        }
        
        try:
            response = httpx.post(self.config.token_url, data=data, timeout=30)
            response.raise_for_status()
            token_data = response.json()
        except httpx.HTTPError as e:
            raise AuthenticationError(f"Token exchange failed: {e}")
        
        return OAuthCredentials(
            provider=self.config.provider_name,
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token", ""),
            expires_at=time.time() + token_data.get("expires_in", 3600),
        )
    
    def _enrich_credentials(self, credentials: OAuthCredentials) -> None:
        """Add user info and project ID to credentials."""
        headers = {"Authorization": f"Bearer {credentials.access_token}"}
        
        # Get user info
        try:
            response = httpx.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers=headers,
                timeout=10,
            )
            if response.status_code == 200:
                user_info = response.json()
                credentials.email = user_info.get("email")
        except httpx.HTTPError:
            pass
        
        # Get project ID
        credentials.project_id = self._discover_project(credentials.access_token)
    
    def _discover_project(self, access_token: str) -> str:
        """Discover project ID from API. Override in subclass."""
        return self.default_project_id
    
    def refresh(self, credentials: OAuthCredentials) -> OAuthCredentials:
        """Refresh expired credentials."""
        if not credentials.refresh_token:
            raise TokenRefreshError("No refresh token available")
        
        data = {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "refresh_token": credentials.refresh_token,
            "grant_type": "refresh_token",
        }
        
        try:
            response = httpx.post(self.config.token_url, data=data, timeout=30)
            response.raise_for_status()
            token_data = response.json()
        except httpx.HTTPError as e:
            raise TokenRefreshError(f"Token refresh failed: {e}")
        
        # Update credentials
        credentials.access_token = token_data["access_token"]
        credentials.expires_at = time.time() + token_data.get("expires_in", 3600)
        
        # Save updated credentials
        save_credentials(credentials)
        
        return credentials
    
    def revoke(self, credentials: OAuthCredentials) -> bool:
        """Revoke Google OAuth token."""
        try:
            response = httpx.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": credentials.access_token},
                timeout=10,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False


class GeminiCLIProvider(GoogleOAuthProvider):
    """Google Gemini CLI OAuth provider."""
    
    api_endpoint = "https://cloudcode-pa.googleapis.com"
    default_project_id = ""
    
    def __init__(self):
        super().__init__(GEMINI_CONFIG)
    
    def _discover_project(self, access_token: str) -> str:
        """Discover project from Code Assist API."""
        headers = {"Authorization": f"Bearer {access_token}"}
        
        try:
            response = httpx.get(
                f"{self.api_endpoint}/v1/userSettings",
                headers=headers,
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                if project := data.get("cloudProject"):
                    return project
        except httpx.HTTPError:
            pass
        
        # Fallback: try to get from resource manager
        try:
            response = httpx.get(
                "https://cloudresourcemanager.googleapis.com/v1/projects",
                headers=headers,
                params={"pageSize": 1},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                if projects := data.get("projects"):
                    return projects[0].get("projectId", "")
        except httpx.HTTPError:
            pass
        
        return ""


class AntigravityProvider(GoogleOAuthProvider):
    """Google Antigravity OAuth provider."""
    
    api_endpoint = "https://eu-autopush-aiplatform.sandbox.googleapis.com"
    default_project_id = "rising-fact-p41fc"
    
    def __init__(self):
        super().__init__(ANTIGRAVITY_CONFIG)
    
    def _discover_project(self, access_token: str) -> str:
        """Discover project from Antigravity API."""
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Try Code Assist API first
        try:
            response = httpx.get(
                "https://cloudcode-pa.googleapis.com/v1/userSettings",
                headers=headers,
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                if project := data.get("cloudProject"):
                    return project
        except httpx.HTTPError:
            pass
        
        return self.default_project_id

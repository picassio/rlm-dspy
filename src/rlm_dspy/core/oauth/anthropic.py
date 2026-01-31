"""Anthropic OAuth provider (Claude Pro/Max via claude.ai).

Implements OAuth flow for Anthropic Claude models using the same
credentials as Claude Code.
"""

from __future__ import annotations

import base64
import logging
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


# Anthropic OAuth configuration (from Claude Code)
# These are public OAuth client credentials - NOT secrets
_ANTHROPIC_CLIENT_ID = base64.b64decode(
    "OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl"
).decode()

ANTHROPIC_CONFIG = OAuthConfig(
    provider_name="anthropic",
    client_id=_ANTHROPIC_CLIENT_ID,
    client_secret="",  # Anthropic uses PKCE, no client secret
    auth_url="https://claude.ai/oauth/authorize",
    token_url="https://console.anthropic.com/v1/oauth/token",
    redirect_uri="https://console.anthropic.com/oauth/code/callback",
    scopes=["org:create_api_key", "user:profile", "user:inference"],
    callback_path="/oauth/code/callback",
    callback_port=0,  # Uses redirect to console.anthropic.com
    extra_auth_params={},
)


class AnthropicProvider(OAuthProvider):
    """Anthropic OAuth provider for Claude Pro/Max."""
    
    def __init__(self):
        super().__init__(ANTHROPIC_CONFIG)
    
    def authenticate(self) -> OAuthCredentials:
        """Run OAuth flow and return credentials.
        
        Note: This uses a manual code entry flow since Anthropic
        redirects to console.anthropic.com, not localhost.
        """
        verifier, challenge = generate_pkce()
        
        # Build authorization URL (use verifier as state for simplicity)
        params = {
            "code": "true",
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": verifier,
        }
        
        auth_url = f"{self.config.auth_url}?{urlencode(params)}"
        
        # Open browser
        print("\n" + "=" * 60)
        print("Anthropic OAuth Login (Claude Pro/Max)")
        print("=" * 60)
        print("\nOpening browser for authentication...")
        print(f"\nIf browser doesn't open, visit:\n{auth_url}\n")
        
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass
        
        # Ask user to paste the authorization code
        print("After authorizing, you'll see a page with an authorization code.")
        print("Copy the FULL code (format: CODE#STATE) and paste it below.\n")
        
        try:
            auth_code = input("Authorization code: ").strip()
        except (KeyboardInterrupt, EOFError):
            raise AuthenticationError("Authentication cancelled by user")
        
        if not auth_code:
            raise AuthenticationError("No authorization code provided")
        
        # Parse code and state (format: CODE#STATE)
        if "#" in auth_code:
            code, state = auth_code.split("#", 1)
        else:
            code = auth_code
            state = verifier
        
        # Exchange code for tokens
        credentials = self._exchange_code(code, state, verifier)
        
        # Save credentials
        save_credentials(credentials)
        
        print(f"\n✓ Authenticated with Anthropic!")
        return credentials
    
    def _exchange_code(self, code: str, state: str, verifier: str) -> OAuthCredentials:
        """Exchange authorization code for tokens."""
        # Anthropic expects JSON payload, not form-encoded
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.config.client_id,
            "code": code,
            "state": state,
            "redirect_uri": self.config.redirect_uri,
            "code_verifier": verifier,
        }
        
        try:
            response = httpx.post(
                self.config.token_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            
            if response.status_code != 200:
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error_description", error_data.get("error", error_msg))
                except Exception:
                    pass
                raise AuthenticationError(f"Token exchange failed: {error_msg}")
            
            token_data = response.json()
        except httpx.HTTPError as e:
            raise AuthenticationError(f"Token exchange failed: {e}")
        
        # Create credentials (with 5 min buffer like old code)
        expires_in = token_data.get("expires_in", 3600)
        
        return OAuthCredentials(
            provider=self.config.provider_name,
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token", ""),
            expires_at=time.time() + expires_in - 300,
        )
    
    def refresh(self, credentials: OAuthCredentials) -> OAuthCredentials:
        """Refresh expired credentials."""
        if not credentials.refresh_token:
            raise TokenRefreshError("No refresh token available")
        
        # Anthropic expects JSON payload
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.config.client_id,
            "refresh_token": credentials.refresh_token,
        }
        
        try:
            response = httpx.post(
                self.config.token_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            
            if response.status_code != 200:
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error_description", error_data.get("error", error_msg))
                except Exception:
                    pass
                raise TokenRefreshError(f"Token refresh failed: {error_msg}")
            
            token_data = response.json()
        except httpx.HTTPError as e:
            raise TokenRefreshError(f"Token refresh failed: {e}")
        
        # Update credentials (with 5 min buffer)
        expires_in = token_data.get("expires_in", 3600)
        credentials.access_token = token_data["access_token"]
        credentials.expires_at = time.time() + expires_in - 300
        
        # Update refresh token if provided
        if new_refresh := token_data.get("refresh_token"):
            credentials.refresh_token = new_refresh
        
        # Save updated credentials
        save_credentials(credentials)
        
        logger.info("Refreshed Anthropic OAuth token")
        
        return credentials
    
    def revoke(self, credentials: OAuthCredentials) -> bool:
        """Revoke Anthropic OAuth token.
        
        Note: Anthropic may not support token revocation.
        """
        # Anthropic doesn't have a public revocation endpoint
        # Just return True to allow local cleanup
        return True


def is_oauth_token(token: str) -> bool:
    """Check if a token is an Anthropic OAuth token."""
    return token.startswith("sk-ant-oat")

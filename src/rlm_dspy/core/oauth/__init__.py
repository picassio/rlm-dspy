"""Unified OAuth authentication for LLM providers.

This module provides a consistent interface for OAuth flows across providers:
- Google Gemini CLI
- Google Antigravity
- Generic OAuth (future providers)

Example:
    from rlm_dspy.core.oauth import get_provider, authenticate, get_credentials
    
    # Authenticate with a provider
    creds = authenticate("google-gemini")
    
    # Get existing credentials
    creds = get_credentials("antigravity")
    
    # Check if authenticated
    if is_authenticated("google-gemini"):
        print("Ready to use Gemini")
"""

from .base import (
    OAuthCredentials,
    OAuthConfig,
    OAuthProvider,
    OAuthError,
    AuthenticationError,
    TokenRefreshError,
    # Utility functions for tests/backward compat
    generate_pkce,
    save_credentials,
    load_credentials,
    delete_credentials,
    OAUTH_DIR,
    CREDENTIALS_FILE,
)
from .manager import (
    get_provider,
    get_credentials,
    authenticate,
    refresh_credentials,
    revoke_credentials,
    is_authenticated,
    list_providers,
    list_authenticated,
)

# Backward compatibility aliases
_generate_pkce = generate_pkce
_save_credentials = save_credentials
_load_credentials = load_credentials
_delete_credentials = delete_credentials


def get_oauth_token(provider: str) -> str | None:
    """Get OAuth access token for a provider (backward compatibility).
    
    Args:
        provider: Provider name
        
    Returns:
        Access token if authenticated, None otherwise
    """
    creds = get_credentials(provider)
    if creds and not creds.is_expired:
        return creds.access_token
    return None


def is_anthropic_authenticated() -> bool:
    """Check if authenticated with Anthropic (backward compatibility).
    
    Note: Anthropic OAuth is no longer supported. Always returns False.
    """
    return False


def oauth_status(provider: str) -> dict:
    """Get OAuth status for a provider (backward compatibility).
    
    Returns:
        Dict with authenticated, is_expired, expires_at keys
    """
    creds = get_credentials(provider)
    if creds:
        return {
            "provider": provider,
            "authenticated": True,
            "is_expired": creds.is_expired,
            "expires_at": creds.expires_at,
            "email": creds.email,
            "project_id": creds.project_id,
        }
    return {
        "provider": provider,
        "authenticated": False,
        "is_expired": False,
        "expires_at": 0,
    }


def oauth_login(provider: str, open_browser: bool = True) -> OAuthCredentials:
    """Login with OAuth (backward compatibility)."""
    return authenticate(provider)


def oauth_logout(provider: str) -> bool:
    """Logout from OAuth (backward compatibility)."""
    return revoke_credentials(provider)


# Providers list for backward compat
OAUTH_PROVIDERS = ["google-gemini", "antigravity"]


__all__ = [
    # Base classes
    "OAuthCredentials",
    "OAuthConfig", 
    "OAuthProvider",
    "OAuthError",
    "AuthenticationError",
    "TokenRefreshError",
    # Utility functions
    "generate_pkce",
    "save_credentials",
    "load_credentials",
    "delete_credentials",
    "OAUTH_DIR",
    "CREDENTIALS_FILE",
    # Backward compat
    "_generate_pkce",
    "_save_credentials",
    "_load_credentials",
    "_delete_credentials",
    "get_oauth_token",
    "is_anthropic_authenticated",
    "oauth_status",
    "oauth_login",
    "oauth_logout",
    "OAUTH_PROVIDERS",
    # Manager functions
    "get_provider",
    "get_credentials",
    "authenticate",
    "refresh_credentials",
    "revoke_credentials",
    "is_authenticated",
    "list_providers",
    "list_authenticated",
]

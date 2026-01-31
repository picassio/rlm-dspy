"""Unified OAuth authentication for LLM providers.

This module provides a consistent interface for OAuth flows across providers:
- Google Gemini CLI
- Google Antigravity
- Anthropic (Claude Pro/Max)

Example:
    from rlm_dspy.core.oauth import authenticate, get_credentials, is_authenticated
    
    # Authenticate with a provider
    creds = authenticate("google-gemini")
    
    # Get existing credentials (auto-refreshes if expired)
    creds = get_credentials("antigravity")
    
    # Check if authenticated
    if is_authenticated("google-gemini"):
        print("Ready to use Gemini")
    
    # Get token for API calls
    token, project = get_google_token()
"""

from .base import (
    OAuthCredentials,
    OAuthConfig,
    OAuthProvider,
    OAuthError,
    AuthenticationError,
    TokenRefreshError,
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


# =============================================================================
# Provider-specific token helpers (used by LM classes)
# =============================================================================

def get_google_token() -> tuple[str, str] | None:
    """Get valid Google Gemini CLI OAuth token and project ID.
    
    Auto-refreshes if expired.
    
    Returns:
        Tuple of (access_token, project_id) or None if not authenticated
    """
    creds = get_credentials("google-gemini")
    if creds and not creds.is_expired:
        return creds.access_token, creds.project_id
    return None


def get_antigravity_token() -> tuple[str, str] | None:
    """Get valid Antigravity OAuth token and project ID.
    
    Auto-refreshes if expired.
    
    Returns:
        Tuple of (access_token, project_id) or None if not authenticated
    """
    creds = get_credentials("antigravity")
    if creds and not creds.is_expired:
        return creds.access_token, creds.project_id
    return None


def is_google_authenticated() -> bool:
    """Check if authenticated with Google Gemini CLI OAuth."""
    return is_authenticated("google-gemini")


def is_antigravity_authenticated() -> bool:
    """Check if authenticated with Antigravity OAuth."""
    return is_authenticated("antigravity")


def get_anthropic_token() -> str | None:
    """Get valid Anthropic OAuth token.
    
    Auto-refreshes if expired.
    
    Returns:
        Access token or None if not authenticated
    """
    creds = get_credentials("anthropic")
    if creds and not creds.is_expired:
        return creds.access_token
    return None


def is_anthropic_authenticated() -> bool:
    """Check if authenticated with Anthropic OAuth."""
    return is_authenticated("anthropic")


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
    # Manager functions
    "get_provider",
    "get_credentials",
    "authenticate",
    "refresh_credentials",
    "revoke_credentials",
    "is_authenticated",
    "list_providers",
    "list_authenticated",
    # Provider-specific helpers
    "get_google_token",
    "get_antigravity_token",
    "get_anthropic_token",
    "is_google_authenticated",
    "is_antigravity_authenticated",
    "is_anthropic_authenticated",
]

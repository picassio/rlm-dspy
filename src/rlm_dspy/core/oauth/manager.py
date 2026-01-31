"""OAuth provider manager.

Provides a unified interface for managing OAuth providers and credentials.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import (
    OAuthCredentials,
    OAuthProvider,
    OAuthError,
    AuthenticationError,
    TokenRefreshError,
    load_credentials,
    delete_credentials,
    CREDENTIALS_FILE,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Provider registry
_providers: dict[str, type[OAuthProvider]] = {}


def _register_builtin_providers() -> None:
    """Register built-in providers."""
    global _providers
    
    if _providers:
        return  # Already registered
    
    from .google import GeminiCLIProvider, AntigravityProvider
    
    _providers = {
        "google-gemini": GeminiCLIProvider,
        "gemini": GeminiCLIProvider,  # Alias
        "antigravity": AntigravityProvider,
    }


def get_provider(name: str) -> OAuthProvider:
    """Get an OAuth provider by name.
    
    Args:
        name: Provider name (e.g., "google-gemini", "antigravity")
        
    Returns:
        OAuth provider instance
        
    Raises:
        ValueError: If provider not found
    """
    _register_builtin_providers()
    
    name_lower = name.lower()
    if name_lower not in _providers:
        available = ", ".join(sorted(set(_providers.keys())))
        raise ValueError(f"Unknown OAuth provider: {name}. Available: {available}")
    
    return _providers[name_lower]()


def list_providers() -> list[str]:
    """List available OAuth providers.
    
    Returns:
        List of provider names
    """
    _register_builtin_providers()
    # Return unique names (not aliases)
    return sorted(set(p().name for p in _providers.values()))


def get_credentials(provider: str) -> OAuthCredentials | None:
    """Get credentials for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        Credentials if found and valid, None otherwise
    """
    # Normalize provider name
    try:
        provider_instance = get_provider(provider)
        provider = provider_instance.name
    except ValueError:
        pass
    
    # Try primary name first
    creds = load_credentials(provider)
    
    # Try aliases if not found
    if not creds:
        aliases = {
            "google-gemini": ["google", "gemini"],
            "antigravity": ["antigravity"],
        }
        for alias in aliases.get(provider, []):
            creds = load_credentials(alias)
            if creds:
                break
    
    if not creds:
        return None
    
    # Auto-refresh if expired
    if creds.is_expired and creds.refresh_token:
        try:
            provider_instance = get_provider(provider)
            creds = provider_instance.refresh(creds)
            logger.info("Refreshed expired credentials for %s", provider)
        except (OAuthError, ValueError) as e:
            logger.warning("Failed to refresh credentials: %s", e)
            return None
    
    return creds


def authenticate(provider: str) -> OAuthCredentials:
    """Authenticate with a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        OAuth credentials
        
    Raises:
        AuthenticationError: If authentication fails
        ValueError: If provider not found
    """
    provider_instance = get_provider(provider)
    return provider_instance.authenticate()


def refresh_credentials(provider: str) -> OAuthCredentials:
    """Refresh credentials for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        Updated credentials
        
    Raises:
        TokenRefreshError: If refresh fails
        ValueError: If provider not found or no credentials
    """
    creds = load_credentials(provider)
    if not creds:
        raise ValueError(f"No credentials found for {provider}")
    
    provider_instance = get_provider(provider)
    return provider_instance.refresh(creds)


def revoke_credentials(provider: str) -> bool:
    """Revoke and delete credentials for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        True if revoked, False if failed or not found
    """
    # Normalize provider name
    try:
        provider_instance = get_provider(provider)
        provider = provider_instance.name
    except ValueError:
        pass
    
    creds = load_credentials(provider)
    if not creds:
        return False
    
    # Try to revoke with provider
    try:
        provider_instance = get_provider(provider)
        provider_instance.revoke(creds)
    except (OAuthError, ValueError):
        pass
    
    # Delete local credentials
    return delete_credentials(provider)


def is_authenticated(provider: str) -> bool:
    """Check if authenticated with a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        True if valid credentials exist
    """
    creds = get_credentials(provider)
    return creds is not None and not creds.is_expired


def list_authenticated() -> list[str]:
    """List providers with valid credentials.
    
    Returns:
        List of authenticated provider names
    """
    import json
    
    if not CREDENTIALS_FILE.exists():
        return []
    
    try:
        all_creds = json.loads(CREDENTIALS_FILE.read_text())
        return [
            name for name in all_creds.keys()
            if is_authenticated(name)
        ]
    except (json.JSONDecodeError, OSError):
        return []

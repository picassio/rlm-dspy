"""Anthropic LM with OAuth token support.

This module provides a custom DSPy LM that supports:
1. Regular API keys (sk-ant-api...)
2. OAuth tokens (sk-ant-oat...) from Claude Pro/Max subscriptions

The OAuth flow mimics Claude Code's authentication to use the same
API endpoint and features.
"""

from __future__ import annotations

import logging
from typing import Any

import dspy

logger = logging.getLogger(__name__)

# Claude Code version for stealth mode
CLAUDE_CODE_VERSION = "2.1.2"

# Headers that mimic Claude Code when using OAuth
CLAUDE_CODE_HEADERS = {
    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14",
    "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
    "x-app": "cli",
}


def is_oauth_token(api_key: str) -> bool:
    """Check if the API key is an OAuth token."""
    return api_key.startswith("sk-ant-oat")


def create_anthropic_lm(
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
    **kwargs,
) -> dspy.LM:
    """Create a DSPy LM for Anthropic with OAuth support.
    
    Args:
        model: Model ID (e.g., "claude-sonnet-4-20250514")
        api_key: API key or OAuth token. If None, tries to get from:
                 1. ANTHROPIC_OAUTH_TOKEN env var
                 2. Stored OAuth credentials
                 3. ANTHROPIC_API_KEY env var
        **kwargs: Additional arguments for dspy.LM
        
    Returns:
        Configured DSPy LM
    """
    import os
    from .oauth import get_anthropic_token
    
    # Resolve API key
    if api_key is None:
        # Try OAuth token first
        api_key = os.environ.get("ANTHROPIC_OAUTH_TOKEN")
        
        if not api_key:
            # Try stored OAuth credentials
            api_key = get_anthropic_token()
        
        if not api_key:
            # Fall back to regular API key
            api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError(
            "No Anthropic API key or OAuth token found. "
            "Run 'rlm-dspy auth login anthropic' or set ANTHROPIC_API_KEY"
        )
    
    # Check if OAuth token
    if is_oauth_token(api_key):
        logger.info("Using Anthropic OAuth token (Claude Pro/Max)")
        return _create_oauth_lm(model, api_key, **kwargs)
    else:
        logger.debug("Using Anthropic API key")
        return dspy.LM(f"anthropic/{model}", api_key=api_key, **kwargs)


def _create_oauth_lm(model: str, oauth_token: str, **kwargs) -> dspy.LM:
    """Create LM using OAuth token with Claude Code headers.
    
    This requires using the Anthropic SDK directly since LiteLLM
    doesn't support OAuth tokens.
    """
    # For OAuth tokens, we need to use the anthropic provider
    # and inject the special headers
    
    # Create LM with custom headers
    lm = dspy.LM(
        f"anthropic/{model}",
        api_key=oauth_token,
        # These headers make it work like Claude Code
        extra_headers=CLAUDE_CODE_HEADERS,
        **kwargs,
    )
    
    return lm


def get_anthropic_api_key() -> str | None:
    """Get the best available Anthropic API key/token.
    
    Priority:
    1. ANTHROPIC_OAUTH_TOKEN env var
    2. Stored OAuth credentials (auto-refreshed)
    3. ANTHROPIC_API_KEY env var
    
    Returns:
        API key/token or None
    """
    import os
    from .oauth import get_anthropic_token
    
    # Try OAuth token first
    token = os.environ.get("ANTHROPIC_OAUTH_TOKEN")
    if token:
        return token
    
    # Try stored OAuth credentials
    token = get_anthropic_token()
    if token:
        return token
    
    # Fall back to regular API key
    return os.environ.get("ANTHROPIC_API_KEY")


def create_lm_with_oauth_fallback(model: str, api_key: str | None = None, **kwargs) -> dspy.LM:
    """Create a DSPy LM with automatic OAuth fallback for Anthropic.
    
    For Anthropic models, this will:
    1. Check for OAuth tokens
    2. Use OAuth authentication if available
    3. Fall back to API key if no OAuth
    
    For other providers, creates a standard LM.
    
    Args:
        model: Model string (e.g., "anthropic/claude-3-opus", "openrouter/...")
        api_key: Optional API key
        **kwargs: Additional LM arguments
        
    Returns:
        Configured DSPy LM
    """
    # Check if Anthropic model
    if model.startswith("anthropic/"):
        # Extract model ID
        model_id = model.split("/", 1)[1]
        
        # Try to get OAuth token
        from .oauth import get_anthropic_token
        oauth_token = get_anthropic_token()
        
        if oauth_token and not api_key:
            logger.info("Using Anthropic OAuth token")
            return create_anthropic_lm(model_id, api_key=oauth_token, **kwargs)
    
    # Standard LM creation
    return dspy.LM(model, api_key=api_key, **kwargs)

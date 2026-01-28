"""Secret masking utilities.

Learned from modaic: automatic secret cleaning for safe logging/saving.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

logger = logging.getLogger(__name__)

SECRET_MASK = "********"

# Common secret keys to auto-mask
COMMON_SECRETS = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "openai_api_key",
        "anthropic_api_key",
        "openrouter_api_key",
        "hf_token",
        "huggingface_token",
        "access_token",
        "secret_key",
        "secret",
        "password",
        "passwd",
        "token",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "credentials",
    }
)


class MissingSecretError(Exception):
    """Raised when a required secret is not found."""

    def __init__(self, key: str, env_var: str | None = None):
        self.key = key
        self.env_var = env_var
        msg = f"Missing required secret: {key}"
        if env_var:
            msg += f" (set via {env_var} environment variable)"
        super().__init__(msg)


def is_secret_key(key: str) -> bool:
    """Check if a key name looks like a secret."""
    key_lower = key.lower().replace("-", "_")
    return key_lower in COMMON_SECRETS or any(
        s in key_lower for s in ("key", "token", "secret", "password", "credential")
    )


def mask_value(value: Any, reveal_prefix: bool = False) -> str:
    """Mask a secret value for safe display.

    Args:
        value: The secret value to mask
        reveal_prefix: If True, shows first 4 chars (use sparingly, e.g., for debugging).
                      Default False for security - fully masks the value.

    Security: By default, fully masks secrets to prevent entropy exposure.
    Even partial reveals help attackers narrow down the keyspace.
    """
    if value is None:
        return "[None]"
    s = str(value)
    if len(s) == 0:
        return "[empty]"
    
    # Full masking by default for security
    if not reveal_prefix:
        return SECRET_MASK
    
    # Optional prefix reveal (use only when necessary for debugging)
    if len(s) <= 8:
        return SECRET_MASK
    return f"{s[:4]}{'*' * min(8, len(s) - 4)}"


def clean_secrets(data: dict[str, Any], in_place: bool = False) -> dict[str, Any]:
    """
    Remove or mask secrets from a dictionary.

    Learned from modaic's _clean_secrets pattern:
    - Recursively walks nested dicts
    - Masks values for secret-looking keys
    - Safe for logging/saving to disk

    Args:
        data: Dictionary to clean
        in_place: If True, modify original dict

    Returns:
        Cleaned dictionary
    """
    if not in_place:
        data = _deep_copy_dict(data)

    return _clean_secrets_recursive(data)


def _deep_copy_dict(d: dict) -> dict:
    """Simple deep copy for dicts."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = [_deep_copy_dict(i) if isinstance(i, dict) else i for i in v]
        else:
            result[k] = v
    return result


def _clean_secrets_recursive(data: dict) -> dict:
    """Recursively clean secrets from dict."""
    for key, value in list(data.items()):
        if is_secret_key(key):
            if value and value != SECRET_MASK:
                data[key] = SECRET_MASK
        elif isinstance(value, dict):
            _clean_secrets_recursive(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _clean_secrets_recursive(item)
    return data


def inject_secrets(
    data: dict[str, Any],
    secrets: dict[str, str] | None = None,
    env_prefix: str = "RLM_",
) -> dict[str, Any]:
    """
    Inject secrets from environment or provided dict.

    Learned from modaic's _get_state_with_secrets pattern:
    - Re-injects secrets that were masked
    - Falls back to environment variables

    Args:
        data: Dictionary with masked secrets
        secrets: Optional dict of secrets to inject
        env_prefix: Prefix for environment variables

    Returns:
        Dictionary with secrets injected
    """
    secrets = secrets or {}

    for key, value in data.items():
        if value == SECRET_MASK:
            # Try provided secrets first
            if key in secrets:
                data[key] = secrets[key]
            else:
                # Try environment variable
                env_key = f"{env_prefix}{key.upper()}"
                env_value = os.environ.get(env_key)
                if env_value:
                    data[key] = env_value
                else:
                    logger.warning(f"Secret {key} is masked but no value provided")
        elif isinstance(value, dict):
            inject_secrets(value, secrets, env_prefix)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    inject_secrets(item, secrets, env_prefix)

    return data


def check_for_exposed_secrets(data: dict[str, Any], source: str = "config") -> None:
    """
    Warn if secrets appear to be stored in plain text.

    Args:
        data: Dictionary to check
        source: Description of where data came from
    """
    exposed = []

    def _check(d: dict, path: str = ""):
        for key, value in d.items():
            current_path = f"{path}.{key}" if path else key
            if is_secret_key(key) and value and value != SECRET_MASK:
                if isinstance(value, str) and len(value) > 10:
                    exposed.append(current_path)
            elif isinstance(value, dict):
                _check(value, current_path)

    _check(data)

    if exposed:
        warnings.warn(
            f"Potential secrets exposed in {source}: {', '.join(exposed)}. "
            "Consider using environment variables instead.",
            UserWarning,
            stacklevel=2,
        )


def get_api_key(
    key_name: str = "api_key",
    env_vars: list[str] | None = None,
    required: bool = True,
) -> str | None:
    """
    Get API key from environment with fallbacks.

    Args:
        key_name: Name of the key (for error messages)
        env_vars: List of env var names to try, in order
        required: If True, raise error when not found

    Returns:
        API key string or None

    Raises:
        MissingSecretError: If required and not found
    """
    if env_vars is None:
        env_vars = [
            "RLM_API_KEY",
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ]

    for env_var in env_vars:
        value = os.environ.get(env_var)
        if value:
            return value

    if required:
        raise MissingSecretError(key_name, env_vars[0] if env_vars else None)

    return None

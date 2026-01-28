"""Configuration utilities with atomic writes and secure defaults.

Patterns learned from microcode and llm-tldr:
- Atomic file writes (tempfile + os.replace)
- Secure permissions for sensitive files
- Hierarchical config resolution (CLI > Env > Cache > Defaults)
- Graceful degradation with fallbacks
"""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_config_dir() -> Path:
    """Get the RLM config directory.

    Uses XDG_CONFIG_HOME if set, otherwise ~/.config/rlm-dspy
    """
    if xdg := os.environ.get("XDG_CONFIG_HOME"):
        return Path(xdg) / "rlm-dspy"
    return Path.home() / ".config" / "rlm-dspy"


def atomic_write_json(path: Path | str, data: dict[str, Any], secure: bool = False) -> None:
    """Write JSON atomically to prevent corruption.

    Uses tempfile + os.replace pattern for crash-safe writes.

    Args:
        path: Destination path
        data: Data to write as JSON
        secure: If True, set file permissions to 0o600 (owner only)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)

        # Set permissions before moving (if secure)
        if secure:
            os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600

        # Atomic replace
        os.replace(tmp_path, path)
        logger.debug(f"Wrote config to {path}")

    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError as cleanup_err:
            logger.debug("Failed to remove temp file during cleanup: %s", cleanup_err)
        raise


def atomic_read_json(path: Path | str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read JSON file with fallback to default.

    Args:
        path: Path to read
        default: Default value if file doesn't exist or is invalid

    Returns:
        Parsed JSON or default
    """
    path = Path(path)
    if not path.exists():
        return default or {}

    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read {path}: {e}")
        return default or {}


@dataclass
class ConfigResolver:
    """Hierarchical configuration resolver.

    Resolution order (highest priority first):
    1. Explicit values (passed to get())
    2. Environment variables
    3. Cached/persisted values
    4. Default values

    Example:
        resolver = ConfigResolver(
            env_prefix="RLM_",
            cache_path=Path("~/.cache/rlm-dspy/config.json")
        )

        # Gets from env RLM_MODEL, then cache, then default
        model = resolver.get("MODEL", default="gpt-4")
    """

    env_prefix: str = "RLM_"
    cache_path: Path | None = None
    _cache: dict[str, Any] = field(default_factory=dict)
    _loaded: bool = False

    def _ensure_loaded(self) -> None:
        """Lazy load cache on first access."""
        if self._loaded:
            return
        if self.cache_path:
            self._cache = atomic_read_json(self.cache_path)
        self._loaded = True

    def get(
        self,
        key: str,
        default: T = None,
        explicit: T | None = None,
        type_: type[T] | None = None,
    ) -> T:
        """Get configuration value with hierarchical resolution.

        Args:
            key: Config key (will be prefixed for env lookup)
            default: Default value if not found anywhere
            explicit: Explicit override (highest priority)
            type_: Type to coerce value to (str, int, float, bool)

        Returns:
            Resolved value
        """
        # 1. Explicit override
        if explicit is not None:
            return explicit

        # 2. Environment variable
        env_key = f"{self.env_prefix}{key}"
        if env_val := os.environ.get(env_key):
            return self._coerce(env_val, type_ or type(default) if default is not None else str)

        # 3. Cached value
        self._ensure_loaded()
        if key in self._cache:
            return self._cache[key]

        # 4. Default
        return default

    def _coerce(self, value: str, type_: type) -> Any:
        """Coerce string value to target type."""
        if type_ is bool:
            return value.lower() in ("true", "1", "yes", "on")
        if type_ is int:
            return int(value)
        if type_ is float:
            return float(value)
        return value

    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """Set a configuration value.

        Args:
            key: Config key
            value: Value to set
            persist: If True, save to cache file
        """
        self._ensure_loaded()
        self._cache[key] = value

        if persist and self.cache_path:
            atomic_write_json(self.cache_path, self._cache)

    def clear(self, key: str | None = None, persist: bool = True) -> None:
        """Clear configuration value(s).

        Args:
            key: Specific key to clear, or None to clear all
            persist: If True, save to cache file
        """
        self._ensure_loaded()

        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()

        if persist and self.cache_path:
            atomic_write_json(self.cache_path, self._cache)


def format_user_error(error: Exception, context: str = "") -> str:
    """Format an exception as a user-friendly error message.

    Transforms low-level API errors into actionable instructions.

    Args:
        error: The exception to format
        context: Additional context about what was being attempted

    Returns:
        User-friendly error message
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # API authentication errors
    if "401" in error_str or "unauthorized" in error_str or "invalid api key" in error_str:
        return (
            "❌ Authentication failed. Please check your API key:\n"
            "   1. Verify RLM_API_KEY or OPENROUTER_API_KEY is set\n"
            "   2. Ensure the key hasn't expired\n"
            "   3. Check you have credits remaining"
        )

    # Rate limiting
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return (
            "⏳ Rate limited. Options:\n"
            "   1. Wait a moment and retry\n"
            "   2. Reduce RLM_PARALLEL_CHUNKS\n"
            "   3. Use a different model or provider"
        )

    # Context length
    if "context" in error_str and ("length" in error_str or "too long" in error_str):
        return (
            "📏 Context too long. Options:\n"
            "   1. Reduce RLM_CHUNK_SIZE\n"
            "   2. Use a model with larger context (e.g., gemini-1.5-pro)\n"
            "   3. Filter input to relevant files only"
        )

    # Model not found
    if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
        return (
            "🔍 Model not found. Check:\n"
            "   1. Model name spelling in RLM_MODEL\n"
            "   2. Provider prefix (e.g., openrouter/anthropic/claude-3)\n"
            "   3. Model availability on your provider"
        )

    # Network errors
    if "connection" in error_str or "timeout" in error_str or "network" in error_str:
        return (
            "🌐 Network error. Check:\n"
            "   1. Internet connection\n"
            "   2. RLM_API_BASE URL is correct\n"
            "   3. Firewall/proxy settings"
        )

    # Generic fallback
    prefix = f"Error during {context}: " if context else "Error: "
    return f"{prefix}{error_type}: {error}"


def inject_context(
    task: str,
    include_cwd: bool = True,
    include_time: bool = True,
    extra: dict[str, str] | None = None,
) -> str:
    """Inject system context into a task description.

    Automatically adds working directory, timestamp, and custom context.

    Args:
        task: The original task/query
        include_cwd: Include current working directory
        include_time: Include current timestamp
        extra: Additional context key-value pairs

    Returns:
        Task with injected context
    """
    from datetime import datetime

    lines = []

    if include_cwd:
        lines.append(f"Working Directory: {os.getcwd()}")

    if include_time:
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if extra:
        for key, value in extra.items():
            lines.append(f"{key}: {value}")

    if lines:
        context_block = "\n".join(lines)
        return f"## Context\n{context_block}\n\n## Task\n{task}"

    return task


__all__ = [
    "get_config_dir",
    "atomic_write_json",
    "atomic_read_json",
    "ConfigResolver",
    "format_user_error",
    "inject_context",
]

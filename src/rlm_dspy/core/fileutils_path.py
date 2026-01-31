"""Path utilities - validation, OS detection, caching."""

from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class PathTraversalError(Exception):
    """Raised when a path traversal attack is detected."""
    pass


def validate_path_safety(path: Path, base_dir: Path | None = None) -> Path:
    """Validate path is safe and resolve it.

    Args:
        path: Path to validate
        base_dir: Optional base directory to restrict to

    Returns:
        Resolved safe path

    Raises:
        PathTraversalError: If path escapes base_dir or is unsafe
    """
    resolved = path.resolve()

    if ".." in str(path):
        if base_dir:
            base_resolved = base_dir.resolve()
            try:
                resolved.relative_to(base_resolved)
            except ValueError:
                raise PathTraversalError(
                    f"Path '{path}' escapes base directory '{base_dir}'"
                )

    SENSITIVE_PATHS = [
        "/etc/passwd", "/etc/shadow", "/etc/sudoers",
        "~/.ssh", "~/.gnupg", "~/.aws",
    ]

    resolved_str = str(resolved)
    for sensitive in SENSITIVE_PATHS:
        sensitive_resolved = str(Path(sensitive).expanduser().resolve())
        if resolved_str.startswith(sensitive_resolved):
            raise PathTraversalError(
                f"Access to sensitive path '{sensitive}' is not allowed"
            )

    return resolved


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def is_macos() -> bool:
    """Check if running on macOS."""
    return sys.platform == "darwin"


def is_linux() -> bool:
    """Check if running on Linux."""
    return sys.platform.startswith("linux")


def get_cache_dir(app_name: str = "rlm_dspy") -> Path:
    """Get platform-specific cache directory."""
    if is_windows():
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif is_macos():
        base = Path.home() / "Library" / "Caches"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            base = Path(xdg_cache)
        else:
            base = Path.home() / ".cache"

    cache_dir = base / app_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def path_to_module(path: Path, root: Path | None = None) -> str:
    """Convert file path to Python module path.

    Args:
        path: Path to Python file
        root: Root directory to calculate relative module from

    Returns:
        Module path (e.g., 'package.module')
    """
    if root:
        try:
            rel_path = path.resolve().relative_to(root.resolve())
        except ValueError:
            rel_path = path
    else:
        rel_path = path

    parts = list(rel_path.parts)

    if parts and parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]
        if parts[-1] == '__init__':
            parts = parts[:-1]

    return '.'.join(parts)

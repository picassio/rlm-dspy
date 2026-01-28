"""Cross-platform file utilities.

Learned from modaic: smart linking, robust deletion, platform handling.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
    """
    Get platform-appropriate cache directory.

    Learned from modaic: platformdirs-style cache locations.

    Args:
        app_name: Application name for the cache folder

    Returns:
        Path to cache directory
    """
    if is_windows():
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / app_name / "Cache"
    elif is_macos():
        return Path.home() / "Library" / "Caches" / app_name
    else:
        # Linux and others: XDG spec
        xdg_cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return Path(xdg_cache) / app_name


def smart_link(source: Path, target: Path, force: bool = False) -> None:
    """
    Create a smart link (symlink on Unix, hardlink on Windows).

    Learned from modaic's smart_link pattern:
    - Symlinks on macOS/Linux (preserves structure)
    - Hardlinks on Windows (better compatibility)

    Args:
        source: Source path (must exist)
        target: Target path to create
        force: Remove existing target if it exists
    """
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    if target.exists():
        if force:
            if target.is_dir():
                smart_rmtree(target)
            else:
                target.unlink()
        else:
            raise FileExistsError(f"Target already exists: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)

    if is_windows():
        if source.is_dir():
            _recursive_hard_link(source, target)
        else:
            target.hardlink_to(source)
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())

    logger.debug("Linked %s -> %s", source, target)


def _recursive_hard_link(source: Path, target: Path) -> None:
    """
    Recursively create hard links for a directory.

    Windows doesn't support directory symlinks well, so we
    create hard links for each file.
    """
    target.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        dest = target / item.name
        if item.is_dir():
            _recursive_hard_link(item, dest)
        else:
            dest.hardlink_to(item)


def smart_rmtree(path: Path, aggressive: bool = False) -> bool:
    """
    Robustly remove a directory tree.

    Learned from modaic: handles Windows permission issues and locked files.

    Args:
        path: Directory to remove
        aggressive: On Windows, try to kill blocking processes

    Returns:
        True if successful
    """
    if not path.exists():
        return True

    # Try standard removal first
    try:
        shutil.rmtree(path)
        return True
    except (PermissionError, OSError) as e:
        logger.debug("Standard rmtree failed: %s", e)

    # Windows fallback
    if is_windows():
        if aggressive:
            _kill_blocking_processes(path)

        try:
            # Use cmd /c rmdir which can handle more cases
            result = subprocess.run(
                ["cmd", "/c", "rmdir", "/s", "/q", str(path)],
                capture_output=True,
                timeout=60,
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning("Windows rmdir failed: %s", e)

    # Unix fallback
    else:
        try:
            result = subprocess.run(
                ["rm", "-rf", str(path)],
                capture_output=True,
                timeout=60,
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning("rm -rf failed: %s", e)

    return False


def _kill_blocking_processes(path: Path) -> None:
    """
    Kill processes that might be blocking file deletion on Windows.

    Learned from modaic's aggressive cleanup for git.exe locks.
    """
    if not is_windows():
        return

    try:
        # Kill any git.exe processes that might be holding locks
        subprocess.run(
            ["taskkill", "/f", "/im", "git.exe"],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


def sync_directory(
    source: Path,
    target: Path,
    delete_extra: bool = True,
    ignore_times: bool = False,
) -> bool:
    """
    Synchronize directories using platform-appropriate tools.

    Learned from modaic: rsync on Unix, robocopy on Windows.

    Args:
        source: Source directory
        target: Target directory
        delete_extra: Delete files in target not in source
        ignore_times: Compare by content, not timestamps

    Returns:
        True if successful
    """
    target.mkdir(parents=True, exist_ok=True)

    if is_windows():
        cmd = ["robocopy", str(source), str(target), "/E"]
        if delete_extra:
            cmd.append("/MIR")
        # robocopy returns non-zero for success, 0-7 are OK
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        return result.returncode < 8
    else:
        cmd = ["rsync", "-a"]
        if delete_extra:
            cmd.append("--delete")
        if ignore_times:
            cmd.append("--ignore-times")
        cmd.extend([f"{source}/", str(target)])
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        return result.returncode == 0


def path_to_module(path: Path, root: Path | None = None) -> str:
    """
    Convert a file path to Python module notation.

    Learned from modaic: handles Windows backslashes.

    Args:
        path: File path (e.g., src/rlm_dspy/core/rlm.py)
        root: Root to make path relative to

    Returns:
        Module path (e.g., rlm_dspy.core.rlm)
    """
    if root:
        path = path.relative_to(root)

    # Remove .py extension
    if path.suffix == ".py":
        path = path.with_suffix("")

    # Convert to module notation
    parts = path.parts

    # Remove __init__ from the end
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write(
    path: Path,
    content: str | bytes,
    mode: str = "w",
    retries: int = 3,
    retry_delay: float = 0.1,
) -> None:
    """
    Atomically write to a file (write to temp, then rename).

    Prevents partial writes on crash. On Windows, includes retry logic
    for cases where the target file is temporarily locked.

    Args:
        path: Target file path
        content: Content to write
        mode: File mode ('w' for text, 'wb' for binary)
        retries: Number of retry attempts for rename (Windows)
        retry_delay: Delay between retries in seconds
    """
    import tempfile
    import time

    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (for atomic rename)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    temp = Path(temp_path)

    try:
        with os.fdopen(fd, mode) as f:
            f.write(content)

        # Retry loop for rename (helps with Windows file locking)
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                temp.replace(path)
                return  # Success
            except PermissionError as e:
                last_error = e
                if attempt < retries:
                    logger.debug(
                        "atomic_write: rename failed (attempt %d/%d), retrying...",
                        attempt + 1, retries + 1
                    )
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
            except OSError as e:
                # On Windows, file in use raises OSError
                if is_windows() and attempt < retries:
                    last_error = e
                    logger.debug(
                        "atomic_write: rename failed with OSError (attempt %d/%d), retrying...",
                        attempt + 1, retries + 1
                    )
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error

    except Exception:
        # Clean up temp file on any failure
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass  # Best effort cleanup
        raise

"""Cross-platform file utilities.

Learned from modaic: smart linking, robust deletion, platform handling.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathspec

logger = logging.getLogger(__name__)

# Pre-compiled regex for file marker extraction (used in smart_truncate_context)
_FILE_MARKER_PATTERN = re.compile(r'(=== FILE: .+? ===\n.*?=== END FILE ===\n)', re.DOTALL)


class PathTraversalError(Exception):
    """Raised when a path traversal attack is detected."""
    pass


def validate_path_safety(path: Path, base_dir: Path | None = None) -> Path:
    """
    Validate that a path is safe (no traversal attacks).

    Args:
        path: The path to validate
        base_dir: Optional base directory - path must be within this directory

    Returns:
        The resolved (absolute) path

    Raises:
        PathTraversalError: If path contains traversal sequences or escapes base_dir
    """
    # Resolve to absolute path
    resolved = path.resolve()

    # Check for traversal patterns in the original path string
    path_str = str(path)
    if ".." in path_str:
        logger.warning("Path traversal attempt detected: %s", path_str)
        raise PathTraversalError(f"Path contains traversal sequence: {path_str}")

    # If base_dir specified, ensure path is within it
    if base_dir is not None:
        base_resolved = base_dir.resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            logger.warning(
                "Path escapes base directory: %s not in %s",
                resolved, base_resolved
            )
            raise PathTraversalError(
                f"Path {resolved} is outside base directory {base_resolved}"
            )

        # Additional symlink-aware check using commonpath
        try:
            common = os.path.commonpath([str(base_resolved), str(resolved)])
            if common != str(base_resolved):
                raise PathTraversalError(
                    f"Path {resolved} escapes base directory via symlink"
                )
        except ValueError:
            # Different drives on Windows
            raise PathTraversalError(
                f"Path {resolved} is on different drive than {base_resolved}"
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

    Raises:
        PathTraversalError: If paths contain traversal sequences
        FileNotFoundError: If source does not exist
        FileExistsError: If target exists and force=False
    """
    # Validate paths for traversal attacks
    source = validate_path_safety(source)
    target = validate_path_safety(target)

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
    create hard links for each file. Falls back to copy if
    hardlink fails (e.g., cross-volume on Windows).
    """
    target.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        dest = target / item.name
        if item.is_dir():
            _recursive_hard_link(item, dest)
        else:
            try:
                dest.hardlink_to(item)
            except OSError:
                # Hardlink failed (cross-volume, permissions, etc.) - fall back to copy
                shutil.copy2(item, dest)


def smart_rmtree(path: Path, aggressive: bool = False) -> bool:
    """
    Robustly remove a directory tree.

    Learned from modaic: handles Windows permission issues and locked files.

    Args:
        path: Directory to remove
        aggressive: On Windows, try to kill blocking processes

    Returns:
        True if successful

    Raises:
        PathTraversalError: If path contains traversal sequences
    """
    # Validate path for traversal attacks
    path = validate_path_safety(path)

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
        except (subprocess.TimeoutExpired, OSError) as e:
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
        except (subprocess.TimeoutExpired, OSError) as e:
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
    except Exception as e:
        logger.debug("Failed to kill blocking processes: %s", e)


def sync_directory(
    source: Path,
    target: Path,
    delete_extra: bool = False,  # CHANGED: Default to safe copy-only mode
    ignore_times: bool = False,
) -> bool:
    """
    Synchronize directories using platform-appropriate tools.

    Learned from modaic: rsync on Unix, robocopy on Windows.

    SECURITY: delete_extra defaults to False to prevent accidental data loss.
    When True, files in target not in source will be PERMANENTLY DELETED.

    Args:
        source: Source directory
        target: Target directory
        delete_extra: Delete files in target not in source (DANGEROUS - default False)
        ignore_times: Compare by content, not timestamps

    Returns:
        True if successful

    Raises:
        PathTraversalError: If paths contain traversal sequences
        ValueError: If target is a protected system directory
    """
    # Validate paths for traversal attacks
    source = validate_path_safety(source)
    target = validate_path_safety(target)

    # Prevent syncing to dangerous system directories
    dangerous_paths = {
        Path.home(),
        Path("/"),
        Path("/home"),
        Path("/etc"),
        Path("/var"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
    }
    target_resolved = target.resolve()
    for dangerous in dangerous_paths:
        try:
            if target_resolved == dangerous.resolve():
                raise ValueError(f"Cannot sync to protected directory: {target}")
        except (OSError, RuntimeError) as e:
            logger.debug("Could not resolve protected path %s: %s", dangerous, e)

    target.mkdir(parents=True, exist_ok=True)

    if is_windows():
        cmd = ["robocopy", str(source), str(target), "/E"]
        if delete_extra:
            cmd.append("/MIR")
            logger.warning("sync_directory: delete_extra=True - files may be deleted from %s", target)
        # robocopy returns non-zero for success, 0-7 are OK
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        return result.returncode < 8
    else:
        cmd = ["rsync", "-a"]
        if delete_extra:
            cmd.append("--delete")
            logger.warning("sync_directory: delete_extra=True - files may be deleted from %s", target)
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

    Raises:
        PathTraversalError: If path contains traversal sequences
    """
    import tempfile
    import time

    # Validate path for traversal attacks
    path = validate_path_safety(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (for atomic rename)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    temp = Path(temp_path)
    fd_closed = False

    try:
        with os.fdopen(fd, mode) as f:
            fd_closed = True  # fdopen takes ownership of fd
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk before rename

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
        # Clean up fd if fdopen failed
        if not fd_closed:
            try:
                os.close(fd)
            except Exception as cleanup_err:
                logger.debug("Failed to close fd during cleanup: %s", cleanup_err)
        # Clean up temp file on any failure
        try:
            temp.unlink(missing_ok=True)
        except Exception as cleanup_err:
            logger.debug("Failed to remove temp file during cleanup: %s", cleanup_err)
        raise


# Common directories to always skip (performance optimization)
SKIP_DIRS = frozenset({
    '.git', '__pycache__', 'node_modules', '.venv', 'venv',
    '.tox', 'dist', 'build', '.eggs', '*.egg-info', '.mypy_cache',
    '.pytest_cache', '.ruff_cache', '.coverage', 'htmlcov',
})


def load_gitignore_patterns(paths: list[Path]) -> list[str]:
    """Load .gitignore patterns from paths.

    Args:
        paths: List of file or directory paths

    Returns:
        List of gitignore pattern strings
    """
    patterns = []
    for path in paths:
        p = Path(path)
        gitignore_path = (p if p.is_dir() else p.parent) / ".gitignore"
        if gitignore_path.exists():
            try:
                patterns.extend(gitignore_path.read_text(encoding="utf-8").splitlines())
            except (OSError, UnicodeDecodeError):
                logger.debug("Failed to read .gitignore: %s", gitignore_path)
    return patterns


def should_skip_entry(
    entry_name: str,
    entry_path: Path,
    root_path: Path,
    spec: "pathspec.PathSpec | None" = None,
    is_dir: bool = False,
) -> bool:
    """Check if an entry should be skipped based on gitignore and common patterns.

    Args:
        entry_name: Name of the file/directory
        entry_path: Full path to the entry
        root_path: Root path for relative path calculation
        spec: Optional pathspec for gitignore matching
        is_dir: Whether the entry is a directory

    Returns:
        True if entry should be skipped
    """
    # Skip common ignored directories
    if is_dir and entry_name in SKIP_DIRS:
        return True

    # Check gitignore patterns
    if spec:
        try:
            rel_path = entry_path.relative_to(root_path)
        except ValueError:
            rel_path = entry_path
        if spec.match_file(str(rel_path)):
            return True

    return False


def collect_files(
    paths: list[Path | str],
    spec: "pathspec.PathSpec | None" = None,
) -> list[Path]:
    """Collect files from paths, respecting gitignore patterns.

    Uses iterative traversal to avoid RecursionError on deep trees.

    Args:
        paths: List of file or directory paths
        spec: Optional pathspec for gitignore matching

    Returns:
        List of file paths
    """
    from collections import deque

    files: list[Path] = []

    for path in paths:
        p = Path(path)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            # Iterative directory traversal
            dirs_to_process: deque[Path] = deque([p])
            root_path = p

            while dirs_to_process:
                current_path = dirs_to_process.popleft()

                try:
                    entries = list(os.scandir(current_path))
                except PermissionError:
                    logger.debug("Permission denied: %s", current_path)
                    continue

                for entry in entries:
                    entry_path = Path(entry.path)
                    is_dir = entry.is_dir(follow_symlinks=False)

                    if should_skip_entry(entry.name, entry_path, root_path, spec, is_dir):
                        continue

                    if entry.is_file(follow_symlinks=False):
                        files.append(entry_path)
                    elif is_dir:
                        dirs_to_process.append(entry_path)

    return sorted(files)


def format_file_context(
    files: list[Path],
    add_line_numbers: bool = True,
) -> tuple[str, list[tuple[Path, str]]]:
    """Format files into a context string for LLM consumption.

    Args:
        files: List of file paths to read
        add_line_numbers: Whether to add line numbers (helps LLM report locations)

    Returns:
        Tuple of (context_string, list of (skipped_file, reason) tuples)
    """
    context_parts = []
    skipped_files = []

    for f in files:
        try:
            content = f.read_text(encoding="utf-8")

            if add_line_numbers:
                numbered_lines = [
                    f"{i+1:4d} | {line}"
                    for i, line in enumerate(content.splitlines())
                ]
                formatted_content = "\n".join(numbered_lines)
            else:
                formatted_content = content

            context_parts.append(
                f"=== FILE: {f} ===\n{formatted_content}\n=== END FILE ===\n"
            )
        except UnicodeDecodeError:
            skipped_files.append((f, "binary/encoding"))
        except PermissionError:
            skipped_files.append((f, "permission denied"))
        except OSError as e:
            skipped_files.append((f, str(e)))

    return "\n".join(context_parts), skipped_files


def load_context_from_paths(
    paths: list[Path | str],
    gitignore: bool = True,
    add_line_numbers: bool = True,
) -> str:
    """Load and format context from file/directory paths.

    This is the main entry point for loading context for LLM analysis.

    Args:
        paths: List of file or directory paths
        gitignore: Whether to respect .gitignore patterns
        add_line_numbers: Whether to add line numbers

    Returns:
        Formatted context string
    """
    import pathspec

    # Load gitignore patterns
    spec = None
    if gitignore:
        patterns = load_gitignore_patterns([Path(p) for p in paths])
        if patterns:
            spec = pathspec.PathSpec.from_lines("gitignore", patterns)

    # Collect files
    files = collect_files(paths, spec)

    # Format context
    context, skipped = format_file_context(files, add_line_numbers)

    if skipped:
        logger.warning(
            "Skipped %d files: %s",
            len(skipped),
            ", ".join(f"{f.name} ({reason})" for f, reason in skipped[:5]),
        )

    return context


# Context cache for repeated loads (key: frozenset of (path, mtime) tuples)
_context_cache: dict[tuple, tuple[str, float]] = {}
_CONTEXT_CACHE_MAX_SIZE = 50  # Max number of cached contexts
_CONTEXT_CACHE_MAX_AGE = 300  # Max age in seconds


def _get_cache_key(paths: list[Path], files: list[Path]) -> tuple:
    """Generate a cache key based on input paths and collected file mtimes.

    Args:
        paths: Original input paths (for identity)
        files: Collected files (for mtime checking)
    """
    # Include input paths for identity
    key_parts = [("input", tuple(str(p.resolve()) for p in sorted(paths)))]

    # Include file mtimes for invalidation
    for f in sorted(files):
        try:
            mtime = f.stat().st_mtime if f.exists() else 0
            key_parts.append((str(f), mtime))
        except OSError:
            key_parts.append((str(f), 0))

    return tuple(key_parts)


def load_context_from_paths_cached(
    paths: list[Path | str],
    gitignore: bool = True,
    add_line_numbers: bool = True,
) -> str:
    """Load context with caching based on file paths and mtimes.

    This is a cached version of load_context_from_paths that avoids
    re-reading files that haven't changed.

    Args:
        paths: List of file or directory paths
        gitignore: Whether to respect .gitignore patterns
        add_line_numbers: Whether to add line numbers

    Returns:
        Formatted context string (from cache if available and valid)
    """
    import pathspec
    import time

    global _context_cache

    # Convert to Path objects
    path_objs = [Path(p) for p in paths]

    # Load gitignore patterns and collect files
    spec = None
    if gitignore:
        patterns = load_gitignore_patterns(path_objs)
        if patterns:
            spec = pathspec.PathSpec.from_lines("gitignore", patterns)

    files = collect_files(path_objs, spec)

    # Generate cache key using collected files
    cache_key = _get_cache_key(path_objs, files)

    # Check cache
    now = time.time()
    if cache_key in _context_cache:
        cached_context, cached_time = _context_cache[cache_key]
        if now - cached_time < _CONTEXT_CACHE_MAX_AGE:
            logger.debug("Context cache hit for %d paths (%d files)", len(paths), len(files))
            return cached_context

    # Cache miss - format context (files already collected)
    context, skipped = format_file_context(files, add_line_numbers)

    if skipped:
        logger.warning(
            "Skipped %d files: %s",
            len(skipped),
            ", ".join(f"{f.name} ({reason})" for f, reason in skipped[:5]),
        )

    # Evict old entries if cache is full
    if len(_context_cache) >= _CONTEXT_CACHE_MAX_SIZE:
        # Remove oldest entries
        sorted_keys = sorted(_context_cache.keys(), key=lambda k: _context_cache[k][1])
        for key in sorted_keys[:len(_context_cache) - _CONTEXT_CACHE_MAX_SIZE + 1]:
            del _context_cache[key]

    # Store in cache
    _context_cache[cache_key] = (context, now)
    logger.debug("Context cached for %d paths (%d files, %d chars)", len(paths), len(files), len(context))

    return context


def clear_context_cache() -> None:
    """Clear the context cache."""
    global _context_cache
    _context_cache.clear()
    logger.debug("Context cache cleared")


def get_context_cache_stats() -> dict:
    """Get context cache statistics."""
    return {
        "entries": len(_context_cache),
        "max_size": _CONTEXT_CACHE_MAX_SIZE,
        "max_age_seconds": _CONTEXT_CACHE_MAX_AGE,
    }


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count for text.

    Uses a simple heuristic (4 chars per token on average for code).
    For more accurate counts, use tiktoken directly.

    Args:
        text: Text to estimate
        chars_per_token: Average characters per token (default 4.0 for code)

    Returns:
        Estimated token count
    """
    return int(len(text) / chars_per_token)


def truncate_context(
    context: str,
    max_tokens: int = 100_000,
    strategy: str = "tail",
    chars_per_token: float = 4.0,
) -> tuple[str, bool]:
    """Truncate context to fit within token limit.

    Args:
        context: Context string to truncate
        max_tokens: Maximum tokens allowed
        strategy: Truncation strategy:
            - "tail": Keep end (most recent code)
            - "head": Keep start
            - "middle": Keep start and end, remove middle
        chars_per_token: Chars per token for estimation

    Returns:
        Tuple of (truncated_context, was_truncated)
    """
    estimated_tokens = estimate_tokens(context, chars_per_token)

    if estimated_tokens <= max_tokens:
        return context, False

    max_chars = int(max_tokens * chars_per_token)

    if strategy == "tail":
        # Keep the end (most recent files)
        truncated = "...[TRUNCATED]...\n" + context[-max_chars:]
    elif strategy == "head":
        # Keep the start
        truncated = context[:max_chars] + "\n...[TRUNCATED]..."
    elif strategy == "middle":
        # Keep start and end, remove middle
        half = max_chars // 2
        truncated = context[:half] + "\n...[TRUNCATED]...\n" + context[-half:]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return truncated, True


def smart_truncate_context(
    context: str,
    max_tokens: int = 100_000,
    chars_per_token: float = 4.0,
) -> tuple[str, bool]:
    """Intelligently truncate context preserving file boundaries.

    Removes complete files from the middle to preserve context coherence.

    Args:
        context: Context with file markers (=== FILE: ... ===)
        max_tokens: Maximum tokens allowed
        chars_per_token: Chars per token for estimation

    Returns:
        Tuple of (truncated_context, was_truncated)
    """
    import re

    estimated_tokens = estimate_tokens(context, chars_per_token)

    if estimated_tokens <= max_tokens:
        return context, False

    # Split by file markers (pattern cached at module level)
    files = _FILE_MARKER_PATTERN.findall(context)

    if not files:
        # No file markers, use simple truncation
        return truncate_context(context, max_tokens, "tail", chars_per_token)

    # Calculate tokens per file
    file_tokens = [(f, estimate_tokens(f, chars_per_token)) for f in files]
    sum(t for _, t in file_tokens)

    # Remove files from middle until under limit
    # Keep first 25% and last 25% of files, remove from middle
    target_tokens = max_tokens - 100  # Buffer for truncation message

    result_files = []
    current_tokens = 0

    # Always include first and last files
    if len(files) >= 2:
        first_quarter = max(1, len(files) // 4)
        last_quarter = max(1, len(files) // 4)

        # Add first files
        for f, t in file_tokens[:first_quarter]:
            result_files.append(f)
            current_tokens += t

        # Add truncation marker
        result_files.append("\n...[TRUNCATED: removed middle files to fit context limit]...\n\n")

        # Add last files (as many as fit)
        for f, t in reversed(file_tokens[-last_quarter:]):
            if current_tokens + t <= target_tokens:
                result_files.insert(-1, f)  # Insert before truncation marker... wait, need to fix
                current_tokens += t

        # Fix order - rebuild properly
        result_files = []
        current_tokens = 0

        # First quarter
        for f, t in file_tokens[:first_quarter]:
            if current_tokens + t <= target_tokens * 0.5:
                result_files.append(f)
                current_tokens += t

        result_files.append("\n...[TRUNCATED: removed middle files to fit context limit]...\n\n")

        # Last quarter
        last_files = []
        last_tokens = 0
        for f, t in reversed(file_tokens[-last_quarter:]):
            if last_tokens + t <= target_tokens * 0.5:
                last_files.insert(0, f)
                last_tokens += t

        result_files.extend(last_files)
    else:
        # Only one file, use simple truncation
        return truncate_context(context, max_tokens, "tail", chars_per_token)

    return "".join(result_files), True

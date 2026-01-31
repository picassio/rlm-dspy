"""Context loading utilities - file collection, gitignore, truncation."""

from __future__ import annotations

import hashlib
import logging
import time
from functools import lru_cache
from pathlib import Path

import pathspec

logger = logging.getLogger(__name__)


def load_gitignore_patterns(paths: list[Path]) -> list[str]:
    """Load gitignore patterns from .gitignore files."""
    patterns = []
    for path in paths:
        gitignore = path / ".gitignore" if path.is_dir() else path.parent / ".gitignore"
        if gitignore.exists():
            try:
                content = gitignore.read_text(encoding="utf-8", errors="replace")
                patterns.extend(line.strip() for line in content.splitlines() 
                               if line.strip() and not line.startswith("#"))
            except Exception:
                pass
    return patterns


def should_skip_entry(
    entry: Path,
    gitignore_spec: pathspec.PathSpec | None,
    skip_hidden: bool = True,
    allowed_extensions: set[str] | None = None,
) -> bool:
    """Check if a file/directory should be skipped."""
    name = entry.name

    # Skip hidden files/dirs
    if skip_hidden and name.startswith(".") and name not in {".env.example", ".gitignore"}:
        return True

    # Skip common non-code directories
    skip_dirs = {"node_modules", "__pycache__", ".git", ".venv", "venv", 
                 "dist", "build", ".tox", ".pytest_cache", "*.egg-info"}
    if entry.is_dir() and name in skip_dirs:
        return True

    # Check gitignore
    if gitignore_spec:
        try:
            rel_path = str(entry.relative_to(entry.parent.parent)) if entry.parent.parent != entry else name
            if gitignore_spec.match_file(rel_path):
                return True
        except (ValueError, Exception):
            pass

    # Check extensions for files
    if entry.is_file() and allowed_extensions:
        if entry.suffix.lower() not in allowed_extensions:
            return True

    return False


def collect_files(
    paths: list[Path],
    extensions: set[str] | None = None,
    max_files: int = 1000,
    respect_gitignore: bool = True,
) -> list[Path]:
    """Collect files from paths, respecting gitignore and extensions.

    Args:
        paths: List of files/directories to collect from
        extensions: Allowed file extensions (e.g., {'.py', '.js'})
        max_files: Maximum number of files to collect
        respect_gitignore: Whether to respect .gitignore patterns

    Returns:
        List of collected file paths
    """
    files: list[Path] = []
    gitignore_patterns = load_gitignore_patterns(paths) if respect_gitignore else []
    gitignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore_patterns) if gitignore_patterns else None

    for path in paths:
        if len(files) >= max_files:
            break

        path = path.resolve()

        if path.is_file():
            if not should_skip_entry(path, gitignore_spec, allowed_extensions=extensions):
                files.append(path)
        elif path.is_dir():
            for entry in sorted(path.rglob("*")):
                if len(files) >= max_files:
                    break
                if entry.is_file() and not should_skip_entry(entry, gitignore_spec, allowed_extensions=extensions):
                    files.append(entry)

    return files


def format_file_context(
    files: list[Path],
    base_path: Path | None = None,
    include_line_numbers: bool = True,
    max_file_size: int = 100_000,
) -> str:
    """Format files into context string.

    Args:
        files: List of file paths
        base_path: Base path for relative file names
        include_line_numbers: Add line numbers to content
        max_file_size: Skip files larger than this

    Returns:
        Formatted context string
    """
    context_parts = []

    for file_path in files:
        try:
            if file_path.stat().st_size > max_file_size:
                continue

            content = file_path.read_text(encoding="utf-8", errors="replace")

            if base_path:
                try:
                    display_path = file_path.relative_to(base_path)
                except ValueError:
                    display_path = file_path
            else:
                display_path = file_path

            if include_line_numbers:
                lines = content.splitlines()
                numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
                content = "\n".join(numbered)

            context_parts.append(f"=== FILE: {display_path} ===\n{content}")

        except Exception as e:
            logger.debug("Skipping %s: %s", file_path, e)

    return "\n\n".join(context_parts)


def load_context_from_paths(
    paths: list[Path],
    extensions: set[str] | None = None,
    max_files: int = 1000,
    max_tokens: int | None = None,
) -> str:
    """Load context from file paths.

    Args:
        paths: List of files/directories
        extensions: Allowed extensions
        max_files: Maximum files to include
        max_tokens: Maximum tokens (approximate)

    Returns:
        Context string
    """
    files = collect_files(paths, extensions=extensions, max_files=max_files)

    base_path = paths[0] if len(paths) == 1 and paths[0].is_dir() else None
    context = format_file_context(files, base_path=base_path)

    if max_tokens:
        context = truncate_context(context, max_tokens)

    return context


# Context cache
_context_cache: dict[tuple, tuple[str, float]] = {}
_CACHE_TTL = 60.0


def _get_cache_key(paths: list[Path], files: list[Path]) -> tuple:
    """Generate cache key from paths and file mtimes."""
    path_strs = tuple(str(p) for p in sorted(paths))
    mtimes = []
    for f in sorted(files)[:50]:
        try:
            mtimes.append((str(f), f.stat().st_mtime))
        except Exception:
            pass
    return (path_strs, tuple(mtimes))


def load_context_from_paths_cached(
    paths: list[Path],
    extensions: set[str] | None = None,
    max_files: int = 1000,
    max_tokens: int | None = None,
) -> str:
    """Load context with caching."""
    files = collect_files(paths, extensions=extensions, max_files=max_files)
    cache_key = _get_cache_key(paths, files)

    now = time.time()
    if cache_key in _context_cache:
        cached_context, cached_time = _context_cache[cache_key]
        if now - cached_time < _CACHE_TTL:
            if max_tokens:
                return truncate_context(cached_context, max_tokens)
            return cached_context

    base_path = paths[0] if len(paths) == 1 and paths[0].is_dir() else None
    context = format_file_context(files, base_path=base_path)
    _context_cache[cache_key] = (context, now)

    if max_tokens:
        context = truncate_context(context, max_tokens)

    return context


def clear_context_cache() -> None:
    """Clear the context cache."""
    global _context_cache
    _context_cache = {}


def get_context_cache_stats() -> dict:
    """Get context cache statistics."""
    return {
        "entries": len(_context_cache),
        "keys": list(_context_cache.keys())[:5],
    }


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from text length."""
    return int(len(text) / chars_per_token)


def truncate_context(
    context: str,
    max_tokens: int,
    chars_per_token: float = 4.0,
    preserve_structure: bool = True,
) -> str:
    """Truncate context to fit within token limit.

    Args:
        context: Full context string
        max_tokens: Maximum tokens
        chars_per_token: Characters per token estimate
        preserve_structure: Try to preserve file boundaries

    Returns:
        Truncated context
    """
    max_chars = int(max_tokens * chars_per_token)

    if len(context) <= max_chars:
        return context

    if preserve_structure:
        files = context.split("=== FILE: ")
        result_parts = []
        current_length = 0

        for i, file_section in enumerate(files):
            if not file_section.strip():
                continue

            section = ("=== FILE: " + file_section) if i > 0 else file_section

            if current_length + len(section) <= max_chars:
                result_parts.append(section)
                current_length += len(section)
            else:
                remaining = max_chars - current_length - 100
                if remaining > 500:
                    truncated = section[:remaining] + "\n... (truncated)"
                    result_parts.append(truncated)
                break

        if result_parts:
            return "".join(result_parts)

    return context[:max_chars - 50] + "\n... (truncated to fit token limit)"


def smart_truncate_context(
    context: str,
    max_tokens: int,
    priority_patterns: list[str] | None = None,
) -> str:
    """Truncate context with priority-based file selection.

    Args:
        context: Full context
        max_tokens: Token limit
        priority_patterns: Patterns for high-priority files

    Returns:
        Truncated context with priority files first
    """
    import re
    
    if estimate_tokens(context) <= max_tokens:
        return context

    files = context.split("=== FILE: ")
    if len(files) <= 1:
        return truncate_context(context, max_tokens)

    priority_patterns = priority_patterns or ["README", "main", "__init__", "cli", "app"]

    high_priority = []
    normal_priority = []

    for i, section in enumerate(files):
        if not section.strip():
            continue

        full_section = ("=== FILE: " + section) if i > 0 else section

        is_priority = any(p.lower() in section.lower()[:100] for p in priority_patterns)
        if is_priority:
            high_priority.append(full_section)
        else:
            normal_priority.append(full_section)

    ordered = high_priority + normal_priority
    result_parts = []
    current_tokens = 0
    max_chars = int(max_tokens * 4.0)

    for section in ordered:
        section_tokens = estimate_tokens(section)
        if current_tokens + section_tokens <= max_tokens:
            result_parts.append(section)
            current_tokens += section_tokens
        else:
            remaining_chars = max_chars - len("".join(result_parts)) - 100
            if remaining_chars > 500:
                result_parts.append(section[:remaining_chars] + "\n... (truncated)")
            break

    return "".join(result_parts) if result_parts else truncate_context(context, max_tokens)

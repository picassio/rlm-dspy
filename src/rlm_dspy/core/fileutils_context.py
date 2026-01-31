"""Context loading utilities - file collection, formatting, caching."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathspec

logger = logging.getLogger(__name__)

# Pre-compiled regex for file marker extraction
_FILE_MARKER_PATTERN = re.compile(r'(=== FILE: .+? ===\n.*?=== END FILE ===\n)', re.DOTALL)

# Directories to always skip during file traversal
SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".tox",
    ".pytest_cache", ".mypy_cache", "dist", "build", ".eggs", "eggs",
    ".hg", ".svn", ".nox", "htmlcov", ".coverage", ".cache",
})


def load_gitignore_patterns(paths: list[Path]) -> list[str]:
    """Load gitignore patterns from .gitignore files in the given paths."""
    patterns = []
    seen_gitignores = set()

    for p in paths:
        p = Path(p)
        search_dirs = []

        if p.is_file():
            search_dirs.append(p.parent)
        else:
            search_dirs.append(p)

        for d in search_dirs:
            gitignore = d / ".gitignore"
            if gitignore.exists() and str(gitignore) not in seen_gitignores:
                seen_gitignores.add(str(gitignore))
                try:
                    content = gitignore.read_text(encoding="utf-8", errors="replace")
                    for line in content.splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)
                except Exception as e:
                    logger.debug("Failed to read %s: %s", gitignore, e)

    return patterns


def should_skip_entry(entry: Path, gitignore_spec: "pathspec.PathSpec | None", base_path: Path) -> bool:
    """Check if an entry should be skipped based on gitignore patterns."""
    name = entry.name

    always_skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".pytest_cache", ".mypy_cache"}
    if name in always_skip:
        return True

    if name.startswith(".") and not name.startswith(".env"):
        return True

    if gitignore_spec:
        try:
            rel_path = str(entry.relative_to(base_path))
            if entry.is_dir():
                rel_path += "/"
            if gitignore_spec.match_file(rel_path):
                return True
        except (ValueError, Exception):
            pass

    return False


def collect_files(paths: list[Path | str], gitignore_spec: "pathspec.PathSpec | None" = None) -> list[Path]:
    """Collect all files from paths, respecting gitignore patterns."""
    files = []
    seen = set()

    for p in paths:
        p = Path(p).resolve()

        if p.is_file():
            if str(p) not in seen:
                seen.add(str(p))
                files.append(p)
        elif p.is_dir():
            base_path = p
            for entry in sorted(p.rglob("*")):
                if entry.is_file() and str(entry) not in seen:
                    if not should_skip_entry(entry, gitignore_spec, base_path):
                        if not should_skip_entry(entry.parent, gitignore_spec, base_path):
                            seen.add(str(entry))
                            files.append(entry)

    return files


def format_file_context(files: list[Path], add_line_numbers: bool = True) -> tuple[str, list[tuple[Path, str]]]:
    """Format files into context string."""
    context_parts = []
    skipped = []

    code_extensions = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", 
                       ".rb", ".cs", ".kt", ".scala", ".php", ".lua", ".sh", ".bash", ".zsh", ".yaml", ".yml",
                       ".json", ".toml", ".md", ".txt", ".sql", ".html", ".css", ".scss", ".vue", ".svelte"}

    for file_path in files:
        try:
            if file_path.stat().st_size > 500_000:
                skipped.append((file_path, "too large"))
                continue

            if file_path.suffix.lower() not in code_extensions:
                skipped.append((file_path, "unsupported extension"))
                continue

            content = file_path.read_text(encoding="utf-8", errors="replace")

            if add_line_numbers:
                lines = content.splitlines()
                numbered_lines = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
                content = "\n".join(numbered_lines)

            context_parts.append(f"=== FILE: {file_path} ===\n{content}\n=== END FILE ===")

        except Exception as e:
            skipped.append((file_path, str(e)))

    return "\n\n".join(context_parts), skipped


def load_context_from_paths(
    paths: list[Path | str],
    gitignore: bool = True,
    add_line_numbers: bool = True,
) -> str:
    """Load and format context from file/directory paths."""
    import pathspec

    spec = None
    if gitignore:
        patterns = load_gitignore_patterns([Path(p) for p in paths])
        if patterns:
            spec = pathspec.PathSpec.from_lines("gitignore", patterns)

    files = collect_files(paths, spec)
    context, skipped = format_file_context(files, add_line_numbers)

    if skipped:
        logger.warning("Skipped %d files: %s", len(skipped),
                      ", ".join(f"{f.name} ({reason})" for f, reason in skipped[:5]))

    return context


# Context cache
_context_cache: dict[tuple, tuple[str, float]] = {}
_CONTEXT_CACHE_MAX_SIZE = 50
_CONTEXT_CACHE_MAX_AGE = 300


def _get_cache_key(paths: list[Path], files: list[Path]) -> tuple:
    """Generate cache key based on paths and file mtimes."""
    key_parts = [("input", tuple(str(p.resolve()) for p in sorted(paths)))]
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
    """Load context with caching based on file paths and mtimes."""
    import pathspec

    global _context_cache

    path_objs = [Path(p) for p in paths]

    spec = None
    if gitignore:
        patterns = load_gitignore_patterns(path_objs)
        if patterns:
            spec = pathspec.PathSpec.from_lines("gitignore", patterns)

    files = collect_files(path_objs, spec)
    cache_key = _get_cache_key(path_objs, files)

    now = time.time()
    if cache_key in _context_cache:
        cached_context, cached_time = _context_cache[cache_key]
        if now - cached_time < _CONTEXT_CACHE_MAX_AGE:
            logger.debug("Context cache hit for %d paths (%d files)", len(paths), len(files))
            return cached_context

    context, skipped = format_file_context(files, add_line_numbers)

    if skipped:
        logger.warning("Skipped %d files: %s", len(skipped),
                      ", ".join(f"{f.name} ({reason})" for f, reason in skipped[:5]))

    if len(_context_cache) >= _CONTEXT_CACHE_MAX_SIZE:
        sorted_keys = sorted(_context_cache.keys(), key=lambda k: _context_cache[k][1])
        for key in sorted_keys[:len(_context_cache) - _CONTEXT_CACHE_MAX_SIZE + 1]:
            del _context_cache[key]

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
    """Estimate token count for text."""
    return int(len(text) / chars_per_token)


def truncate_context(
    context: str,
    max_tokens: int,
    chars_per_token: float = 4.0,
    preserve_structure: bool = True,
) -> tuple[str, bool]:
    """Truncate context to fit within token limit."""
    max_chars = int(max_tokens * chars_per_token)

    if len(context) <= max_chars:
        return context, False

    if not preserve_structure:
        return context[:max_chars - 50] + "\n... (truncated)", True

    files = context.split("=== FILE: ")
    result = []
    current_len = 0

    for i, section in enumerate(files):
        if not section.strip():
            continue

        full_section = ("=== FILE: " + section) if i > 0 else section

        if current_len + len(full_section) <= max_chars:
            result.append(full_section)
            current_len += len(full_section)
        else:
            remaining = max_chars - current_len - 100
            if remaining > 200:
                result.append(full_section[:remaining] + "\n... (truncated)")
            break

    return "".join(result) if result else context[:max_chars - 50] + "\n... (truncated)", True


def smart_truncate_context(
    context: str,
    max_tokens: int,
    priority_patterns: list[str] | None = None,
    chars_per_token: float = 4.0,
) -> tuple[str, bool]:
    """Truncate context with priority-based file selection."""
    if estimate_tokens(context, chars_per_token) <= max_tokens:
        return context, False

    priority_patterns = priority_patterns or ["README", "main", "__init__", "cli", "app", "index"]

    files = context.split("=== FILE: ")
    if len(files) <= 1:
        return truncate_context(context, max_tokens, chars_per_token)

    high_priority = []
    normal_priority = []

    for i, section in enumerate(files):
        if not section.strip():
            continue
        full_section = ("=== FILE: " + section) if i > 0 else section
        header = section[:200].lower()
        is_priority = any(p.lower() in header for p in priority_patterns)
        if is_priority:
            high_priority.append(full_section)
        else:
            normal_priority.append(full_section)

    ordered = high_priority + normal_priority
    max_chars = int(max_tokens * chars_per_token)
    result = []
    current_len = 0

    for section in ordered:
        if current_len + len(section) <= max_chars:
            result.append(section)
            current_len += len(section)
        else:
            remaining = max_chars - current_len - 100
            if remaining > 200:
                result.append(section[:remaining] + "\n... (truncated)")
            break

    return "".join(result) if result else context[:max_chars - 50] + "\n... (truncated)", True

"""Cross-platform file utilities.

This module re-exports all file utilities for backward compatibility.
"""

from __future__ import annotations

# Re-export from base module
from .fileutils_base import (
    PathTraversalError,
    validate_path_safety,
    is_windows,
    is_macos,
    is_linux,
    get_cache_dir,
    smart_link,
    smart_rmtree,
    sync_directory,
    path_to_module,
    ensure_dir,
    atomic_write,
)

# Re-export from context module
from .fileutils_context import (
    SKIP_DIRS,
    load_gitignore_patterns,
    should_skip_entry,
    collect_files,
    format_file_context,
    load_context_from_paths,
    load_context_from_paths_cached,
    clear_context_cache,
    get_context_cache_stats,
    estimate_tokens,
    truncate_context,
    smart_truncate_context,
)

__all__ = [
    # Base utilities
    "PathTraversalError",
    "validate_path_safety",
    "is_windows",
    "is_macos",
    "is_linux",
    "get_cache_dir",
    "smart_link",
    "smart_rmtree",
    "sync_directory",
    "path_to_module",
    "ensure_dir",
    "atomic_write",
    # Context utilities
    "SKIP_DIRS",
    "load_gitignore_patterns",
    "should_skip_entry",
    "collect_files",
    "format_file_context",
    "load_context_from_paths",
    "load_context_from_paths_cached",
    "clear_context_cache",
    "get_context_cache_stats",
    "estimate_tokens",
    "truncate_context",
    "smart_truncate_context",
]

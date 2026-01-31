"""File utilities for RLM-DSPy.

This module provides file system utilities for path validation, I/O operations,
and context loading from code files.
"""

from __future__ import annotations

# Re-export from submodules
from .fileutils_path import (
    PathTraversalError,
    validate_path_safety,
    is_windows,
    is_macos,
    is_linux,
    get_cache_dir,
    ensure_dir,
    path_to_module,
)

from .fileutils_io import (
    smart_link,
    smart_rmtree,
    sync_directory,
    atomic_write,
)

from .fileutils_context import (
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
    # Path utilities
    "PathTraversalError",
    "validate_path_safety",
    "is_windows",
    "is_macos", 
    "is_linux",
    "get_cache_dir",
    "ensure_dir",
    "path_to_module",
    # I/O utilities
    "smart_link",
    "smart_rmtree",
    "sync_directory",
    "atomic_write",
    # Context utilities
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

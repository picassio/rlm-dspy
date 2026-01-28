"""Salsa-style incremental computation for rlm-dspy.

Inspired by rust-analyzer's Salsa framework and llm-tldr's implementation.
Provides automatic memoization with dependency tracking and cascade invalidation.

Key concepts:
1. **Queries as Functions**: Everything is a query with automatic memoization
2. **Automatic Dependency Tracking**: Queries record which other queries they call
3. **Minimal Re-computation**: Only affected queries re-run on change

Example usage:
    from rlm_dspy.core.salsa import SalsaDB, salsa_query

    @salsa_query
    def read_file(db: SalsaDB, path: str) -> str:
        return db.get_file(path)

    @salsa_query
    def parse_file(db: SalsaDB, path: str) -> dict:
        content = db.query(read_file, path)
        return parse(content)

    db = SalsaDB()
    db.set_file("auth.py", "def login(): pass")
    result = db.query(parse_file, "auth.py")

    # When file changes, dependent queries auto-invalidate
    db.set_file("auth.py", "def login(): pass\\ndef logout(): pass")
    result = db.query(parse_file, "auth.py")  # Recomputes automatically
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
QueryKey = tuple[str, tuple[Any, ...]]  # (func_name, args)

# Marker for salsa queries
_SALSA_QUERY_MARKER = "_is_salsa_query"


def salsa_query(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to mark a function as a Salsa query.

    Salsa queries:
    - Are automatically memoized when called through SalsaDB.query()
    - Track their dependencies on other queries
    - Can be invalidated, cascading to dependents

    Example:
        @salsa_query
        def get_functions(db: SalsaDB, path: str) -> list[str]:
            content = db.query(read_file, path)
            return extract_functions(content)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # When called directly (not through db.query), just execute
        return func(*args, **kwargs)

    # Mark as salsa query
    setattr(wrapper, _SALSA_QUERY_MARKER, True)
    setattr(wrapper, "_original_func", func)
    setattr(wrapper, "_query_name", func.__name__)

    return wrapper


def is_salsa_query(func: Callable) -> bool:
    """Check if a function is decorated with @salsa_query."""
    return getattr(func, _SALSA_QUERY_MARKER, False)


@dataclass
class CacheEntry:
    """Cache entry for a query result."""

    result: Any
    dependencies: set[QueryKey] = field(default_factory=set)
    file_dependencies: dict[str, int] = field(default_factory=dict)  # path -> revision
    computed_at: float = field(default_factory=time.time)


@dataclass
class QueryStats:
    """Statistics for query execution."""

    cache_hits: int = 0
    cache_misses: int = 0
    invalidations: int = 0
    recomputations: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0.0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "invalidations": self.invalidations,
            "recomputations": self.recomputations,
            "hit_rate_percent": round(hit_rate, 2),
        }


class SalsaDB:
    """Database for Salsa-style query memoization.

    Tracks:
    - File contents and revisions
    - Query results and their dependencies
    - Reverse dependency graph for invalidation cascading

    Thread-safe for concurrent access.
    """

    def __init__(self, persist_path: Path | str | None = None):
        """Initialize SalsaDB.

        Args:
            persist_path: Optional path to persist cache to disk
        """
        self._lock = threading.RLock()

        # File storage
        self._file_contents: dict[str, str] = {}
        self._file_revisions: dict[str, int] = {}

        # Query cache: query_key -> CacheEntry
        self._query_cache: dict[QueryKey, CacheEntry] = {}

        # Reverse dependencies: query_key -> set of dependent query_keys
        self._reverse_deps: dict[QueryKey, set[QueryKey]] = {}

        # File to query dependencies: file_path -> set of query_keys
        self._file_to_queries: dict[str, set[QueryKey]] = {}

        # Stats
        self._stats = QueryStats()

        # Active query stack for dependency tracking
        self._query_stack: list[QueryKey] = []

        # Persistence
        self._persist_path = Path(persist_path) if persist_path else None

    # -------------------------------------------------------------------------
    # File Management
    # -------------------------------------------------------------------------

    def set_file(self, path: str, content: str) -> None:
        """Set or update file content.

        This increments the file's revision and invalidates any queries
        that depend on this file.

        Args:
            path: File path (used as key)
            content: File content
        """
        with self._lock:
            old_revision = self._file_revisions.get(path, 0)
            self._file_contents[path] = content
            self._file_revisions[path] = old_revision + 1

            # Invalidate queries that depend on this file
            self._invalidate_file_dependents(path)

            logger.debug(f"File updated: {path} (rev {old_revision + 1})")

    def get_file(self, path: str) -> str | None:
        """Get file content.

        If called during a query, registers the file as a dependency.

        Args:
            path: File path

        Returns:
            File content or None if not found
        """
        with self._lock:
            # Track file dependency if in a query context
            if self._query_stack:
                current_query = self._query_stack[-1]
                if path not in self._file_to_queries:
                    self._file_to_queries[path] = set()
                self._file_to_queries[path].add(current_query)

                # Also track in the cache entry
                if current_query in self._query_cache:
                    entry = self._query_cache[current_query]
                    entry.file_dependencies[path] = self._file_revisions.get(path, 0)

            return self._file_contents.get(path)

    def get_file_revision(self, path: str) -> int:
        """Get current revision number for a file."""
        with self._lock:
            return self._file_revisions.get(path, 0)

    def load_file(self, path: str | Path) -> str | None:
        """Load file from disk and register it.

        Args:
            path: Path to file on disk

        Returns:
            File content or None if not found
        """
        path = Path(path)
        if not path.exists():
            return None

        content = path.read_text()
        self.set_file(str(path), content)
        return content

    # -------------------------------------------------------------------------
    # Query Execution
    # -------------------------------------------------------------------------

    def query(self, func: Callable[..., T], *args: Any) -> T:
        """Execute a query with memoization and dependency tracking.

        Args:
            func: A @salsa_query decorated function
            *args: Arguments to pass to the function

        Returns:
            Query result (cached or freshly computed)
        """
        if not is_salsa_query(func):
            # Not a salsa query, just execute
            return func(self, *args)

        query_name = getattr(func, "_query_name", func.__name__)
        query_key: QueryKey = (query_name, args)

        with self._lock:
            # Check cache
            if query_key in self._query_cache:
                entry = self._query_cache[query_key]

                # Validate file dependencies haven't changed
                valid = True
                for file_path, cached_rev in entry.file_dependencies.items():
                    current_rev = self._file_revisions.get(file_path, 0)
                    if current_rev != cached_rev:
                        valid = False
                        break

                if valid:
                    self._stats.cache_hits += 1
                    logger.debug(f"Cache hit: {query_name}({args})")
                    return entry.result

            # Cache miss - need to compute
            self._stats.cache_misses += 1
            logger.debug(f"Cache miss: {query_name}({args})")

            # Push onto query stack for dependency tracking
            self._query_stack.append(query_key)

            try:
                # Execute the query
                original_func = getattr(func, "_original_func", func)
                result = original_func(self, *args)

                # Create cache entry
                entry = CacheEntry(result=result)

                # Record reverse dependencies for cache invalidation
                # When parent calls child (query_key), parent depends on child
                # So if child's result changes, parent should be invalidated
                # _reverse_deps maps: child -> set of parents that depend on it
                if len(self._query_stack) > 1:
                    parent_key = self._query_stack[-2]
                    # Track that parent depends on this child query
                    if query_key not in self._reverse_deps:
                        self._reverse_deps[query_key] = set()
                    self._reverse_deps[query_key].add(parent_key)

                self._query_cache[query_key] = entry

                return result

            finally:
                self._query_stack.pop()

    # -------------------------------------------------------------------------
    # Invalidation
    # -------------------------------------------------------------------------

    def invalidate(self, func: Callable, *args: Any) -> int:
        """Manually invalidate a specific query.

        Args:
            func: The query function
            *args: Query arguments

        Returns:
            Number of queries invalidated (including cascades)
        """
        query_name = getattr(func, "_query_name", func.__name__)
        query_key: QueryKey = (query_name, args)

        with self._lock:
            return self._invalidate_query(query_key)

    def _invalidate_query(self, query_key: QueryKey) -> int:
        """Invalidate a query and cascade to dependents."""
        if query_key not in self._query_cache:
            return 0

        count = 1
        self._stats.invalidations += 1

        # Remove from cache
        del self._query_cache[query_key]

        # Cascade to dependent queries
        if query_key in self._reverse_deps:
            for dependent_key in list(self._reverse_deps[query_key]):
                count += self._invalidate_query(dependent_key)
            del self._reverse_deps[query_key]

        logger.debug(f"Invalidated: {query_key[0]}({query_key[1]})")
        return count

    def _invalidate_file_dependents(self, path: str) -> int:
        """Invalidate all queries depending on a file."""
        if path not in self._file_to_queries:
            return 0

        count = 0
        for query_key in list(self._file_to_queries[path]):
            count += self._invalidate_query(query_key)

        # Clear file->query mapping
        self._file_to_queries[path] = set()

        return count

    def invalidate_all(self) -> int:
        """Invalidate entire cache.

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            count = len(self._query_cache)
            self._query_cache.clear()
            self._reverse_deps.clear()
            self._file_to_queries.clear()
            self._stats.invalidations += count
            logger.debug(f"Invalidated all: {count} entries")
            return count

    # -------------------------------------------------------------------------
    # Stats and Inspection
    # -------------------------------------------------------------------------

    @property
    def stats(self) -> QueryStats:
        """Get query statistics."""
        return self._stats

    def cache_size(self) -> int:
        """Number of cached queries."""
        with self._lock:
            return len(self._query_cache)

    def file_count(self) -> int:
        """Number of tracked files."""
        with self._lock:
            return len(self._file_contents)

    def get_dependencies(self, func: Callable, *args: Any) -> set[QueryKey]:
        """Get dependencies of a cached query."""
        query_name = getattr(func, "_query_name", func.__name__)
        query_key: QueryKey = (query_name, args)

        with self._lock:
            if query_key in self._query_cache:
                return self._query_cache[query_key].dependencies.copy()
            return set()

    def get_dependents(self, func: Callable, *args: Any) -> set[QueryKey]:
        """Get queries that depend on this query."""
        query_name = getattr(func, "_query_name", func.__name__)
        query_key: QueryKey = (query_name, args)

        with self._lock:
            return self._reverse_deps.get(query_key, set()).copy()

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> None:
        """Save cache state to disk.

        Args:
            path: Path to save to (uses persist_path if not specified)
        """
        save_path = Path(path) if path else self._persist_path
        if not save_path:
            raise ValueError("No path specified and no persist_path configured")

        with self._lock:
            # Create serializable state
            state = {
                "files": {
                    p: {"content": c, "revision": self._file_revisions.get(p, 0)}
                    for p, c in self._file_contents.items()
                },
                "stats": self._stats.to_dict(),
            }

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(state, f, indent=2)

            logger.info(f"Saved SalsaDB state to {save_path}")

    def load(self, path: Path | str | None = None) -> bool:
        """Load cache state from disk.

        Args:
            path: Path to load from (uses persist_path if not specified)

        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = Path(path) if path else self._persist_path
        if not load_path or not load_path.exists():
            return False

        try:
            with open(load_path) as f:
                state = json.load(f)

            with self._lock:
                # Restore files
                for p, data in state.get("files", {}).items():
                    self._file_contents[p] = data["content"]
                    self._file_revisions[p] = data["revision"]

                logger.info(f"Loaded SalsaDB state from {load_path}")
                return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load SalsaDB state: {e}")
            return False


# -------------------------------------------------------------------------
# Convenience: Global DB instance
# -------------------------------------------------------------------------

_global_db: SalsaDB | None = None
_global_db_lock = threading.Lock()


def get_db() -> SalsaDB:
    """Get or create the global SalsaDB instance (thread-safe)."""
    global _global_db
    if _global_db is None:
        with _global_db_lock:
            # Double-check after acquiring lock
            if _global_db is None:
                _global_db = SalsaDB()
    return _global_db


def reset_db() -> None:
    """Reset the global SalsaDB instance (thread-safe)."""
    global _global_db
    with _global_db_lock:
        _global_db = None


# -------------------------------------------------------------------------
# Common Queries
# -------------------------------------------------------------------------

@salsa_query
def file_content(db: SalsaDB, path: str) -> str | None:
    """Query for file content with caching."""
    return db.get_file(path)


@salsa_query
def file_hash(db: SalsaDB, path: str) -> str | None:
    """Query for file content hash."""
    content = db.query(file_content, path)
    if content is None:
        return None
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@salsa_query
def file_lines(db: SalsaDB, path: str) -> list[str] | None:
    """Query for file content as lines."""
    content = db.query(file_content, path)
    if content is None:
        return None
    return content.split("\n")


@salsa_query
def file_line_count(db: SalsaDB, path: str) -> int:
    """Query for file line count."""
    lines = db.query(file_lines, path)
    return len(lines) if lines else 0


__all__ = [
    "SalsaDB",
    "salsa_query",
    "is_salsa_query",
    "CacheEntry",
    "QueryStats",
    "get_db",
    "reset_db",
    "file_content",
    "file_hash",
    "file_lines",
    "file_line_count",
]

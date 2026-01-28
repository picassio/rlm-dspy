"""Content-hashed indexing for deduplication.

Learned from llm-tldr's ContentHashedIndex pattern:
- Deduplicate identical content regardless of filename
- Save 10-20% storage by sharing entries for duplicate files
- Fast lookup using content hash as key
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def content_hash(content: str | bytes, algorithm: str = "sha256", length: int = 16) -> str:
    """Compute a truncated hash of content.

    Args:
        content: Content to hash (string or bytes)
        algorithm: Hash algorithm (sha256, md5, etc.)
        length: Number of hex characters to return

    Returns:
        Truncated hex digest
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    h = hashlib.new(algorithm)
    h.update(content)
    return h.hexdigest()[:length]


@dataclass
class ContentHashedIndex(Generic[T]):
    """Content-addressed storage with deduplication.

    Stores values indexed by content hash, allowing multiple paths
    to share the same entry if they have identical content.

    Example:
        index = ContentHashedIndex()

        # Store analysis result for a file
        index.set("src/utils.py", file_content, analysis_result)

        # If another file has same content, they share the entry
        index.set("lib/utils_copy.py", same_content, analysis_result)

        # Both paths resolve to same hash
        assert index.get("src/utils.py") == index.get("lib/utils_copy.py")

        # Stats show deduplication
        print(index.stats())  # "2 paths, 1 unique entries (50% dedup)"
    """

    # Hash -> stored value
    _entries: dict[str, T] = field(default_factory=dict)

    # Path -> hash (for reverse lookup)
    _path_to_hash: dict[str, str] = field(default_factory=dict)

    # Hash -> set of paths (for finding duplicates)
    _hash_to_paths: dict[str, set[str]] = field(default_factory=dict)

    # Stats
    _hits: int = 0
    _misses: int = 0

    def set(self, path: str, content: str | bytes, value: T) -> str:
        """Store a value indexed by content hash.

        Args:
            path: File path (for reverse lookup)
            content: Content to hash
            value: Value to store

        Returns:
            The content hash
        """
        h = content_hash(content)

        # Remove old mapping if path existed with different content
        if path in self._path_to_hash:
            old_hash = self._path_to_hash[path]
            if old_hash != h:
                self._hash_to_paths[old_hash].discard(path)
                if not self._hash_to_paths[old_hash]:
                    # No more paths reference this hash, clean up
                    del self._hash_to_paths[old_hash]
                    del self._entries[old_hash]

        # Store new mapping
        self._entries[h] = value
        self._path_to_hash[path] = h

        if h not in self._hash_to_paths:
            self._hash_to_paths[h] = set()
        self._hash_to_paths[h].add(path)

        return h

    def get(self, path: str) -> T | None:
        """Get value by path.

        Args:
            path: File path

        Returns:
            Stored value or None
        """
        h = self._path_to_hash.get(path)
        if h is None:
            self._misses += 1
            return None

        self._hits += 1
        return self._entries.get(h)

    def get_by_hash(self, h: str) -> T | None:
        """Get value by content hash directly."""
        return self._entries.get(h)

    def get_hash(self, path: str) -> str | None:
        """Get content hash for a path."""
        return self._path_to_hash.get(path)

    def get_duplicates(self, path: str) -> set[str]:
        """Get all paths with same content as given path.

        Args:
            path: File path to check

        Returns:
            Set of paths with identical content (excluding input path)
        """
        h = self._path_to_hash.get(path)
        if h is None:
            return set()

        paths = self._hash_to_paths.get(h, set())
        return paths - {path}

    def remove(self, path: str) -> bool:
        """Remove a path from the index.

        Args:
            path: Path to remove

        Returns:
            True if path existed
        """
        h = self._path_to_hash.pop(path, None)
        if h is None:
            return False

        self._hash_to_paths[h].discard(path)
        if not self._hash_to_paths[h]:
            del self._hash_to_paths[h]
            del self._entries[h]

        return True

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._path_to_hash.clear()
        self._hash_to_paths.clear()
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        """Number of unique entries (by content)."""
        return len(self._entries)

    def __contains__(self, path: str) -> bool:
        """Check if path is indexed."""
        return path in self._path_to_hash

    @property
    def path_count(self) -> int:
        """Number of indexed paths."""
        return len(self._path_to_hash)

    @property
    def dedup_ratio(self) -> float:
        """Deduplication ratio (0-1, higher = more dedup)."""
        if self.path_count == 0:
            return 0.0
        return 1 - (len(self._entries) / self.path_count)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "paths": self.path_count,
            "unique_entries": len(self._entries),
            "dedup_ratio": round(self.dedup_ratio * 100, 1),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate * 100, 1),
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        s = self.stats()
        return (
            f"ContentHashedIndex: {s['paths']} paths, "
            f"{s['unique_entries']} unique ({s['dedup_ratio']}% dedup), "
            f"{s['hit_rate']}% hit rate"
        )


@dataclass
class DirtyTracker:
    """Track dirty (modified) files for incremental updates.

    Learned from llm-tldr's dirty_flag.py pattern.

    Example:
        tracker = DirtyTracker()

        # Mark files as dirty when they change
        tracker.mark_dirty("src/main.py")
        tracker.mark_dirty("src/utils.py")

        # Check what needs reprocessing
        for path in tracker.get_dirty():
            process(path)
            tracker.mark_clean(path)
    """

    _dirty: set[str] = field(default_factory=set)
    _clean_count: int = 0

    def mark_dirty(self, path: str) -> None:
        """Mark a file as dirty (needs reprocessing)."""
        self._dirty.add(path)

    def mark_clean(self, path: str) -> None:
        """Mark a file as clean (processed)."""
        if path in self._dirty:
            self._dirty.discard(path)
            self._clean_count += 1

    def is_dirty(self, path: str) -> bool:
        """Check if a file is dirty."""
        return path in self._dirty

    def get_dirty(self) -> set[str]:
        """Get all dirty files."""
        return self._dirty.copy()

    def clear(self) -> None:
        """Clear all dirty flags."""
        self._dirty.clear()

    def __len__(self) -> int:
        """Number of dirty files."""
        return len(self._dirty)

    def __bool__(self) -> bool:
        """True if any files are dirty."""
        return bool(self._dirty)

    def stats(self) -> dict[str, int]:
        """Get tracker statistics."""
        return {
            "dirty": len(self._dirty),
            "cleaned": self._clean_count,
        }


__all__ = [
    "content_hash",
    "ContentHashedIndex",
    "DirtyTracker",
]

"""Paste Store for managing large content without context overflow.

Inspired by microcode's paste handling pattern. When content exceeds a
threshold, it's stored separately and replaced with a placeholder in the
immediate context. The full content is injected at final task construction.

This prevents large code blocks from overwhelming conversation history
while still making them available for analysis.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    """Get environment variable as int with default."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass
class PasteStore:
    """Manages large content placeholders to prevent context overflow.

    When text exceeds the threshold, it's stored and replaced with a
    placeholder like "[paste_1: 5000 chars]". The full content can be
    injected into the final context separately.

    Features:
    - LRU eviction when max_entries is exceeded
    - TTL-based expiration for stale entries
    - Memory-aware storage limits

    Example:
        store = PasteStore(threshold=2000, max_entries=100, ttl_seconds=3600)

        # Large content gets stored
        text, paste_id = store.maybe_store(large_code)
        # text = "[paste_1: 5000 chars]"
        # paste_id = "paste_1"

        # Later, inject all stored content
        full_context = store.inject_context() + "\\n" + query_context
    """

    threshold: int = field(
        default_factory=lambda: _env_int("RLM_PASTE_THRESHOLD", 2000)
    )
    max_entries: int = field(
        default_factory=lambda: _env_int("RLM_PASTE_MAX_ENTRIES", 100)
    )
    ttl_seconds: float = field(
        default_factory=lambda: float(_env_int("RLM_PASTE_TTL", 3600))  # 1 hour default
    )
    max_total_bytes: int = field(
        default_factory=lambda: _env_int("RLM_PASTE_MAX_BYTES", 50_000_000)  # 50MB default
    )

    # Internal state - using OrderedDict for LRU ordering
    _store: OrderedDict[str, str] = field(default_factory=OrderedDict)
    _counter: int = 0
    _metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    _total_bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _evict_expired(self) -> int:
        """Remove entries older than TTL. Returns count of evicted entries."""
        if self.ttl_seconds <= 0:
            return 0  # TTL disabled

        now = time.time()
        expired = []
        for paste_id, meta in self._metadata.items():
            if now - meta.get("created_at", now) > self.ttl_seconds:
                expired.append(paste_id)

        for paste_id in expired:
            self._remove(paste_id)

        if expired:
            logger.debug("PasteStore: evicted %d expired entries", len(expired))

        return len(expired)

    def _evict_lru(self, needed_bytes: int = 0) -> int:
        """Evict least recently used entries until under limits. Returns count evicted."""
        evicted = 0

        # Evict until under max_entries
        while len(self._store) >= self.max_entries:
            if not self._store:
                break
            # Pop oldest (first) item from OrderedDict
            oldest_id = next(iter(self._store))
            self._remove(oldest_id)
            evicted += 1

        # Evict until under max_total_bytes (considering needed_bytes for new entry)
        while self._total_bytes + needed_bytes > self.max_total_bytes and self._store:
            oldest_id = next(iter(self._store))
            self._remove(oldest_id)
            evicted += 1

        if evicted:
            logger.debug("PasteStore: LRU evicted %d entries", evicted)

        return evicted

    def _remove(self, paste_id: str) -> None:
        """Remove a paste by ID."""
        if paste_id in self._store:
            content = self._store.pop(paste_id)
            self._total_bytes -= len(content.encode('utf-8'))
        self._metadata.pop(paste_id, None)

    def _touch(self, paste_id: str) -> None:
        """Move entry to end of OrderedDict (most recently used)."""
        if paste_id in self._store:
            self._store.move_to_end(paste_id)

    def maybe_store(
        self,
        text: str,
        label: str | None = None,
    ) -> tuple[str, str | None]:
        """Store text if it exceeds threshold, returning placeholder (thread-safe).

        Args:
            text: Content to potentially store
            label: Optional label for the paste (e.g., "file:auth.py")

        Returns:
            Tuple of (text_or_placeholder, paste_id_or_none)
            If text is under threshold, returns (text, None)
            If stored, returns ("[paste_N: X chars]", "paste_N")
        """
        if len(text) <= self.threshold:
            return text, None

        with self._lock:
            # Evict expired entries first
            self._evict_expired()

            # Calculate bytes needed
            text_bytes = len(text.encode('utf-8'))

            # Evict LRU entries if needed
            self._evict_lru(needed_bytes=text_bytes)

            self._counter += 1
            paste_id = f"paste_{self._counter}"
            self._store[paste_id] = text
            self._total_bytes += text_bytes
            self._metadata[paste_id] = {
                "chars": len(text),
                "lines": text.count("\n") + 1,
                "label": label,
                "created_at": time.time(),
                "bytes": text_bytes,
            }

            # Create informative placeholder
            meta = self._metadata[paste_id]
            label_str = f" ({label})" if label else ""
            placeholder = f"[{paste_id}{label_str}: {meta['chars']} chars, {meta['lines']} lines]"

            return placeholder, paste_id

    def store(self, text: str, label: str | None = None) -> str:
        """Force store text regardless of threshold (thread-safe).

        Args:
            text: Content to store
            label: Optional label for the paste

        Returns:
            The paste_id
        """
        with self._lock:
            # Evict expired entries first
            self._evict_expired()

            # Calculate bytes needed
            text_bytes = len(text.encode('utf-8'))

            # Evict LRU entries if needed
            self._evict_lru(needed_bytes=text_bytes)

            self._counter += 1
            paste_id = f"paste_{self._counter}"
            self._store[paste_id] = text
        self._total_bytes += text_bytes
        self._metadata[paste_id] = {
            "chars": len(text),
            "lines": text.count("\n") + 1,
            "label": label,
            "created_at": time.time(),
            "bytes": text_bytes,
        }
        return paste_id

    def get(self, paste_id: str) -> str | None:
        """Retrieve stored content by ID. Marks as recently used (thread-safe)."""
        with self._lock:
            # Check expiration
            if paste_id in self._metadata:
                meta = self._metadata[paste_id]
                if self.ttl_seconds > 0:
                    age = time.time() - meta.get("created_at", 0)
                    if age > self.ttl_seconds:
                        self._remove(paste_id)
                        return None

            content = self._store.get(paste_id)
            if content is not None:
                self._touch(paste_id)  # Mark as recently used
            return content

    def inject_context(self, max_per_paste: int | None = None) -> str:
        """Generate context section with all stored pastes (thread-safe).

        Args:
            max_per_paste: Optional max chars per paste (truncates if exceeded)

        Returns:
            Formatted string with all stored content
        """
        with self._lock:
            # Evict expired before generating context
            self._evict_expired()

            if not self._store:
                return ""

            lines = ["## Stored Content\n"]
            for paste_id, content in self._store.items():
                meta = self._metadata.get(paste_id, {})
                label = meta.get("label", "")
                label_str = f" ({label})" if label else ""

                lines.append(f"### [{paste_id}]{label_str}")
                lines.append("```")

                if max_per_paste and len(content) > max_per_paste:
                    lines.append(content[:max_per_paste])
                    lines.append(f"\n... truncated ({len(content) - max_per_paste} more chars)")
                else:
                    lines.append(content)

                lines.append("```\n")

            return "\n".join(lines)

    def summary(self) -> str:
        """Get a summary of stored content without the full text (thread-safe)."""
        with self._lock:
            # Evict expired before summary
            self._evict_expired()

            if not self._store:
                return "No stored content"

            lines = ["Stored content:"]
            total_chars = 0
            for paste_id in self._store:
                meta = self._metadata.get(paste_id, {})
                chars = meta.get("chars", 0)
                label = meta.get("label", "")
                total_chars += chars
                label_str = f" ({label})" if label else ""
                lines.append(f"  - [{paste_id}]{label_str}: {chars} chars")

            lines.append(f"Total: {total_chars} chars in {len(self._store)} pastes")
            lines.append(f"Memory: {self._total_bytes:,} bytes")
            return "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored content (thread-safe)."""
        with self._lock:
            self._store.clear()
            self._metadata.clear()
            self._counter = 0
            self._total_bytes = 0

    def stats(self) -> dict[str, Any]:
        """Get storage statistics (thread-safe)."""
        with self._lock:
            return {
                "entries": len(self._store),
                "total_bytes": self._total_bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_total_bytes,
                "ttl_seconds": self.ttl_seconds,
                "utilization_percent": round(self._total_bytes / self.max_total_bytes * 100, 1)
                if self.max_total_bytes > 0 else 0,
            }

    def __len__(self) -> int:
        """Number of stored pastes."""
        with self._lock:
            return len(self._store)

    def __bool__(self) -> bool:
        """True if any pastes are stored."""
        with self._lock:
            return bool(self._store)


# Convenience function for one-off use
def store_large_content(
    text: str,
    threshold: int = 2000,
    label: str | None = None,
) -> tuple[str, PasteStore]:
    """Store large content and return placeholder + store.

    Args:
        text: Content to potentially store
        threshold: Character threshold for storing
        label: Optional label

    Returns:
        Tuple of (text_or_placeholder, store_instance)
    """
    store = PasteStore(threshold=threshold)
    result, _ = store.maybe_store(text, label)
    return result, store


__all__ = ["PasteStore", "store_large_content"]

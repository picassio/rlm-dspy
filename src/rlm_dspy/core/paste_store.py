"""Paste Store for managing large content without context overflow.

Inspired by microcode's paste handling pattern. When content exceeds a
threshold, it's stored separately and replaced with a placeholder in the
immediate context. The full content is injected at final task construction.

This prevents large code blocks from overwhelming conversation history
while still making them available for analysis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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

    Example:
        store = PasteStore(threshold=2000)

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
    _store: dict[str, str] = field(default_factory=dict)
    _counter: int = 0
    _metadata: dict[str, dict] = field(default_factory=dict)

    def maybe_store(
        self,
        text: str,
        label: str | None = None,
    ) -> tuple[str, str | None]:
        """Store text if it exceeds threshold, returning placeholder.

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

        self._counter += 1
        paste_id = f"paste_{self._counter}"
        self._store[paste_id] = text
        self._metadata[paste_id] = {
            "chars": len(text),
            "lines": text.count("\n") + 1,
            "label": label,
        }

        # Create informative placeholder
        meta = self._metadata[paste_id]
        label_str = f" ({label})" if label else ""
        placeholder = f"[{paste_id}{label_str}: {meta['chars']} chars, {meta['lines']} lines]"

        return placeholder, paste_id

    def store(self, text: str, label: str | None = None) -> str:
        """Force store text regardless of threshold.

        Args:
            text: Content to store
            label: Optional label for the paste

        Returns:
            The paste_id
        """
        self._counter += 1
        paste_id = f"paste_{self._counter}"
        self._store[paste_id] = text
        self._metadata[paste_id] = {
            "chars": len(text),
            "lines": text.count("\n") + 1,
            "label": label,
        }
        return paste_id

    def get(self, paste_id: str) -> str | None:
        """Retrieve stored content by ID."""
        return self._store.get(paste_id)

    def inject_context(self, max_per_paste: int | None = None) -> str:
        """Generate context section with all stored pastes.

        Args:
            max_per_paste: Optional max chars per paste (truncates if exceeded)

        Returns:
            Formatted string with all stored content
        """
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
        """Get a summary of stored content without the full text."""
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
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored content."""
        self._store.clear()
        self._metadata.clear()
        self._counter = 0

    def __len__(self) -> int:
        """Number of stored pastes."""
        return len(self._store)

    def __bool__(self) -> bool:
        """True if any pastes are stored."""
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

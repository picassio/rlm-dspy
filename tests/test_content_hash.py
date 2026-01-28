"""Tests for content-hashed indexing."""

import pytest

from rlm_dspy.core.content_hash import (
    ContentHashedIndex,
    DirtyTracker,
    content_hash,
)


class TestContentHash:
    """Tests for content_hash function."""

    def test_string_hash(self):
        """String content hashes correctly."""
        h = content_hash("hello world")
        assert len(h) == 16
        assert h.isalnum()

    def test_bytes_hash(self):
        """Bytes content hashes correctly."""
        h = content_hash(b"hello world")
        assert len(h) == 16

    def test_same_content_same_hash(self):
        """Same content produces same hash."""
        h1 = content_hash("test content")
        h2 = content_hash("test content")
        assert h1 == h2

    def test_different_content_different_hash(self):
        """Different content produces different hash."""
        h1 = content_hash("content a")
        h2 = content_hash("content b")
        assert h1 != h2

    def test_custom_length(self):
        """Custom hash length works."""
        h = content_hash("test", length=8)
        assert len(h) == 8


class TestContentHashedIndex:
    """Tests for ContentHashedIndex."""

    def test_set_and_get(self):
        """Basic set and get works."""
        index = ContentHashedIndex[str]()

        index.set("file.txt", "content", "result")
        result = index.get("file.txt")

        assert result == "result"

    def test_get_missing(self):
        """Get missing path returns None."""
        index = ContentHashedIndex[str]()
        assert index.get("missing.txt") is None

    def test_deduplication(self):
        """Same content shares entry."""
        index = ContentHashedIndex[str]()
        content = "shared content"

        index.set("file1.txt", content, "result")
        index.set("file2.txt", content, "result")

        # Only 1 unique entry
        assert len(index) == 1
        # But 2 paths
        assert index.path_count == 2
        # Both return same result
        assert index.get("file1.txt") == index.get("file2.txt")

    def test_dedup_ratio(self):
        """Dedup ratio is calculated correctly."""
        index = ContentHashedIndex[str]()

        # Add 4 files with 2 unique contents
        index.set("a.txt", "content1", "r1")
        index.set("b.txt", "content1", "r1")
        index.set("c.txt", "content2", "r2")
        index.set("d.txt", "content2", "r2")

        # 4 paths, 2 unique = 50% dedup
        assert index.dedup_ratio == 0.5

    def test_get_duplicates(self):
        """Get duplicates returns paths with same content."""
        index = ContentHashedIndex[str]()
        content = "shared"

        index.set("a.txt", content, "r")
        index.set("b.txt", content, "r")
        index.set("c.txt", content, "r")

        duplicates = index.get_duplicates("a.txt")

        assert duplicates == {"b.txt", "c.txt"}

    def test_remove(self):
        """Remove works correctly."""
        index = ContentHashedIndex[str]()

        index.set("file.txt", "content", "result")
        assert "file.txt" in index

        removed = index.remove("file.txt")

        assert removed
        assert "file.txt" not in index
        assert index.get("file.txt") is None

    def test_remove_with_duplicates(self):
        """Remove one duplicate keeps others."""
        index = ContentHashedIndex[str]()
        content = "shared"

        index.set("a.txt", content, "r")
        index.set("b.txt", content, "r")

        index.remove("a.txt")

        # b.txt still works
        assert index.get("b.txt") == "r"
        assert len(index) == 1

    def test_update_content(self):
        """Updating content updates hash."""
        index = ContentHashedIndex[str]()

        index.set("file.txt", "old content", "old result")
        h1 = index.get_hash("file.txt")

        index.set("file.txt", "new content", "new result")
        h2 = index.get_hash("file.txt")

        assert h1 != h2
        assert index.get("file.txt") == "new result"

    def test_hit_rate(self):
        """Hit rate is tracked correctly."""
        index = ContentHashedIndex[str]()

        index.set("file.txt", "content", "result")

        index.get("file.txt")  # hit
        index.get("file.txt")  # hit
        index.get("missing.txt")  # miss

        assert index.hit_rate == pytest.approx(2/3, rel=0.01)

    def test_stats(self):
        """Stats returns correct info."""
        index = ContentHashedIndex[str]()

        index.set("a.txt", "c1", "r1")
        index.set("b.txt", "c1", "r1")
        index.get("a.txt")

        stats = index.stats()

        assert stats["paths"] == 2
        assert stats["unique_entries"] == 1
        assert stats["dedup_ratio"] == 50.0
        assert stats["hits"] == 1

    def test_clear(self):
        """Clear removes everything."""
        index = ContentHashedIndex[str]()

        index.set("a.txt", "c1", "r1")
        index.set("b.txt", "c2", "r2")

        index.clear()

        assert len(index) == 0
        assert index.path_count == 0


class TestDirtyTracker:
    """Tests for DirtyTracker."""

    def test_mark_dirty(self):
        """Mark dirty works."""
        tracker = DirtyTracker()

        tracker.mark_dirty("file.txt")

        assert tracker.is_dirty("file.txt")
        assert len(tracker) == 1

    def test_mark_clean(self):
        """Mark clean works."""
        tracker = DirtyTracker()

        tracker.mark_dirty("file.txt")
        tracker.mark_clean("file.txt")

        assert not tracker.is_dirty("file.txt")
        assert len(tracker) == 0

    def test_get_dirty(self):
        """Get dirty returns all dirty files."""
        tracker = DirtyTracker()

        tracker.mark_dirty("a.txt")
        tracker.mark_dirty("b.txt")
        tracker.mark_dirty("c.txt")

        dirty = tracker.get_dirty()

        assert dirty == {"a.txt", "b.txt", "c.txt"}

    def test_bool(self):
        """Bool conversion works."""
        tracker = DirtyTracker()

        assert not tracker

        tracker.mark_dirty("file.txt")

        assert tracker

    def test_stats(self):
        """Stats tracks cleaned count."""
        tracker = DirtyTracker()

        tracker.mark_dirty("a.txt")
        tracker.mark_dirty("b.txt")
        tracker.mark_clean("a.txt")

        stats = tracker.stats()

        assert stats["dirty"] == 1
        assert stats["cleaned"] == 1

    def test_clear(self):
        """Clear removes all dirty flags."""
        tracker = DirtyTracker()

        tracker.mark_dirty("a.txt")
        tracker.mark_dirty("b.txt")

        tracker.clear()

        assert len(tracker) == 0

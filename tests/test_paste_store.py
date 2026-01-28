"""Tests for PasteStore large content handling."""

import pytest

from rlm_dspy.core.paste_store import PasteStore, store_large_content


class TestPasteStore:
    """Tests for PasteStore class."""

    def test_small_content_not_stored(self):
        """Content under threshold is returned as-is."""
        store = PasteStore(threshold=100)
        text = "small content"
        result, paste_id = store.maybe_store(text)
        assert result == text
        assert paste_id is None
        assert len(store) == 0

    def test_large_content_stored(self):
        """Content over threshold is stored and placeholder returned."""
        store = PasteStore(threshold=100)
        text = "x" * 200
        result, paste_id = store.maybe_store(text)
        
        assert paste_id == "paste_1"
        assert "[paste_1:" in result
        assert "200 chars" in result
        assert len(store) == 1
        assert store.get("paste_1") == text

    def test_multiple_pastes(self):
        """Multiple large contents get unique IDs."""
        store = PasteStore(threshold=50)
        
        text1 = "a" * 100
        text2 = "b" * 100
        
        _, id1 = store.maybe_store(text1)
        _, id2 = store.maybe_store(text2)
        
        assert id1 == "paste_1"
        assert id2 == "paste_2"
        assert len(store) == 2
        assert store.get("paste_1") == text1
        assert store.get("paste_2") == text2

    def test_force_store(self):
        """Force store works regardless of threshold."""
        store = PasteStore(threshold=1000)
        text = "small"
        paste_id = store.store(text)
        
        assert paste_id == "paste_1"
        assert store.get("paste_1") == text

    def test_with_label(self):
        """Labels are included in placeholder."""
        store = PasteStore(threshold=50)
        text = "x" * 100
        result, _ = store.maybe_store(text, label="file:auth.py")
        
        assert "(file:auth.py)" in result

    def test_inject_context(self):
        """Inject context formats all stored content."""
        store = PasteStore(threshold=50)
        store.maybe_store("a" * 100, label="first")
        store.maybe_store("b" * 100, label="second")
        
        context = store.inject_context()
        
        assert "## Stored Content" in context
        assert "[paste_1]" in context
        assert "[paste_2]" in context
        assert "a" * 100 in context
        assert "b" * 100 in context

    def test_inject_context_with_truncation(self):
        """Inject context respects max_per_paste."""
        store = PasteStore(threshold=50)
        store.maybe_store("x" * 1000)
        
        context = store.inject_context(max_per_paste=100)
        
        assert "truncated" in context
        assert "900 more chars" in context

    def test_summary(self):
        """Summary shows overview without full content."""
        store = PasteStore(threshold=50)
        store.maybe_store("a" * 100, label="test")
        
        summary = store.summary()
        
        assert "paste_1" in summary
        assert "100 chars" in summary
        assert "test" in summary
        # Should not contain the actual content
        assert "a" * 100 not in summary

    def test_clear(self):
        """Clear removes all stored content."""
        store = PasteStore(threshold=50)
        store.maybe_store("x" * 100)
        store.maybe_store("y" * 100)
        
        assert len(store) == 2
        store.clear()
        assert len(store) == 0
        assert store.get("paste_1") is None

    def test_bool(self):
        """Boolean conversion works correctly."""
        store = PasteStore(threshold=50)
        assert not store
        
        store.maybe_store("x" * 100)
        assert store

    def test_empty_inject_context(self):
        """Empty store returns empty string."""
        store = PasteStore()
        assert store.inject_context() == ""


class TestStoreLargeContent:
    """Tests for convenience function."""

    def test_store_large_content(self):
        """Convenience function works correctly."""
        text = "x" * 5000
        result, store = store_large_content(text, threshold=1000)
        
        assert "[paste_1:" in result
        assert len(store) == 1
        assert store.get("paste_1") == text

    def test_store_small_content(self):
        """Small content is not stored."""
        text = "small"
        result, store = store_large_content(text, threshold=1000)
        
        assert result == text
        assert len(store) == 0

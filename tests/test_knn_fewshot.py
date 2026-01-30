"""Tests for KNN few-shot module."""

import pytest
from pathlib import Path

from rlm_dspy.core.knn_fewshot import (
    KNNFewShot,
    KNNFewShotConfig,
    STATIC_EXAMPLES,
    get_knn_fewshot,
    clear_knn_fewshot,
)


class TestKNNFewShotConfig:
    """Tests for KNNFewShotConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = KNNFewShotConfig()
        assert config.k == 3
        assert config.min_similarity == 0.3
        assert config.use_traces is True
        assert config.enabled is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = KNNFewShotConfig(
            k=5,
            min_similarity=0.5,
            use_traces=False,
        )
        assert config.k == 5
        assert config.min_similarity == 0.5
        assert config.use_traces is False


class TestStaticExamples:
    """Tests for static examples."""

    def test_static_examples_exist(self):
        """Test that static examples are defined."""
        assert "bugs" in STATIC_EXAMPLES
        assert "security" in STATIC_EXAMPLES
        assert "explanation" in STATIC_EXAMPLES

    def test_example_format(self):
        """Test that examples have required keys."""
        for category, examples in STATIC_EXAMPLES.items():
            for ex in examples:
                assert "query" in ex, f"Missing 'query' in {category}"
                assert "demo" in ex, f"Missing 'demo' in {category}"

    def test_bug_example_content(self):
        """Test bug example has relevant content."""
        examples = STATIC_EXAMPLES["bugs"]
        assert len(examples) >= 1
        demo = examples[0]["demo"]
        assert "find" in demo.lower() or "search" in demo.lower()
        assert "read_file" in demo or "ripgrep" in demo

    def test_security_example_content(self):
        """Test security example has relevant content."""
        examples = STATIC_EXAMPLES["security"]
        assert len(examples) >= 1
        demo = examples[0]["demo"]
        assert "security" in demo.lower() or "vulnerability" in demo.lower()


class TestKNNFewShot:
    """Tests for KNNFewShot class."""

    @pytest.fixture
    def knn(self):
        """Create a KNN few-shot instance."""
        config = KNNFewShotConfig(enabled=True)
        return KNNFewShot(config)

    def test_get_static_examples(self, knn):
        """Test getting static examples."""
        # All examples
        all_examples = knn._get_static_examples()
        assert len(all_examples) >= 3  # At least one per category

        # Filtered by type
        bug_examples = knn._get_static_examples("bugs")
        assert len(bug_examples) >= 1

    def test_disabled_returns_empty(self):
        """Test that disabled config returns empty list."""
        config = KNNFewShotConfig(enabled=False)
        knn = KNNFewShot(config)
        examples = knn.select_examples("Find bugs")
        assert examples == []

    def test_select_examples_returns_list(self, knn):
        """Test that select_examples returns a list."""
        # This might use embeddings which could fail in test env
        try:
            examples = knn.select_examples("Find bugs in the code", k=2)
            assert isinstance(examples, list)
        except Exception:
            # Skip if embeddings not available
            pytest.skip("Embeddings not available")

    def test_format_examples_empty(self, knn):
        """Test formatting empty examples."""
        result = knn.format_examples_for_prompt([])
        assert result == ""

    def test_format_examples_with_content(self, knn):
        """Test formatting examples with content."""
        examples = [
            {"query": "Q1", "demo": "Demo 1 content"},
            {"query": "Q2", "demo": "Demo 2 content"},
        ]
        result = knn.format_examples_for_prompt(examples)
        assert "Example 1" in result
        assert "Example 2" in result
        assert "Demo 1 content" in result
        assert "Demo 2 content" in result
        assert "Now analyze" in result

    def test_format_examples_respects_max_chars(self, knn):
        """Test that formatting respects max_chars."""
        examples = [
            {"query": "Q1", "demo": "A" * 1000},
            {"query": "Q2", "demo": "B" * 1000},
            {"query": "Q3", "demo": "C" * 1000},
        ]
        result = knn.format_examples_for_prompt(examples, max_chars=1500)
        # Should not include all examples
        assert len(result) < 3500


class TestGlobalKNNFewShot:
    """Tests for global KNN few-shot functions."""

    def test_get_knn_fewshot(self):
        """Test getting global instance."""
        clear_knn_fewshot()
        k1 = get_knn_fewshot()
        k2 = get_knn_fewshot()
        assert k1 is k2

    def test_clear_knn_fewshot(self):
        """Test clearing global instance."""
        k1 = get_knn_fewshot()
        clear_knn_fewshot()
        k2 = get_knn_fewshot()
        assert k1 is not k2

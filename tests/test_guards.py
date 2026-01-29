"""Tests for hallucination guards (LLM-as-judge).

Note: These tests require LLM API access and are marked for integration testing.
Unit tests only verify the data structures.
"""

import pytest
from rlm_dspy.guards import GroundednessResult


class TestGroundednessResult:
    """Tests for GroundednessResult dataclass."""

    def test_grounded_result(self):
        """Result with high score should be grounded."""
        result = GroundednessResult(
            score=0.9,
            claims="claim1, claim2",
            discussion="All claims supported",
            is_grounded=True,
            threshold=0.66,
        )
        assert result.is_grounded
        assert result.score == 0.9

    def test_not_grounded_result(self):
        """Result with low score should not be grounded."""
        result = GroundednessResult(
            score=0.4,
            claims="fake claim",
            discussion="Claim not supported",
            is_grounded=False,
            threshold=0.66,
        )
        assert not result.is_grounded
        assert result.score == 0.4

    def test_threshold_boundary(self):
        """Score at threshold should be grounded."""
        result = GroundednessResult(
            score=0.66,
            claims="claims",
            discussion="discussion",
            is_grounded=True,
            threshold=0.66,
        )
        assert result.is_grounded

    def test_default_threshold(self):
        """Default threshold should be 0.66."""
        result = GroundednessResult(
            score=0.5,
            claims="",
            discussion="",
            is_grounded=False,
        )
        assert result.threshold == 0.66


# Integration tests (require LLM API)
# These are skipped by default - run with: pytest -m integration

@pytest.mark.integration
class TestValidateGroundedness:
    """Integration tests for validate_groundedness (require LLM API)."""

    def test_grounded_output(self):
        """Output matching context should be grounded."""
        from rlm_dspy.guards import validate_groundedness

        context = "def add(a, b):\n    return a + b"
        output = "The add function takes two parameters and returns their sum."
        query = "What does this code do?"

        result = validate_groundedness(output, context, query)
        assert result.is_grounded
        assert result.score >= 0.66

    def test_hallucinated_output(self):
        """Output with fake claims should not be grounded."""
        from rlm_dspy.guards import validate_groundedness

        context = "def add(a, b):\n    return a + b"
        output = "The multiply() function handles data processing on line 500."
        query = "What does this code do?"

        result = validate_groundedness(output, context, query)
        assert not result.is_grounded
        assert result.score < 0.66

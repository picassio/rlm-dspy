"""Tests for hallucination guards (LLM-as-judge).

Note: These tests require LLM API access and are marked for integration testing.
Unit tests only verify the data structures.
"""

import pytest
from rlm_dspy.guards import (
    GroundednessResult,
    TrajectoryValidationResult,
    _extract_keywords,
    _extract_specific_terms,
    _find_relevant_snippets,
    validate_trajectory,
    LARGE_CONTEXT_THRESHOLD,
)


class TestKeywordExtraction:
    """Tests for keyword extraction functions."""

    def test_extract_keywords_basic(self):
        """Should extract meaningful keywords."""
        text = "The function process_data handles input validation"
        keywords = _extract_keywords(text)
        assert "function" in keywords
        assert "process_data" in keywords
        assert "handles" in keywords
        assert "validation" in keywords
        # Stop words should be excluded
        assert "the" not in keywords
        assert "of" not in keywords  # stop word

    def test_extract_keywords_code(self):
        """Should extract identifiers from code."""
        text = "def calculate_total(items): return sum(items)"
        keywords = _extract_keywords(text)
        assert "calculate_total" in keywords
        assert "items" in keywords
        assert "return" in keywords

    def test_extract_specific_terms_filenames(self):
        """Should extract filenames."""
        text = "The file main.py imports from utils.py and config.json"
        terms = _extract_specific_terms(text)
        assert "main.py" in terms
        assert "utils.py" in terms
        assert "config.json" in terms

    def test_extract_specific_terms_identifiers(self):
        """Should extract identifiers with underscores."""
        text = "process_data calls validate_input and returns parsed_result"
        terms = _extract_specific_terms(text)
        assert "process_data" in terms
        assert "validate_input" in terms
        assert "parsed_result" in terms

    def test_extract_specific_terms_quoted(self):
        """Should extract quoted strings."""
        text = 'The error message is "Invalid input" and status is "failed"'
        terms = _extract_specific_terms(text)
        assert "invalid input" in terms
        assert "failed" in terms

    def test_extract_specific_terms_numbers(self):
        """Should extract significant numbers."""
        text = "Version 1.2.3, line 42, and error code 500"
        terms = _extract_specific_terms(text)
        assert "1.2.3" in terms
        assert "42" in terms
        assert "500" in terms


class TestSnippetExtraction:
    """Tests for relevant snippet extraction."""

    def test_small_context_unchanged(self):
        """Small contexts should be returned as-is."""
        context = "def add(a, b): return a + b"
        output = "The add function returns the sum"
        query = "What does add do?"
        
        result = _find_relevant_snippets(context, output, query)
        assert "add" in result
        assert "return" in result

    def test_large_context_extracts_relevant(self):
        """Large contexts should extract relevant snippets."""
        # Create a large context with relevant content in the middle
        padding = "x" * 10000
        relevant = "def process_data(items): return [x*2 for x in items]"
        context = padding + "\n" + relevant + "\n" + padding
        
        output = "process_data doubles each item"
        query = "What does process_data do?"
        
        result = _find_relevant_snippets(context, output, query)
        assert "process_data" in result
        assert "items" in result

    def test_finds_filenames_in_context(self):
        """Should find snippets containing mentioned filenames."""
        context = """
        File: main.py
        def main(): pass
        
        File: utils.py
        def helper(): pass
        
        File: config.json
        {"key": "value"}
        """
        output = "main.py contains the main function"
        query = "What's in main.py?"
        
        result = _find_relevant_snippets(context, output, query)
        assert "main.py" in result
        assert "def main" in result


class TestTrajectoryValidation:
    """Tests for trajectory-based validation."""

    def test_grounded_answer(self):
        """Answer with terms from trajectory should be grounded."""
        # Mock RLMResult-like object
        class MockResult:
            answer = "The version is 0.1.0"
            trajectory = [
                {"output": "version = '0.1.0'"},
                {"output": "Found: 0.1.0"},
            ]
        
        result = validate_trajectory(MockResult())
        assert result.is_grounded
        assert result.score >= 0.5
        assert "0.1.0" in result.found_terms

    def test_ungrounded_answer(self):
        """Answer with terms NOT in trajectory should not be grounded."""
        class MockResult:
            answer = "The file main.py has a bug on line 42"
            trajectory = [
                {"output": "utils.py: no issues"},
                {"output": "config.json: valid"},
            ]
        
        result = validate_trajectory(MockResult())
        assert not result.is_grounded
        assert "main.py" in result.missing_terms

    def test_empty_trajectory(self):
        """Empty trajectory should return grounded (no validation possible)."""
        class MockResult:
            answer = "Some answer"
            trajectory = []
        
        result = validate_trajectory(MockResult())
        assert result.is_grounded
        assert result.score == 1.0

    def test_partial_match(self):
        """Partial matches should give partial score."""
        class MockResult:
            answer = "Found main.py and utils.py"
            trajectory = [{"output": "main.py exists"}]  # Only main.py in output
        
        result = validate_trajectory(MockResult())
        # main.py found, utils.py not found
        assert 0 < result.score < 1.0
        assert "main.py" in result.found_terms
        assert "utils.py" in result.missing_terms


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

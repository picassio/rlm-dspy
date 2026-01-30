"""Tests for grounded proposer module."""

import pytest
from pathlib import Path

from rlm_dspy.core.grounded_proposer import (
    GroundedProposer,
    ProposerConfig,
    FailureRecord,
    SuccessRecord,
    DEFAULT_TIPS,
    get_grounded_proposer,
    clear_grounded_proposer,
)


class TestFailureRecord:
    """Tests for FailureRecord dataclass."""

    def test_create_record(self):
        """Test creating a failure record."""
        record = FailureRecord(
            query="Find bugs",
            query_type="bugs",
            failure_reason="Claims not verified",
            grounded_score=0.3,
            ungrounded_claims=["Bug at line 10"],
        )
        assert record.query == "Find bugs"
        assert record.failure_reason == "Claims not verified"
        assert record.grounded_score == 0.3

    def test_record_to_dict(self):
        """Test converting record to dict."""
        record = FailureRecord(
            query="Test",
            query_type="general",
            failure_reason="Low score",
            grounded_score=0.2,
        )
        data = record.to_dict()
        assert data["query"] == "Test"
        assert data["failure_reason"] == "Low score"

    def test_record_from_dict(self):
        """Test creating record from dict."""
        data = {
            "query": "Test",
            "query_type": "bugs",
            "failure_reason": "Missing verification",
            "grounded_score": 0.25,
            "ungrounded_claims": ["Claim 1"],
            "tools_used": ["ripgrep"],
            "tools_missing": ["read_file"],
            "timestamp": "2025-01-30T00:00:00Z",
        }
        record = FailureRecord.from_dict(data)
        assert record.query_type == "bugs"
        assert "read_file" in record.tools_missing


class TestSuccessRecord:
    """Tests for SuccessRecord dataclass."""

    def test_create_record(self):
        """Test creating a success record."""
        record = SuccessRecord(
            query="Find bugs",
            query_type="bugs",
            grounded_score=0.95,
            tools_used=["read_file", "ripgrep"],
        )
        assert record.grounded_score == 0.95
        assert "read_file" in record.tools_used

    def test_record_to_dict(self):
        """Test converting to dict."""
        record = SuccessRecord(
            query="Test",
            query_type="security",
            grounded_score=0.9,
            key_patterns=["verified with read_file"],
        )
        data = record.to_dict()
        assert data["grounded_score"] == 0.9
        assert "verified with read_file" in data["key_patterns"]


class TestProposerConfig:
    """Tests for ProposerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ProposerConfig()
        assert config.max_failures == 200
        assert config.max_successes == 200
        assert config.tip_refresh_interval == 50
        assert config.enabled is True

    def test_custom_config(self, tmp_path):
        """Test custom configuration."""
        config = ProposerConfig(
            storage_path=tmp_path / "proposer",
            max_failures=100,
            tip_refresh_interval=20,
        )
        assert config.max_failures == 100
        assert config.tip_refresh_interval == 20


class TestGroundedProposer:
    """Tests for GroundedProposer."""

    @pytest.fixture
    def proposer(self, tmp_path):
        """Create a proposer with temp storage."""
        config = ProposerConfig(
            storage_path=tmp_path / "proposer",
            max_failures=100,
            max_successes=100,
            tip_refresh_interval=100,  # High to avoid auto-refresh
            enabled=True,
        )
        return GroundedProposer(config)

    def test_record_failure(self, proposer):
        """Test recording a failure."""
        proposer.record_failure(
            query="Find bugs",
            query_type="bugs",
            failure_reason="Claims not verified",
            grounded_score=0.3,
            tools_used=["ripgrep"],
        )
        assert len(proposer.failures) == 1
        assert proposer.failures[0].failure_reason == "Claims not verified"

    def test_record_success(self, proposer):
        """Test recording a success."""
        proposer.record_success(
            query="Find bugs",
            query_type="bugs",
            grounded_score=0.95,
            tools_used=["read_file", "ripgrep"],
        )
        assert len(proposer.successes) == 1
        assert proposer.successes[0].grounded_score == 0.95

    def test_infer_missing_tools(self, proposer):
        """Test inferring missing tools."""
        # Line number claim without read_file
        missing = proposer._infer_missing_tools(
            failure_reason="Line number wrong",
            tools_used=["ripgrep"],
            ungrounded_claims=["Bug at line 50"],
        )
        assert "read_file" in missing

        # Not found without search
        missing = proposer._infer_missing_tools(
            failure_reason="Function not found",
            tools_used=[],
            ungrounded_claims=[],
        )
        assert any("ripgrep" in m or "find" in m for m in missing)

    def test_extract_success_patterns(self, proposer):
        """Test extracting success patterns."""
        patterns = proposer._extract_success_patterns(["read_file", "ripgrep", "find_functions"])
        assert "verified with read_file" in patterns
        assert "used structural search first" in patterns
        assert "multiple verification steps" in patterns

    def test_default_tips(self, proposer):
        """Test that default tips are loaded."""
        tips = proposer.get_tips()
        assert len(tips) > 0
        assert tips == DEFAULT_TIPS

    def test_persistence(self, tmp_path):
        """Test that records persist across instances."""
        config = ProposerConfig(
            storage_path=tmp_path / "proposer",
            enabled=True,
        )

        # Create and record
        p1 = GroundedProposer(config)
        p1.record_failure("Q1", "bugs", "reason", 0.3)
        p1.record_success("Q2", "bugs", 0.9)

        # Create new instance
        p2 = GroundedProposer(config)
        assert len(p2.failures) == 1
        assert len(p2.successes) == 1

    def test_enforce_limits(self, tmp_path):
        """Test that limits are enforced."""
        config = ProposerConfig(
            storage_path=tmp_path / "proposer",
            max_failures=5,
            max_successes=5,
            enabled=True,
        )
        proposer = GroundedProposer(config)

        # Add more than max
        for i in range(10):
            proposer.record_failure(f"Q{i}", "bugs", "reason", 0.3)
            proposer.record_success(f"S{i}", "bugs", 0.9)

        assert len(proposer.failures) <= 5
        assert len(proposer.successes) <= 5

    def test_format_tips_for_prompt(self, proposer):
        """Test formatting tips for prompt."""
        formatted = proposer.format_tips_for_prompt()
        assert "IMPORTANT TIPS" in formatted
        for tip in DEFAULT_TIPS:
            assert tip in formatted

    def test_augment_prompt(self, proposer):
        """Test augmenting a prompt with tips."""
        base = "Analyze this code."
        augmented = proposer.augment_prompt(base)
        assert base in augmented
        assert "IMPORTANT TIPS" in augmented

    def test_analyze_failure_patterns(self, proposer):
        """Test analyzing failure patterns."""
        proposer.record_failure("Q1", "bugs", "not verified", 0.3, tools_used=[])
        proposer.record_failure("Q2", "bugs", "not verified", 0.2, tools_used=[])
        proposer.record_failure("Q3", "security", "wrong tool", 0.4, tools_used=["ripgrep"])

        patterns = proposer._analyze_failure_patterns()
        assert patterns["total_failures"] == 3
        assert ("not verified", 2) in patterns["top_reasons"]

    def test_analyze_success_patterns(self, proposer):
        """Test analyzing success patterns."""
        proposer.record_success("Q1", "bugs", 0.9, ["read_file", "ripgrep"])
        proposer.record_success("Q2", "bugs", 0.95, ["read_file"])

        patterns = proposer._analyze_success_patterns()
        assert patterns["total_successes"] == 2
        assert patterns["avg_grounded_score"] >= 0.9

    def test_generate_heuristic_tips(self, proposer):
        """Test heuristic tip generation."""
        failure_patterns = {
            "missing_tools": [("read_file", 5), ("ripgrep", 3)],
        }
        success_patterns = {
            "top_patterns": [("verified with read_file", 10), ("used structural search first", 8)],
        }

        tips = proposer._generate_heuristic_tips(failure_patterns, success_patterns)
        assert len(tips) >= 3
        assert any("read_file" in tip.lower() or "verify" in tip.lower() for tip in tips)

    def test_get_stats(self, proposer):
        """Test getting statistics."""
        proposer.record_failure("Q1", "bugs", "reason", 0.3)
        proposer.record_success("Q2", "bugs", 0.9)

        stats = proposer.get_stats()
        assert stats["total_failures"] == 1
        assert stats["total_successes"] == 1
        assert stats["current_tips_count"] == len(DEFAULT_TIPS)

    def test_clear(self, proposer):
        """Test clearing records."""
        proposer.record_failure("Q1", "bugs", "reason", 0.3)
        proposer.record_success("Q2", "bugs", 0.9)

        failures, successes = proposer.clear()
        assert failures == 1
        assert successes == 1
        assert len(proposer.failures) == 0
        assert len(proposer.successes) == 0

    def test_reset_tips(self, proposer):
        """Test resetting tips to defaults."""
        proposer.current_tips = ["Custom tip"]
        proposer.reset_tips()
        assert proposer.current_tips == DEFAULT_TIPS


class TestGlobalProposer:
    """Tests for global proposer functions."""

    def test_get_grounded_proposer(self):
        """Test getting global proposer."""
        clear_grounded_proposer()
        p1 = get_grounded_proposer()
        p2 = get_grounded_proposer()
        assert p1 is p2

    def test_clear_grounded_proposer(self):
        """Test clearing global proposer."""
        p1 = get_grounded_proposer()
        clear_grounded_proposer()
        p2 = get_grounded_proposer()
        assert p1 is not p2

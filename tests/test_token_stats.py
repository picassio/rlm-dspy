"""Tests for token stats tracking."""

import json
import tempfile
from pathlib import Path

import pytest

from rlm_dspy.core.token_stats import (
    SessionStats,
    TokenStats,
    count_tokens,
    estimate_cost,
    get_session,
    record_operation,
)


class TestCountTokens:
    """Tests for token counting."""

    def test_empty_string(self):
        """Empty string returns 0."""
        assert count_tokens("") == 0

    def test_simple_text(self):
        """Simple text counts tokens."""
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10  # Should be around 4-5 tokens

    def test_code(self):
        """Code counts tokens."""
        code = "def hello():\n    print('world')"
        tokens = count_tokens(code)
        assert tokens > 0
        assert tokens < 20


class TestEstimateCost:
    """Tests for cost estimation."""

    def test_gpt4_pricing(self):
        """GPT-4 pricing calculation."""
        cost = estimate_cost(1_000_000, 500_000, "gpt-4")
        # $30/M input + $60/M * 0.5 = $30 + $30 = $60
        assert cost == pytest.approx(60.0, rel=0.1)

    def test_claude_sonnet_pricing(self):
        """Claude Sonnet pricing calculation."""
        cost = estimate_cost(1_000_000, 1_000_000, "claude-sonnet-4")
        # $3/M input + $15/M output = $18
        assert cost == pytest.approx(18.0, rel=0.1)

    def test_unknown_model_uses_default(self):
        """Unknown model uses GPT-4 pricing."""
        cost = estimate_cost(1_000_000, 0, "unknown-model")
        # Should use GPT-4 pricing: $30/M
        assert cost == pytest.approx(30.0, rel=0.1)


class TestTokenStats:
    """Tests for TokenStats dataclass."""

    def test_context_savings(self):
        """Context savings percentage calculation."""
        stats = TokenStats(raw_context_tokens=1000, processed_tokens=200)
        assert stats.context_savings == 80.0

    def test_context_savings_zero_raw(self):
        """Zero raw tokens gives 0% savings."""
        stats = TokenStats(raw_context_tokens=0, processed_tokens=0)
        assert stats.context_savings == 0.0

    def test_chunk_relevance_rate(self):
        """Chunk relevance rate calculation."""
        stats = TokenStats(chunks_processed=10, chunks_relevant=3)
        assert stats.chunk_relevance_rate == 30.0

    def test_chunk_relevance_zero_chunks(self):
        """Zero chunks gives 0% relevance."""
        stats = TokenStats(chunks_processed=0, chunks_relevant=0)
        assert stats.chunk_relevance_rate == 0.0

    def test_total_tokens(self):
        """Total tokens calculation."""
        stats = TokenStats(llm_input_tokens=1000, llm_output_tokens=500)
        assert stats.total_tokens == 1500

    def test_to_dict(self):
        """Serialization to dict."""
        stats = TokenStats(
            raw_context_tokens=1000,
            processed_tokens=200,
            llm_input_tokens=150,
            llm_output_tokens=50,
            chunks_processed=5,
            chunks_relevant=2,
        )
        d = stats.to_dict()

        assert d["raw_context_tokens"] == 1000
        assert d["processed_tokens"] == 200
        assert d["context_savings_percent"] == 80.0
        assert d["chunk_relevance_percent"] == 40.0
        assert d["total_tokens"] == 200

    def test_str(self):
        """String representation."""
        stats = TokenStats(
            raw_context_tokens=10000,
            processed_tokens=2000,
            llm_input_tokens=1500,
            llm_output_tokens=500,
            chunks_processed=10,
            chunks_relevant=3,
        )
        s = str(stats)

        assert "Token Stats" in s
        assert "10,000" in s
        assert "2,000" in s
        assert "80.0%" in s


class TestSessionStats:
    """Tests for SessionStats class."""

    def test_record_operation(self):
        """Recording operations accumulates stats."""
        session = SessionStats(session_id="test")

        stats1 = TokenStats(
            raw_context_tokens=1000,
            processed_tokens=200,
            llm_input_tokens=150,
            llm_output_tokens=50,
            chunks_processed=5,
            chunks_relevant=2,
        )
        stats2 = TokenStats(
            raw_context_tokens=2000,
            processed_tokens=400,
            llm_input_tokens=300,
            llm_output_tokens=100,
            chunks_processed=10,
            chunks_relevant=4,
        )

        session.record(stats1)
        session.record(stats2)

        assert len(session.operations) == 2
        assert session.total_raw_tokens == 3000
        assert session.total_processed_tokens == 600
        assert session.total_llm_input == 450
        assert session.total_llm_output == 150
        assert session.total_chunks == 15
        assert session.total_relevant_chunks == 6

    def test_total_savings(self):
        """Total savings calculation."""
        session = SessionStats(session_id="test")
        session.record(TokenStats(raw_context_tokens=1000, processed_tokens=200))
        session.record(TokenStats(raw_context_tokens=1000, processed_tokens=200))

        assert session.total_savings == 80.0

    def test_average_chunk_relevance(self):
        """Average chunk relevance calculation."""
        session = SessionStats(session_id="test")
        session.record(TokenStats(chunks_processed=10, chunks_relevant=5))
        session.record(TokenStats(chunks_processed=10, chunks_relevant=3))

        assert session.average_chunk_relevance == 40.0

    def test_to_dict(self):
        """Serialization to dict."""
        session = SessionStats(session_id="test123")
        session.record(TokenStats(raw_context_tokens=1000, processed_tokens=200))

        d = session.to_dict()

        assert d["session_id"] == "test123"
        assert d["operation_count"] == 1
        assert d["total_raw_tokens"] == 1000
        assert d["total_savings_percent"] == 80.0

    def test_save_to_file(self):
        """Save stats to JSONL file."""
        session = SessionStats(session_id="test")
        session.record(TokenStats(raw_context_tokens=1000, processed_tokens=200))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.jsonl"
            session.save(path)

            assert path.exists()
            with open(path) as f:
                data = json.loads(f.readline())

            assert data["session_id"] == "test"
            assert data["total_raw_tokens"] == 1000

    def test_str(self):
        """String representation."""
        session = SessionStats(session_id="test")
        session.record(TokenStats(
            raw_context_tokens=10000,
            processed_tokens=2000,
            llm_input_tokens=1500,
            llm_output_tokens=500,
        ))

        s = str(session)
        assert "test" in s
        assert "Operations: 1" in s
        assert "80.0%" in s


class TestGlobalSession:
    """Tests for global session management."""

    def test_get_session_creates_new(self):
        """get_session creates new session if needed."""
        session = get_session("new_session_id")
        assert session.session_id == "new_session_id"

    def test_record_operation_global(self):
        """record_operation uses global session."""
        session = get_session("global_test")
        record_operation(TokenStats(raw_context_tokens=500, processed_tokens=100))

        assert session.total_raw_tokens == 500

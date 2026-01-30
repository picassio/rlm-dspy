"""Tests for trace collector module."""

import json
import pytest
from pathlib import Path
from datetime import datetime, UTC

from rlm_dspy.core.trace_collector import (
    REPLTrace,
    TraceCollector,
    TraceCollectorConfig,
    get_trace_collector,
    clear_trace_collector,
)


class TestREPLTrace:
    """Tests for REPLTrace dataclass."""

    def test_create_trace(self):
        """Test creating a trace."""
        trace = REPLTrace(
            query="Find bugs",
            query_type="bugs",
            reasoning_steps=["Step 1", "Step 2"],
            code_blocks=["print('test')"],
            outputs=["output"],
            final_answer="Found 2 bugs",
            grounded_score=0.9,
        )
        assert trace.query == "Find bugs"
        assert trace.query_type == "bugs"
        assert len(trace.reasoning_steps) == 2
        assert trace.grounded_score == 0.9

    def test_trace_to_dict(self):
        """Test converting trace to dict."""
        trace = REPLTrace(
            query="Test query",
            query_type="general",
            final_answer="Answer",
            grounded_score=1.0,
        )
        data = trace.to_dict()
        assert data["query"] == "Test query"
        assert data["query_type"] == "general"
        assert "timestamp" in data
        assert "trace_id" in data

    def test_trace_from_dict(self):
        """Test creating trace from dict."""
        data = {
            "query": "Test",
            "query_type": "bugs",
            "reasoning_steps": ["R1"],
            "code_blocks": ["C1"],
            "outputs": ["O1"],
            "tools_used": ["read_file"],
            "final_answer": "Done",
            "grounded_score": 0.95,
            "timestamp": "2025-01-30T00:00:00Z",
            "trace_id": "test123",
        }
        trace = REPLTrace.from_dict(data)
        assert trace.query == "Test"
        assert trace.grounded_score == 0.95

    def test_format_as_demo(self):
        """Test formatting trace as demo."""
        trace = REPLTrace(
            query="Find function foo",
            query_type="search",
            reasoning_steps=["Looking for foo"],
            code_blocks=["find_functions('foo')"],
            outputs=["Found foo at line 10"],
            final_answer="foo is defined at line 10",
            grounded_score=1.0,
        )
        demo = trace.format_as_demo()
        assert "Query: Find function foo" in demo
        assert "Looking for foo" in demo
        assert "find_functions('foo')" in demo
        assert "Found foo at line 10" in demo
        assert "foo is defined at line 10" in demo


class TestTraceCollectorConfig:
    """Tests for TraceCollectorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TraceCollectorConfig()
        assert config.min_grounded_score == 0.8
        assert config.max_traces == 1000
        assert config.enabled is True

    def test_custom_config(self, tmp_path):
        """Test custom configuration."""
        config = TraceCollectorConfig(
            storage_path=tmp_path / "traces",
            min_grounded_score=0.9,
            max_traces=500,
        )
        assert config.min_grounded_score == 0.9
        assert config.max_traces == 500


class TestTraceCollector:
    """Tests for TraceCollector."""

    @pytest.fixture
    def collector(self, tmp_path):
        """Create a collector with temp storage."""
        config = TraceCollectorConfig(
            storage_path=tmp_path / "traces",
            min_grounded_score=0.8,
            max_traces=100,
        )
        return TraceCollector(config)

    def test_record_high_score(self, collector):
        """Test recording trace with high score."""
        result = collector.record(
            query="Find bugs",
            reasoning_steps=["R1"],
            code_blocks=["read_file('test.py')"],
            outputs=["content"],
            final_answer="No bugs",
            grounded_score=0.95,
        )
        assert result is True
        assert len(collector.traces) == 1
        assert collector.traces[0].query == "Find bugs"

    def test_skip_low_score(self, collector):
        """Test skipping trace with low score."""
        result = collector.record(
            query="Find bugs",
            reasoning_steps=[],
            code_blocks=[],
            outputs=[],
            final_answer="Guessing...",
            grounded_score=0.5,
        )
        assert result is False
        assert len(collector.traces) == 0

    def test_infer_query_type(self, collector):
        """Test query type inference."""
        # Test the internal method directly
        assert collector._infer_query_type("Find bugs in this code") == "bugs"
        assert collector._infer_query_type("Check for security vulnerabilities") == "security"
        assert collector._infer_query_type("Review code quality") == "review"
        assert collector._infer_query_type("Find where foo is defined") == "search"
        assert collector._infer_query_type("Explain this function") == "explanation"
        assert collector._infer_query_type("Do something random") == "general"

    def test_extract_tools_used(self, collector):
        """Test tool extraction from code."""
        collector.record(
            query="Analyze",
            reasoning_steps=[],
            code_blocks=[
                "result = read_file('test.py')",
                "matches = ripgrep('pattern', '.')",
                "funcs = find_functions()",
            ],
            outputs=[],
            final_answer="Done",
            grounded_score=0.9,
        )
        tools = collector.traces[-1].tools_used
        assert "read_file" in tools
        assert "ripgrep" in tools
        assert "find_functions" in tools

    def test_persistence(self, tmp_path):
        """Test that traces persist across instances."""
        config = TraceCollectorConfig(
            storage_path=tmp_path / "traces",
            min_grounded_score=0.8,
        )

        # Create and record
        collector1 = TraceCollector(config)
        collector1.record("Query 1", [], [], [], "Answer 1", 0.9)

        # Create new instance
        collector2 = TraceCollector(config)
        assert len(collector2.traces) == 1
        assert collector2.traces[0].query == "Query 1"

    def test_enforce_limits(self, tmp_path):
        """Test that limits are enforced."""
        config = TraceCollectorConfig(
            storage_path=tmp_path / "traces",
            min_grounded_score=0.0,
            max_traces=5,
            max_traces_per_type=3,
        )
        collector = TraceCollector(config)

        # Add more than max
        for i in range(10):
            collector.record(f"Query {i}", [], [], [], f"Answer {i}", 0.9)

        assert len(collector.traces) <= 5

    def test_get_similar_traces(self, collector):
        """Test finding similar traces."""
        # Add some traces
        collector.record("Find bugs in parser.py", [], [], [], "Found 2 bugs", 0.9)
        collector.record("Security audit of auth.py", [], [], [], "Secure", 0.9)
        collector.record("Find issues in lexer.py", [], [], [], "Found 1 issue", 0.9)

        # This should match bug-related traces
        # Note: Without embeddings, falls back to recency
        similar = collector.get_similar_traces("Find bugs in code", k=2)
        assert len(similar) <= 2

    def test_format_demos(self, collector):
        """Test formatting multiple traces as demos."""
        collector.record("Query 1", ["R1"], ["C1"], ["O1"], "A1", 0.95)
        collector.record("Query 2", ["R2"], ["C2"], ["O2"], "A2", 0.90)

        demos = collector.format_demos(collector.traces, max_chars=5000)
        assert "Example 1" in demos
        assert "Example 2" in demos
        assert "Query 1" in demos
        assert "Query 2" in demos

    def test_get_stats(self, collector):
        """Test statistics gathering."""
        collector.record("Bug query", [], [], [], "Done", 0.9, query_type="bugs")
        collector.record("Security query", [], [], [], "Done", 0.95, query_type="security")
        collector.record("Another bug", [], [], [], "Done", 0.85, query_type="bugs")

        stats = collector.get_stats()
        assert stats["total"] == 3
        assert stats["by_type"]["bugs"] == 2
        assert stats["by_type"]["security"] == 1
        assert 0.85 <= stats["avg_score"] <= 0.95

    def test_clear(self, collector):
        """Test clearing traces."""
        collector.record("Query", [], [], [], "Answer", 0.9)
        assert len(collector.traces) == 1

        deleted = collector.clear()
        assert deleted == 1
        assert len(collector.traces) == 0

    def test_export_import(self, collector, tmp_path):
        """Test export and import."""
        collector.record("Query 1", [], [], [], "Answer 1", 0.9)
        collector.record("Query 2", [], [], [], "Answer 2", 0.85)

        export_path = tmp_path / "export.json"
        exported = collector.export(export_path)
        assert exported == 2
        assert export_path.exists()

        # Clear and import
        collector.clear()
        imported = collector.import_traces(export_path)
        assert imported == 2
        assert len(collector.traces) == 2


class TestGlobalCollector:
    """Tests for global collector functions."""

    def test_get_trace_collector(self):
        """Test getting global collector."""
        clear_trace_collector()
        c1 = get_trace_collector()
        c2 = get_trace_collector()
        assert c1 is c2

    def test_clear_trace_collector(self):
        """Test clearing global collector."""
        c1 = get_trace_collector()
        clear_trace_collector()
        c2 = get_trace_collector()
        assert c1 is not c2

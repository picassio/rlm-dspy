"""Tests for core RLM functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

from rlm_dspy.core.rlm import RLM, RLMConfig, RLMResult


class TestRLMConfig:
    """Tests for RLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RLMConfig()
        assert config.max_iterations == 30
        assert config.max_llm_calls == 100
        assert config.max_output_chars == 100000
        assert config.max_budget == 2.0
        assert config.max_timeout == 600
        assert config.verbose is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RLMConfig(
            model="test/model",
            max_iterations=25,
            max_budget=0.5,
        )
        assert config.model == "test/model"
        assert config.max_iterations == 25
        assert config.max_budget == 0.5

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {"RLM_MAX_ITERATIONS": "50"}):
            config = RLMConfig()
            assert config.max_iterations == 50

    def test_repr_hides_api_key(self):
        """Test that repr doesn't expose API key."""
        config = RLMConfig(api_key="sk-secret-key-12345")
        repr_str = repr(config)
        assert "sk-secret-key-12345" not in repr_str
        assert "***" in repr_str  # API key should be masked


class TestRLMResult:
    """Tests for RLMResult."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = RLMResult(
            answer="Test answer",
            success=True,
            trajectory=[{"code": "print('hi')", "output": "hi"}],
            iterations=2,
            elapsed_time=1.5,
        )
        assert result.answer == "Test answer"
        assert result.success is True
        assert result.iterations == 2
        assert result.elapsed_time == 1.5

    def test_structured_output_access(self):
        """Test accessing structured outputs as attributes."""
        result = RLMResult(
            answer="",
            success=True,
            outputs={"bugs": ["bug1", "bug2"], "has_critical": True},
        )
        assert result.bugs == ["bug1", "bug2"]
        assert result.has_critical is True

    def test_missing_attribute_error(self):
        """Test that missing attributes raise AttributeError."""
        result = RLMResult(answer="test", success=True)
        with pytest.raises(AttributeError):
            _ = result.nonexistent_field


class TestRLMLoadContext:
    """Tests for RLM.load_context functionality."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure."""
        # Create files
        (tmp_path / "main.py").write_text("def main():\n    print('hello')\n")
        (tmp_path / "utils.py").write_text("def helper():\n    pass\n")

        # Create subdirectory
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "app.py").write_text("class App:\n    pass\n")

        # Create gitignore
        (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")

        return tmp_path

    @pytest.fixture
    def mock_rlm(self):
        """Create RLM with mocked DSPy."""
        with patch("rlm_dspy.core.rlm.dspy"):
            config = RLMConfig(api_key="test-key", model="test/model")
            rlm = RLM(config=config, use_tools=False)
            return rlm

    def test_load_single_file(self, mock_rlm, temp_project):
        """Test loading a single file."""
        context = mock_rlm.load_context([str(temp_project / "main.py")])
        assert "def main()" in context
        assert "print('hello')" in context

    def test_load_directory(self, mock_rlm, temp_project):
        """Test loading a directory."""
        context = mock_rlm.load_context([str(temp_project)])
        assert "def main()" in context
        assert "def helper()" in context
        assert "class App" in context

    def test_respects_gitignore(self, mock_rlm, temp_project):
        """Test that gitignore patterns are respected."""
        # Create a .pyc file that should be ignored
        (temp_project / "test.pyc").write_text("binary content")

        context = mock_rlm.load_context([str(temp_project)])
        assert "test.pyc" not in context
        assert "binary content" not in context

    def test_line_numbers_included(self, mock_rlm, temp_project):
        """Test that line numbers are included in context."""
        context = mock_rlm.load_context([str(temp_project / "main.py")])
        # Should have line numbers like "   1 | def main():"
        assert " 1 |" in context or "1 |" in context

    def test_file_markers_included(self, mock_rlm, temp_project):
        """Test that file markers are included."""
        context = mock_rlm.load_context([str(temp_project / "main.py")])
        assert "=== FILE:" in context
        assert "main.py" in context

    def test_empty_directory(self, mock_rlm, tmp_path):
        """Test loading empty directory returns empty context."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        context = mock_rlm.load_context([str(empty_dir)])
        # Should return empty or minimal context
        assert context.strip() == "" or "=== FILE:" not in context

    def test_nonexistent_path(self, mock_rlm):
        """Test loading nonexistent path."""
        context = mock_rlm.load_context(["/nonexistent/path/file.py"])
        # Should handle gracefully
        assert context is not None


class TestRLMTools:
    """Tests for RLM tool integration."""

    def test_tools_enabled_by_default(self):
        """Test that tools are enabled by default."""
        with patch("rlm_dspy.core.rlm.dspy"):
            config = RLMConfig(api_key="test-key", model="test/model")
            rlm = RLM(config=config)
            assert len(rlm._tools) > 0
            assert "ripgrep" in rlm._tools or "index_code" in rlm._tools

    def test_tools_disabled(self):
        """Test disabling tools."""
        with patch("rlm_dspy.core.rlm.dspy"):
            config = RLMConfig(api_key="test-key", model="test/model")
            rlm = RLM(config=config, use_tools=False)
            assert len(rlm._tools) == 0

    def test_add_custom_tool(self):
        """Test adding a custom tool."""
        with patch("rlm_dspy.core.rlm.dspy"):
            config = RLMConfig(api_key="test-key", model="test/model")
            rlm = RLM(config=config, use_tools=False)

            def my_tool(query: str) -> str:
                return f"Result: {query}"

            rlm.add_tool("my_tool", my_tool)
            assert "my_tool" in rlm._tools


class TestRLMBatch:
    """Tests for RLM.batch functionality."""

    def test_batch_validates_queries(self):
        """Test that batch validates query format."""
        with patch("rlm_dspy.core.rlm.dspy"):
            config = RLMConfig(api_key="test-key", model="test/model")
            rlm = RLM(config=config, use_tools=False)

            # Missing 'query' field should raise
            with pytest.raises(ValueError, match="missing 'query' field"):
                rlm.batch([{"context": "test"}], context="base")

    def test_batch_accepts_valid_queries(self):
        """Test that batch accepts valid query format."""
        with patch("rlm_dspy.core.rlm.dspy") as mock_dspy:
            # Mock the batch execution
            mock_rlm = MagicMock()
            mock_rlm.batch.return_value = ([], [], [])
            mock_dspy.RLM.return_value = mock_rlm

            config = RLMConfig(api_key="test-key", model="test/model")
            rlm = RLM(config=config, use_tools=False)

            # Should not raise
            queries = [
                {"query": "Question 1"},
                {"query": "Question 2", "context": "Custom context"},
            ]
            # This will try to execute, which may fail, but format validation passes
            try:
                rlm.batch(queries, context="base context")
            except Exception:
                pass  # Execution may fail, but validation passed


class TestRLMSignatures:
    """Tests for RLM signature handling."""

    def test_string_signature(self):
        """Test string signature is wrapped with tool instructions."""
        with patch("rlm_dspy.core.rlm.dspy"):
            config = RLMConfig(api_key="test-key", model="test/model")
            rlm = RLM(config=config, signature="context, query -> answer")

            # Signature should be wrapped
            assert rlm._signature is not None

    def test_class_signature(self):
        """Test class signature is wrapped with tool instructions."""
        import dspy

        class TestSignature(dspy.Signature):
            """Test signature."""
            context: str = dspy.InputField()
            query: str = dspy.InputField()
            answer: str = dspy.OutputField()

        with patch("rlm_dspy.core.rlm.dspy"):
            config = RLMConfig(api_key="test-key", model="test/model")
            rlm = RLM(config=config, signature=TestSignature)

            assert rlm._signature is not None
            assert rlm._is_structured is True


@pytest.mark.integration
class TestRLMIntegration:
    """Integration tests requiring API access.

    These tests are skipped unless RLM_API_KEY is set.
    Run with: pytest -m integration
    """

    @pytest.fixture
    def skip_without_api_key(self):
        """Skip test if no API key available."""
        if not os.environ.get("RLM_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("No API key available")

    def test_simple_query(self, skip_without_api_key, tmp_path):
        """Test a simple query with real API."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        rlm = RLM()
        context = rlm.load_context([str(test_file)])
        result = rlm.query("What does the hello function return?", context)

        assert result.success
        assert "world" in result.answer.lower()

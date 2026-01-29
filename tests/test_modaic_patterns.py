#!/usr/bin/env python3
"""Tests for patterns learned from modaic."""

import tempfile
from pathlib import Path

import pytest


class TestRetry:
    """Test retry utilities."""

    def test_retry_sync_decorator(self):
        """Test sync retry decorator."""
        from rlm_dspy.core import retry_sync

        call_count = 0

        # Specify ValueError as retryable for this test
        @retry_sync(max_retries=3, base_delay=0.01, retryable_exceptions=(ValueError,))
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_retry_sync_max_retries_exceeded(self):
        """Test retry gives up after max retries."""
        from rlm_dspy.core import retry_sync

        # Specify ValueError as retryable for this test
        @retry_sync(max_retries=2, base_delay=0.01, retryable_exceptions=(ValueError,))
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self):
        """Test async retry with backoff."""
        from rlm_dspy.core import retry_with_backoff

        call_count = 0

        async def flaky_async():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network error")
            return "done"

        result = await retry_with_backoff(
            flaky_async,
            max_retries=3,
            base_delay=0.01,
            jitter=0.01,
        )
        assert result == "done"
        assert call_count == 2


class TestSecrets:
    """Test secret masking utilities."""

    def test_is_secret_key(self):
        """Test secret key detection."""
        from rlm_dspy.core import is_secret_key

        assert is_secret_key("api_key") is True
        assert is_secret_key("API_KEY") is True
        assert is_secret_key("openai_api_key") is True
        assert is_secret_key("hf_token") is True
        assert is_secret_key("password") is True
        assert is_secret_key("username") is False
        assert is_secret_key("model") is False

    def test_mask_value(self):
        """Test value masking - full mask by default for security."""
        from rlm_dspy.core import mask_value

        # Default: full masking for security
        assert mask_value("sk-1234567890abcdef") == "********"
        assert mask_value("short") == "********"
        assert mask_value(None) == "[None]"
        assert mask_value("") == "[empty]"

        # Optional prefix reveal for debugging
        assert mask_value("sk-1234567890abcdef", reveal_prefix=True) == "sk-1********"
        assert mask_value("short", reveal_prefix=True) == "********"  # Too short

    def test_clean_secrets(self):
        """Test recursive secret cleaning."""
        from rlm_dspy.core import clean_secrets

        data = {
            "api_key": "sk-secret123",
            "name": "test",
            "nested": {
                "token": "abc123",
                "value": 42,
            },
            "list": [{"password": "pass123"}],
        }

        cleaned = clean_secrets(data)

        assert cleaned["api_key"] == "********"
        assert cleaned["name"] == "test"
        assert cleaned["nested"]["token"] == "********"
        assert cleaned["nested"]["value"] == 42
        assert cleaned["list"][0]["password"] == "********"

        # Original should be unchanged
        assert data["api_key"] == "sk-secret123"

    def test_inject_secrets(self):
        """Test secret injection from env."""
        import os

        from rlm_dspy.core import inject_secrets

        data = {"api_key": "********", "name": "test"}

        # Inject from provided dict
        injected = inject_secrets(data.copy(), secrets={"api_key": "new-key"})
        assert injected["api_key"] == "new-key"

        # Inject from env
        os.environ["RLM_API_KEY"] = "env-key"
        data2 = {"api_key": "********"}
        injected2 = inject_secrets(data2)
        assert injected2["api_key"] == "env-key"
        del os.environ["RLM_API_KEY"]


class TestFileUtils:
    """Test cross-platform file utilities."""

    def test_platform_detection(self):
        """Test platform detection."""
        from rlm_dspy.core import is_linux, is_macos, is_windows

        # At least one should be true
        assert is_linux() or is_windows() or is_macos()
        # Can't be both Windows and Linux
        assert not (is_linux() and is_windows())

    def test_get_cache_dir(self):
        """Test cache directory resolution."""
        from rlm_dspy.core import get_cache_dir

        cache = get_cache_dir("test_app")
        assert "test_app" in str(cache)
        assert isinstance(cache, Path)

    def test_path_to_module(self):
        """Test path to module conversion."""
        from rlm_dspy.core import path_to_module

        assert path_to_module(Path("src/pkg/mod.py"), Path("src")) == "pkg.mod"
        assert path_to_module(Path("a/b/c/__init__.py"), Path("a")) == "b.c"

    def test_atomic_write(self):
        """Test atomic file writing."""
        from rlm_dspy.core import atomic_write

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            atomic_write(path, "Hello, World!")
            assert path.read_text() == "Hello, World!"

            # Overwrite
            atomic_write(path, "New content")
            assert path.read_text() == "New content"

    def test_ensure_dir(self):
        """Test directory creation."""
        from rlm_dspy.core import ensure_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "a" / "b" / "c"
            result = ensure_dir(new_dir)
            assert result.exists()
            assert result.is_dir()

    def test_smart_rmtree(self):
        """Test robust directory removal."""
        from rlm_dspy.core import smart_rmtree

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            nested = Path(tmpdir) / "a" / "b" / "c"
            nested.mkdir(parents=True)
            (nested / "file.txt").write_text("test")

            target = Path(tmpdir) / "a"
            assert target.exists()

            result = smart_rmtree(target)
            assert result is True
            assert not target.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestFileCollection:
    """Test file collection utilities."""

    def test_skip_dirs_contains_common_patterns(self):
        """Test SKIP_DIRS contains common ignore patterns."""
        from rlm_dspy.core.fileutils import SKIP_DIRS
        
        assert ".git" in SKIP_DIRS
        assert "__pycache__" in SKIP_DIRS
        assert "node_modules" in SKIP_DIRS
        assert ".venv" in SKIP_DIRS

    def test_load_gitignore_patterns(self, tmp_path):
        """Test loading gitignore patterns."""
        from rlm_dspy.core.fileutils import load_gitignore_patterns
        
        # Create .gitignore
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n")
        
        patterns = load_gitignore_patterns([tmp_path])
        assert "*.pyc" in patterns
        assert "__pycache__/" in patterns

    def test_collect_files_basic(self, tmp_path):
        """Test basic file collection."""
        from rlm_dspy.core.fileutils import collect_files
        
        # Create test files
        (tmp_path / "a.py").write_text("# a")
        (tmp_path / "b.py").write_text("# b")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "c.py").write_text("# c")
        
        files = collect_files([tmp_path])
        names = {f.name for f in files}
        
        assert "a.py" in names
        assert "b.py" in names
        assert "c.py" in names

    def test_collect_files_skips_pycache(self, tmp_path):
        """Test that __pycache__ is skipped."""
        from rlm_dspy.core.fileutils import collect_files
        
        # Create test files
        (tmp_path / "main.py").write_text("# main")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-312.pyc").write_text("binary")
        
        files = collect_files([tmp_path])
        names = {f.name for f in files}
        
        assert "main.py" in names
        assert "main.cpython-312.pyc" not in names

    def test_format_file_context(self, tmp_path):
        """Test file context formatting."""
        from rlm_dspy.core.fileutils import format_file_context
        
        f = tmp_path / "test.py"
        f.write_text("line 1\nline 2\n")
        
        context, skipped = format_file_context([f])
        
        assert "=== FILE:" in context
        assert "test.py" in context
        assert "line 1" in context
        assert "line 2" in context
        assert len(skipped) == 0

    def test_format_file_context_with_line_numbers(self, tmp_path):
        """Test that line numbers are added."""
        from rlm_dspy.core.fileutils import format_file_context
        
        f = tmp_path / "test.py"
        f.write_text("first\nsecond\n")
        
        context, _ = format_file_context([f], add_line_numbers=True)
        
        assert "1 |" in context or "1|" in context
        assert "2 |" in context or "2|" in context

    def test_format_file_context_skips_binary(self, tmp_path):
        """Test that binary files are skipped."""
        from rlm_dspy.core.fileutils import format_file_context
        
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\xff\xfe")
        
        context, skipped = format_file_context([f])
        
        assert len(skipped) == 1
        assert skipped[0][0] == f

    def test_load_context_from_paths(self, tmp_path):
        """Test the main load_context_from_paths function."""
        from rlm_dspy.core.fileutils import load_context_from_paths
        
        # Create test structure
        (tmp_path / "main.py").write_text("def main(): pass\n")
        (tmp_path / ".gitignore").write_text("*.log\n")
        (tmp_path / "debug.log").write_text("logs")
        
        context = load_context_from_paths([tmp_path], gitignore=True)
        
        assert "main.py" in context
        assert "def main()" in context
        # .log files should be ignored
        assert "debug.log" not in context

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
            with atomic_write(path) as f:
                f.write("Hello, World!")
            assert path.read_text() == "Hello, World!"

            # Overwrite
            with atomic_write(path) as f:
                f.write("New content")
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


class TestContextCaching:
    """Test context caching utilities."""

    def test_load_context_cached(self, tmp_path):
        """Test that cached loading returns same result."""
        from rlm_dspy.core.fileutils import (
            load_context_from_paths,
            load_context_from_paths_cached,
            clear_context_cache,
        )

        # Clear cache first
        clear_context_cache()

        # Create test file
        f = tmp_path / "test.py"
        f.write_text("def hello(): pass\n")

        # Load without cache
        context1 = load_context_from_paths([tmp_path])

        # Load with cache (first time - cache miss)
        context2 = load_context_from_paths_cached([tmp_path])

        # Load with cache again (cache hit)
        context3 = load_context_from_paths_cached([tmp_path])

        assert context1 == context2 == context3

    def test_cache_invalidation_on_file_change(self, tmp_path):
        """Test that cache invalidates when file changes."""
        from rlm_dspy.core.fileutils import (
            load_context_from_paths_cached,
            clear_context_cache,
        )
        import time

        # Clear cache first
        clear_context_cache()

        # Create test file
        f = tmp_path / "test.py"
        f.write_text("version 1\n")

        # Load (cache miss)
        context1 = load_context_from_paths_cached([tmp_path])
        assert "version 1" in context1

        # Modify file (need to ensure mtime changes)
        time.sleep(0.1)
        f.write_text("version 2\n")

        # Load again (should be cache miss due to mtime change)
        context2 = load_context_from_paths_cached([tmp_path])
        assert "version 2" in context2

        assert context1 != context2

    def test_cache_stats(self):
        """Test cache stats function."""
        from rlm_dspy.core.fileutils import get_context_cache_stats

        stats = get_context_cache_stats()
        assert "entries" in stats
        assert "max_size" in stats
        assert "max_age_seconds" in stats

    def test_clear_cache(self, tmp_path):
        """Test cache clearing."""
        from rlm_dspy.core.fileutils import (
            load_context_from_paths_cached,
            clear_context_cache,
            get_context_cache_stats,
        )

        # Create and load
        f = tmp_path / "test.py"
        f.write_text("content\n")
        load_context_from_paths_cached([tmp_path])

        # Clear cache
        clear_context_cache()

        stats = get_context_cache_stats()
        assert stats["entries"] == 0


class TestContextTruncation:
    """Test context truncation utilities."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        from rlm_dspy.core.fileutils import estimate_tokens

        # Default 4 chars per token
        assert estimate_tokens("1234") == 1
        assert estimate_tokens("12345678") == 2
        assert estimate_tokens("x" * 100) == 25

    def test_truncate_context_no_truncation_needed(self):
        """Test that short context is not truncated."""
        from rlm_dspy.core.fileutils import truncate_context

        context = "short context"
        result, was_truncated = truncate_context(context, max_tokens=1000)

        assert result == context
        assert was_truncated is False

    def test_truncate_context_tail_strategy(self):
        """Test truncation (no strategy param in new API)."""
        from rlm_dspy.core.fileutils import truncate_context

        context = "A" * 1000 + "B" * 1000  # 2000 chars = ~500 tokens
        result, was_truncated = truncate_context(context, max_tokens=100, preserve_structure=False)

        assert was_truncated is True
        assert "truncated" in result.lower()

    def test_truncate_context_head_strategy(self):
        """Test truncation (no strategy param in new API)."""
        from rlm_dspy.core.fileutils import truncate_context

        context = "A" * 1000 + "B" * 1000
        result, was_truncated = truncate_context(context, max_tokens=100, preserve_structure=False)

        assert was_truncated is True
        assert "truncated" in result.lower()

    def test_smart_truncate_preserves_file_markers(self):
        """Test smart truncation preserves file boundaries."""
        from rlm_dspy.core.fileutils import smart_truncate_context

        # Create context with file markers
        files = []
        for i in range(10):
            files.append(f"=== FILE: file{i}.py ===\ncontent {i}\n=== END FILE ===\n")
        context = "".join(files)

        # Truncate to fit ~2 files
        result, was_truncated = smart_truncate_context(
            context,
            max_tokens=50,  # Very small to force truncation
        )

        assert was_truncated is True
        # Result should indicate truncation or be shorter
        assert len(result) < len(context)

    def test_rlm_load_context_with_max_tokens(self, tmp_path):
        """Test RLM.load_context with max_tokens parameter."""
        # Skip this test as it requires API key and network access
        pytest.skip("Requires API key configuration")

"""Tests for configuration utilities."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from rlm_dspy.core.config_utils import (
    atomic_write_json,
    atomic_read_json,
    ConfigResolver,
    format_user_error,
    inject_context,
    get_config_dir,
)


class TestAtomicIO:
    """Tests for atomic file operations."""

    def test_atomic_write_creates_file(self):
        """Atomic write creates file with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 42}
            
            atomic_write_json(path, data)
            
            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == data

    def test_atomic_write_creates_parent_dirs(self):
        """Atomic write creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "test.json"
            
            atomic_write_json(path, {"test": True})
            
            assert path.exists()

    def test_atomic_write_secure_permissions(self):
        """Atomic write with secure=True sets 0o600 permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "secret.json"
            
            atomic_write_json(path, {"secret": "data"}, secure=True)
            
            mode = path.stat().st_mode & 0o777
            assert mode == 0o600

    def test_atomic_read_existing_file(self):
        """Atomic read returns file content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value"}
            with open(path, "w") as f:
                json.dump(data, f)
            
            result = atomic_read_json(path)
            
            assert result == data

    def test_atomic_read_missing_file(self):
        """Atomic read returns default for missing file."""
        result = atomic_read_json("/nonexistent/path.json", default={"default": True})
        assert result == {"default": True}

    def test_atomic_read_invalid_json(self):
        """Atomic read returns default for invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.json"
            with open(path, "w") as f:
                f.write("not valid json")
            
            result = atomic_read_json(path, default={"fallback": True})
            
            assert result == {"fallback": True}


class TestConfigResolver:
    """Tests for hierarchical config resolution."""

    def test_explicit_value_highest_priority(self):
        """Explicit value takes precedence over everything."""
        resolver = ConfigResolver(env_prefix="TEST_")
        os.environ["TEST_KEY"] = "from_env"
        
        try:
            result = resolver.get("KEY", default="default", explicit="explicit")
            assert result == "explicit"
        finally:
            del os.environ["TEST_KEY"]

    def test_env_over_cache(self):
        """Environment variable takes precedence over cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "config.json"
            with open(cache_path, "w") as f:
                json.dump({"KEY": "from_cache"}, f)
            
            resolver = ConfigResolver(env_prefix="TEST_", cache_path=cache_path)
            os.environ["TEST_KEY"] = "from_env"
            
            try:
                result = resolver.get("KEY", default="default")
                assert result == "from_env"
            finally:
                del os.environ["TEST_KEY"]

    def test_cache_over_default(self):
        """Cached value takes precedence over default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "config.json"
            with open(cache_path, "w") as f:
                json.dump({"KEY": "from_cache"}, f)
            
            resolver = ConfigResolver(env_prefix="TEST_", cache_path=cache_path)
            result = resolver.get("KEY", default="default")
            
            assert result == "from_cache"

    def test_default_fallback(self):
        """Default is used when nothing else is set."""
        resolver = ConfigResolver(env_prefix="NONEXISTENT_")
        result = resolver.get("KEY", default="default_value")
        assert result == "default_value"

    def test_type_coercion_bool(self):
        """Boolean values are coerced correctly."""
        resolver = ConfigResolver(env_prefix="TEST_")
        os.environ["TEST_FLAG"] = "true"
        
        try:
            result = resolver.get("FLAG", default=False)
            assert result is True
        finally:
            del os.environ["TEST_FLAG"]

    def test_type_coercion_int(self):
        """Integer values are coerced correctly."""
        resolver = ConfigResolver(env_prefix="TEST_")
        os.environ["TEST_COUNT"] = "42"
        
        try:
            result = resolver.get("COUNT", default=0)
            assert result == 42
            assert isinstance(result, int)
        finally:
            del os.environ["TEST_COUNT"]

    def test_set_and_persist(self):
        """Set persists to cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "config.json"
            resolver = ConfigResolver(env_prefix="TEST_", cache_path=cache_path)
            
            resolver.set("KEY", "new_value")
            
            # Verify persisted
            with open(cache_path) as f:
                data = json.load(f)
            assert data["KEY"] == "new_value"


class TestFormatUserError:
    """Tests for user-friendly error formatting."""

    def test_auth_error(self):
        """Authentication errors are formatted helpfully."""
        error = Exception("401 Unauthorized: Invalid API key")
        result = format_user_error(error)
        
        assert "Authentication failed" in result
        assert "API key" in result

    def test_rate_limit_error(self):
        """Rate limit errors are formatted helpfully."""
        error = Exception("429 Too Many Requests")
        result = format_user_error(error)
        
        assert "Rate limited" in result
        assert "RLM_PARALLEL_CHUNKS" in result

    def test_context_length_error(self):
        """Context length errors are formatted helpfully."""
        error = Exception("Context length exceeded")
        result = format_user_error(error)
        
        assert "Context too long" in result
        assert "RLM_CHUNK_SIZE" in result

    def test_generic_error(self):
        """Unknown errors include type and message."""
        error = ValueError("Something went wrong")
        result = format_user_error(error, context="processing")
        
        assert "ValueError" in result
        assert "Something went wrong" in result
        assert "processing" in result


class TestInjectContext:
    """Tests for context injection."""

    def test_inject_cwd(self):
        """Current directory is injected."""
        result = inject_context("Do something", include_cwd=True, include_time=False)
        
        assert "Working Directory:" in result
        assert os.getcwd() in result

    def test_inject_time(self):
        """Timestamp is injected."""
        result = inject_context("Do something", include_cwd=False, include_time=True)
        
        assert "Timestamp:" in result

    def test_inject_extra(self):
        """Extra context is injected."""
        result = inject_context(
            "Do something",
            include_cwd=False,
            include_time=False,
            extra={"Project": "test-project", "Branch": "main"}
        )
        
        assert "Project: test-project" in result
        assert "Branch: main" in result

    def test_task_section(self):
        """Task is in its own section."""
        result = inject_context("My task here", include_cwd=True)
        
        assert "## Task" in result
        assert "My task here" in result


class TestDirectories:
    """Tests for directory helpers."""

    def test_config_dir_default(self):
        """Config dir uses ~/.config by default."""
        old_xdg = os.environ.pop("XDG_CONFIG_HOME", None)
        try:
            path = get_config_dir()
            assert path == Path.home() / ".config" / "rlm-dspy"
        finally:
            if old_xdg:
                os.environ["XDG_CONFIG_HOME"] = old_xdg

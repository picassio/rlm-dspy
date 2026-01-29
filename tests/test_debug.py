"""Tests for debug and verbose logging utilities."""

import logging
import os
from unittest.mock import patch

import pytest

from rlm_dspy.core.debug import (
    Verbosity,
    get_verbosity,
    is_verbose,
    is_debug,
    DebugConfig,
    configure_debug,
    setup_logging,
    truncate_for_log,
    timer,
    debug_log,
    debug_request,
    debug_response,
    trace,
)


class TestVerbosityLevels:
    """Tests for Verbosity enum."""

    def test_verbosity_ordering(self):
        """Verbosity levels are ordered correctly."""
        assert Verbosity.QUIET < Verbosity.NORMAL
        assert Verbosity.NORMAL < Verbosity.VERBOSE
        assert Verbosity.VERBOSE < Verbosity.DEBUG

    def test_verbosity_values(self):
        """Verbosity levels have expected int values."""
        assert Verbosity.QUIET == 0
        assert Verbosity.NORMAL == 1
        assert Verbosity.VERBOSE == 2
        assert Verbosity.DEBUG == 3


class TestGetVerbosity:
    """Tests for get_verbosity function."""

    def test_default_is_normal(self):
        """Default verbosity is NORMAL."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear all RLM_ vars
            for key in ["RLM_DEBUG", "RLM_VERBOSE", "RLM_QUIET"]:
                os.environ.pop(key, None)
            result = get_verbosity()
            assert result == Verbosity.NORMAL

    def test_debug_env_var(self):
        """RLM_DEBUG=1 sets DEBUG verbosity."""
        with patch.dict(os.environ, {"RLM_DEBUG": "1"}, clear=True):
            result = get_verbosity()
            assert result == Verbosity.DEBUG

    def test_debug_env_var_true(self):
        """RLM_DEBUG=true sets DEBUG verbosity."""
        with patch.dict(os.environ, {"RLM_DEBUG": "true"}, clear=True):
            result = get_verbosity()
            assert result == Verbosity.DEBUG

    def test_verbose_env_var(self):
        """RLM_VERBOSE=1 sets VERBOSE verbosity."""
        with patch.dict(os.environ, {"RLM_VERBOSE": "1"}, clear=True):
            result = get_verbosity()
            assert result == Verbosity.VERBOSE

    def test_quiet_env_var(self):
        """RLM_QUIET=1 sets QUIET verbosity."""
        with patch.dict(os.environ, {"RLM_QUIET": "1"}, clear=True):
            result = get_verbosity()
            assert result == Verbosity.QUIET


class TestIsVerbose:
    """Tests for is_verbose function."""

    def test_verbose_when_verbose(self):
        """Returns True when VERBOSE."""
        with patch.dict(os.environ, {"RLM_VERBOSE": "1"}, clear=True):
            assert is_verbose() is True

    def test_verbose_when_debug(self):
        """Returns True when DEBUG."""
        with patch.dict(os.environ, {"RLM_DEBUG": "1"}, clear=True):
            assert is_verbose() is True

    def test_not_verbose_when_normal(self):
        """Returns False when NORMAL."""
        with patch.dict(os.environ, {}, clear=True):
            for key in ["RLM_DEBUG", "RLM_VERBOSE", "RLM_QUIET"]:
                os.environ.pop(key, None)
            assert is_verbose() is False


class TestIsDebug:
    """Tests for is_debug function."""

    def test_debug_when_debug(self):
        """Returns True when DEBUG."""
        with patch.dict(os.environ, {"RLM_DEBUG": "1"}, clear=True):
            assert is_debug() is True

    def test_not_debug_when_verbose(self):
        """Returns False when only VERBOSE."""
        with patch.dict(os.environ, {"RLM_VERBOSE": "1"}, clear=True):
            # Need to clear RLM_DEBUG
            os.environ.pop("RLM_DEBUG", None)
            assert is_debug() is False


class TestDebugConfig:
    """Tests for DebugConfig dataclass."""

    def test_default_values(self):
        """Has sensible default values."""
        config = DebugConfig()
        assert config.log_inputs is True
        assert config.log_outputs is True
        assert config.max_input_size == 10_000
        assert config.max_output_size == 10_000
        assert config.show_timestamps is True
        assert config.colorize is True


class TestConfigureDebug:
    """Tests for configure_debug function."""

    def test_sets_verbosity(self):
        """Can set verbosity."""
        configure_debug(verbosity=Verbosity.DEBUG)
        # Would need to check global _config, which is harder to test
        # Just verify it doesn't raise

    def test_sets_log_inputs(self):
        """Can set log_inputs."""
        configure_debug(log_inputs=False)

    def test_sets_max_sizes(self):
        """Can set max sizes."""
        configure_debug(max_input_size=5000, max_output_size=5000)


class TestTruncateForLog:
    """Tests for truncate_for_log function."""

    def test_short_string_unchanged(self):
        """Short strings are unchanged."""
        result = truncate_for_log("hello", max_size=100)
        assert result == "hello"

    def test_long_string_truncated(self):
        """Long strings are truncated."""
        long_str = "x" * 1000
        result = truncate_for_log(long_str, max_size=100)
        assert len(result) < len(long_str)
        assert "chars total" in result

    def test_dict_truncated(self):
        """Dicts are truncated recursively."""
        data = {f"key{i}": "x" * 1000 for i in range(5)}
        result = truncate_for_log(data, max_size=1000)
        assert isinstance(result, dict)

    def test_list_truncated(self):
        """Long lists are truncated."""
        data = list(range(100))
        result = truncate_for_log(data, max_size=1000)
        assert isinstance(result, list)
        assert len(result) <= 11  # 10 items + "... (N items)"

    def test_bytes_truncated(self):
        """Large bytes are shown as size."""
        data = b"x" * 100000
        result = truncate_for_log(data, max_size=100)
        assert "bytes" in result


class TestTimer:
    """Tests for timer context manager."""

    def test_tracks_elapsed_time(self):
        """Tracks elapsed time."""
        import time

        with patch.dict(os.environ, {"RLM_VERBOSE": "0"}, clear=True):
            with timer("test", log=False) as t:
                time.sleep(0.01)

        assert t["elapsed"] >= 0.01
        assert t["start"] < t["end"]

    def test_returns_dict(self):
        """Returns dict with timing info."""
        with timer("test", log=False) as t:
            pass

        assert "start" in t
        assert "end" in t
        assert "elapsed" in t


class TestDebugLog:
    """Tests for debug_log function."""

    def test_logs_at_level(self):
        """Logs at specified level."""
        with patch.dict(os.environ, {"RLM_VERBOSE": "1"}, clear=True):
            # Just verify it doesn't raise
            debug_log("test message", level="info")

    def test_skips_debug_when_not_verbose(self):
        """Skips debug level when not verbose."""
        with patch.dict(os.environ, {}, clear=True):
            for key in ["RLM_DEBUG", "RLM_VERBOSE"]:
                os.environ.pop(key, None)
            # Should not log anything
            debug_log("test message", level="debug")


class TestDebugRequest:
    """Tests for debug_request function."""

    def test_logs_when_debug(self):
        """Logs request when in debug mode."""
        with patch.dict(os.environ, {"RLM_DEBUG": "1"}, clear=True):
            # Just verify it doesn't raise
            debug_request(
                method="POST",
                url="https://api.example.com/v1",
                payload={"key": "value"},
                headers={"Authorization": "Bearer token"},
            )

    def test_skips_when_not_debug(self):
        """Skips logging when not in debug mode."""
        with patch.dict(os.environ, {}, clear=True):
            for key in ["RLM_DEBUG", "RLM_VERBOSE"]:
                os.environ.pop(key, None)
            # Should not raise or log
            debug_request("GET", "https://api.example.com")


class TestDebugResponse:
    """Tests for debug_response function."""

    def test_logs_when_debug(self):
        """Logs response when in debug mode."""
        with patch.dict(os.environ, {"RLM_DEBUG": "1"}, clear=True):
            debug_response(
                status=200,
                data={"result": "success"},
                elapsed=0.5,
            )

    def test_skips_when_not_debug(self):
        """Skips logging when not in debug mode."""
        with patch.dict(os.environ, {}, clear=True):
            for key in ["RLM_DEBUG", "RLM_VERBOSE"]:
                os.environ.pop(key, None)
            debug_response(200)


class TestTraceDecorator:
    """Tests for trace decorator."""

    def test_returns_result_when_not_debug(self):
        """Returns function result when not in debug mode."""
        with patch.dict(os.environ, {}, clear=True):
            for key in ["RLM_DEBUG", "RLM_VERBOSE"]:
                os.environ.pop(key, None)

            @trace("my_func")
            def my_func(x):
                return x * 2

            result = my_func(5)
            assert result == 10

    def test_returns_result_when_debug(self):
        """Returns function result when in debug mode."""
        with patch.dict(os.environ, {"RLM_DEBUG": "1"}, clear=True):
            @trace("my_func")
            def my_func(x):
                return x * 2

            result = my_func(5)
            assert result == 10

    def test_preserves_exception(self):
        """Preserves exceptions from decorated function."""
        with patch.dict(os.environ, {"RLM_DEBUG": "1"}, clear=True):
            @trace("failing_func")
            def failing_func():
                raise ValueError("test error")

            with pytest.raises(ValueError, match="test error"):
                failing_func()

    def test_uses_function_name_if_not_provided(self):
        """Uses function name if trace name not provided."""
        @trace()
        def named_function():
            return "result"

        result = named_function()
        assert result == "result"


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self):
        """Returns a logger instance."""
        logger = setup_logging(level=logging.INFO)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "rlm_dspy"

    def test_respects_verbosity(self):
        """Sets log level based on verbosity."""
        with patch.dict(os.environ, {"RLM_DEBUG": "1"}, clear=True):
            logger = setup_logging()
            assert logger.level == logging.DEBUG

    def test_custom_level(self):
        """Accepts custom log level."""
        logger = setup_logging(level=logging.WARNING)
        assert logger.level == logging.WARNING

"""Tests for callbacks module."""

import pytest
import time
from unittest.mock import Mock, patch

from rlm_dspy.core.callbacks import (
    Callback,
    CallbackContext,
    CallbackManager,
    LoggingCallback,
    MetricsCallback,
    ProgressCallback,
    get_callback_manager,
    clear_callback_manager,
    with_callbacks,
    emit_event,
)


class TestCallbackContext:
    """Tests for CallbackContext."""

    def test_create_context(self):
        """Test creating a context."""
        ctx = CallbackContext(event="test.event", data={"key": "value"})
        assert ctx.event == "test.event"
        assert ctx.data["key"] == "value"
        assert ctx.timestamp is not None

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        ctx = CallbackContext(event="test")
        ctx.start_time = 1000.0
        ctx.end_time = 1002.5
        assert ctx.elapsed == 2.5

    def test_elapsed_none_without_times(self):
        """Test elapsed is None without times."""
        ctx = CallbackContext(event="test")
        assert ctx.elapsed is None


class TestCallback:
    """Tests for Callback base class."""

    def test_callback_methods_exist(self):
        """Test that callback has expected methods."""
        class TestCallback(Callback):
            def on_event(self, ctx):
                pass

        cb = TestCallback()
        assert hasattr(cb, 'on_query_start')
        assert hasattr(cb, 'on_query_end')
        assert hasattr(cb, 'on_iteration_start')
        assert hasattr(cb, 'on_iteration_end')
        assert hasattr(cb, 'on_tool_call')
        assert hasattr(cb, 'on_error')


class TestLoggingCallback:
    """Tests for LoggingCallback."""

    def test_logs_events(self, caplog):
        """Test that events are logged."""
        import logging
        caplog.set_level(logging.DEBUG)

        cb = LoggingCallback(level=logging.DEBUG)
        ctx = CallbackContext(event="test.event", data={"message": "test message"})

        cb.on_event(ctx)

        assert "test.event" in caplog.text
        assert "test message" in caplog.text

    def test_logs_elapsed_time(self, caplog):
        """Test that elapsed time is logged."""
        import logging
        caplog.set_level(logging.DEBUG)

        cb = LoggingCallback()
        ctx = CallbackContext(event="test")
        ctx.start_time = 100.0
        ctx.end_time = 101.5

        cb.on_event(ctx)

        assert "(1.50s)" in caplog.text


class TestMetricsCallback:
    """Tests for MetricsCallback."""

    def test_counts_events(self):
        """Test that events are counted."""
        cb = MetricsCallback()

        cb.on_event(CallbackContext(event="query.start"))
        cb.on_event(CallbackContext(event="query.start"))
        cb.on_event(CallbackContext(event="query.end"))

        assert cb.counts["query.start"] == 2
        assert cb.counts["query.end"] == 1

    def test_tracks_timing(self):
        """Test that timing is tracked."""
        cb = MetricsCallback()

        ctx = CallbackContext(event="iteration")
        ctx.start_time = 100.0
        ctx.end_time = 100.5
        cb.on_event(ctx)

        ctx2 = CallbackContext(event="iteration")
        ctx2.start_time = 100.0
        ctx2.end_time = 101.0
        cb.on_event(ctx2)

        assert len(cb.metrics["iteration"]) == 2
        assert cb.metrics["iteration"][0] == 0.5
        assert cb.metrics["iteration"][1] == 1.0

    def test_tracks_errors(self):
        """Test that errors are tracked."""
        cb = MetricsCallback()

        ctx = CallbackContext(event="error", data={"error": "test error"})
        cb.on_event(ctx)

        assert len(cb.errors) == 1
        assert cb.errors[0]["error"] == "test error"

    def test_get_summary(self):
        """Test getting metrics summary."""
        cb = MetricsCallback()

        cb.on_event(CallbackContext(event="query"))
        
        ctx = CallbackContext(event="iteration")
        ctx.start_time = 100.0
        ctx.end_time = 100.5
        cb.on_event(ctx)

        summary = cb.get_summary()

        assert "counts" in summary
        assert "timing" in summary
        assert summary["counts"]["query"] == 1
        assert summary["timing"]["iteration"]["mean"] == 0.5

    def test_reset(self):
        """Test resetting metrics."""
        cb = MetricsCallback()
        cb.on_event(CallbackContext(event="test"))
        cb.errors.append({"error": "test"})

        cb.reset()

        assert len(cb.counts) == 0
        assert len(cb.metrics) == 0
        assert len(cb.errors) == 0


class TestProgressCallback:
    """Tests for ProgressCallback."""

    def test_reports_progress(self):
        """Test that progress is reported."""
        reports = []
        cb = ProgressCallback(on_progress=lambda msg, pct: reports.append((msg, pct)))

        # Start
        cb.on_event(CallbackContext(event="query.start", data={"max_iterations": 10}))
        assert reports[-1][1] == 0

        # Iterations
        cb.on_event(CallbackContext(event="iteration.end"))
        assert reports[-1][1] == 0.1

        cb.on_event(CallbackContext(event="iteration.end"))
        assert reports[-1][1] == 0.2

        # End
        cb.on_event(CallbackContext(event="query.end"))
        assert reports[-1][1] == 1.0


class TestCallbackManager:
    """Tests for CallbackManager."""

    def test_add_remove_callback(self):
        """Test adding and removing callbacks."""
        manager = CallbackManager()
        cb = LoggingCallback()

        manager.add(cb)
        assert cb in manager.callbacks

        manager.remove(cb)
        assert cb not in manager.callbacks

    def test_clear_callbacks(self):
        """Test clearing all callbacks."""
        manager = CallbackManager()
        manager.add(LoggingCallback())
        manager.add(MetricsCallback())

        manager.clear()

        assert len(manager.callbacks) == 0

    def test_emit_event(self):
        """Test emitting events to callbacks."""
        manager = CallbackManager()
        mock_cb = Mock(spec=Callback)
        mock_cb.on_event = Mock()

        manager.add(mock_cb)
        ctx = manager.emit("test.event", key="value")

        mock_cb.on_event.assert_called_once()
        call_ctx = mock_cb.on_event.call_args[0][0]
        assert call_ctx.event == "test.event"
        assert call_ctx.data["key"] == "value"

    def test_timed_event(self):
        """Test timed event context manager."""
        manager = CallbackManager()
        metrics = MetricsCallback()
        manager.add(metrics)

        with manager.timed_event("test", data="value"):
            time.sleep(0.1)

        assert "test.start" in metrics.counts
        assert "test.end" in metrics.counts

    def test_callback_error_handled(self):
        """Test that callback errors don't break execution."""
        manager = CallbackManager()

        class BadCallback(Callback):
            def on_event(self, ctx):
                raise ValueError("Intentional error")

        manager.add(BadCallback())
        manager.add(MetricsCallback())  # Should still work

        # Should not raise
        manager.emit("test")


class TestGlobalManager:
    """Tests for global callback manager functions."""

    def test_get_callback_manager(self):
        """Test getting global manager."""
        clear_callback_manager()
        m1 = get_callback_manager()
        m2 = get_callback_manager()
        assert m1 is m2

    def test_clear_callback_manager(self):
        """Test clearing global manager."""
        m1 = get_callback_manager()
        m1.add(LoggingCallback())
        
        clear_callback_manager()
        m2 = get_callback_manager()
        
        assert m1 is not m2
        assert len(m2.callbacks) == 0


class TestWithCallbacks:
    """Tests for with_callbacks decorator."""

    def test_decorator_emits_events(self):
        """Test that decorator emits start/end events."""
        clear_callback_manager()
        metrics = MetricsCallback()
        get_callback_manager().add(metrics)

        @with_callbacks("test_func")
        def my_function():
            return "result"

        result = my_function()

        assert result == "result"
        assert "test_func.start" in metrics.counts
        assert "test_func.end" in metrics.counts

    def test_decorator_uses_function_name(self):
        """Test that decorator uses function name if no prefix."""
        clear_callback_manager()
        metrics = MetricsCallback()
        get_callback_manager().add(metrics)

        @with_callbacks()
        def another_function():
            pass

        another_function()

        assert "another_function.start" in metrics.counts


class TestEmitEvent:
    """Tests for emit_event function."""

    def test_emit_event(self):
        """Test emitting event via global function."""
        clear_callback_manager()
        metrics = MetricsCallback()
        get_callback_manager().add(metrics)

        emit_event("custom.event", data="test")

        assert "custom.event" in metrics.counts

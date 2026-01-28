#!/usr/bin/env python3
"""Tests for patterns learned from modaic."""

import json
import tempfile
import time
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


class TestTypes:
    """Test type definitions."""

    def test_failed_chunk(self):
        """Test FailedChunk dataclass."""
        from rlm_dspy.core import FailedChunk

        fc = FailedChunk(error="timeout", index=5, chunk_preview="def foo()...")
        assert fc.error == "timeout"
        assert fc.index == 5
        assert fc.retryable is True
        assert "index=5" in str(fc)

    def test_chunk_result(self):
        """Test ChunkResult dataclass."""
        from rlm_dspy.core import ChunkResult

        cr = ChunkResult(index=0, relevant_info="Found X", confidence="high")
        assert cr.success is True
        assert cr.has_info is True

        cr_empty = ChunkResult(index=1, relevant_info="", confidence="none")
        assert cr_empty.has_info is False

    def test_batch_result(self):
        """Test BatchResult with partial success."""
        from rlm_dspy.core import BatchResult, ChunkResult, FailedChunk

        result = BatchResult(
            results=[
                ChunkResult(index=0, relevant_info="A", confidence="high"),
                ChunkResult(index=1, relevant_info="B", confidence="medium"),
            ],
            failed=[
                FailedChunk(error="timeout", index=2),
            ],
        )

        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.success_rate == pytest.approx(2 / 3)
        assert len(result.get_answers()) == 2


class TestProgress:
    """Test progress display."""

    def test_batch_progress(self):
        """Test BatchProgress state tracking."""
        from rlm_dspy.core import BatchProgress

        progress = BatchProgress(total_chunks=100, model="gemini")
        assert progress.progress_pct == 0

        progress.update(processed=50)
        assert progress.progress_pct == 50
        assert progress.processed == 50

        progress.update(processed=90, failed=10, status="completed")
        assert progress.progress_pct == 100
        assert progress.status == "completed"

    def test_batch_progress_rate(self):
        """Test rate calculation."""
        from rlm_dspy.core import BatchProgress

        progress = BatchProgress(total_chunks=10, model="test")
        progress.start_time = time.time() - 10  # 10 seconds ago
        progress.update(processed=5)

        assert progress.rate == pytest.approx(0.5, rel=0.1)  # 5 chunks / 10 sec


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
        """Test value masking."""
        from rlm_dspy.core import mask_value

        assert mask_value("sk-1234567890abcdef") == "sk-1...cdef"
        assert mask_value("short") == "********"
        assert mask_value(None) == "[None]"

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


class TestObservability:
    """Test observability utilities."""

    def test_tracker_span(self):
        """Test span tracking."""
        from rlm_dspy.core import SpanType, Tracker

        tracker = Tracker(enabled=True)

        with tracker.span("test_op", SpanType.LLM):
            time.sleep(0.01)

        assert len(tracker.spans) == 1
        assert tracker.spans[0].name == "test_op"
        assert tracker.spans[0].span_type == SpanType.LLM
        assert tracker.spans[0].duration_ms > 0

    def test_tracker_summary(self):
        """Test summary statistics."""
        from rlm_dspy.core import SpanType, Tracker

        tracker = Tracker(enabled=True)

        with tracker.span("llm1", SpanType.LLM):
            pass
        with tracker.span("llm2", SpanType.LLM):
            pass
        with tracker.span("chunk1", SpanType.CHUNK):
            pass

        summary = tracker.get_summary()
        assert summary["total_spans"] == 3
        assert summary["llm_count"] == 2
        assert summary["chunk_count"] == 1

    def test_track_decorator(self):
        """Test @track decorator."""
        from rlm_dspy.core import SpanType, enable_tracking, get_tracker, track

        enable_tracking(True)
        tracker = get_tracker()
        tracker.clear()

        @track("my_func", SpanType.TOOL)
        def my_function(x):
            return x * 2

        result = my_function(21)
        assert result == 42

    def test_span_error_handling(self):
        """Test span captures errors."""
        from rlm_dspy.core import Tracker

        tracker = Tracker(enabled=True)

        with pytest.raises(ValueError):
            with tracker.span("failing_op"):
                raise ValueError("Something went wrong")

        assert len(tracker.spans) == 1
        assert tracker.spans[0].error == "Something went wrong"


class TestRegistry:
    """Test registry pattern."""

    def test_registry_register(self):
        """Test basic registration."""
        from rlm_dspy.core import Registry

        reg = Registry[object]("test")

        @reg.register("item1")
        class Item1:
            name = "one"

        assert "item1" in reg
        assert len(reg) == 1

    def test_registry_get(self):
        """Test getting items."""
        from rlm_dspy.core import Registry

        reg = Registry[object]("test")

        @reg.register("thing")
        class Thing:
            value = 42

        instance = reg.get("thing")
        assert instance.value == 42

        # Cached instance
        instance2 = reg.get("thing")
        assert instance is instance2

    def test_registry_freeze(self):
        """Test freezing prevents modifications."""
        from rlm_dspy.core import Registry, RegistryError

        reg = Registry[object]("test")
        reg.add("a", object)
        reg.freeze()

        with pytest.raises(RegistryError):
            reg.add("b", object)

    def test_load_class(self):
        """Test dynamic class loading."""
        from rlm_dspy.core import load_class

        cls = load_class("rlm_dspy.core.RLM")
        assert cls.__name__ == "RLM"

        # Cached
        cls2 = load_class("rlm_dspy.core.RLM")
        assert cls is cls2


class TestBatch:
    """Test batch processing utilities."""

    def test_batch_request_formats(self):
        """Test request formatting."""
        from rlm_dspy.core import BatchRequest

        req = BatchRequest(
            custom_id="req-0",
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
        )

        openai = req.to_openai_format()
        assert openai["custom_id"] == "req-0"
        assert openai["method"] == "POST"
        assert openai["body"]["model"] == "gpt-4"

        anthropic = req.to_anthropic_format()
        assert anthropic["custom_id"] == "req-0"
        assert anthropic["params"]["model"] == "gpt-4"

    def test_batch_status(self):
        """Test batch status states."""
        from rlm_dspy.core import BatchStatus

        pending = BatchStatus(id="b1", status="pending")
        assert not pending.is_terminal

        completed = BatchStatus(id="b2", status="completed")
        assert completed.is_terminal
        assert completed.is_success

        failed = BatchStatus(id="b3", status="failed")
        assert failed.is_terminal
        assert not failed.is_success

    def test_create_parse_jsonl(self):
        """Test JSONL creation and parsing."""
        from rlm_dspy.core import BatchRequest, create_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            requests = [
                BatchRequest(
                    custom_id=f"req-{i}",
                    messages=[{"role": "user", "content": f"Test {i}"}],
                )
                for i in range(5)
            ]

            path = create_jsonl(requests, output_path=Path(tmpdir) / "test.jsonl")

            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 5

            # Verify content
            first = json.loads(lines[0])
            assert first["custom_id"] == "req-0"

    def test_sort_results_by_custom_id(self):
        """Test result sorting."""
        from rlm_dspy.core.batch import BatchResult, sort_results_by_custom_id

        results = [
            BatchResult(custom_id="request-2", content="C"),
            BatchResult(custom_id="request-0", content="A"),
            BatchResult(custom_id="request-1", content="B"),
        ]

        sorted_results = sort_results_by_custom_id(results)
        assert [r.content for r in sorted_results] == ["A", "B", "C"]


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


class TestIntegration:
    """Integration tests combining multiple patterns."""

    def test_tracked_batch_with_secrets(self):
        """Test combining tracking, batching, and secret handling."""
        from rlm_dspy.core import (
            BatchRequest,
            SpanType,
            Tracker,
            clean_secrets,
            create_jsonl,
        )

        tracker = Tracker(enabled=True)

        with tracker.span("batch_creation", SpanType.TOOL):
            requests = [
                BatchRequest(
                    custom_id=f"req-{i}",
                    messages=[{"role": "user", "content": f"Query {i}"}],
                    model="gpt-4",
                    metadata={"api_key": "sk-secret123"},
                )
                for i in range(3)
            ]

            # Clean secrets before logging
            for req in requests:
                req.metadata = clean_secrets(req.metadata)

            with tempfile.TemporaryDirectory() as tmpdir:
                path = create_jsonl(requests, output_path=Path(tmpdir) / "batch.jsonl")
                assert path.exists()

        assert len(tracker.spans) == 1
        assert tracker.spans[0].name == "batch_creation"

        # Verify secrets were cleaned
        assert requests[0].metadata["api_key"] == "********"

    def test_retry_with_progress(self):
        """Test retry with progress tracking."""
        from rlm_dspy.core import BatchProgress, retry_sync

        progress = BatchProgress(total_chunks=10, model="test")
        attempts = []

        @retry_sync(max_retries=2, base_delay=0.01)
        def process_with_retry(chunk_id: int):
            attempts.append(chunk_id)
            if len(attempts) < 2:
                raise ConnectionError("Retry me")
            progress.update(processed=progress.processed + 1)
            return f"done-{chunk_id}"

        result = process_with_retry(0)
        assert result == "done-0"
        assert progress.processed == 1
        assert len(attempts) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

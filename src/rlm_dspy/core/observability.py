"""Observability and tracing utilities.

Learned from modaic: @track decorator and span tracking.
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Span types for categorization
class SpanType:
    LLM = "llm"
    CHUNK = "chunk"
    AGGREGATE = "aggregate"
    TOOL = "tool"
    GUARDRAIL = "guardrail"
    CUSTOM = "custom"


@dataclass
class Span:
    """A traced span of execution."""

    name: str
    span_type: str = SpanType.CUSTOM
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    parent_id: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def finish(self, error: str | None = None) -> None:
        self.end_time = time.perf_counter()
        if error:
            self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.span_type,
            "duration_ms": self.duration_ms,
            "metadata": _truncate_for_logging(self.metadata),
            "error": self.error,
        }


def _truncate_for_logging(data: Any, max_str_len: int = 500) -> Any:
    """
    Truncate large values for logging.

    Learned from modaic: prevent performance degradation during serialization.
    """
    if isinstance(data, str):
        if len(data) > max_str_len:
            return data[:max_str_len] + f"... ({len(data)} chars)"
        return data
    elif isinstance(data, dict):
        return {k: _truncate_for_logging(v, max_str_len) for k, v in data.items()}
    elif isinstance(data, list):
        return [_truncate_for_logging(item, max_str_len) for item in data[:10]]
    return data


class Tracker:
    """
    Simple tracing tracker for observability.

    Learned from modaic's Trackable pattern.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.spans: list[Span] = []
        self._current_span: Span | None = None

    @contextmanager
    def span(
        self,
        name: str,
        span_type: str = SpanType.CUSTOM,
        **metadata: Any,
    ) -> Generator[Span, None, None]:
        """Create a tracked span context."""
        if not self.enabled:
            yield Span(name=name, span_type=span_type)
            return

        span = Span(
            name=name,
            span_type=span_type,
            metadata=metadata,
            parent_id=self._current_span.name if self._current_span else None,
        )

        prev_span = self._current_span
        self._current_span = span

        try:
            yield span
        except Exception as e:
            span.finish(error=str(e))
            raise
        finally:
            span.finish()
            self.spans.append(span)
            self._current_span = prev_span

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Span %s (%s): %.1fms",
                    span.name,
                    span.span_type,
                    span.duration_ms,
                )

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        if not self.spans:
            return {"total_spans": 0}

        by_type: dict[str, list[float]] = {}
        for span in self.spans:
            by_type.setdefault(span.span_type, []).append(span.duration_ms)

        summary = {
            "total_spans": len(self.spans),
            "total_time_ms": sum(s.duration_ms for s in self.spans),
            "errors": len([s for s in self.spans if s.error]),
        }

        for span_type, durations in by_type.items():
            summary[f"{span_type}_count"] = len(durations)
            summary[f"{span_type}_avg_ms"] = sum(durations) / len(durations)

        return summary

    def clear(self) -> None:
        """Clear all recorded spans."""
        self.spans = []
        self._current_span = None


# Global tracker instance
_global_tracker = Tracker(enabled=False)


def get_tracker() -> Tracker:
    """Get the global tracker instance."""
    return _global_tracker


def enable_tracking(enabled: bool = True) -> None:
    """Enable or disable global tracking."""
    _global_tracker.enabled = enabled


def track(
    name: str | None = None,
    span_type: str = SpanType.CUSTOM,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to track function execution.

    Learned from modaic's @track pattern.

    Usage:
        @track("my_function", SpanType.LLM)
        def my_function():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            tracker = get_tracker()
            with tracker.span(span_name, span_type):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            tracker = get_tracker()
            with tracker.span(span_name, span_type):
                return await func(*args, **kwargs)

        # Check if function is async
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper

    return decorator


@dataclass
class OperationMetrics:
    """Metrics for a single RLM operation."""

    query: str
    context_chars: int
    chunks_processed: int
    chunks_failed: int
    total_tokens: int
    total_cost: float
    elapsed_ms: float
    strategy: str
    model: str

    def log(self) -> None:
        """Log metrics at INFO level."""
        logger.info(
            "RLM query completed: %d chunks (%.1f%% success), %.1fs, %d tokens, $%.4f",
            self.chunks_processed + self.chunks_failed,
            self.chunks_processed / max(1, self.chunks_processed + self.chunks_failed) * 100,
            self.elapsed_ms / 1000,
            self.total_tokens,
            self.total_cost,
        )

"""
Callback Middleware for extensibility.

Provides a callback system for hooking into RLM execution lifecycle,
following DSPy's @with_callbacks pattern.
"""

import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, UTC
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class CallbackContext:
    """Context passed to callbacks."""
    
    event: str  # Event name (e.g., "query.start", "iteration.end")
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = field(default_factory=dict)
    
    # Timing info
    start_time: float | None = None
    end_time: float | None = None
    
    @property
    def elapsed(self) -> float | None:
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class Callback(ABC):
    """Base class for callbacks."""
    
    @abstractmethod
    def on_event(self, ctx: CallbackContext) -> None:
        """Handle a callback event.
        
        Args:
            ctx: Callback context with event details
        """
        pass
    
    def on_query_start(self, ctx: CallbackContext) -> None:
        """Called when a query starts."""
        self.on_event(ctx)
    
    def on_query_end(self, ctx: CallbackContext) -> None:
        """Called when a query ends."""
        self.on_event(ctx)
    
    def on_iteration_start(self, ctx: CallbackContext) -> None:
        """Called when an RLM iteration starts."""
        self.on_event(ctx)
    
    def on_iteration_end(self, ctx: CallbackContext) -> None:
        """Called when an RLM iteration ends."""
        self.on_event(ctx)
    
    def on_tool_call(self, ctx: CallbackContext) -> None:
        """Called when a tool is invoked."""
        self.on_event(ctx)
    
    def on_llm_call(self, ctx: CallbackContext) -> None:
        """Called when an LLM call is made."""
        self.on_event(ctx)
    
    def on_validation(self, ctx: CallbackContext) -> None:
        """Called when validation runs."""
        self.on_event(ctx)
    
    def on_error(self, ctx: CallbackContext) -> None:
        """Called when an error occurs."""
        self.on_event(ctx)


class LoggingCallback(Callback):
    """Callback that logs all events."""
    
    def __init__(self, level: int = logging.DEBUG):
        self.level = level
    
    def on_event(self, ctx: CallbackContext) -> None:
        elapsed = f" ({ctx.elapsed:.2f}s)" if ctx.elapsed else ""
        logger.log(
            self.level,
            "[%s]%s %s",
            ctx.event,
            elapsed,
            ctx.data.get("message", ""),
        )


class MetricsCallback(Callback):
    """Callback that collects metrics."""
    
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}
        self.errors: list[dict] = []
    
    def on_event(self, ctx: CallbackContext) -> None:
        # Count events
        self.counts[ctx.event] = self.counts.get(ctx.event, 0) + 1
        
        # Track timing
        if ctx.elapsed is not None:
            if ctx.event not in self.metrics:
                self.metrics[ctx.event] = []
            self.metrics[ctx.event].append(ctx.elapsed)
        
        # Track errors
        if ctx.event == "error":
            self.errors.append({
                "timestamp": ctx.timestamp.isoformat(),
                "error": ctx.data.get("error"),
                "traceback": ctx.data.get("traceback"),
            })
    
    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        summary = {
            "counts": self.counts.copy(),
            "errors": len(self.errors),
            "timing": {},
        }
        
        for event, times in self.metrics.items():
            if times:
                summary["timing"][event] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
        
        return summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()
        self.errors.clear()


class ProgressCallback(Callback):
    """Callback for progress reporting."""
    
    def __init__(self, on_progress: Callable[[str, float], None] | None = None):
        """
        Args:
            on_progress: Callback function(message, progress_pct)
        """
        self.on_progress = on_progress
        self.total_iterations = 0
        self.current_iteration = 0
    
    def on_event(self, ctx: CallbackContext) -> None:
        if ctx.event == "query.start":
            self.total_iterations = ctx.data.get("max_iterations", 20)
            self.current_iteration = 0
            self._report("Starting query...", 0)
        
        elif ctx.event == "iteration.end":
            self.current_iteration += 1
            progress = self.current_iteration / self.total_iterations if self.total_iterations > 0 else 0
            self._report(f"Iteration {self.current_iteration}/{self.total_iterations}", progress)
        
        elif ctx.event == "query.end":
            self._report("Query complete", 1.0)
    
    def _report(self, message: str, progress: float) -> None:
        if self.on_progress:
            self.on_progress(message, progress)


class CallbackManager:
    """Manages a collection of callbacks."""
    
    def __init__(self):
        self.callbacks: list[Callback] = []
    
    def add(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def remove(self, callback: Callback) -> None:
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def clear(self) -> None:
        """Remove all callbacks."""
        self.callbacks.clear()
    
    def emit(self, event: str, **data: Any) -> CallbackContext:
        """Emit an event to all callbacks.
        
        Args:
            event: Event name
            **data: Event data
            
        Returns:
            The callback context
        """
        ctx = CallbackContext(event=event, data=data)
        
        for callback in self.callbacks:
            try:
                # Route to specific handler if exists
                handler_name = f"on_{event.replace('.', '_')}"
                handler = getattr(callback, handler_name, None)
                if handler:
                    handler(ctx)
                else:
                    callback.on_event(ctx)
            except Exception as e:
                logger.warning("Callback %s failed for %s: %s", type(callback).__name__, event, e)
        
        return ctx
    
    @contextmanager
    def timed_event(self, event: str, **data: Any):
        """Context manager for timed events.
        
        Usage:
            with manager.timed_event("query", query=query):
                # do work
        """
        ctx = CallbackContext(event=f"{event}.start", data=data)
        ctx.start_time = time.time()
        
        # Emit start event
        for callback in self.callbacks:
            try:
                handler = getattr(callback, f"on_{event}_start", callback.on_event)
                handler(ctx)
            except Exception as e:
                logger.warning("Callback start failed: %s", e)
        
        try:
            yield ctx
        finally:
            ctx.event = f"{event}.end"
            ctx.end_time = time.time()
            
            # Emit end event
            for callback in self.callbacks:
                try:
                    handler = getattr(callback, f"on_{event}_end", callback.on_event)
                    handler(ctx)
                except Exception as e:
                    logger.warning("Callback end failed: %s", e)


# Global callback manager
_callback_manager: CallbackManager | None = None


def get_callback_manager() -> CallbackManager:
    """Get the global callback manager."""
    global _callback_manager
    if _callback_manager is None:
        _callback_manager = CallbackManager()
    return _callback_manager


def clear_callback_manager() -> None:
    """Clear the global callback manager."""
    global _callback_manager
    if _callback_manager:
        _callback_manager.clear()
    _callback_manager = None


def with_callbacks(event_prefix: str = "") -> Callable[[F], F]:
    """
    Decorator to add callback support to a function.
    
    Usage:
        @with_callbacks("query")
        def my_function(query: str) -> str:
            # Will emit query.start and query.end events
            return "result"
    
    Args:
        event_prefix: Prefix for event names
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_callback_manager()
            event = event_prefix or func.__name__
            
            with manager.timed_event(event, args=args, kwargs=kwargs):
                return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


def emit_event(event: str, **data: Any) -> None:
    """Emit an event to the global callback manager.
    
    Args:
        event: Event name
        **data: Event data
    """
    manager = get_callback_manager()
    manager.emit(event, **data)

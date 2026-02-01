"""Debug and verbose logging utilities.

Learned from modaic: comprehensive debug mode with multiple verbosity levels.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from functools import wraps
from typing import Any, Callable, Generator, TypeVar

from rich.console import Console
from rich.logging import RichHandler
from rich.syntax import Syntax

T = TypeVar("T")

# Global console for rich output
console = Console(stderr=True)


class Verbosity(IntEnum):
    """Verbosity levels."""

    QUIET = 0  # Only errors
    NORMAL = 1  # Info + warnings
    VERBOSE = 2  # Debug info
    DEBUG = 3  # Full trace with payloads


def get_verbosity() -> Verbosity:
    """Get current verbosity level from environment."""
    if os.environ.get("RLM_DEBUG", "").lower() in ("1", "true", "yes"):
        return Verbosity.DEBUG
    if os.environ.get("RLM_VERBOSE", "").lower() in ("1", "true", "yes"):
        return Verbosity.VERBOSE
    if os.environ.get("RLM_QUIET", "").lower() in ("1", "true", "yes"):
        return Verbosity.QUIET
    return Verbosity.NORMAL


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return get_verbosity() >= Verbosity.VERBOSE


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return get_verbosity() >= Verbosity.DEBUG


@dataclass
class DebugConfig:
    """Configuration for debug output."""

    verbosity: Verbosity = field(default_factory=get_verbosity)
    log_inputs: bool = True
    log_outputs: bool = True
    max_input_size: int = 10_000  # Truncate inputs larger than this
    max_output_size: int = 10_000  # Truncate outputs larger than this
    show_timestamps: bool = True
    show_tokens: bool = True
    show_cost: bool = True
    colorize: bool = True


# Global debug config
_config = DebugConfig()


def configure_debug(
    verbosity: Verbosity | int | None = None,
    log_inputs: bool | None = None,
    log_outputs: bool | None = None,
    max_input_size: int | None = None,
    max_output_size: int | None = None,
) -> None:
    """Configure debug settings."""
    global _config

    if verbosity is not None:
        try:
            _config.verbosity = Verbosity(verbosity)
        except ValueError:
            valid = [v.value for v in Verbosity]
            raise ValueError(f"verbosity must be one of {valid}, got {verbosity}")
    if log_inputs is not None:
        _config.log_inputs = log_inputs
    if log_outputs is not None:
        _config.log_outputs = log_outputs
    if max_input_size is not None:
        _config.max_input_size = max_input_size
    if max_output_size is not None:
        _config.max_output_size = max_output_size


def setup_logging(
    level: int | str | None = None,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """
    Setup logging with rich formatting.

    Args:
        level: Log level (defaults based on verbosity)
        rich_tracebacks: Use rich for exception formatting

    Returns:
        Configured logger
    """
    if level is None:
        verbosity = get_verbosity()
        level = {
            Verbosity.QUIET: logging.ERROR,
            Verbosity.NORMAL: logging.INFO,
            Verbosity.VERBOSE: logging.DEBUG,
            Verbosity.DEBUG: logging.DEBUG,
        }[verbosity]

    # Configure root logger for rlm_dspy
    logger = logging.getLogger("rlm_dspy")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Add rich handler
    handler = RichHandler(
        console=console,
        show_time=_config.show_timestamps,
        show_path=get_verbosity() >= Verbosity.DEBUG,
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=get_verbosity() >= Verbosity.DEBUG,
    )
    handler.setLevel(level)
    logger.addHandler(handler)

    return logger


def truncate_for_log(data: Any, max_size: int = 10_000) -> Any:
    """
    Truncate data for logging.

    Learned from modaic: prevent log bloat with large payloads.
    """
    if isinstance(data, str):
        if len(data) > max_size:
            return f"{data[:max_size]}... ({len(data):,} chars total)"
        return data
    elif isinstance(data, bytes):
        if len(data) > max_size:
            return f"<{len(data):,} bytes>"
        return data
    elif isinstance(data, dict):
        return {k: truncate_for_log(v, max_size // 10) for k, v in list(data.items())[:20]}
    elif isinstance(data, list):
        if len(data) > 10:
            return [truncate_for_log(x, max_size // 10) for x in data[:10]] + [f"... ({len(data)} items)"]
        return [truncate_for_log(x, max_size // 10) for x in data]
    return data


@contextmanager
def timer(name: str, log: bool = True) -> Generator[dict[str, float], None, None]:
    """
    Context manager to time a block of code.

    Learned from modaic: Timer for performance logging.

    Usage:
        with timer("my_operation") as t:
            do_something()
        print(f"Took {t['elapsed']:.2f}s")
    """
    result: dict[str, float] = {"start": 0, "end": 0, "elapsed": 0}
    result["start"] = time.perf_counter()

    try:
        yield result
    finally:
        result["end"] = time.perf_counter()
        result["elapsed"] = result["end"] - result["start"]

        if log and is_verbose():
            console.print(f"[dim]⏱ {name}: {result['elapsed']:.3f}s[/dim]")


def debug_log(
    message: str,
    data: Any = None,
    level: str = "debug",
    show_data: bool = True,
) -> None:
    """
    Log a debug message with optional data.

    Args:
        message: Log message
        data: Optional data to display
        level: Log level (debug, info, warning, error)
        show_data: Whether to show the data
    """
    if not is_verbose() and level == "debug":
        return
    if not is_debug() and data is not None:
        data = None

    logger = logging.getLogger("rlm_dspy")
    log_func = getattr(logger, level, logger.debug)

    log_func(message)

    if show_data and data is not None and is_debug():
        truncated = truncate_for_log(data, _config.max_output_size)
        if isinstance(truncated, (dict, list)):
            import json

            console.print(Syntax(json.dumps(truncated, indent=2, default=str), "json", theme="monokai"))
        else:
            console.print(f"[dim]{truncated}[/dim]")


def debug_request(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """Log an API request in debug mode."""
    if not is_debug():
        return

    console.print(f"\n[bold blue]→ {method}[/bold blue] {url}")

    if headers and _config.log_inputs:
        # Mask authorization header
        safe_headers = {k: "***" if "auth" in k.lower() else v for k, v in headers.items()}
        console.print(f"[dim]Headers: {safe_headers}[/dim]")

    if payload and _config.log_inputs:
        truncated = truncate_for_log(payload, _config.max_input_size)
        console.print(Syntax(str(truncated), "json", theme="monokai"))


def debug_response(
    status: int,
    data: Any = None,
    elapsed: float = 0,
) -> None:
    """Log an API response in debug mode."""
    if not is_debug():
        return

    color = "green" if 200 <= status < 300 else "red"
    console.print(f"[bold {color}]← {status}[/bold {color}] ({elapsed:.2f}s)")

    if data and _config.log_outputs:
        truncated = truncate_for_log(data, _config.max_output_size)
        if isinstance(truncated, str) and len(truncated) > 200:
            console.print(f"[dim]{truncated[:200]}...[/dim]")
        else:
            console.print(f"[dim]{truncated}[/dim]")


def trace(name: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to trace function execution in debug mode.

    Usage:
        @trace("my_function")
        def my_function(x):
            return x * 2
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not is_debug():
                return func(*args, **kwargs)

            # Log entry
            console.print(f"[dim]▶ {func_name}[/dim]")

            if _config.log_inputs and (args or kwargs):
                inputs = {"args": args[:3], "kwargs": {k: v for k, v in list(kwargs.items())[:5]}}
                truncated = truncate_for_log(inputs, 500)
                console.print(f"[dim]  inputs: {truncated}[/dim]")

            with timer(func_name, log=False) as t:
                try:
                    result = func(*args, **kwargs)

                    if _config.log_outputs:
                        truncated = truncate_for_log(result, 200)
                        console.print(f"[dim]  output: {truncated}[/dim]")

                    console.print(f"[dim]◀ {func_name} ({t['elapsed']:.3f}s)[/dim]")
                    return result

                except Exception as e:
                    console.print(f"[red]✗ {func_name} failed: {e}[/red]")
                    raise

        return wrapper

    return decorator


# Auto-setup logging on import if verbose/debug mode
if is_verbose():
    setup_logging()

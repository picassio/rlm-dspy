"""Retry utilities with exponential backoff and jitter.

Learned from modaic: robust network resilience patterns.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from functools import wraps
from typing import Any, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


def parse_retry_after(response: httpx.Response | None) -> float | None:
    """
    Parse Retry-After header from HTTP 429 response.

    Learned from modaic: intelligent rate limit handling.

    Supports:
    - Seconds: "Retry-After: 120"
    - HTTP date: "Retry-After: Wed, 21 Oct 2015 07:28:00 GMT"

    Returns:
        Delay in seconds, or None if not present/parseable
    """
    if response is None:
        return None

    retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
    if not retry_after:
        return None

    # Try as integer seconds
    try:
        return float(retry_after)
    except ValueError:
        pass

    # Try as HTTP date
    try:
        import time
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(retry_after)
        delay = dt.timestamp() - time.time()
        return max(0, delay)
    except Exception:
        pass

    # Try to extract number from error message (some APIs include it)
    match = re.search(r"(\d+)\s*(?:seconds?|s)", retry_after, re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None


def is_rate_limit_error(response: httpx.Response | None = None, error: Exception | None = None) -> bool:
    """Check if an error is a rate limit (429) error."""
    if response is not None and response.status_code == 429:
        return True

    if error is not None:
        error_str = str(error).lower()
        if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
            return True

    return False


async def retry_with_backoff(
    coro_func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    jitter: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    Retry an async function with exponential backoff and jitter.

    Pattern: delay = min(base_delay * 2^attempt + random(0, jitter), max_delay)

    Args:
        coro_func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        jitter: Random jitter range (0 to jitter seconds)
        retryable_exceptions: Tuple of exceptions to retry on

    Returns:
        Result from successful function call

    Raises:
        Last exception if all retries fail
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(
                    "All %d retries failed: %s",
                    max_retries,
                    str(e)[:100],
                )
                raise

            # Exponential backoff with jitter
            delay = min(
                base_delay * (2**attempt) + (random.random() * jitter),
                max_delay,
            )

            logger.warning(
                "Retry %d/%d after error: %s. Waiting %.2fs",
                attempt + 1,
                max_retries,
                str(e)[:50],
                delay,
            )

            await asyncio.sleep(delay)

    # Should not reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry loop exit")


def retry_sync(
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for synchronous retry with backoff.

    Usage:
        @retry_sync(max_retries=3)
        def fetch_data():
            ...
    """
    import time

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        raise

                    delay = base_delay * (2**attempt) + random.random()
                    logger.warning(
                        "Retry %d/%d: %s. Waiting %.2fs",
                        attempt + 1,
                        max_retries,
                        str(e)[:50],
                        delay,
                    )
                    time.sleep(delay)

            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected")

        return wrapper

    return decorator

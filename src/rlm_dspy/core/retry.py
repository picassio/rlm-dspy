"""Retry utilities with exponential backoff and jitter.

Learned from modaic: robust network resilience patterns.
"""

from __future__ import annotations

import asyncio
import logging
import random
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


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

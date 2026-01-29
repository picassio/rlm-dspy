"""Retry utilities with exponential backoff and jitter.

Learned from modaic: robust network resilience patterns.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from email.utils import parsedate_to_datetime
from functools import wraps
from typing import Any, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Network/API errors that are safe to retry
RETRYABLE_NETWORK_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    ConnectionError,
    TimeoutError,
    OSError,  # Includes socket errors
)

# HTTP status codes that indicate transient errors worth retrying
RETRYABLE_STATUS_CODES = {
    408,  # Request Timeout
    429,  # Too Many Requests (rate limit)
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
    520,  # Cloudflare: Unknown Error
    521,  # Cloudflare: Web Server Is Down
    522,  # Cloudflare: Connection Timed Out
    523,  # Cloudflare: Origin Is Unreachable
    524,  # Cloudflare: A Timeout Occurred
}


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable (transient network/server issue).

    Does NOT retry:
    - Programming errors (TypeError, ValueError, AttributeError, etc.)
    - Authentication errors (401)
    - Client errors (400-499 except 408, 429)
    - Malformed response errors

    Does retry:
    - Network connectivity issues
    - Timeouts
    - Server errors (5xx)
    - Rate limits (429)
    """
    # Check for retryable exception types
    if isinstance(error, RETRYABLE_NETWORK_EXCEPTIONS):
        return True

    # Check HTTP status errors
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS_CODES

    # Don't retry programming errors
    if isinstance(error, (TypeError, ValueError, AttributeError, KeyError, IndexError)):
        return False

    # Don't retry assertion errors
    if isinstance(error, AssertionError):
        return False

    # For unknown exceptions, check the message for hints
    error_str = str(error).lower()

    # Non-retryable patterns (check first)
    non_retryable_hints = ["invalid", "unauthorized", "forbidden", "not found", "bad request"]
    if any(hint in error_str for hint in non_retryable_hints):
        return False

    # Retryable keyword patterns
    retryable_keywords = ["timeout", "connection", "temporarily", "retry", "unavailable"]
    if any(hint in error_str for hint in retryable_keywords):
        return True

    # Check for HTTP status codes with word boundaries (avoid matching IDs like "user 503")
    # Match patterns like "503", "HTTP 503", "status 503", "error 503", "(503)"
    # Include all codes from RETRYABLE_STATUS_CODES
    retryable_codes = "|".join(str(c) for c in RETRYABLE_STATUS_CODES)
    status_pattern = rf'\b(?:http\s*)?(?:status\s*)?(?:code\s*)?(?:error\s*)?({retryable_codes})\b'
    if re.search(status_pattern, error_str, re.IGNORECASE):
        return True

    # Default: don't retry unknown errors (safer)
    return False


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

    # Maximum retry delay (1 hour) to prevent unbounded waits
    MAX_RETRY_DELAY = 3600.0
    
    retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
    if not retry_after:
        return None

    delay = None
    
    # Try as integer seconds
    try:
        delay = float(retry_after)
    except ValueError:
        pass

    # Try as HTTP date
    if delay is None:
        try:
            dt = parsedate_to_datetime(retry_after)
            delay = dt.timestamp() - time.time()
        except (ValueError, TypeError) as e:
            logger.debug("Failed to parse retry-after as HTTP date '%s': %s", retry_after, e)

    # Try to extract number from error message (some APIs include it)
    if delay is None:
        match = re.search(r"(\d+)\s*(?:seconds?|s)", retry_after, re.IGNORECASE)
        if match:
            delay = float(match.group(1))

    # Apply bounds: must be positive and capped at MAX_RETRY_DELAY
    if delay is not None:
        return max(0.0, min(delay, MAX_RETRY_DELAY))
    
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
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    use_smart_retry: bool = True,
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
        retryable_exceptions: Tuple of exceptions to retry on (if None, uses smart retry)
        use_smart_retry: If True and retryable_exceptions is None, use is_retryable_error()

    Returns:
        Result from successful function call

    Raises:
        Last exception if all retries fail
    """
    # Default to network exceptions if not specified
    if retryable_exceptions is None:
        retryable_exceptions = RETRYABLE_NETWORK_EXCEPTIONS

    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            # Never retry these - always re-raise immediately
            raise
        except Exception as e:
            # Check if this is a retryable exception
            should_retry = False
            if isinstance(e, retryable_exceptions):
                should_retry = True
            elif use_smart_retry:
                should_retry = is_retryable_error(e)

            if not should_retry:
                # Not retryable - raise immediately
                raise

            last_exception = e

            if attempt == max_retries:
                logger.error(
                    "All %d retries failed: %s",
                    max_retries,
                    str(e)[:100],
                )
                raise

            # Check for Retry-After header if it's an HTTP error
            delay = None
            if isinstance(e, httpx.HTTPStatusError):
                delay = parse_retry_after(e.response)

            # Fall back to exponential backoff with jitter
            if delay is None:
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
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    use_smart_retry: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for synchronous retry with backoff.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        retryable_exceptions: Specific exceptions to retry (if None, uses smart retry)
        use_smart_retry: If True and retryable_exceptions is None, use is_retryable_error()

    Usage:
        @retry_sync(max_retries=3)
        def fetch_data():
            ...
    """
    import time

    if retryable_exceptions is None:
        retryable_exceptions = RETRYABLE_NETWORK_EXCEPTIONS

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (KeyboardInterrupt, SystemExit, GeneratorExit):
                    # Never retry these - always re-raise immediately
                    raise
                except Exception as e:
                    # Check if this is a retryable exception
                    should_retry = False
                    if isinstance(e, retryable_exceptions):  # type: ignore
                        should_retry = True
                    elif use_smart_retry:
                        should_retry = is_retryable_error(e)

                    if not should_retry:
                        raise

                    last_exception = e

                    if attempt == max_retries:
                        raise

                    # Check for Retry-After header if it's an HTTP error
                    delay = None
                    if isinstance(e, httpx.HTTPStatusError):
                        delay = parse_retry_after(e.response)
                    
                    # Fall back to exponential backoff with jitter
                    if delay is None:
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


__all__ = [
    "retry_with_backoff",
    "retry_sync",
    "parse_retry_after",
    "is_rate_limit_error",
    "is_retryable_error",
    "RETRYABLE_NETWORK_EXCEPTIONS",
    "RETRYABLE_STATUS_CODES",
]

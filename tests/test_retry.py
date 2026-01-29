"""Tests for retry utilities with exponential backoff."""

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest

from rlm_dspy.core.retry import (
    RETRYABLE_NETWORK_EXCEPTIONS,
    RETRYABLE_STATUS_CODES,
    is_retryable_error,
    parse_retry_after,
    is_rate_limit_error,
    retry_with_backoff,
    retry_sync,
)


class TestRetryableStatusCodes:
    """Tests for retryable status code constants."""
    
    def test_includes_rate_limit(self):
        """Includes 429 rate limit."""
        assert 429 in RETRYABLE_STATUS_CODES
    
    def test_includes_server_errors(self):
        """Includes 5xx server errors."""
        assert 500 in RETRYABLE_STATUS_CODES
        assert 502 in RETRYABLE_STATUS_CODES
        assert 503 in RETRYABLE_STATUS_CODES
        assert 504 in RETRYABLE_STATUS_CODES
    
    def test_includes_timeout(self):
        """Includes 408 request timeout."""
        assert 408 in RETRYABLE_STATUS_CODES
    
    def test_excludes_client_errors(self):
        """Excludes most client errors."""
        assert 400 not in RETRYABLE_STATUS_CODES  # Bad Request
        assert 401 not in RETRYABLE_STATUS_CODES  # Unauthorized
        assert 403 not in RETRYABLE_STATUS_CODES  # Forbidden
        assert 404 not in RETRYABLE_STATUS_CODES  # Not Found


class TestRetryableNetworkExceptions:
    """Tests for retryable exception types."""
    
    def test_includes_connect_errors(self):
        """Includes connection errors."""
        assert httpx.ConnectError in RETRYABLE_NETWORK_EXCEPTIONS
        assert ConnectionError in RETRYABLE_NETWORK_EXCEPTIONS
    
    def test_includes_timeouts(self):
        """Includes timeout errors."""
        assert httpx.ConnectTimeout in RETRYABLE_NETWORK_EXCEPTIONS
        assert httpx.ReadTimeout in RETRYABLE_NETWORK_EXCEPTIONS
        assert TimeoutError in RETRYABLE_NETWORK_EXCEPTIONS
    
    def test_includes_network_errors(self):
        """Includes network errors."""
        assert httpx.NetworkError in RETRYABLE_NETWORK_EXCEPTIONS
        assert OSError in RETRYABLE_NETWORK_EXCEPTIONS


class TestIsRetryableError:
    """Tests for is_retryable_error function."""
    
    def test_timeout_error_retryable(self):
        """Timeout errors are retryable."""
        assert is_retryable_error(TimeoutError("timeout")) is True
    
    def test_connection_error_retryable(self):
        """Connection errors are retryable."""
        assert is_retryable_error(ConnectionError("refused")) is True
    
    def test_type_error_not_retryable(self):
        """TypeError is not retryable."""
        assert is_retryable_error(TypeError("bad type")) is False
    
    def test_value_error_not_retryable(self):
        """ValueError is not retryable."""
        assert is_retryable_error(ValueError("bad value")) is False
    
    def test_assertion_error_not_retryable(self):
        """AssertionError is not retryable."""
        assert is_retryable_error(AssertionError("failed")) is False
    
    def test_unauthorized_not_retryable(self):
        """401 unauthorized is not retryable."""
        error = Exception("unauthorized access")
        assert is_retryable_error(error) is False
    
    def test_retry_keyword_in_message_retryable(self):
        """Errors with retry-related keywords are retryable."""
        # These contain keywords from retryable_keywords list
        assert is_retryable_error(Exception("connection refused")) is True
        assert is_retryable_error(Exception("timeout occurred")) is True
        assert is_retryable_error(Exception("temporarily unavailable")) is True
        assert is_retryable_error(Exception("please retry")) is True
    
    def test_503_in_message_retryable(self):
        """Errors with 503 status in message are retryable."""
        error = Exception("server returned 503")
        assert is_retryable_error(error) is True
    
    def test_http_status_error_429_retryable(self):
        """HTTP 429 status error is retryable."""
        response = MagicMock()
        response.status_code = 429
        error = httpx.HTTPStatusError("rate limited", request=MagicMock(), response=response)
        assert is_retryable_error(error) is True
    
    def test_http_status_error_401_not_retryable(self):
        """HTTP 401 status error is not retryable."""
        response = MagicMock()
        response.status_code = 401
        error = httpx.HTTPStatusError("unauthorized", request=MagicMock(), response=response)
        assert is_retryable_error(error) is False


class TestParseRetryAfter:
    """Tests for parse_retry_after function."""
    
    def test_parses_seconds(self):
        """Parses integer seconds."""
        response = MagicMock()
        response.headers = {"retry-after": "120"}
        result = parse_retry_after(response)
        assert result == 120.0
    
    def test_parses_float_seconds(self):
        """Parses float seconds."""
        response = MagicMock()
        response.headers = {"retry-after": "30.5"}
        result = parse_retry_after(response)
        assert result == 30.5
    
    def test_returns_none_for_missing_header(self):
        """Returns None when header is missing."""
        response = MagicMock()
        response.headers = {}
        result = parse_retry_after(response)
        assert result is None
    
    def test_returns_none_for_none_response(self):
        """Returns None for None response."""
        result = parse_retry_after(None)
        assert result is None
    
    def test_caps_at_max_delay(self):
        """Caps delay at 1 hour maximum."""
        response = MagicMock()
        response.headers = {"retry-after": "999999"}
        result = parse_retry_after(response)
        assert result == 3600.0  # Max 1 hour
    
    def test_ensures_non_negative(self):
        """Ensures delay is non-negative."""
        response = MagicMock()
        response.headers = {"retry-after": "-10"}
        result = parse_retry_after(response)
        assert result == 0.0


class TestIsRateLimitError:
    """Tests for is_rate_limit_error function."""
    
    def test_response_429(self):
        """Detects 429 response."""
        response = MagicMock()
        response.status_code = 429
        assert is_rate_limit_error(response=response) is True
    
    def test_response_200(self):
        """Does not detect 200 response."""
        response = MagicMock()
        response.status_code = 200
        assert is_rate_limit_error(response=response) is False
    
    def test_error_with_429(self):
        """Detects 429 in error message."""
        error = Exception("error code 429")
        assert is_rate_limit_error(error=error) is True
    
    def test_error_with_rate_limit(self):
        """Detects 'rate limit' in error message."""
        error = Exception("rate limit exceeded")
        assert is_rate_limit_error(error=error) is True
    
    def test_error_with_too_many_requests(self):
        """Detects 'too many requests' in error message."""
        error = Exception("too many requests")
        assert is_rate_limit_error(error=error) is True


class TestRetrySyncDecorator:
    """Tests for retry_sync decorator."""
    
    def test_succeeds_first_try(self):
        """Returns immediately on success."""
        call_count = 0
        
        @retry_sync(max_retries=3)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = succeed()
        assert result == "success"
        assert call_count == 1
    
    def test_retries_on_retryable_error(self):
        """Retries on retryable errors."""
        call_count = 0
        
        @retry_sync(max_retries=3, base_delay=0.01)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("network issue")
            return "success"
        
        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 3
    
    def test_raises_non_retryable_immediately(self):
        """Raises non-retryable errors immediately."""
        call_count = 0
        
        @retry_sync(max_retries=3)
        def programming_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")
        
        with pytest.raises(ValueError):
            programming_error()
        
        assert call_count == 1  # No retries
    
    def test_raises_after_max_retries(self):
        """Raises after exhausting retries."""
        call_count = 0
        
        @retry_sync(max_retries=2, base_delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("network issue")
        
        with pytest.raises(ConnectionError):
            always_fail()
        
        assert call_count == 3  # Initial + 2 retries


class TestRetryWithBackoff:
    """Tests for async retry_with_backoff function."""
    
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        """Returns immediately on success."""
        call_count = 0
        
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await retry_with_backoff(succeed, max_retries=3)
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retries_on_retryable_error(self):
        """Retries on retryable errors."""
        call_count = 0
        
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("timeout")
            return "success"
        
        result = await retry_with_backoff(
            fail_then_succeed, 
            max_retries=3,
            base_delay=0.01,
        )
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_raises_non_retryable_immediately(self):
        """Raises non-retryable errors immediately."""
        call_count = 0
        
        async def type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("bad type")
        
        with pytest.raises(TypeError):
            await retry_with_backoff(type_error, max_retries=3)
        
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Raises after exhausting retries."""
        call_count = 0
        
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("timeout")
        
        with pytest.raises(TimeoutError):
            await retry_with_backoff(
                always_fail,
                max_retries=2,
                base_delay=0.01,
            )
        
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_keyboard_interrupt_not_retried(self):
        """KeyboardInterrupt is never retried."""
        call_count = 0
        
        async def keyboard_interrupt():
            nonlocal call_count
            call_count += 1
            raise KeyboardInterrupt()
        
        with pytest.raises(KeyboardInterrupt):
            await retry_with_backoff(keyboard_interrupt, max_retries=3)
        
        assert call_count == 1

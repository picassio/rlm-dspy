"""Tests for validation utilities."""

import os
from unittest.mock import patch, MagicMock

import pytest

from rlm_dspy.core.validation import (
    ValidationResult,
    PreflightResult,
    check_api_key,
    check_model_format,
    check_api_endpoint,
    check_budget,
    check_context_size,
    preflight_check,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_create_passed(self):
        """Can create a passing result."""
        result = ValidationResult(
            name="Test",
            passed=True,
            message="All good",
        )
        assert result.passed is True
        assert result.name == "Test"
        assert result.severity == "error"  # default
    
    def test_create_failed_with_suggestion(self):
        """Can create a failed result with suggestion."""
        result = ValidationResult(
            name="API Key",
            passed=False,
            message="Missing",
            severity="error",
            suggestion="Set OPENAI_API_KEY",
        )
        assert result.passed is False
        assert result.suggestion == "Set OPENAI_API_KEY"
    
    def test_warning_severity(self):
        """Can create a warning result."""
        result = ValidationResult(
            name="Budget",
            passed=False,
            message="Low budget",
            severity="warning",
        )
        assert result.severity == "warning"


class TestPreflightResult:
    """Tests for PreflightResult class."""
    
    def test_empty_result_passes(self):
        """Empty result passes."""
        result = PreflightResult()
        assert result.passed is True
        assert result.errors == []
        assert result.warnings == []
    
    def test_passed_with_all_passing_checks(self):
        """Passes when all checks pass."""
        result = PreflightResult()
        result.add(ValidationResult("Check1", True, "OK"))
        result.add(ValidationResult("Check2", True, "OK"))
        assert result.passed is True
    
    def test_fails_with_error(self):
        """Fails when any error check fails."""
        result = PreflightResult()
        result.add(ValidationResult("Check1", True, "OK"))
        result.add(ValidationResult("Check2", False, "Failed", severity="error"))
        assert result.passed is False
        assert len(result.errors) == 1
    
    def test_passes_with_only_warnings(self):
        """Passes when only warnings fail."""
        result = PreflightResult()
        result.add(ValidationResult("Check1", True, "OK"))
        result.add(ValidationResult("Check2", False, "Warning", severity="warning"))
        assert result.passed is True
        assert len(result.warnings) == 1
    
    def test_errors_property(self):
        """errors property returns only failed error checks."""
        result = PreflightResult()
        result.add(ValidationResult("Pass", True, "OK", severity="error"))
        result.add(ValidationResult("Fail", False, "Bad", severity="error"))
        result.add(ValidationResult("Warn", False, "Meh", severity="warning"))
        
        errors = result.errors
        assert len(errors) == 1
        assert errors[0].name == "Fail"
    
    def test_warnings_property(self):
        """warnings property returns only failed warning checks."""
        result = PreflightResult()
        result.add(ValidationResult("Fail", False, "Bad", severity="error"))
        result.add(ValidationResult("Warn1", False, "Meh1", severity="warning"))
        result.add(ValidationResult("Warn2", False, "Meh2", severity="warning"))
        
        warnings = result.warnings
        assert len(warnings) == 2


class TestCheckApiKey:
    """Tests for check_api_key function."""
    
    def test_finds_key_in_env(self):
        """Finds API key in environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=True):
            result = check_api_key(env_vars=["OPENAI_API_KEY"])
            assert result.passed is True
            assert "OPENAI_API_KEY" in result.message
    
    def test_tries_multiple_env_vars(self):
        """Tries multiple environment variables."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True):
            result = check_api_key(env_vars=["NONEXISTENT", "OPENROUTER_API_KEY"])
            assert result.passed is True
            assert "OPENROUTER_API_KEY" in result.message
    
    def test_fails_when_required_and_missing(self):
        """Fails when required and no key found."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove all possible API keys
            for key in ["RLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
                os.environ.pop(key, None)
            
            result = check_api_key(required=True)
            assert result.passed is False
            assert result.severity == "error"
    
    def test_warns_when_optional_and_missing(self):
        """Warns when optional and no key found."""
        with patch.dict(os.environ, {}, clear=True):
            for key in ["RLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
                os.environ.pop(key, None)
            
            result = check_api_key(required=False)
            assert result.passed is True  # Optional, so passes
            assert result.severity == "warning"


class TestCheckModelFormat:
    """Tests for check_model_format function."""
    
    def test_valid_provider_model(self):
        """Accepts valid provider/model format."""
        result = check_model_format("openai/gpt-4")
        assert result.passed is True
        assert "openai" in result.message.lower()
    
    def test_openrouter_format(self):
        """Accepts OpenRouter provider/org/model format."""
        result = check_model_format("openrouter/google/gemini-2.0-flash")
        assert result.passed is True
        # Accepts either message format
        assert "openrouter" in result.message.lower() or "OpenRouter" in result.message
    
    def test_simple_model_name(self):
        """Accepts simple model name."""
        result = check_model_format("gpt-4")
        assert result.passed is True
    
    def test_anthropic_provider(self):
        """Accepts Anthropic provider."""
        result = check_model_format("anthropic/claude-3-opus")
        assert result.passed is True


class TestCheckApiEndpoint:
    """Tests for check_api_endpoint function."""
    
    def test_unreachable_endpoint(self):
        """Fails for unreachable endpoint."""
        result = check_api_endpoint("http://nonexistent.invalid:12345")
        assert result.passed is False
        assert result.severity == "error"
    
    @patch('httpx.Client')
    def test_reachable_endpoint(self, mock_client_class):
        """Passes for reachable endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client
        
        result = check_api_endpoint("https://api.openai.com/v1")
        assert result.passed is True
    
    @patch('httpx.Client')
    def test_auth_required_still_passes(self, mock_client_class):
        """401 response still means endpoint is reachable."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client
        
        result = check_api_endpoint("https://api.example.com")
        assert result.passed is True


class TestCheckBudget:
    """Tests for check_budget function."""
    
    def test_sufficient_budget(self):
        """Passes with sufficient budget."""
        result = check_budget(budget=10.0, context_size=1000)
        assert result.passed is True
        assert "$10.00" in result.message
    
    def test_insufficient_budget(self):
        """Warns with insufficient budget."""
        # Very large context with tiny budget
        result = check_budget(budget=0.001, context_size=1_000_000)
        assert result.passed is False
        assert result.severity == "warning"


class TestCheckContextSize:
    """Tests for check_context_size function."""
    
    def test_empty_context_fails(self):
        """Fails for empty context."""
        result = check_context_size("")
        assert result.passed is False
        assert result.severity == "error"
        assert "Empty" in result.message
    
    def test_normal_context_passes(self):
        """Passes for normal context."""
        result = check_context_size("some code content" * 100)
        assert result.passed is True
        assert "chars" in result.message
    
    def test_large_context_warns(self):
        """Warns for very large context."""
        result = check_context_size("x" * 15_000_000)
        assert result.passed is True  # Still passes, just warns
        assert result.severity == "warning"


class TestPreflightCheck:
    """Tests for preflight_check function."""
    
    def test_minimal_check(self):
        """Runs minimal preflight check."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            result = preflight_check(
                api_key_required=True,
                check_network=False,  # Skip network check for speed
            )
            assert isinstance(result, PreflightResult)
            assert len(result.checks) >= 1
    
    def test_full_check_with_context(self):
        """Runs full preflight check with context."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            result = preflight_check(
                api_key_required=True,
                model="openai/gpt-4",
                budget=5.0,
                context="some code",
                check_network=False,
            )
            assert len(result.checks) >= 3  # API key, model, context
    
    def test_skips_network_check_when_disabled(self):
        """Skips network check when disabled."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            result = preflight_check(
                api_key_required=True,
                api_base="https://api.openai.com/v1",
                check_network=False,
            )
            # Should not have API endpoint check
            check_names = [c.name for c in result.checks]
            assert "API Endpoint" not in check_names

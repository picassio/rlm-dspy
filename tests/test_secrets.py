"""Tests for secret masking utilities."""

import os
import warnings
from unittest.mock import patch

import pytest

from rlm_dspy.core.secrets import (
    SECRET_MASK,
    COMMON_SECRETS,
    MissingSecretError,
    is_secret_key,
    mask_value,
    clean_secrets,
    inject_secrets,
    check_for_exposed_secrets,
    get_api_key,
)


class TestCommonSecrets:
    """Tests for COMMON_SECRETS constant."""

    def test_includes_api_keys(self):
        """Includes common API key names."""
        assert "api_key" in COMMON_SECRETS
        assert "openai_api_key" in COMMON_SECRETS
        assert "anthropic_api_key" in COMMON_SECRETS

    def test_includes_tokens(self):
        """Includes common token names."""
        assert "access_token" in COMMON_SECRETS
        assert "hf_token" in COMMON_SECRETS

    def test_includes_passwords(self):
        """Includes password-related names."""
        assert "password" in COMMON_SECRETS
        assert "secret_key" in COMMON_SECRETS

    def test_includes_connection_strings(self):
        """Includes database connection strings."""
        assert "database_url" in COMMON_SECRETS
        assert "redis_url" in COMMON_SECRETS


class TestIsSecretKey:
    """Tests for is_secret_key function."""

    def test_exact_matches(self):
        """Matches exact common secret names."""
        assert is_secret_key("api_key") is True
        assert is_secret_key("password") is True
        assert is_secret_key("secret_key") is True

    def test_case_insensitive(self):
        """Matches case-insensitively."""
        assert is_secret_key("API_KEY") is True
        assert is_secret_key("Api_Key") is True
        assert is_secret_key("PASSWORD") is True

    def test_partial_matches(self):
        """Matches keys containing secret-related substrings."""
        assert is_secret_key("my_api_key") is True
        assert is_secret_key("auth_token") is True
        assert is_secret_key("db_password") is True

    def test_non_secrets(self):
        """Does not match non-secret keys."""
        assert is_secret_key("name") is False
        assert is_secret_key("count") is False
        assert is_secret_key("user_id") is False


class TestMaskValue:
    """Tests for mask_value function."""

    def test_masks_by_default(self):
        """Fully masks value by default."""
        result = mask_value("sk-secret123456")
        assert result == SECRET_MASK
        assert "sk-" not in result

    def test_masks_none(self):
        """Handles None value."""
        result = mask_value(None)
        assert result == "[None]"

    def test_masks_empty(self):
        """Handles empty value."""
        result = mask_value("")
        assert result == "[empty]"

    def test_optional_prefix_reveal(self):
        """Can reveal prefix when explicitly requested."""
        result = mask_value("sk-secret123456", reveal_prefix=True)
        assert result.startswith("sk-s")
        assert "secret" not in result

    def test_short_value_fully_masked(self):
        """Short values are fully masked even with reveal_prefix."""
        result = mask_value("short", reveal_prefix=True)
        assert result == SECRET_MASK


class TestCleanSecrets:
    """Tests for clean_secrets function."""

    def test_masks_secret_keys(self):
        """Masks values for secret keys."""
        data = {"api_key": "sk-secret123", "name": "test"}
        result = clean_secrets(data)

        assert result["api_key"] == SECRET_MASK
        assert result["name"] == "test"

    def test_leaves_original_unchanged(self):
        """Does not modify original by default."""
        data = {"api_key": "sk-secret123"}
        clean_secrets(data)

        assert data["api_key"] == "sk-secret123"

    def test_in_place_modifies_original(self):
        """in_place=True modifies original."""
        data = {"api_key": "sk-secret123"}
        clean_secrets(data, in_place=True)

        assert data["api_key"] == SECRET_MASK

    def test_nested_dicts(self):
        """Cleans nested dictionaries."""
        data = {
            "config": {
                "api_key": "sk-secret123",
                "timeout": 30,
            }
        }
        result = clean_secrets(data)

        assert result["config"]["api_key"] == SECRET_MASK
        assert result["config"]["timeout"] == 30

    def test_lists_of_dicts(self):
        """Cleans lists of dictionaries."""
        data = {
            "providers": [
                {"name": "openai", "api_key": "sk-key1"},
                {"name": "anthropic", "api_key": "sk-key2"},
            ]
        }
        result = clean_secrets(data)

        assert result["providers"][0]["api_key"] == SECRET_MASK
        assert result["providers"][1]["api_key"] == SECRET_MASK
        assert result["providers"][0]["name"] == "openai"

    def test_already_masked_unchanged(self):
        """Already masked values are not changed."""
        data = {"api_key": SECRET_MASK}
        result = clean_secrets(data)

        assert result["api_key"] == SECRET_MASK


class TestInjectSecrets:
    """Tests for inject_secrets function."""

    def test_injects_from_dict(self):
        """Injects secrets from provided dict."""
        data = {"api_key": SECRET_MASK}
        secrets = {"api_key": "sk-restored"}

        result = inject_secrets(data, secrets=secrets)

        assert result["api_key"] == "sk-restored"

    def test_injects_from_env(self):
        """Injects secrets from environment."""
        data = {"api_key": SECRET_MASK}

        with patch.dict(os.environ, {"RLM_API_KEY": "sk-from-env"}):
            result = inject_secrets(data)

        assert result["api_key"] == "sk-from-env"

    def test_dict_takes_precedence(self):
        """Provided dict takes precedence over env."""
        data = {"api_key": SECRET_MASK}
        secrets = {"api_key": "sk-from-dict"}

        with patch.dict(os.environ, {"RLM_API_KEY": "sk-from-env"}):
            result = inject_secrets(data, secrets=secrets)

        assert result["api_key"] == "sk-from-dict"

    def test_nested_injection(self):
        """Injects into nested dicts."""
        data = {"config": {"api_key": SECRET_MASK}}
        secrets = {"api_key": "sk-nested"}

        result = inject_secrets(data, secrets=secrets)

        assert result["config"]["api_key"] == "sk-nested"


class TestCheckForExposedSecrets:
    """Tests for check_for_exposed_secrets function."""

    def test_warns_on_exposed_secret(self):
        """Warns when secrets are exposed."""
        data = {"api_key": "sk-this-is-a-long-secret-key"}

        with pytest.warns(UserWarning, match="Potential secrets exposed"):
            check_for_exposed_secrets(data)

    def test_no_warning_when_masked(self):
        """No warning when secrets are masked."""
        data = {"api_key": SECRET_MASK}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_for_exposed_secrets(data)
            assert len(w) == 0

    def test_no_warning_for_short_values(self):
        """No warning for short values (might not be real secrets)."""
        data = {"api_key": "short"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_for_exposed_secrets(data)
            assert len(w) == 0


class TestMissingSecretError:
    """Tests for MissingSecretError exception."""

    def test_error_message(self):
        """Error has informative message."""
        error = MissingSecretError("api_key")
        assert "api_key" in str(error)

    def test_error_with_env_var(self):
        """Error includes env var suggestion."""
        error = MissingSecretError("api_key", "OPENAI_API_KEY")
        assert "api_key" in str(error)
        assert "OPENAI_API_KEY" in str(error)

    def test_error_attributes(self):
        """Error has key and env_var attributes."""
        error = MissingSecretError("api_key", "OPENAI_API_KEY")
        assert error.key == "api_key"
        assert error.env_var == "OPENAI_API_KEY"


class TestGetApiKey:
    """Tests for get_api_key function."""

    def test_finds_key_in_env(self):
        """Finds API key in environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=True):
            # Clear other possible keys
            for key in ["RLM_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(key, None)
            os.environ["OPENAI_API_KEY"] = "sk-test123"

            result = get_api_key()
            assert result == "sk-test123"

    def test_tries_env_vars_in_order(self):
        """Tries env vars in specified order."""
        with patch.dict(os.environ, {
            "SECOND_KEY": "second",
            "FIRST_KEY": "first",
        }, clear=True):
            result = get_api_key(env_vars=["FIRST_KEY", "SECOND_KEY"])
            assert result == "first"

    def test_raises_when_required_and_missing(self):
        """Raises MissingSecretError when required and missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove all possible keys
            for key in ["RLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(key, None)

            with pytest.raises(MissingSecretError):
                get_api_key(required=True)

    def test_returns_none_when_optional_and_missing(self):
        """Returns None when optional and missing."""
        with patch.dict(os.environ, {}, clear=True):
            for key in ["RLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(key, None)

            result = get_api_key(required=False)
            assert result is None

    def test_custom_env_vars(self):
        """Uses custom env var list."""
        with patch.dict(os.environ, {"MY_CUSTOM_KEY": "custom-value"}, clear=True):
            result = get_api_key(env_vars=["MY_CUSTOM_KEY"])
            assert result == "custom-value"

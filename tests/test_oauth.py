"""Tests for OAuth authentication."""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from rlm_dspy.core.oauth import (
    OAuthCredentials,
    _generate_pkce,
    _save_credentials,
    _load_credentials,
    _delete_credentials,
    is_anthropic_authenticated,
    oauth_status,
    ANTHROPIC_CLIENT_ID,
)


class TestPKCE:
    """Test PKCE generation."""
    
    def test_generate_pkce_returns_tuple(self):
        """PKCE should return verifier and challenge."""
        verifier, challenge = _generate_pkce()
        
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)
        assert len(verifier) > 20
        assert len(challenge) > 20
    
    def test_generate_pkce_unique(self):
        """Each PKCE generation should be unique."""
        v1, c1 = _generate_pkce()
        v2, c2 = _generate_pkce()
        
        assert v1 != v2
        assert c1 != c2
    
    def test_pkce_challenge_is_base64url(self):
        """Challenge should be URL-safe base64."""
        _, challenge = _generate_pkce()
        
        # Should not contain padding or non-URL-safe chars
        assert "=" not in challenge
        assert "+" not in challenge
        assert "/" not in challenge


class TestOAuthCredentials:
    """Test OAuth credentials dataclass."""
    
    def test_credentials_creation(self):
        """Test creating credentials."""
        creds = OAuthCredentials(
            provider="anthropic",
            access_token="sk-ant-oat-test",
            refresh_token="refresh-test",
            expires_at=time.time() + 3600,
        )
        
        assert creds.provider == "anthropic"
        assert creds.access_token == "sk-ant-oat-test"
        assert creds.is_oauth_token is True
        assert creds.is_expired is False
    
    def test_expired_credentials(self):
        """Test expired credentials detection."""
        creds = OAuthCredentials(
            provider="anthropic",
            access_token="sk-ant-oat-test",
            refresh_token="refresh-test",
            expires_at=time.time() - 100,  # Already expired
        )
        
        assert creds.is_expired is True
    
    def test_expiring_soon(self):
        """Credentials should be considered expired with 5 min buffer."""
        creds = OAuthCredentials(
            provider="anthropic",
            access_token="sk-ant-oat-test",
            refresh_token="refresh-test",
            expires_at=time.time() + 200,  # Less than 5 min buffer
        )
        
        assert creds.is_expired is True
    
    def test_api_key_not_oauth(self):
        """Regular API key should not be detected as OAuth."""
        creds = OAuthCredentials(
            provider="anthropic",
            access_token="sk-ant-api-test",  # Not OAuth format
            refresh_token="refresh-test",
            expires_at=time.time() + 3600,
        )
        
        assert creds.is_oauth_token is False
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        original = OAuthCredentials(
            provider="anthropic",
            access_token="sk-ant-oat-test",
            refresh_token="refresh-test",
            expires_at=time.time() + 3600,
        )
        
        data = original.to_dict()
        restored = OAuthCredentials.from_dict(data)
        
        assert restored.provider == original.provider
        assert restored.access_token == original.access_token
        assert restored.refresh_token == original.refresh_token
        assert restored.expires_at == original.expires_at


class TestCredentialsStorage:
    """Test credentials file storage."""
    
    @pytest.fixture
    def temp_oauth_dir(self, tmp_path):
        """Use a temporary directory for credentials."""
        oauth_dir = tmp_path / ".rlm" / "oauth"
        creds_file = oauth_dir / "credentials.json"
        
        with patch("rlm_dspy.core.oauth.OAUTH_DIR", oauth_dir), \
             patch("rlm_dspy.core.oauth.CREDENTIALS_FILE", creds_file):
            yield oauth_dir, creds_file
    
    def test_save_and_load_credentials(self, temp_oauth_dir):
        """Test saving and loading credentials."""
        oauth_dir, creds_file = temp_oauth_dir
        
        creds = OAuthCredentials(
            provider="anthropic",
            access_token="sk-ant-oat-test",
            refresh_token="refresh-test",
            expires_at=time.time() + 3600,
        )
        
        _save_credentials(creds)
        
        assert creds_file.exists()
        
        loaded = _load_credentials("anthropic")
        
        assert loaded is not None
        assert loaded.access_token == creds.access_token
        assert loaded.refresh_token == creds.refresh_token
    
    def test_load_nonexistent(self, temp_oauth_dir):
        """Loading non-existent credentials should return None."""
        loaded = _load_credentials("anthropic")
        assert loaded is None
    
    def test_delete_credentials(self, temp_oauth_dir):
        """Test deleting credentials."""
        oauth_dir, creds_file = temp_oauth_dir
        
        creds = OAuthCredentials(
            provider="anthropic",
            access_token="sk-ant-oat-test",
            refresh_token="refresh-test",
            expires_at=time.time() + 3600,
        )
        
        _save_credentials(creds)
        assert _load_credentials("anthropic") is not None
        
        result = _delete_credentials("anthropic")
        assert result is True
        
        assert _load_credentials("anthropic") is None
    
    def test_delete_nonexistent(self, temp_oauth_dir):
        """Deleting non-existent credentials should return False."""
        result = _delete_credentials("anthropic")
        assert result is False
    
    def test_multiple_providers(self, temp_oauth_dir):
        """Test storing credentials for multiple providers."""
        oauth_dir, creds_file = temp_oauth_dir
        
        creds1 = OAuthCredentials(
            provider="anthropic",
            access_token="sk-ant-oat-test1",
            refresh_token="refresh-test1",
            expires_at=time.time() + 3600,
        )
        
        creds2 = OAuthCredentials(
            provider="google",
            access_token="google-token",
            refresh_token="google-refresh",
            expires_at=time.time() + 3600,
        )
        
        _save_credentials(creds1)
        _save_credentials(creds2)
        
        loaded1 = _load_credentials("anthropic")
        loaded2 = _load_credentials("google")
        
        assert loaded1.access_token == "sk-ant-oat-test1"
        assert loaded2.access_token == "google-token"


class TestOAuthStatus:
    """Test OAuth status functions."""
    
    @pytest.fixture
    def temp_oauth_dir(self, tmp_path):
        """Use a temporary directory for credentials."""
        oauth_dir = tmp_path / ".rlm" / "oauth"
        creds_file = oauth_dir / "credentials.json"
        
        with patch("rlm_dspy.core.oauth.OAUTH_DIR", oauth_dir), \
             patch("rlm_dspy.core.oauth.CREDENTIALS_FILE", creds_file):
            yield oauth_dir, creds_file
    
    def test_not_authenticated(self, temp_oauth_dir):
        """Test status when not authenticated."""
        assert is_anthropic_authenticated() is False
        
        status = oauth_status("anthropic")
        assert status["authenticated"] is False
        assert status["provider"] == "anthropic"
    
    def test_authenticated(self, temp_oauth_dir):
        """Test status when authenticated."""
        creds = OAuthCredentials(
            provider="anthropic",
            access_token="sk-ant-oat-test",
            refresh_token="refresh-test",
            expires_at=time.time() + 3600,
        )
        _save_credentials(creds)
        
        assert is_anthropic_authenticated() is True
        
        status = oauth_status("anthropic")
        assert status["authenticated"] is True
        assert status["is_expired"] is False
        assert "expires_at" in status


class TestAnthropicConfig:
    """Test Anthropic OAuth configuration."""
    
    def test_client_id_format(self):
        """Client ID should be a valid UUID format."""
        # UUID format: 8-4-4-4-12 hex chars
        import re
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(uuid_pattern, ANTHROPIC_CLIENT_ID)

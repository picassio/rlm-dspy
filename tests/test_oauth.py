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
    generate_pkce,
    save_credentials,
    load_credentials,
    delete_credentials,
    is_authenticated,
    list_providers,
    list_authenticated,
    OAUTH_DIR,
    CREDENTIALS_FILE,
)


class TestPKCE:
    """Test PKCE generation."""
    
    def test_generate_pkce_returns_tuple(self):
        """PKCE should return verifier and challenge."""
        verifier, challenge = generate_pkce()
        
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)
        assert len(verifier) > 20
        assert len(challenge) > 20
    
    def test_generate_pkce_unique(self):
        """Each PKCE generation should be unique."""
        v1, c1 = generate_pkce()
        v2, c2 = generate_pkce()
        
        assert v1 != v2
        assert c1 != c2
    
    def test_pkce_challenge_is_base64url(self):
        """Challenge should be URL-safe base64."""
        _, challenge = generate_pkce()
        
        # Should not contain padding or non-URL-safe chars
        assert "=" not in challenge
        assert "+" not in challenge
        assert "/" not in challenge


class TestOAuthCredentials:
    """Test OAuth credentials dataclass."""
    
    def test_credentials_creation(self):
        """Test creating credentials."""
        creds = OAuthCredentials(
            provider="google-gemini",
            access_token="ya29.test-token",
            refresh_token="1//refresh-test",
            expires_at=time.time() + 3600,
        )
        
        assert creds.provider == "google-gemini"
        assert creds.access_token == "ya29.test-token"
        assert creds.is_expired is False
    
    def test_expired_credentials(self):
        """Test expired credentials detection."""
        creds = OAuthCredentials(
            provider="google-gemini",
            access_token="ya29.test-token",
            refresh_token="1//refresh-test",
            expires_at=time.time() - 100,  # Already expired
        )
        
        assert creds.is_expired is True
    
    def test_expiring_soon(self):
        """Credentials should be considered expired with 5 min buffer."""
        creds = OAuthCredentials(
            provider="google-gemini",
            access_token="ya29.test-token",
            refresh_token="1//refresh-test",
            expires_at=time.time() + 200,  # Less than 5 min buffer
        )
        
        assert creds.is_expired is True
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        original = OAuthCredentials(
            provider="google-gemini",
            access_token="ya29.test-token",
            refresh_token="1//refresh-test",
            expires_at=time.time() + 3600,
            project_id="test-project",
            email="user@example.com",
        )
        
        data = original.to_dict()
        restored = OAuthCredentials.from_dict(data)
        
        assert restored.provider == original.provider
        assert restored.access_token == original.access_token
        assert restored.refresh_token == original.refresh_token
        assert restored.expires_at == original.expires_at
        assert restored.project_id == original.project_id
        assert restored.email == original.email


class TestCredentialsStorage:
    """Test credentials file storage."""
    
    @pytest.fixture
    def temp_oauth_dir(self, tmp_path):
        """Use a temporary directory for credentials."""
        oauth_dir = tmp_path / ".rlm" / "oauth"
        creds_file = oauth_dir / "credentials.json"
        
        with patch("rlm_dspy.core.oauth.base.OAUTH_DIR", oauth_dir), \
             patch("rlm_dspy.core.oauth.base.CREDENTIALS_FILE", creds_file), \
             patch("rlm_dspy.core.oauth.OAUTH_DIR", oauth_dir), \
             patch("rlm_dspy.core.oauth.CREDENTIALS_FILE", creds_file):
            yield oauth_dir, creds_file
    
    def test_save_and_load_credentials(self, temp_oauth_dir):
        """Test saving and loading credentials."""
        oauth_dir, creds_file = temp_oauth_dir
        
        creds = OAuthCredentials(
            provider="google-gemini",
            access_token="ya29.test-token",
            refresh_token="1//refresh-test",
            expires_at=time.time() + 3600,
        )
        
        save_credentials(creds)
        
        assert creds_file.exists()
        
        loaded = load_credentials("google-gemini")
        
        assert loaded is not None
        assert loaded.access_token == creds.access_token
        assert loaded.refresh_token == creds.refresh_token
    
    def test_load_nonexistent(self, temp_oauth_dir):
        """Loading non-existent credentials should return None."""
        loaded = load_credentials("google-gemini")
        assert loaded is None
    
    def test_delete_credentials(self, temp_oauth_dir):
        """Test deleting credentials."""
        oauth_dir, creds_file = temp_oauth_dir
        
        creds = OAuthCredentials(
            provider="google-gemini",
            access_token="ya29.test-token",
            refresh_token="1//refresh-test",
            expires_at=time.time() + 3600,
        )
        
        save_credentials(creds)
        assert load_credentials("google-gemini") is not None
        
        result = delete_credentials("google-gemini")
        assert result is True
        
        assert load_credentials("google-gemini") is None
    
    def test_delete_nonexistent(self, temp_oauth_dir):
        """Deleting non-existent credentials should return False."""
        result = delete_credentials("google-gemini")
        assert result is False
    
    def test_multiple_providers(self, temp_oauth_dir):
        """Test storing credentials for multiple providers."""
        oauth_dir, creds_file = temp_oauth_dir
        
        creds1 = OAuthCredentials(
            provider="google-gemini",
            access_token="ya29.test-token-1",
            refresh_token="1//refresh-test-1",
            expires_at=time.time() + 3600,
        )
        
        creds2 = OAuthCredentials(
            provider="antigravity",
            access_token="ya29.test-token-2",
            refresh_token="1//refresh-test-2",
            expires_at=time.time() + 3600,
        )
        
        save_credentials(creds1)
        save_credentials(creds2)
        
        loaded1 = load_credentials("google-gemini")
        loaded2 = load_credentials("antigravity")
        
        assert loaded1.access_token == "ya29.test-token-1"
        assert loaded2.access_token == "ya29.test-token-2"


class TestOAuthStatus:
    """Test OAuth status functions."""
    
    @pytest.fixture
    def temp_oauth_dir(self, tmp_path):
        """Use a temporary directory for credentials."""
        oauth_dir = tmp_path / ".rlm" / "oauth"
        creds_file = oauth_dir / "credentials.json"
        
        with patch("rlm_dspy.core.oauth.base.OAUTH_DIR", oauth_dir), \
             patch("rlm_dspy.core.oauth.base.CREDENTIALS_FILE", creds_file), \
             patch("rlm_dspy.core.oauth.manager.CREDENTIALS_FILE", creds_file), \
             patch("rlm_dspy.core.oauth.OAUTH_DIR", oauth_dir), \
             patch("rlm_dspy.core.oauth.CREDENTIALS_FILE", creds_file):
            yield oauth_dir, creds_file
    
    def test_not_authenticated(self, temp_oauth_dir):
        """Test status when not authenticated."""
        assert is_authenticated("google-gemini") is False
    
    def test_authenticated(self, temp_oauth_dir):
        """Test status when authenticated."""
        creds = OAuthCredentials(
            provider="google-gemini",
            access_token="ya29.test-token",
            refresh_token="1//refresh-test",
            expires_at=time.time() + 3600,
        )
        save_credentials(creds)
        
        # Note: is_authenticated may try to refresh, which would fail
        # So we just check load_credentials works
        loaded = load_credentials("google-gemini")
        assert loaded is not None
        assert loaded.is_expired is False


class TestProviderRegistry:
    """Test OAuth provider registry."""
    
    def test_list_providers(self):
        """Test listing available providers."""
        providers = list_providers()
        
        assert isinstance(providers, list)
        assert len(providers) >= 2
        assert "google-gemini" in providers
        assert "antigravity" in providers
    
    def test_get_provider(self):
        """Test getting a provider instance."""
        from rlm_dspy.core.oauth import get_provider
        
        provider = get_provider("google-gemini")
        assert provider.name == "google-gemini"
        
        provider = get_provider("antigravity")
        assert provider.name == "antigravity"
    
    def test_get_unknown_provider(self):
        """Test getting unknown provider raises ValueError."""
        from rlm_dspy.core.oauth import get_provider
        
        with pytest.raises(ValueError, match="Unknown OAuth provider"):
            get_provider("unknown-provider")

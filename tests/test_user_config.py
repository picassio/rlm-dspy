"""Tests for user configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from rlm_dspy.core.user_config import (
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_CONFIG,
    ensure_config_dir,
    load_config,
    save_config,
    load_env_file,
    get_config_value,
    set_config_value,
    is_configured,
    get_config_status,
)


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG constant."""
    
    def test_has_model(self):
        """Has model setting."""
        assert "model" in DEFAULT_CONFIG
        assert isinstance(DEFAULT_CONFIG["model"], str)
    
    def test_has_execution_limits(self):
        """Has execution limit settings."""
        assert "max_iterations" in DEFAULT_CONFIG
        assert "max_llm_calls" in DEFAULT_CONFIG
        assert "max_output_chars" in DEFAULT_CONFIG
    
    def test_has_budget_limits(self):
        """Has budget/safety limits."""
        assert "max_budget" in DEFAULT_CONFIG
        assert "max_timeout" in DEFAULT_CONFIG
    
    def test_has_embedding_settings(self):
        """Has embedding settings."""
        assert "embedding_model" in DEFAULT_CONFIG
        assert "local_embedding_model" in DEFAULT_CONFIG
        assert "embedding_batch_size" in DEFAULT_CONFIG
    
    def test_has_index_settings(self):
        """Has vector index settings."""
        assert "index_dir" in DEFAULT_CONFIG
        assert "use_faiss" in DEFAULT_CONFIG
        assert "auto_update_index" in DEFAULT_CONFIG


class TestEnsureConfigDir:
    """Tests for ensure_config_dir function."""
    
    def test_creates_dir(self):
        """Creates config directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / ".rlm"
            
            with patch("rlm_dspy.core.user_config.CONFIG_DIR", test_dir):
                result = ensure_config_dir()
                assert result == test_dir
                assert test_dir.exists()
    
    def test_returns_existing_dir(self):
        """Returns existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / ".rlm"
            test_dir.mkdir()
            
            with patch("rlm_dspy.core.user_config.CONFIG_DIR", test_dir):
                result = ensure_config_dir()
                assert result == test_dir


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_returns_defaults_when_no_file(self):
        """Returns default config when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "nonexistent.yaml"
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                config = load_config()
                assert config == DEFAULT_CONFIG
    
    def test_loads_from_file(self):
        """Loads config from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_file.write_text("model: custom/model\nmax_iterations: 50\n")
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                config = load_config()
                assert config["model"] == "custom/model"
                assert config["max_iterations"] == 50
    
    def test_merges_with_defaults(self):
        """Merges loaded config with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_file.write_text("model: custom/model\n")  # Only model, not other settings
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                config = load_config()
                # Custom value loaded
                assert config["model"] == "custom/model"
                # Defaults filled in
                assert config["max_iterations"] == DEFAULT_CONFIG["max_iterations"]
    
    def test_handles_invalid_yaml(self):
        """Handles invalid YAML gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_file.write_text("{ invalid yaml [")
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                config = load_config()
                # Falls back to defaults
                assert config == DEFAULT_CONFIG


class TestSaveConfig:
    """Tests for save_config function."""
    
    def test_saves_config(self):
        """Saves config to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_dir = Path(tmpdir)
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                with patch("rlm_dspy.core.user_config.CONFIG_DIR", test_dir):
                    save_config({"model": "test/model"})
                    
                    assert test_file.exists()
                    content = test_file.read_text()
                    assert "test/model" in content
    
    def test_uses_template_by_default(self):
        """Uses template with comments by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_dir = Path(tmpdir)
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                with patch("rlm_dspy.core.user_config.CONFIG_DIR", test_dir):
                    save_config({"model": "test/model"})
                    
                    content = test_file.read_text()
                    assert "# RLM-DSPy Configuration" in content
                    assert "# Model Settings" in content
    
    def test_simple_mode(self):
        """Can save without template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_dir = Path(tmpdir)
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                with patch("rlm_dspy.core.user_config.CONFIG_DIR", test_dir):
                    save_config({"model": "test/model"}, use_template=False)
                    
                    content = test_file.read_text()
                    # Should be simpler, not have full template comments
                    assert "test/model" in content


class TestLoadEnvFile:
    """Tests for load_env_file function."""
    
    def test_loads_env_vars(self):
        """Loads environment variables from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("TEST_VAR=test_value\nANOTHER_VAR=another\n")
            
            # Clear any existing values
            os.environ.pop("TEST_VAR", None)
            os.environ.pop("ANOTHER_VAR", None)
            
            loaded = load_env_file(env_file)
            
            assert loaded["TEST_VAR"] == "test_value"
            assert loaded["ANOTHER_VAR"] == "another"
            assert os.environ.get("TEST_VAR") == "test_value"
            
            # Cleanup
            os.environ.pop("TEST_VAR", None)
            os.environ.pop("ANOTHER_VAR", None)
    
    def test_skips_comments(self):
        """Skips comment lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("# This is a comment\nTEST_VAR=value\n")
            
            os.environ.pop("TEST_VAR", None)
            
            loaded = load_env_file(env_file)
            
            assert "TEST_VAR" in loaded
            assert "#" not in str(loaded.keys())
            
            os.environ.pop("TEST_VAR", None)
    
    def test_returns_empty_for_missing_file(self):
        """Returns empty dict for missing file."""
        loaded = load_env_file("/nonexistent/path/.env")
        assert loaded == {}
    
    def test_handles_quoted_values(self):
        """Handles quoted values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text('TEST_VAR="quoted value"\nANOTHER=\'single quoted\'\n')
            
            os.environ.pop("TEST_VAR", None)
            os.environ.pop("ANOTHER", None)
            
            loaded = load_env_file(env_file)
            
            assert loaded["TEST_VAR"] == "quoted value"
            assert loaded["ANOTHER"] == "single quoted"
            
            os.environ.pop("TEST_VAR", None)
            os.environ.pop("ANOTHER", None)


class TestGetConfigValue:
    """Tests for get_config_value function."""
    
    def test_gets_value(self):
        """Gets a config value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_file.write_text("model: test/model\n")
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                value = get_config_value("model")
                assert value == "test/model"
    
    def test_returns_default(self):
        """Returns default for missing key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_file.write_text("")
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                value = get_config_value("nonexistent", default="fallback")
                assert value == "fallback"


class TestSetConfigValue:
    """Tests for set_config_value function."""
    
    def test_sets_value(self):
        """Sets a config value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_dir = Path(tmpdir)
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                with patch("rlm_dspy.core.user_config.CONFIG_DIR", test_dir):
                    set_config_value("model", "new/model")
                    
                    # Verify it was saved
                    assert test_file.exists()
                    content = test_file.read_text()
                    assert "new/model" in content


class TestIsConfigured:
    """Tests for is_configured function."""
    
    def test_configured_with_api_key(self):
        """Returns True when API key is available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_file.write_text("model: openai/gpt-4\n")
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
                    result = is_configured()
                    assert result is True
    
    def test_not_configured_without_api_key(self):
        """Returns False when no API key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_file.write_text("model: openai/gpt-4\n")
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                with patch.dict(os.environ, {}, clear=True):
                    # Remove all API keys
                    for key in ["RLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"]:
                        os.environ.pop(key, None)
                    
                    result = is_configured()
                    assert result is False


class TestGetConfigStatus:
    """Tests for get_config_status function."""
    
    def test_returns_status_dict(self):
        """Returns status dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_file.write_text("model: test/model\n")
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
                    status = get_config_status()
                    
                    assert "config" in status
                    assert "model" in status
                    assert "is_configured" in status
                    assert "api_key_found" in status
    
    def test_reports_config_file(self):
        """Reports config file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.yaml"
            test_file.write_text("model: test/model\n")
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                status = get_config_status()
                assert status["config_file"] == str(test_file)
    
    def test_reports_missing_config_file(self):
        """Reports None for missing config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "nonexistent.yaml"
            
            with patch("rlm_dspy.core.user_config.CONFIG_FILE", test_file):
                status = get_config_status()
                assert status["config_file"] is None

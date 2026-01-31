"""Tests for unified configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


class TestConfigDefaults:
    """Test default configuration values."""
    
    def test_default_model(self):
        """Test default model value."""
        from rlm_dspy.core.config import Config, DEFAULTS
        
        with patch.dict(os.environ, {}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            assert config.model == DEFAULTS["model"]
    
    def test_default_iterations(self):
        """Test default max_iterations."""
        from rlm_dspy.core.config import Config, DEFAULTS
        
        with patch.dict(os.environ, {}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            assert config.max_iterations == DEFAULTS["max_iterations"]
    
    def test_sub_model_defaults_to_model(self):
        """Test that sub_model defaults to model when not set."""
        from rlm_dspy.core.config import Config
        
        with patch.dict(os.environ, {"RLM_MODEL": "test/model"}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            assert config.sub_model == "test/model"


class TestConfigEnvPrecedence:
    """Test environment variable precedence."""
    
    def test_env_overrides_default(self):
        """Test that env vars override defaults."""
        from rlm_dspy.core.config import Config
        
        with patch.dict(os.environ, {"RLM_MODEL": "env/model"}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            assert config.model == "env/model"
    
    def test_env_overrides_file(self):
        """Test that env vars override file config."""
        from rlm_dspy.core.config import Config
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"model": "file/model"}, f)
            f.flush()
            
            with patch.dict(os.environ, {"RLM_MODEL": "env/model"}, clear=True):
                config = Config(config_file=Path(f.name))
                assert config.model == "env/model"
            
            os.unlink(f.name)
    
    def test_env_int_casting(self):
        """Test environment variable integer casting."""
        from rlm_dspy.core.config import Config
        
        with patch.dict(os.environ, {"RLM_MAX_ITERATIONS": "30"}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            assert config.max_iterations == 30
            assert isinstance(config.max_iterations, int)
    
    def test_env_float_casting(self):
        """Test environment variable float casting."""
        from rlm_dspy.core.config import Config
        
        with patch.dict(os.environ, {"RLM_MAX_BUDGET": "2.5"}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            assert config.max_budget == 2.5
            assert isinstance(config.max_budget, float)
    
    def test_env_bool_casting(self):
        """Test environment variable boolean casting."""
        from rlm_dspy.core.config import Config
        
        with patch.dict(os.environ, {"RLM_VERBOSE": "true"}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            assert config.verbose is True
        
        with patch.dict(os.environ, {"RLM_VERBOSE": "false"}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            assert config.verbose is False


class TestConfigFile:
    """Test config file loading."""
    
    def test_file_config_loaded(self):
        """Test that file config is loaded."""
        from rlm_dspy.core.config import Config
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"model": "file/model", "max_iterations": 40}, f)
            f.flush()
            
            with patch.dict(os.environ, {}, clear=True):
                config = Config(config_file=Path(f.name))
                assert config.model == "file/model"
                assert config.max_iterations == 40
            
            os.unlink(f.name)
    
    def test_missing_file_uses_defaults(self):
        """Test that missing file falls back to defaults."""
        from rlm_dspy.core.config import Config, DEFAULTS
        
        with patch.dict(os.environ, {}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            assert config.model == DEFAULTS["model"]


class TestNestedConfigs:
    """Test nested configuration objects."""
    
    def test_index_config(self):
        """Test index configuration."""
        from rlm_dspy.core.config import Config
        
        with patch.dict(os.environ, {}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            index = config.index
            
            assert index.use_faiss is True
            assert index.faiss_threshold == 5000
            assert index.auto_update is True
    
    def test_daemon_config(self):
        """Test daemon configuration."""
        from rlm_dspy.core.config import Config
        
        with patch.dict(os.environ, {}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            daemon = config.daemon
            
            assert daemon.debounce_seconds == 5.0
            assert daemon.max_concurrent_indexes == 2
    
    def test_optimization_config(self):
        """Test optimization configuration."""
        from rlm_dspy.core.config import Config
        
        with patch.dict(os.environ, {}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            opt = config.optimization
            
            assert opt.enabled is True
            assert opt.optimizer == "simba"
            assert opt.min_new_traces == 50
    
    def test_embedding_config(self):
        """Test embedding configuration."""
        from rlm_dspy.core.config import Config
        
        with patch.dict(os.environ, {}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            embed = config.embedding
            
            assert "embedding" in embed.model.lower()
            assert embed.batch_size == 100


class TestConfigSingleton:
    """Test global config singleton."""
    
    def test_get_config_returns_same_instance(self):
        """Test that get_config returns singleton."""
        from rlm_dspy.core.config import get_config, reload_config
        
        # Reset singleton
        reload_config()
        
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_reload_config_creates_new_instance(self):
        """Test that reload_config creates new instance."""
        from rlm_dspy.core.config import get_config, reload_config
        
        config1 = get_config()
        config2 = reload_config()
        
        assert config1 is not config2


class TestConfigToRLMConfig:
    """Test conversion to RLMConfig."""
    
    def test_to_rlm_config(self):
        """Test converting to RLMConfig."""
        from rlm_dspy.core.config import Config
        
        with patch.dict(os.environ, {"RLM_MODEL": "test/model", "RLM_MAX_ITERATIONS": "30"}, clear=True):
            config = Config(config_file=Path("/nonexistent/config.yaml"))
            rlm_config = config.to_rlm_config()
            
            assert rlm_config.model == "test/model"
            assert rlm_config.max_iterations == 30

"""Tests for RLM builder pattern."""

import pytest
from unittest.mock import patch, MagicMock


class TestRLMBuilder:
    """Test RLMBuilder fluent interface."""
    
    def test_import_builder(self):
        """Test that builder can be imported."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder()
        assert builder is not None
    
    def test_import_from_package(self):
        """Test that builder can be imported from main package."""
        from rlm_dspy import RLMBuilder
        builder = RLMBuilder()
        assert builder is not None
    
    def test_builder_from_rlm_class(self):
        """Test RLM.builder() class method."""
        from rlm_dspy import RLM
        builder = RLM.builder()
        assert builder is not None
        assert builder.__class__.__name__ == "RLMBuilder"
    
    def test_model_setting(self):
        """Test setting model."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().model("kimi/k2p5")
        assert builder._model == "kimi/k2p5"
    
    def test_sub_model_setting(self):
        """Test setting sub_model."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().sub_model("kimi/k2p5")
        assert builder._sub_model == "kimi/k2p5"
    
    def test_iterations_setting(self):
        """Test setting max_iterations."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().iterations(30)
        assert builder._max_iterations == 30
    
    def test_llm_calls_setting(self):
        """Test setting max_llm_calls."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().llm_calls(100)
        assert builder._max_llm_calls == 100
    
    def test_timeout_setting(self):
        """Test setting timeout."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().timeout(600)
        assert builder._max_timeout == 600
    
    def test_budget_setting(self):
        """Test setting budget."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().budget(2.0)
        assert builder._max_budget == 2.0
    
    def test_workers_setting(self):
        """Test setting workers."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().workers(16)
        assert builder._max_workers == 16
    
    def test_verbose_flag(self):
        """Test verbose flag."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().verbose()
        assert builder._verbose is True
        
        builder2 = RLMBuilder().verbose(False)
        assert builder2._verbose is False
    
    def test_validate_flag(self):
        """Test validate flag."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().validate(False)
        assert builder._validate is False
    
    def test_no_tools(self):
        """Test disabling tools."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().no_tools()
        assert builder._use_tools is False
    
    def test_use_tools_all(self):
        """Test enabling all tools."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().use_tools("all")
        assert builder._use_tools == "all"
    
    def test_custom_tools(self):
        """Test setting custom tools."""
        from rlm_dspy.core.builder import RLMBuilder
        
        def my_tool(x: str) -> str:
            return x
        
        builder = RLMBuilder().tools({"my_tool": my_tool})
        assert "my_tool" in builder._tools
    
    def test_signature_string(self):
        """Test setting signature as string."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder().signature("query -> result")
        assert builder._signature == "query -> result"
    
    def test_chaining(self):
        """Test method chaining."""
        from rlm_dspy.core.builder import RLMBuilder
        
        builder = (RLMBuilder()
            .model("kimi/k2p5")
            .sub_model("kimi/k2p5")
            .iterations(30)
            .llm_calls(100)
            .timeout(600)
            .budget(2.0)
            .verbose())
        
        assert builder._model == "kimi/k2p5"
        assert builder._sub_model == "kimi/k2p5"
        assert builder._max_iterations == 30
        assert builder._max_llm_calls == 100
        assert builder._max_timeout == 600
        assert builder._max_budget == 2.0
        assert builder._verbose is True
    
    def test_fast_preset(self):
        """Test fast preset configuration."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder.fast()
        
        assert builder._max_iterations == 20
        assert builder._max_llm_calls == 30
        assert builder._max_timeout == 120
    
    def test_thorough_preset(self):
        """Test thorough preset configuration."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder.thorough()
        
        assert builder._max_iterations == 50
        assert builder._max_llm_calls == 150
        assert builder._max_timeout == 600
        assert builder._max_budget == 5.0
    
    def test_debug_preset(self):
        """Test debug preset configuration."""
        from rlm_dspy.core.builder import RLMBuilder
        builder = RLMBuilder.debug()
        
        assert builder._verbose is True
        assert builder._enable_logging is True
        assert builder._enable_metrics is True


class TestRLMBuilderBuild:
    """Test building RLM instances from builder."""
    
    @pytest.fixture
    def mock_env(self):
        """Mock environment with API key."""
        with patch.dict("os.environ", {"KIMI_API_KEY": "test-key"}):
            yield
    
    def test_build_with_model(self, mock_env):
        """Test building RLM with model set."""
        from rlm_dspy.core.builder import RLMBuilder
        
        rlm = (RLMBuilder()
            .model("kimi/k2p5")
            .build())
        
        assert rlm.config.model == "kimi/k2p5"
    
    def test_build_with_all_settings(self, mock_env):
        """Test building RLM with all settings."""
        from rlm_dspy.core.builder import RLMBuilder
        
        rlm = (RLMBuilder()
            .model("kimi/k2p5")
            .sub_model("kimi/k2p5")
            .iterations(30)
            .llm_calls(100)
            .timeout(600)
            .budget(2.0)
            .workers(16)
            .verbose()
            .build())
        
        assert rlm.config.model == "kimi/k2p5"
        assert rlm.config.sub_model == "kimi/k2p5"
        assert rlm.config.max_iterations == 30
        assert rlm.config.max_llm_calls == 100
        assert rlm.config.max_timeout == 600
        assert rlm.config.max_budget == 2.0
        assert rlm.config.max_workers == 16
        assert rlm.config.verbose is True
    
    def test_build_with_preset_then_override(self, mock_env):
        """Test using preset then overriding values."""
        from rlm_dspy.core.builder import RLMBuilder
        
        rlm = (RLMBuilder.fast()
            .model("kimi/k2p5")
            .iterations(25)  # Override fast preset
            .build())
        
        assert rlm.config.model == "kimi/k2p5"
        assert rlm.config.max_iterations == 25  # Overridden
        assert rlm.config.max_llm_calls == 30   # From preset
    
    def test_build_uses_defaults_when_not_set(self, mock_env):
        """Test that defaults are used when values not set."""
        from rlm_dspy.core.builder import RLMBuilder
        
        rlm = (RLMBuilder()
            .model("kimi/k2p5")
            .build())
        
        # These should have defaults from RLMConfig
        assert rlm.config.max_iterations >= 20
        assert rlm.config.max_llm_calls >= 1

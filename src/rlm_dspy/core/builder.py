"""RLM Builder pattern for fluent configuration.

Provides a fluent interface for creating RLM instances:

    rlm = (RLM.builder()
        .model("kimi/k2p5")
        .iterations(30)
        .verbose()
        .build())

Or with more options:

    rlm = (RLM.builder()
        .model("kimi/k2p5")
        .sub_model("kimi/k2p5")
        .iterations(30)
        .llm_calls(100)
        .timeout(600)
        .budget(2.0)
        .tools(my_custom_tools)
        .signature(MyCustomSignature)
        .verbose()
        .build())
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .rlm import RLM
    from .rlm_types import ProgressCallback


class RLMBuilder:
    """Fluent builder for RLM instances.
    
    Example:
        rlm = (RLMBuilder()
            .model("kimi/k2p5")
            .iterations(30)
            .verbose()
            .build())
    """
    
    def __init__(self):
        """Initialize builder with defaults."""
        self._model: str | None = None
        self._sub_model: str | None = None
        self._api_key: str | None = None
        self._api_base: str | None = None
        
        self._max_iterations: int | None = None
        self._max_llm_calls: int | None = None
        self._max_output_chars: int | None = None
        self._max_timeout: float | None = None
        self._max_budget: float | None = None
        self._max_workers: int | None = None
        
        self._verbose: bool = False
        self._validate: bool = True
        self._enable_logging: bool = False
        self._enable_metrics: bool = False
        
        self._tools: dict[str, Callable[..., str]] | None = None
        self._use_tools: bool | str = True
        self._signature: str | type = "context, query -> answer"
        self._interpreter: Any = None
        self._progress_callback: ProgressCallback | None = None
    
    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    
    def model(self, model: str) -> "RLMBuilder":
        """Set the primary model.
        
        Args:
            model: Model ID (e.g., "kimi/k2p5", "openai/gpt-4o")
        """
        self._model = model
        return self
    
    def sub_model(self, model: str) -> "RLMBuilder":
        """Set the sub-model for llm_query() calls.
        
        Args:
            model: Model ID for sub-queries
        """
        self._sub_model = model
        return self
    
    def api_key(self, key: str) -> "RLMBuilder":
        """Set the API key.
        
        Args:
            key: API key for the provider
        """
        self._api_key = key
        return self
    
    def api_base(self, base: str) -> "RLMBuilder":
        """Set custom API base URL.
        
        Args:
            base: Base URL for API calls
        """
        self._api_base = base
        return self
    
    # ==========================================================================
    # Execution Limits
    # ==========================================================================
    
    def iterations(self, max_iterations: int) -> "RLMBuilder":
        """Set maximum REPL iterations.
        
        Args:
            max_iterations: Max iterations (20-100)
            
        Raises:
            ValueError: If max_iterations is out of range
        """
        if not isinstance(max_iterations, int) or not 20 <= max_iterations <= 100:
            raise ValueError("max_iterations must be an integer between 20 and 100")
        self._max_iterations = max_iterations
        return self
    
    def llm_calls(self, max_calls: int) -> "RLMBuilder":
        """Set maximum sub-LLM calls.
        
        Args:
            max_calls: Max llm_query() calls (1-500)
            
        Raises:
            ValueError: If max_calls is out of range
        """
        if not isinstance(max_calls, int) or not 1 <= max_calls <= 500:
            raise ValueError("max_calls must be an integer between 1 and 500")
        self._max_llm_calls = max_calls
        return self
    
    def output_chars(self, max_chars: int) -> "RLMBuilder":
        """Set maximum output characters per iteration.
        
        Args:
            max_chars: Max characters in REPL output (1000-1000000)
            
        Raises:
            ValueError: If max_chars is out of range
        """
        if not isinstance(max_chars, int) or not 1000 <= max_chars <= 1_000_000:
            raise ValueError("max_chars must be an integer between 1000 and 1000000")
        self._max_output_chars = max_chars
        return self
    
    def timeout(self, seconds: float) -> "RLMBuilder":
        """Set maximum execution timeout.
        
        Args:
            seconds: Timeout in seconds (1-3600)
            
        Raises:
            ValueError: If seconds is out of range
        """
        if not isinstance(seconds, (int, float)) or not 1 <= seconds <= 3600:
            raise ValueError("timeout must be a number between 1 and 3600")
        self._max_timeout = float(seconds)
        return self
    
    def budget(self, usd: float) -> "RLMBuilder":
        """Set maximum cost budget.
        
        Args:
            usd: Maximum cost in USD (0.01-100)
            
        Raises:
            ValueError: If usd is out of range
        """
        if not isinstance(usd, (int, float)) or not 0.01 <= usd <= 100:
            raise ValueError("budget must be a number between 0.01 and 100")
        self._max_budget = float(usd)
        return self
    
    def workers(self, num_workers: int) -> "RLMBuilder":
        """Set number of parallel workers.
        
        Args:
            num_workers: Number of workers (1-32)
            
        Raises:
            ValueError: If num_workers is out of range
        """
        if not isinstance(num_workers, int) or not 1 <= num_workers <= 32:
            raise ValueError("num_workers must be an integer between 1 and 32")
        self._max_workers = num_workers
        return self
    
    # ==========================================================================
    # Behavior Flags
    # ==========================================================================
    
    def verbose(self, enabled: bool = True) -> "RLMBuilder":
        """Enable verbose output.
        
        Args:
            enabled: Whether to enable verbose mode
        """
        self._verbose = enabled
        return self
    
    def validate(self, enabled: bool = True) -> "RLMBuilder":
        """Enable output validation.
        
        Args:
            enabled: Whether to validate outputs
        """
        self._validate = enabled
        return self
    
    def logging(self, enabled: bool = True) -> "RLMBuilder":
        """Enable logging callback.
        
        Args:
            enabled: Whether to enable logging
        """
        self._enable_logging = enabled
        return self
    
    def metrics(self, enabled: bool = True) -> "RLMBuilder":
        """Enable metrics collection.
        
        Args:
            enabled: Whether to collect metrics
        """
        self._enable_metrics = enabled
        return self
    
    # ==========================================================================
    # Tools and Signature
    # ==========================================================================
    
    def tools(self, tools: dict[str, Callable[..., str]]) -> "RLMBuilder":
        """Set custom tools.
        
        Args:
            tools: Dictionary of tool name -> function
        """
        self._tools = tools
        return self
    
    def use_tools(self, mode: bool | str = True) -> "RLMBuilder":
        """Configure built-in tools.
        
        Args:
            mode: True for safe tools, "all" for all tools, False to disable
        """
        self._use_tools = mode
        return self
    
    def no_tools(self) -> "RLMBuilder":
        """Disable all built-in tools."""
        self._use_tools = False
        return self
    
    def signature(self, sig: str | type) -> "RLMBuilder":
        """Set the output signature.
        
        Args:
            sig: Signature string or DSPy Signature class
        """
        self._signature = sig
        return self
    
    def interpreter(self, interp: Any) -> "RLMBuilder":
        """Set custom code interpreter.
        
        Args:
            interp: Custom interpreter instance
        """
        self._interpreter = interp
        return self
    
    def progress(self, callback: "ProgressCallback") -> "RLMBuilder":
        """Set progress callback.
        
        Args:
            callback: Progress callback instance
        """
        self._progress_callback = callback
        return self
    
    # ==========================================================================
    # Build
    # ==========================================================================
    
    def build(self) -> "RLM":
        """Build the RLM instance.
        
        Returns:
            Configured RLM instance
            
        Raises:
            ValueError: If configuration is invalid
        """
        from .rlm import RLM
        from .rlm_types import RLMConfig
        
        # Build config with only non-None values
        config_kwargs = {}
        
        if self._model is not None:
            config_kwargs["model"] = self._model
        if self._sub_model is not None:
            config_kwargs["sub_model"] = self._sub_model
        if self._api_key is not None:
            config_kwargs["api_key"] = self._api_key
        if self._api_base is not None:
            config_kwargs["api_base"] = self._api_base
        if self._max_iterations is not None:
            config_kwargs["max_iterations"] = self._max_iterations
        if self._max_llm_calls is not None:
            config_kwargs["max_llm_calls"] = self._max_llm_calls
        if self._max_output_chars is not None:
            config_kwargs["max_output_chars"] = self._max_output_chars
        if self._max_timeout is not None:
            config_kwargs["max_timeout"] = self._max_timeout
        if self._max_budget is not None:
            config_kwargs["max_budget"] = self._max_budget
        if self._max_workers is not None:
            config_kwargs["max_workers"] = self._max_workers
        
        config_kwargs["verbose"] = self._verbose
        config_kwargs["validate"] = self._validate
        config_kwargs["enable_logging"] = self._enable_logging
        config_kwargs["enable_metrics"] = self._enable_metrics
        
        config = RLMConfig(**config_kwargs)
        
        return RLM(
            config=config,
            tools=self._tools,
            signature=self._signature,
            interpreter=self._interpreter,
            use_tools=self._use_tools,
            progress_callback=self._progress_callback,
        )
    
    # ==========================================================================
    # Preset Configurations
    # ==========================================================================
    
    @classmethod
    def fast(cls) -> "RLMBuilder":
        """Create builder with fast defaults (fewer iterations, smaller limits).
        
        Good for quick queries on small files.
        """
        return (cls()
            .iterations(20)
            .llm_calls(30)
            .timeout(120))
    
    @classmethod
    def thorough(cls) -> "RLMBuilder":
        """Create builder with thorough defaults (more iterations, higher limits).
        
        Good for comprehensive analysis of large codebases.
        """
        return (cls()
            .iterations(50)
            .llm_calls(150)
            .timeout(600)
            .budget(5.0))
    
    @classmethod
    def debug(cls) -> "RLMBuilder":
        """Create builder with debug settings enabled.
        
        Enables verbose output, logging, and metrics.
        """
        return (cls()
            .verbose()
            .logging()
            .metrics())

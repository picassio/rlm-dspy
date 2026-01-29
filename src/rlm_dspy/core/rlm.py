"""Core RLM class using DSPy's native RLM module.

This module wraps dspy.RLM to provide a unified interface for recursive
language model processing with proper configuration management.

Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import dspy

from .secrets import COMMON_SECRETS

_logger = logging.getLogger(__name__)


# =============================================================================
# Environment Helpers
# =============================================================================

def _env(key: str, default: str) -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)


# Pre-compiled regex patterns for secret detection (compiled once at import)
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'), '[REDACTED_SK]'),  # OpenAI style
    (re.compile(r'sk-ant-[a-zA-Z0-9-]{20,}'), '[REDACTED_ANTHROPIC]'),  # Anthropic
    (re.compile(r'sk-or-v1-[a-zA-Z0-9]{20,}'), '[REDACTED_OPENROUTER]'),  # OpenRouter
    (re.compile(r'gsk_[a-zA-Z0-9]{20,}'), '[REDACTED_GROQ]'),  # Groq
    (re.compile(r'AIza[a-zA-Z0-9_-]{35}'), '[REDACTED_GOOGLE]'),  # Google
]


# Cache resolved secret values (lazy initialization)
_cached_env_secrets: list[str] | None = None


def _get_env_secrets() -> list[str]:
    """Get secret values from environment (cached for performance)."""
    global _cached_env_secrets
    if _cached_env_secrets is None:
        _cached_env_secrets = [
            os.environ.get(key, "")
            for key in COMMON_SECRETS
            if os.environ.get(key) and len(os.environ.get(key, "")) > 8
        ]
    return _cached_env_secrets


def _sanitize_secrets(text: str, extra_secrets: list[str] | None = None) -> str:
    """Remove any leaked secrets from output text.
    
    Scans for common secret patterns and replaces them with masks.
    This prevents API keys from appearing in logs, trajectory, or answers.
    
    Args:
        text: Text to sanitize
        extra_secrets: Additional secret values to mask (e.g., config.api_key)
    """
    if not text:
        return text
    
    result = text
    
    # Check for additional secrets passed explicitly (e.g., from config)
    if extra_secrets:
        for secret in extra_secrets:
            if secret and len(secret) > 8 and secret in result:
                result = result.replace(secret, "[REDACTED]")
    
    # Check for actual secret values from environment (cached)
    for value in _get_env_secrets():
        if value in result:
            result = result.replace(value, "[REDACTED]")
    
    # Apply pre-compiled regex patterns
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    
    return result


def _sanitize_trajectory(trajectory: list, extra_secrets: list[str] | None = None) -> list:
    """Sanitize all strings in a trajectory list."""
    if not trajectory:
        return trajectory
    
    sanitized = []
    for item in trajectory:
        if isinstance(item, str):
            sanitized.append(_sanitize_secrets(item, extra_secrets))
        elif isinstance(item, dict):
            sanitized.append({
                k: _sanitize_secrets(v, extra_secrets) if isinstance(v, str) else v
                for k, v in item.items()
            })
        else:
            sanitized.append(item)
    return sanitized


T = TypeVar("T", int, float, bool, str)


def _env_get(key: str, default: T, cast: type[T] | None = None) -> T:
    """Get environment variable with type casting and default.
    
    Args:
        key: Environment variable name
        default: Default value (also determines type if cast not specified)
        cast: Type to cast to (int, float, bool, str). If None, inferred from default.
        
    Returns:
        The environment variable value cast to the appropriate type, or default.
    """
    val = os.environ.get(key)
    if val is None:
        return default
    
    target_type = cast or type(default)
    
    # Special handling for bool
    if target_type is bool:
        return val.lower() in ("true", "1", "yes", "on")  # type: ignore[return-value]
    
    try:
        return target_type(val)  # type: ignore[return-value]
    except ValueError:
        _logger.warning("Invalid %s for %s=%r, using default %r", target_type.__name__, key, val, default)
        return default


# =============================================================================
# Provider API Key Resolution
# =============================================================================

PROVIDER_API_KEYS: dict[str, str] = {
    "minimax/": "MINIMAX_API_KEY",
    "deepseek/": "DEEPSEEK_API_KEY",
    "moonshot/": "MOONSHOT_API_KEY",
    "dashscope/": "DASHSCOPE_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "openai/": "OPENAI_API_KEY",
    "gemini/": "GEMINI_API_KEY",
    "groq/": "GROQ_API_KEY",
    "together_ai/": "TOGETHER_API_KEY",
    "fireworks_ai/": "FIREWORKS_API_KEY",
    "openrouter/": "OPENROUTER_API_KEY",
    "bedrock/": "AWS_ACCESS_KEY_ID",
    "vertex_ai/": "GOOGLE_APPLICATION_CREDENTIALS",
    "ollama/": "",  # No API key needed for local
}


def get_provider_env_var(model: str) -> str | None:
    """Get the environment variable name for a model's provider."""
    model_lower = model.lower()
    for prefix, env_var in PROVIDER_API_KEYS.items():
        if model_lower.startswith(prefix):
            return env_var if env_var else None
    return None


def _load_user_env() -> None:
    """Load environment variables from user's configured env file."""
    try:
        from .user_config import load_env_file
        load_env_file()
    except ImportError:
        _logger.debug("user_config module not available, skipping env file loading")


def _resolve_api_key() -> str | None:
    """Resolve API key from environment."""
    _load_user_env()

    if key := os.environ.get("RLM_API_KEY"):
        return key

    model = os.environ.get("RLM_MODEL", "")
    if env_var := get_provider_env_var(model):
        if key := os.environ.get(env_var):
            return key

    return os.environ.get("OPENROUTER_API_KEY")


def _get_user_config_default(key: str, default: Any) -> Any:
    """Get default from user config, falling back to provided default."""
    try:
        from .user_config import get_config_value
        return get_config_value(key, default)
    except ImportError:
        return default


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RLMConfig:
    """Configuration for RLM execution.

    Settings priority (highest to lowest):
    1. Constructor arguments
    2. Environment variables (RLM_*)
    3. User config (~/.rlm/config.yaml)
    4. Built-in defaults

    Environment variables:
    - RLM_MODEL: Primary model (e.g., openai/gpt-4o)
    - RLM_SUB_MODEL: Model for sub-queries in REPL (defaults to RLM_MODEL)
    - RLM_API_BASE: Custom API endpoint (optional)
    - RLM_API_KEY: API key (or use provider-specific keys)
    - RLM_MAX_ITERATIONS: Max REPL iterations (default: 20)
    - RLM_MAX_LLM_CALLS: Max sub-LLM calls per execution (default: 50)
    - RLM_MAX_OUTPUT_CHARS: Max chars in REPL output (default: 100000)
    """

    # Model settings
    model: str = field(
        default_factory=lambda: _env(
            "RLM_MODEL", _get_user_config_default("model", "openai/gpt-4o-mini")
        )
    )
    sub_model: str = field(
        default_factory=lambda: _env(
            "RLM_SUB_MODEL", 
            _get_user_config_default(
                "sub_model",
                _env("RLM_MODEL", _get_user_config_default("model", "openai/gpt-4o-mini"))
            )
        )
    )
    api_base: str | None = field(default_factory=lambda: os.environ.get("RLM_API_BASE"))
    api_key: str | None = field(default_factory=_resolve_api_key)

    # RLM execution settings (maps to dspy.RLM parameters)
    # Priority: env var > config.yaml > default
    max_iterations: int = field(
        default_factory=lambda: _env_get(
            "RLM_MAX_ITERATIONS", _get_user_config_default("max_iterations", 20)
        )
    )
    max_llm_calls: int = field(
        default_factory=lambda: _env_get(
            "RLM_MAX_LLM_CALLS", _get_user_config_default("max_llm_calls", 50)
        )
    )
    max_output_chars: int = field(
        default_factory=lambda: _env_get(
            "RLM_MAX_OUTPUT_CHARS", _get_user_config_default("max_output_chars", 100_000)
        )
    )
    verbose: bool = field(
        default_factory=lambda: _env_get("RLM_VERBOSE", False)
    )

    # Budget/safety limits
    max_budget: float = field(
        default_factory=lambda: _env_get(
            "RLM_MAX_BUDGET", _get_user_config_default("max_budget", 1.0)
        )
    )
    max_timeout: float = field(
        default_factory=lambda: _env_get(
            "RLM_MAX_TIMEOUT", _get_user_config_default("max_timeout", 300.0)
        )
    )
    
    # Parallelism settings
    max_workers: int = field(
        default_factory=lambda: _env_get(
            "RLM_MAX_WORKERS", _get_user_config_default("max_workers", 8)
        )
    )

    def __repr__(self) -> str:
        key_display = "***" if self.api_key else None
        return (
            f"RLMConfig(model={self.model!r}, sub_model={self.sub_model!r}, "
            f"api_key={key_display!r}, max_iterations={self.max_iterations}, ...)"
        )


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class RLMResult:
    """Result from RLM execution.
    
    For standard signatures (context, query -> answer), access result.answer.
    For custom signatures with structured output, access fields via:
    - result.outputs dict: result.outputs["bugs"], result.outputs["score"]
    - attribute access: result.bugs, result.score (raises AttributeError if missing)
    """

    answer: str
    success: bool

    # Execution metadata
    total_tokens: int = 0
    total_cost: float = 0.0
    elapsed_time: float = 0.0
    iterations: int = 0

    # RLM-specific
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    final_reasoning: str = ""

    # Error info
    error: str | None = None

    # Token stats (for compatibility)
    token_stats: Any = None
    
    # Structured output fields (for custom signatures)
    outputs: dict[str, Any] = field(default_factory=dict)
    
    def __getattr__(self, name: str) -> Any:
        """Allow accessing output fields as attributes.
        
        Example:
            result.bugs  # same as result.outputs["bugs"]
        """
        # This is only called for attributes not found normally
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        outputs = object.__getattribute__(self, "outputs")
        if name in outputs:
            return outputs[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Available outputs: {list(outputs.keys())}"
        )


# =============================================================================
# Progress Callback
# =============================================================================

class ProgressCallback:
    """Callback for RLM progress updates.
    
    Subclass this to receive progress updates during RLM execution.
    
    Example:
        ```python
        class MyCallback(ProgressCallback):
            def on_iteration(self, iteration, max_iterations):
                print(f"Iteration {iteration}/{max_iterations}")
            
            def on_lm_call(self, call_type, inputs):
                print(f"LLM call: {call_type}")
        
        rlm = RLM(config=config, progress_callback=MyCallback())
        ```
    """
    
    def on_start(self, query: str, context_tokens: int) -> None:
        """Called when RLM execution starts."""
        pass
    
    def on_iteration(self, iteration: int, max_iterations: int) -> None:
        """Called at the start of each REPL iteration."""
        pass
    
    def on_lm_call(self, call_type: str, inputs: dict | None = None) -> None:
        """Called when an LLM call is made.
        
        Args:
            call_type: Type of call ("main", "sub", "tool")
            inputs: Optional input data
        """
        pass
    
    def on_tool_use(self, tool_name: str, args: dict | None = None) -> None:
        """Called when a tool is invoked."""
        pass
    
    def on_complete(self, result: "RLMResult") -> None:
        """Called when RLM execution completes."""
        pass
    
    def on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        pass


class DspyProgressCallback:
    """DSPy-compatible callback that wraps our ProgressCallback.
    
    This bridges our callback interface with dspy's BaseCallback.
    """
    
    def __init__(self, progress_callback: ProgressCallback):
        self.progress = progress_callback
        self._call_count = 0
    
    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._call_count += 1
        self.progress.on_lm_call("main", inputs)
    
    def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        pass


# =============================================================================
# Main RLM Class
# =============================================================================

class RLM:
    """
    Recursive Language Model using DSPy's native RLM module.

    RLMs treat large contexts as external environments that the LLM explores
    programmatically through a Python REPL. The LLM writes code to:
    - Navigate and examine data
    - Call sub-LLMs for semantic analysis (llm_query)
    - Build up answers iteratively

    This is fundamentally different from chunking approaches - the LLM has
    agency to explore the context as needed.

    Example:
        ```python
        rlm = RLM(config=RLMConfig(model="openai/gpt-4o"))

        # Load context from files
        context = rlm.load_context(["src/"])

        # Query - LLM will explore context via REPL
        result = rlm.query("What does the main function do?", context)
        print(result.answer)
        print(result.trajectory)  # See how LLM explored the context
        ```

    Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
    """

    def __init__(
        self,
        config: RLMConfig | None = None,
        tools: dict[str, Callable[..., str]] | None = None,
        signature: str | type = "context, query -> answer",
        interpreter: Any | None = None,
        use_tools: bool | str = True,
        progress_callback: ProgressCallback | None = None,
    ):
        """
        Initialize RLM.

        Args:
            config: Configuration settings. Uses defaults if not provided.
            tools: Additional tool functions available in the REPL.
                   Built-in tools (llm_query, llm_query_batched) are always available.
            signature: DSPy signature defining inputs and outputs.
                      Can be a string like "context, query -> answer" or
                      a dspy.Signature class for structured output.
                      Default: "context, query -> answer"
            interpreter: Custom code interpreter for REPL execution.
                        Must implement the CodeInterpreter protocol:
                        - tools property
                        - start() method
                        - execute(code, variables) method
                        - shutdown() method
                        If None, uses dspy's default PythonInterpreter (Deno/Pyodide).
            use_tools: Enable built-in code analysis tools.
                      - True or "safe": Safe tools (default) - ripgrep, tree-sitter, file ops
                      - "all": All tools including shell (requires RLM_ALLOW_SHELL=1)
                      - False: No extra tools
            progress_callback: Optional callback for progress updates.
                      Receives on_start, on_iteration, on_lm_call, on_complete events.
                      
        Available tools when enabled:
            - ripgrep(pattern, path, flags): Fast regex search
            - grep_context(pattern, path, context_lines): Search with context
            - find_files(pattern, path, file_type): Find files by pattern
            - read_file(path, start_line, end_line): Read file contents
            - file_stats(path): Get file/directory statistics
            - ast_query(code, query, language): Tree-sitter AST queries
            - find_definitions(path, name): Find function/class definitions
            - find_imports(path): Find all imports
            - find_calls(path, function_name): Find function call sites
            - shell(command, timeout): Run shell commands (disabled by default)
                      
        Example with structured output:
            ```python
            from rlm_dspy.signatures import BugFinder
            
            rlm = RLM(config=config, signature=BugFinder)
            result = rlm.query("Find all bugs", context)
            
            print(result.bugs)          # list[str]
            print(result.has_critical)  # bool
            ```
            
        Example with tools:
            ```python
            rlm = RLM(config=config, use_tools=True)
            result = rlm.query(
                "Find all functions that call 'execute' and check for bugs",
                context
            )
            # LLM can now use ripgrep, find_calls, etc. to explore the codebase
            ```
            
        Example with custom interpreter:
            ```python
            from e2b_code_interpreter import CodeInterpreter
            
            rlm = RLM(config=config, interpreter=CodeInterpreter())
            ```
        """
        self.config = config or RLMConfig()
        self._interpreter = interpreter
        self._progress_callback = progress_callback
        
        # Initialize tools
        self._tools = tools.copy() if tools else {}
        if use_tools:
            from ..tools import BUILTIN_TOOLS, SAFE_TOOLS
            builtin = BUILTIN_TOOLS if use_tools == "all" else SAFE_TOOLS
            # User tools take precedence over built-in
            for name, func in builtin.items():
                if name not in self._tools:
                    self._tools[name] = func
        
        # Validate all tools
        self._validate_tools(self._tools)
        
        # Track if we have a custom signature with structured output
        self._is_structured = not isinstance(signature, str)
        
        # Wrap signature with tool-first instructions if tools are enabled
        self._signature = self._wrap_signature_with_tool_instructions(signature, use_tools)

        # Validate API key early
        requires_api_key = not self.config.model.lower().startswith("ollama/")
        if requires_api_key and not self.config.api_key:
            env_var = get_provider_env_var(self.config.model) or "PROVIDER_API_KEY"
            raise ValueError(
                f"No API key configured for model '{self.config.model}'.\n"
                f"Set one of: RLM_API_KEY, {env_var}, or pass api_key to RLMConfig."
            )

        # Setup DSPy
        self._setup_dspy()

        # Create the dspy.RLM instance
        self._rlm = self._create_rlm()

        # Tracking
        self._start_time: float | None = None

    def _wrap_signature_with_tool_instructions(
        self, 
        signature: str | type, 
        use_tools: bool | str
    ) -> str | type:
        """Wrap signature with instructions to prioritize tool usage.
        
        When tools are enabled, this adds instructions telling the LLM to
        use the provided tools (index_code, ripgrep, etc.) FIRST before
        writing custom code. This improves accuracy for code analysis tasks.
        """
        if not use_tools or not self._tools:
            return signature
        
        tool_instructions = """IMPORTANT: You have access to powerful code analysis tools. USE THEM FIRST before writing custom code:

- For finding classes/functions/methods with EXACT line numbers: use `index_code(path, kind, name)` or `find_classes()`, `find_functions()`, `find_methods()`
- For searching code patterns: use `ripgrep(pattern, path)` - much faster than regex on context
- For reading specific files: use `read_file(path, start_line, end_line)`
- For finding function calls: use `find_calls(path, function_name)`

These tools provide 100% accurate results. Only fall back to manual parsing if tools don't meet your needs.

"""
        if isinstance(signature, str):
            # Convert string signature to class with tool instructions
            # Parse "context, query -> answer" format
            base_sig = dspy.Signature(signature)
            
            class ToolFirstSignature(base_sig):
                pass
            
            ToolFirstSignature.__doc__ = tool_instructions
            return ToolFirstSignature
        else:
            # For class-based signatures, prepend to the docstring
            original_doc = signature.__doc__ or ""
            
            # Create a new signature class with updated docstring
            class WrappedSignature(signature):
                pass
            
            WrappedSignature.__doc__ = tool_instructions + original_doc
            WrappedSignature.__name__ = signature.__name__
            WrappedSignature.__qualname__ = signature.__qualname__
            return WrappedSignature

    def _setup_dspy(self) -> None:
        """Configure DSPy with the primary model (thread-safe).
        
        Note: We do NOT call dspy.configure() here to avoid global state pollution.
        Instead, we use dspy.settings.context(lm=self._lm) in query() for thread-safety.
        """
        lm_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "api_key": self.config.api_key,
        }
        if self.config.api_base:
            lm_kwargs["api_base"] = self.config.api_base

        # Store LM instance for thread-local configuration in query()
        self._lm = dspy.LM(**lm_kwargs)
        
        # Configure parallelism settings (affects batch operations)
        dspy.settings.configure(
            async_max_workers=self.config.max_workers,
            num_threads=self.config.max_workers,
        )
        # NOTE: No lm= in configure() - we use context manager in query() instead

    def _create_sub_lm(self) -> dspy.LM | None:
        """Create sub-LM for llm_query calls if different from primary."""
        if self.config.sub_model == self.config.model:
            return None  # Use primary model

        lm_kwargs: dict[str, Any] = {
            "model": self.config.sub_model,
            "api_key": self.config.api_key,
        }
        if self.config.api_base:
            lm_kwargs["api_base"] = self.config.api_base

        return dspy.LM(**lm_kwargs)

    def _create_rlm(self) -> dspy.RLM:
        """Create the dspy.RLM instance."""
        kwargs: dict[str, Any] = {
            "signature": self._signature,
            "max_iterations": self.config.max_iterations,
            "max_llm_calls": self.config.max_llm_calls,
            "max_output_chars": self.config.max_output_chars,
            "verbose": self.config.verbose,
            "tools": self._tools,
            "sub_lm": self._create_sub_lm(),
        }
        
        # Add custom interpreter if provided
        if self._interpreter is not None:
            kwargs["interpreter"] = self._interpreter
        
        return dspy.RLM(**kwargs)

    def load_context(
        self,
        paths: list[str | Path],
        gitignore: bool = True,
        use_cache: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        """
        Load context from files or directories.

        Args:
            paths: List of file or directory paths
            gitignore: Whether to respect .gitignore patterns
            use_cache: Whether to use context caching (default True)
            max_tokens: Optional max tokens (truncates if exceeded)

        Returns:
            Combined context string with file markers
        """
        from .fileutils import (
            load_context_from_paths, 
            load_context_from_paths_cached,
            smart_truncate_context,
        )
        
        loader = load_context_from_paths_cached if use_cache else load_context_from_paths
        context = loader(
            paths=[Path(p) for p in paths],
            gitignore=gitignore,
            add_line_numbers=True,
        )
        
        # Optionally truncate to fit token limit
        if max_tokens is not None:
            context, was_truncated = smart_truncate_context(context, max_tokens)
            if was_truncated:
                _logger.warning(
                    "Context truncated to fit %d token limit", max_tokens
                )
        
        return context

    def _build_result(self, prediction: Any, elapsed: float) -> RLMResult:
        """Build RLMResult from dspy prediction, handling structured outputs."""
        raw_trajectory = getattr(prediction, "trajectory", [])
        raw_reasoning = getattr(prediction, "final_reasoning", "")
        extra_secrets = [self.config.api_key] if self.config.api_key else None
        
        # Extract structured outputs if using custom signature
        outputs: dict[str, Any] = {}
        answer = ""
        
        if self._is_structured:
            # Get all output fields from the signature class
            sig = self._signature
            output_field_names = []
            
            # dspy.Signature classes have output_fields as a dict property
            if hasattr(sig, "output_fields"):
                fields = sig.output_fields
                if isinstance(fields, dict):
                    output_field_names = list(fields.keys())
            
            # Extract values from prediction
            for field_name in output_field_names:
                value = getattr(prediction, field_name, None)
                if value is not None:
                    # Sanitize string values
                    if isinstance(value, str):
                        value = _sanitize_secrets(value, extra_secrets)
                    elif isinstance(value, list):
                        value = [
                            _sanitize_secrets(v, extra_secrets) if isinstance(v, str) else v
                            for v in value
                        ]
                    outputs[field_name] = value
            
            # Use first output field as "answer" for compatibility
            if outputs:
                first_key = next(iter(outputs))
                first_val = outputs[first_key]
                if isinstance(first_val, str):
                    answer = first_val
                elif isinstance(first_val, list):
                    answer = "\n".join(str(v) for v in first_val)
                else:
                    answer = str(first_val)
        else:
            # Standard signature: just get answer
            answer = _sanitize_secrets(getattr(prediction, "answer", ""), extra_secrets)
        
        return RLMResult(
            answer=answer,
            success=True,
            elapsed_time=elapsed,
            trajectory=_sanitize_trajectory(raw_trajectory, extra_secrets),
            final_reasoning=_sanitize_secrets(raw_reasoning, extra_secrets),
            iterations=len(raw_trajectory),
            outputs=outputs,
        )

    def query(
        self,
        query: str,
        context: str,
    ) -> RLMResult:
        """
        Execute a query against the context using RLM.

        The LLM will explore the context programmatically through a REPL,
        writing Python code to navigate, analyze, and build up an answer.

        Args:
            query: The question to answer
            context: The context to explore (typically from load_context)

        Returns:
            RLMResult with answer, trajectory, and metadata.
            For custom signatures, structured outputs available via result.outputs dict.
            
        Raises:
            ValueError: If query or context is empty/None
        """
        import concurrent.futures
        from .fileutils import estimate_tokens
        
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not context or not context.strip():
            raise ValueError("Context cannot be empty")

        self._start_time = time.time()
        
        # Notify callback of start
        if self._progress_callback:
            context_tokens = estimate_tokens(context)
            self._progress_callback.on_start(query, context_tokens)

        def _execute_rlm() -> Any:
            """Execute RLM with thread-local DSPy configuration."""
            # Use thread-local configuration to avoid global state pollution
            with dspy.settings.context(lm=self._lm):
                return self._rlm(context=context, query=query)

        try:
            # Execute with timeout if configured
            if self.config.max_timeout and self.config.max_timeout > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_execute_rlm)
                    try:
                        prediction = future.result(timeout=self.config.max_timeout)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutExceededError(
                            self.config.max_timeout, self.config.max_timeout
                        )
            else:
                prediction = _execute_rlm()

            elapsed = time.time() - self._start_time

            # Build result from prediction
            result = self._build_result(prediction, elapsed)
            
            # Notify callback of completion
            if self._progress_callback:
                self._progress_callback.on_complete(result)
            
            return result

        except TimeoutExceededError:
            elapsed = time.time() - self._start_time if self._start_time else 0
            _logger.warning("RLM execution timed out after %.1fs", elapsed)
            error_result = RLMResult(
                answer="",
                success=False,
                error=f"Query timed out after {elapsed:.1f}s (limit: {self.config.max_timeout}s)",
                elapsed_time=elapsed,
            )
            if self._progress_callback:
                self._progress_callback.on_error(TimeoutExceededError(self.config.max_timeout, elapsed))
            return error_result

        except Exception as e:
            elapsed = time.time() - self._start_time if self._start_time else 0
            if self._progress_callback:
                self._progress_callback.on_error(e)
            _logger.exception("RLM execution failed")
            return RLMResult(
                answer="",
                success=False,
                error=str(e),
                elapsed_time=elapsed,
            )

    async def query_async(
        self,
        query: str,
        context: str,
    ) -> RLMResult:
        """Async version of query."""
        self._start_time = time.time()

        try:
            with dspy.settings.context(lm=self._lm):
                prediction = await self._rlm.aforward(context=context, query=query)

            elapsed = time.time() - self._start_time

            # Build result from prediction (handles structured outputs)
            return self._build_result(prediction, elapsed)

        except Exception as e:
            elapsed = time.time() - self._start_time if self._start_time else 0
            _logger.exception("RLM async execution failed")
            return RLMResult(
                answer="",
                success=False,
                error=str(e),
                elapsed_time=elapsed,
            )

    def add_tool(self, name: str, func: Callable[..., str]) -> None:
        """
        Add a custom tool function available in the REPL.

        The tool will be callable by name from the LLM's code.

        Args:
            name: Tool name (must be valid Python identifier)
            func: Tool function (should return str)

        Example:
            ```python
            def search_web(query: str) -> str:
                '''Search the web for information.'''
                return requests.get(f"https://api.search.com?q={query}").text

            rlm.add_tool("search_web", search_web)
            ```
        
        Raises:
            ValueError: If name is not a valid Python identifier
            TypeError: If func is not callable
        """
        self._validate_tools({name: func})
        self._tools[name] = func
        # Recreate RLM with updated tools
        self._rlm = self._create_rlm()
    
    def _validate_tools(self, tools: dict[str, Callable]) -> None:
        """Validate tool names and types.
        
        Args:
            tools: Dict of tool name -> function
            
        Raises:
            ValueError: If name is not a valid Python identifier or conflicts with builtins
            TypeError: If tool is not callable
        """
        # Reserved names that conflict with sandbox builtins
        RESERVED_NAMES = {
            'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
            'sorted', 'reversed', 'sum', 'min', 'max', 'abs', 'round',
            'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
            'SUBMIT', 'FINAL',  # RLM special functions
        }
        
        for name, func in tools.items():
            # Check valid identifier
            if not name.isidentifier():
                raise ValueError(f"Invalid tool name '{name}': must be a valid Python identifier")
            
            # Check not reserved
            if name in RESERVED_NAMES:
                raise ValueError(f"Tool name '{name}' conflicts with built-in function")
            
            # Check callable
            if not callable(func):
                raise TypeError(f"Tool '{name}' must be callable, got {type(func).__name__}")

    def batch(
        self,
        queries: list[dict[str, str]],
        context: str | None = None,
        num_threads: int = 4,
        max_errors: int | None = None,
        return_failed: bool = False,
    ) -> list[RLMResult]:
        """
        Process multiple queries in parallel.

        This is much faster than sequential query() calls when you have
        multiple independent questions about the same or different contexts.

        Args:
            queries: List of query dicts. Each dict should have:
                - "query": The question to ask (required)
                - "context": Optional context (uses shared context if not provided)
            context: Shared context for all queries (used if not in query dict)
            num_threads: Number of parallel threads (default: 4)
            max_errors: Maximum failures before stopping (default: None = no limit)
            return_failed: If True, include failed results (default: False)

        Returns:
            List of RLMResult in same order as input queries.
            Failed queries have success=False and error message.

        Example:
            ```python
            rlm = RLM(config=config)
            context = rlm.load_context(["src/"])

            # Same context, different queries (parallel)
            results = rlm.batch([
                {"query": "Summarize the architecture"},
                {"query": "Find security issues"},
                {"query": "Find performance bottlenecks"},
            ], context=context, num_threads=3)

            # Different contexts
            results = rlm.batch([
                {"context": file1, "query": "Find bugs"},
                {"context": file2, "query": "Find bugs"},
            ], num_threads=2)
            ```
        """
        if not queries:
            return []

        # Build dspy.Example list
        examples = []
        for i, q in enumerate(queries):
            query_text = q.get("query")
            if not query_text:
                raise ValueError(f"Query {i} missing 'query' field")

            ctx = q.get("context", context)
            if ctx is None:
                raise ValueError(f"Query {i} has no context (provide in query or as shared context)")

            examples.append(dspy.Example(
                context=ctx,
                query=query_text,
            ).with_inputs("context", "query"))

        # Execute batch with thread-local configuration
        start_time = time.time()
        with dspy.settings.context(lm=self._lm):
            raw_results, failed_examples, exceptions = self._rlm.batch(
                examples,
                num_threads=num_threads,
                max_errors=max_errors,
                return_failed_examples=True,
            )

        elapsed = time.time() - start_time

        # Convert to RLMResult list
        # Note: batch() returns results in order, with None for failed ones
        results: list[RLMResult] = []
        
        # Create a map of failed example indices
        failed_indices = set()
        failed_map: dict[int, Exception] = {}
        if failed_examples:
            for ex, exc in zip(failed_examples, exceptions):
                # Find the index of this failed example
                for i, orig in enumerate(examples):
                    if orig.context == ex.context and orig.query == ex.query:
                        failed_indices.add(i)
                        failed_map[i] = exc
                        break

        # Build results maintaining order
        result_idx = 0
        for i in range(len(examples)):
            if i in failed_indices:
                # Failed query
                error_msg = str(failed_map.get(i, "Unknown error"))
                results.append(RLMResult(
                    answer="",
                    success=False,
                    error=error_msg,
                    elapsed_time=elapsed / len(examples),  # Approximate
                ))
            else:
                # Successful query
                if result_idx < len(raw_results):
                    pred = raw_results[result_idx]
                    result_idx += 1
                    
                    # Sanitize output
                    extra_secrets = [self.config.api_key] if self.config.api_key else None
                    raw_trajectory = getattr(pred, "trajectory", [])
                    raw_reasoning = getattr(pred, "final_reasoning", "")
                    
                    results.append(RLMResult(
                        answer=_sanitize_secrets(pred.answer, extra_secrets),
                        success=True,
                        elapsed_time=elapsed / len(examples),  # Approximate
                        trajectory=_sanitize_trajectory(raw_trajectory, extra_secrets),
                        final_reasoning=_sanitize_secrets(raw_reasoning, extra_secrets),
                        iterations=len(raw_trajectory),
                    ))
                else:
                    # Shouldn't happen, but handle gracefully
                    results.append(RLMResult(
                        answer="",
                        success=False,
                        error="Result missing from batch",
                        elapsed_time=elapsed / len(examples),
                    ))

        # Filter out failed if not requested
        if not return_failed:
            results = [r for r in results if r.success]

        _logger.info(
            "Batch completed: %d/%d successful in %.1fs",
            sum(1 for r in results if r.success),
            len(queries),
            elapsed,
        )

        return results

    def close(self) -> None:
        """Clean up resources."""
        pass  # dspy.RLM handles its own cleanup

    def __enter__(self) -> "RLM":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# =============================================================================
# Custom Exceptions
# =============================================================================

class TimeoutExceededError(Exception):
    """Raised when query execution exceeds the configured timeout."""
    def __init__(self, elapsed: float, timeout: float):
        self.elapsed = elapsed
        self.timeout = timeout
        super().__init__(f"Timeout: {elapsed:.1f}s of {timeout:.1f}s")

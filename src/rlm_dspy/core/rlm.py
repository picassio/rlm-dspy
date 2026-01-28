"""Core RLM class using DSPy's native RLM module.

This module wraps dspy.RLM to provide a unified interface for recursive
language model processing with proper configuration management.

Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import dspy

_logger = logging.getLogger(__name__)


# =============================================================================
# Environment Helpers
# =============================================================================

def _env(key: str, default: str) -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """Get environment variable as int with default."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        _logger.warning("Invalid int for %s=%r, using default %d", key, val, default)
        return default


def _env_float(key: str, default: float) -> float:
    """Get environment variable as float with default."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        _logger.warning("Invalid float for %s=%r, using default %f", key, val, default)
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Get environment variable as bool with default."""
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


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
            _env("RLM_MODEL", _get_user_config_default("model", "openai/gpt-4o-mini"))
        )
    )
    api_base: str | None = field(default_factory=lambda: os.environ.get("RLM_API_BASE"))
    api_key: str | None = field(default_factory=_resolve_api_key)

    # RLM execution settings (maps to dspy.RLM parameters)
    max_iterations: int = field(
        default_factory=lambda: _env_int("RLM_MAX_ITERATIONS", 20)
    )
    max_llm_calls: int = field(
        default_factory=lambda: _env_int("RLM_MAX_LLM_CALLS", 50)
    )
    max_output_chars: int = field(
        default_factory=lambda: _env_int("RLM_MAX_OUTPUT_CHARS", 100_000)
    )
    verbose: bool = field(
        default_factory=lambda: _env_bool("RLM_VERBOSE", False)
    )

    # Budget/safety limits
    max_budget: float = field(
        default_factory=lambda: _env_float(
            "RLM_MAX_BUDGET", _get_user_config_default("max_budget", 1.0)
        )
    )
    max_timeout: float = field(
        default_factory=lambda: _env_float(
            "RLM_MAX_TIMEOUT", _get_user_config_default("max_timeout", 300.0)
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
    """Result from RLM execution."""

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
        signature: str = "context, query -> answer",
    ):
        """
        Initialize RLM.

        Args:
            config: Configuration settings. Uses defaults if not provided.
            tools: Additional tool functions available in the REPL.
                   Built-in tools (llm_query, llm_query_batched) are always available.
            signature: DSPy signature defining inputs and outputs.
                      Default: "context, query -> answer"
        """
        self.config = config or RLMConfig()
        self._tools = tools or {}
        self._signature = signature

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

    def _setup_dspy(self) -> None:
        """Configure DSPy with the primary model (thread-safe)."""
        lm_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "api_key": self.config.api_key,
        }
        if self.config.api_base:
            lm_kwargs["api_base"] = self.config.api_base

        # Store LM instance for thread-local configuration in query()
        self._lm = dspy.LM(**lm_kwargs)
        # Set as default for this instance (will use context manager in query)
        dspy.configure(lm=self._lm)

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
        return dspy.RLM(
            signature=self._signature,
            max_iterations=self.config.max_iterations,
            max_llm_calls=self.config.max_llm_calls,
            max_output_chars=self.config.max_output_chars,
            verbose=self.config.verbose,
            tools=self._tools,
            sub_lm=self._create_sub_lm(),
        )

    def load_context(
        self,
        paths: list[str | Path],
        gitignore: bool = True,
    ) -> str:
        """
        Load context from files or directories.

        Args:
            paths: List of file or directory paths
            gitignore: Whether to respect .gitignore patterns

        Returns:
            Combined context string with file markers
        """
        import pathspec

        # Load gitignore patterns
        patterns = []
        if gitignore:
            for path in paths:
                p = Path(path)
                gitignore_path = (p if p.is_dir() else p.parent) / ".gitignore"
                if gitignore_path.exists():
                    patterns.extend(gitignore_path.read_text(encoding="utf-8").splitlines())

        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns) if patterns else None

        # Collect all files
        files: list[Path] = []
        for path in paths:
            p = Path(path)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        if spec is None or not spec.match_file(str(f.relative_to(p))):
                            files.append(f)

        # Read and combine with clear file markers
        context_parts = []
        skipped_files = []
        for f in sorted(files):
            try:
                content = f.read_text(encoding="utf-8")
                # Add line numbers to help LLM report accurate locations
                numbered_lines = [
                    f"{i+1:4d} | {line}"
                    for i, line in enumerate(content.splitlines())
                ]
                numbered_content = "\n".join(numbered_lines)
                context_parts.append(f"=== FILE: {f} ===\n{numbered_content}\n=== END FILE ===\n")
            except UnicodeDecodeError:
                skipped_files.append((f, "binary/encoding"))
            except PermissionError:
                skipped_files.append((f, "permission denied"))

        if skipped_files:
            _logger.warning(
                "Skipped %d files: %s",
                len(skipped_files),
                ", ".join(f"{f.name} ({reason})" for f, reason in skipped_files[:5]),
            )

        return "\n".join(context_parts)

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
            RLMResult with answer, trajectory, and metadata
        """
        import concurrent.futures

        self._start_time = time.time()

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

            return RLMResult(
                answer=prediction.answer,
                success=True,
                elapsed_time=elapsed,
                trajectory=getattr(prediction, "trajectory", []),
                final_reasoning=getattr(prediction, "final_reasoning", ""),
                iterations=len(getattr(prediction, "trajectory", [])),
            )

        except TimeoutExceededError:
            elapsed = time.time() - self._start_time if self._start_time else 0
            _logger.warning("RLM execution timed out after %.1fs", elapsed)
            return RLMResult(
                answer="",
                success=False,
                error=f"Query timed out after {elapsed:.1f}s (limit: {self.config.max_timeout}s)",
                elapsed_time=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - self._start_time if self._start_time else 0
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
            prediction = await self._rlm.acall(context=context, query=query)

            elapsed = time.time() - self._start_time

            return RLMResult(
                answer=prediction.answer,
                success=True,
                elapsed_time=elapsed,
                trajectory=getattr(prediction, "trajectory", []),
                final_reasoning=getattr(prediction, "final_reasoning", ""),
                iterations=len(getattr(prediction, "trajectory", [])),
            )

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
        """
        self._tools[name] = func
        # Recreate RLM with updated tools
        self._rlm = self._create_rlm()

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

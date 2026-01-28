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
from typing import Any, Callable

import dspy

from .secrets import COMMON_SECRETS, mask_value

_logger = logging.getLogger(__name__)


# =============================================================================
# Environment Helpers
# =============================================================================

def _env(key: str, default: str) -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)


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
    
    # Check for actual secret values from environment
    for key in COMMON_SECRETS:
        value = os.environ.get(key)
        if value and len(value) > 8 and value in result:
            result = result.replace(value, "[REDACTED]")
    
    # Also check for common API key patterns
    import re
    patterns = [
        (r'sk-[a-zA-Z0-9]{20,}', '[REDACTED_SK]'),  # OpenAI style
        (r'sk-ant-[a-zA-Z0-9-]{20,}', '[REDACTED_ANTHROPIC]'),  # Anthropic
        (r'sk-or-v1-[a-zA-Z0-9]{20,}', '[REDACTED_OPENROUTER]'),  # OpenRouter
        (r'gsk_[a-zA-Z0-9]{20,}', '[REDACTED_GROQ]'),  # Groq
        (r'AIza[a-zA-Z0-9_-]{35}', '[REDACTED_GOOGLE]'),  # Google
    ]
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    
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
                      
        Example with structured output:
            ```python
            from rlm_dspy.signatures import BugFinder
            
            rlm = RLM(config=config, signature=BugFinder)
            result = rlm.query("Find all bugs", context)
            
            print(result.bugs)          # list[str]
            print(result.has_critical)  # bool
            ```
            
        Example with custom interpreter:
            ```python
            from e2b_code_interpreter import CodeInterpreter
            
            rlm = RLM(config=config, interpreter=CodeInterpreter())
            ```
        """
        self.config = config or RLMConfig()
        self._tools = tools or {}
        self._signature = signature
        self._interpreter = interpreter
        
        # Track if we have a custom signature with structured output
        self._is_structured = not isinstance(signature, str)

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
        # NOTE: No dspy.configure() call - we use context manager in query() instead

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

        # Common directories to always skip (performance optimization)
        SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.tox', 'dist', 'build'}
        
        def _should_skip_entry(entry, entry_path: Path, root_path: Path, spec) -> bool:
            """Check if an entry should be skipped based on gitignore and common patterns."""
            # Skip common ignored directories
            if entry.is_dir(follow_symlinks=False) and entry.name in SKIP_DIRS:
                return True
            # Check gitignore patterns
            if spec:
                try:
                    rel_path = entry_path.relative_to(root_path)
                except ValueError:
                    rel_path = entry_path
                if spec.match_file(str(rel_path)):
                    return True
            return False

        def collect_files_fast(
            current_path: Path, 
            root_path: Path,
            spec: pathspec.PathSpec | None
        ) -> list[Path]:
            """Recursively collect files, pruning ignored directories early."""
            result: list[Path] = []
            
            try:
                entries = list(os.scandir(current_path))
            except PermissionError:
                _logger.debug("Permission denied: %s", current_path)
                return result
            
            for entry in entries:
                entry_path = Path(entry.path)
                
                if _should_skip_entry(entry, entry_path, root_path, spec):
                    continue
                
                if entry.is_file(follow_symlinks=False):
                    result.append(entry_path)
                elif entry.is_dir(follow_symlinks=False):
                    result.extend(collect_files_fast(entry_path, root_path, spec))
            
            return result

        files: list[Path] = []
        for path in paths:
            p = Path(path)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                # Pass p as both current and root path
                files.extend(collect_files_fast(p, p, spec))

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
            # dspy.Signature classes have output_fields() method or we can inspect
            output_field_names = []
            if hasattr(sig, "output_fields"):
                # Method that returns field names
                try:
                    fields = sig.output_fields()
                    if isinstance(fields, dict):
                        output_field_names = list(fields.keys())
                    else:
                        output_field_names = list(fields)
                except Exception:
                    pass
            
            if not output_field_names:
                # Fallback: inspect the class for OutputField annotations
                import dspy
                for name in dir(sig):
                    if not name.startswith("_"):
                        attr = getattr(sig, name, None)
                        if isinstance(attr, dspy.OutputField):
                            output_field_names.append(name)
            
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

            # Build result from prediction
            return self._build_result(prediction, elapsed)

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
        """
        self._tools[name] = func
        # Recreate RLM with updated tools
        self._rlm = self._create_rlm()

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

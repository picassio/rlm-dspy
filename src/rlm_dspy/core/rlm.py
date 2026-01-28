"""Core RLM class combining recursive decomposition with DSPy optimization."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import dspy

from .signatures import AggregateAnswers, AnalyzeChunk, DecomposeTask

_logger = logging.getLogger(__name__)


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


# Provider prefix -> environment variable mapping (single source of truth)
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
    "bedrock/": "AWS_ACCESS_KEY_ID",  # AWS uses different auth
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
        pass  # user_config not available


def _resolve_api_key() -> str | None:
    """Resolve API key from environment, checking provider-specific keys.

    Priority:
    1. RLM_API_KEY (explicit override)
    2. Provider-specific keys based on RLM_MODEL
    3. User's env_file (from ~/.rlm/config.yaml)
    4. OPENROUTER_API_KEY (legacy fallback)
    """
    # Load user's env file first
    _load_user_env()

    # Explicit override
    if key := os.environ.get("RLM_API_KEY"):
        return key

    # Check provider-specific keys based on model
    model = os.environ.get("RLM_MODEL", "")
    if env_var := get_provider_env_var(model):
        if key := os.environ.get(env_var):
            return key

    # Legacy fallback
    return os.environ.get("OPENROUTER_API_KEY")


def _get_user_config_default(key: str, default: Any) -> Any:
    """Get default from user config, falling back to provided default."""
    try:
        from .user_config import get_config_value
        return get_config_value(key, default)
    except ImportError:
        return default


@dataclass
class RLMConfig:
    """Configuration for RLM execution.

    Settings priority (highest to lowest):
    1. Constructor arguments
    2. Environment variables (RLM_*)
    3. User config (~/.rlm/config.yaml)
    4. Built-in defaults

    Environment variables:
    - RLM_MODEL: Model name (e.g., openai/gpt-4o, deepseek/deepseek-chat)
    - RLM_API_BASE: Custom API endpoint (optional, for self-hosted or proxies)
    - RLM_API_KEY: API key (or use provider-specific: OPENAI_API_KEY, etc.)
    - RLM_MAX_BUDGET: Max cost in USD (default: 1.0)
    - RLM_MAX_TIMEOUT: Max time in seconds (default: 300)
    - RLM_CHUNK_SIZE: Chunk size in chars (default: 100000)

    Run 'rlm-dspy setup' to configure defaults interactively.
    """

    # Model settings - check env first, then user config, then built-in default
    # Model format: provider/model-name (e.g., openai/gpt-4o, deepseek/deepseek-chat)
    model: str = field(
        default_factory=lambda: _env(
            "RLM_MODEL", _get_user_config_default("model", "openai/gpt-4o-mini")
        )
    )
    sub_model: str = field(
        default_factory=lambda: _env("RLM_SUB_MODEL", _env("RLM_MODEL", "openai/gpt-4o-mini"))
    )
    # api_base is optional - only needed for custom endpoints (self-hosted, proxies)
    # Native providers (openai/, anthropic/, deepseek/, etc.) don't need this
    api_base: str | None = field(default_factory=lambda: os.environ.get("RLM_API_BASE"))
    api_key: str | None = field(default_factory=_resolve_api_key)

    # Execution limits
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
    max_tokens: int = field(default_factory=lambda: _env_int("RLM_MAX_TOKENS", 500_000))
    max_iterations: int = field(default_factory=lambda: _env_int("RLM_MAX_ITERATIONS", 30))
    max_depth: int = field(default_factory=lambda: _env_int("RLM_MAX_DEPTH", 3))

    # Chunking settings
    default_chunk_size: int = field(
        default_factory=lambda: _env_int(
            "RLM_CHUNK_SIZE", _get_user_config_default("chunk_size", 100_000)
        )
    )
    overlap: int = field(default_factory=lambda: _env_int("RLM_OVERLAP", 500))
    syntax_aware_chunking: bool = field(
        default_factory=lambda: _env_bool(
            "RLM_SYNTAX_AWARE", _get_user_config_default("syntax_aware", True)
        )
    )

    # Processing settings
    strategy: Literal["auto", "map_reduce", "iterative", "hierarchical"] = "auto"
    parallel_chunks: int = field(default_factory=lambda: _env_int("RLM_PARALLEL_CHUNKS", 20))
    use_async: bool = field(default_factory=lambda: _env_bool("RLM_USE_ASYNC", True))

    # Model-specific settings
    disable_thinking: bool = field(default_factory=lambda: _env_bool("RLM_DISABLE_THINKING", True))
    enable_cache: bool = field(default_factory=lambda: _env_bool("RLM_ENABLE_CACHE", True))

    # DSPy optimization
    use_compiled_prompts: bool = field(default_factory=lambda: _env_bool("RLM_USE_COMPILED_PROMPTS", True))
    prompt_cache_dir: Path | None = None

    def __repr__(self) -> str:
        """Hide API key in repr to prevent accidental logging."""
        key_display = "***" if self.api_key else None
        return (
            f"RLMConfig(model={self.model!r}, api_base={self.api_base!r}, "
            f"api_key={key_display!r}, max_budget={self.max_budget}, "
            f"strategy={self.strategy!r}, ...)"
        )


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
    depth_reached: int = 0

    # Chunk processing stats
    chunks_processed: int = 0
    chunks_with_relevant_info: int = 0

    # Error info
    error: str | None = None
    partial_answer: str | None = None

    # Trace for debugging
    trace: list[dict[str, Any]] = field(default_factory=list)

    # Token stats (from token_stats module)
    token_stats: Any = None  # TokenStats | None


class RLM:
    """
    Recursive Language Model with DSPy optimization.

    Combines RLM's recursive decomposition approach with DSPy's
    automatic prompt optimization for better performance.

    Example:
        ```python
        rlm = RLM(config=RLMConfig(model="claude-sonnet-4"))

        # Load context from files
        context = rlm.load_context(["file1.py", "file2.py"])

        # Run query
        result = rlm.query("What does the main function do?", context)
        print(result.answer)
        ```
    """

    def __init__(self, config: RLMConfig | None = None):
        self.config = config or RLMConfig()

        # Validate API key early to fail fast
        if not self.config.api_key:
            env_var = get_provider_env_var(self.config.model) or "PROVIDER_API_KEY"
            raise ValueError(
                f"No API key configured for model '{self.config.model}'.\n"
                f"Set one of: RLM_API_KEY, {env_var}, or pass api_key to RLMConfig."
            )

        self._setup_dspy()
        self._setup_programs()

        # Execution state
        self._tokens_used = 0
        self._cost_spent = 0.0
        self._start_time: float | None = None

        # Token tracking (from token_stats module)
        from .token_stats import TokenStats
        self._current_stats: TokenStats | None = None

        # Large content handling (from paste_store module)
        from .paste_store import PasteStore
        self.paste_store = PasteStore()

        # Reusable thread pool for async operations (prevents thread exhaustion)
        import threading
        from concurrent.futures import ThreadPoolExecutor
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()

    def _setup_dspy(self) -> None:
        """Configure DSPy with the specified model."""
        # Build kwargs for DSPy LM
        lm_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "api_key": self.config.api_key,
        }

        # Only set api_base if explicitly configured
        # LiteLLM handles routing for native providers automatically
        if self.config.api_base:
            lm_kwargs["api_base"] = self.config.api_base

        lm = dspy.LM(**lm_kwargs)
        dspy.configure(lm=lm)

    def _setup_programs(self) -> None:
        """Initialize DSPy programs with optional compiled prompts."""
        # Use ChainOfThought for better reasoning
        self.decomposer = dspy.ChainOfThought(DecomposeTask)
        self.chunk_analyzer = dspy.ChainOfThought(AnalyzeChunk)
        self.aggregator = dspy.ChainOfThought(AggregateAnswers)

        # Load compiled prompts if available
        if self.config.use_compiled_prompts and self.config.prompt_cache_dir:
            self._load_compiled_prompts()

    def _load_compiled_prompts(self) -> None:
        """Load pre-optimized prompts from cache."""
        cache_dir = self.config.prompt_cache_dir
        if cache_dir and cache_dir.exists():
            for name, program in [
                ("decomposer", self.decomposer),
                ("chunk_analyzer", self.chunk_analyzer),
                ("aggregator", self.aggregator),
            ]:
                prompt_file = cache_dir / f"{name}.json"
                if prompt_file.exists():
                    program.load(str(prompt_file))

    def _check_limits(self) -> None:
        """Check if execution limits have been exceeded."""
        if self._cost_spent >= self.config.max_budget:
            raise BudgetExceededError(self._cost_spent, self.config.max_budget)

        if self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed >= self.config.max_timeout:
                raise TimeoutExceededError(elapsed, self.config.max_timeout)

        if self._tokens_used >= self.config.max_tokens:
            raise TokenLimitExceededError(self._tokens_used, self.config.max_tokens)

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
            Combined context string
        """
        import pathspec

        # Load gitignore patterns
        patterns = []
        if gitignore:
            for path in paths:
                p = Path(path)
                gitignore_path = (p if p.is_dir() else p.parent) / ".gitignore"
                if gitignore_path.exists():
                    patterns.extend(gitignore_path.read_text().splitlines())

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

        # Read and combine
        context_parts = []
        skipped_files = []
        for f in sorted(files):
            try:
                content = f.read_text()
                context_parts.append(f"### {f}\n```\n{content}\n```\n")
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
        depth: int = 0,
    ) -> RLMResult:
        """
        Execute a query against the context.

        Args:
            query: The question to answer
            context: The context to search
            depth: Current recursion depth (internal use)

        Returns:
            RLMResult with answer and metadata
        """
        from .token_stats import TokenStats, count_tokens

        self._start_time = time.time()
        trace: list[dict[str, Any]] = []

        # Track raw context size for stats
        raw_context_tokens = count_tokens(context)
        self._current_stats = TokenStats(raw_context_tokens=raw_context_tokens)

        try:
            # Determine processing strategy
            strategy_result = self.decomposer(
                query=query,
                context_size=len(context),
                context_type=self._detect_context_type(context),
            )

            trace.append(
                {
                    "step": "decompose",
                    "strategy": strategy_result.strategy,
                    "chunk_size": strategy_result.chunk_size,
                    "subtasks": strategy_result.subtasks,
                }
            )

            strategy = strategy_result.strategy if self.config.strategy == "auto" else self.config.strategy
            chunk_size = strategy_result.chunk_size or self.config.default_chunk_size

            # Process based on strategy
            if strategy == "direct" or len(context) <= chunk_size:
                # Small enough to process directly
                answer = self._process_direct(query, context)
            elif strategy == "map_reduce":
                answer = self._process_map_reduce(query, context, chunk_size, trace)
            elif strategy == "iterative":
                answer = self._process_iterative(query, context, chunk_size, trace)
            elif strategy == "hierarchical":
                answer = self._process_hierarchical(query, context, chunk_size, depth, trace)
            else:
                answer = self._process_map_reduce(query, context, chunk_size, trace)

            # Update token stats
            if self._current_stats:
                self._current_stats.llm_input_tokens = self._tokens_used
                self._current_stats.processed_tokens = self._tokens_used

            return RLMResult(
                answer=answer,
                success=True,
                total_tokens=self._tokens_used,
                total_cost=self._cost_spent,
                elapsed_time=time.time() - self._start_time,
                depth_reached=depth,
                trace=trace,
                token_stats=self._current_stats,
            )

        except (BudgetExceededError, TimeoutExceededError, TokenLimitExceededError) as e:
            return RLMResult(
                answer="",
                success=False,
                error=str(e),
                partial_answer=getattr(e, "partial_answer", None),
                total_tokens=self._tokens_used,
                total_cost=self._cost_spent,
                elapsed_time=time.time() - self._start_time,
                trace=trace,
            )

    def _detect_context_type(self, context: str) -> str:
        """Detect the type of context (code, docs, mixed)."""
        code_indicators = ["def ", "class ", "function ", "import ", "```"]
        doc_indicators = ["# ", "## ", "### ", "**", "- "]

        code_score = sum(1 for i in code_indicators if i in context[:5000])
        doc_score = sum(1 for i in doc_indicators if i in context[:5000])

        if code_score > doc_score * 2:
            return "code"
        elif doc_score > code_score * 2:
            return "docs"
        return "mixed"

    def _chunk_context(self, context: str, chunk_size: int) -> list[str]:
        """Split context into chunks, optionally respecting syntax boundaries.

        When syntax_aware_chunking is enabled (default), uses tree-sitter to
        identify function/class boundaries and avoids splitting mid-definition.
        This prevents false positives from truncated code in LLM analysis.
        """
        # Safety check to avoid division by zero or invalid chunking
        if chunk_size <= 0:
            chunk_size = self.config.default_chunk_size
        if chunk_size <= 0:
            chunk_size = 100_000  # Ultimate fallback
        overlap = min(self.config.overlap, chunk_size - 1) if chunk_size > 1 else 0

        # Try syntax-aware chunking if enabled
        if self.config.syntax_aware_chunking:
            try:
                from .syntax_chunker import TREE_SITTER_AVAILABLE, chunk_code_syntax_aware

                if TREE_SITTER_AVAILABLE:
                    code_chunks = chunk_code_syntax_aware(
                        context,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                    if code_chunks:
                        return [c.content for c in code_chunks]
            except ImportError:
                pass  # Fall back to character-based chunking

        # Fallback: character-based chunking
        chunks = []
        start = 0
        while start < len(context):
            end = min(start + chunk_size, len(context))
            chunks.append(context[start:end])
            step = max(1, chunk_size - overlap)
            start += step
            if end >= len(context):
                break
        return chunks

    def _process_direct(self, query: str, context: str) -> str:
        """Process small context directly without chunking."""
        self._check_limits()
        result = self.chunk_analyzer(
            query=query,
            chunk=context,
            chunk_index=0,
            total_chunks=1,
        )
        return result.relevant_info

    def _process_map_reduce(
        self,
        query: str,
        context: str,
        chunk_size: int,
        trace: list[dict[str, Any]],
    ) -> str:
        """Process chunks in parallel (map) then aggregate (reduce)."""
        chunks = self._chunk_context(context, chunk_size)
        trace.append({"step": "chunk", "num_chunks": len(chunks)})

        # Use async client for faster processing
        if self.config.use_async:
            return self._process_map_reduce_async(query, chunks, trace)

        # Fallback to thread pool with DSPy
        return self._process_map_reduce_threads(query, chunks, trace)

    def _process_map_reduce_async(
        self,
        query: str,
        chunks: list[str],
        trace: list[dict[str, Any]],
    ) -> str:
        """Process using async HTTP client - fastest method."""
        from .async_client import aggregate_answers_async, analyze_chunks_async

        # Get model name without openrouter/ prefix for direct API
        model = self.config.model
        if model.startswith("openrouter/"):
            model = model[len("openrouter/") :]

        # Run async analysis - handle both sync and async contexts
        async def run_analysis() -> list[dict[str, Any]]:
            return await analyze_chunks_async(
                query=query,
                chunks=chunks,
                model=model,
                max_concurrent=self.config.parallel_chunks,
                disable_thinking=self.config.disable_thinking,
                enable_cache=self.config.enable_cache,
            )

        async def run_aggregation(partial: list[str]) -> str:
            return await aggregate_answers_async(
                query,
                partial,
                model,
                disable_thinking=self.config.disable_thinking,
                enable_cache=self.config.enable_cache,
            )

        def run_in_new_loop(coro: Any) -> Any:
            """Run coroutine in a new event loop (for sync context)."""
            return asyncio.run(coro)

        def run_coro(coro: Any) -> Any:
            """Run coroutine, handling both sync and async contexts."""
            try:
                asyncio.get_running_loop()
                # We're in an async context - run in thread to avoid blocking
                # Reuse executor to prevent thread exhaustion (thread-safe init)
                with self._executor_lock:
                    if self._executor is None:
                        from concurrent.futures import ThreadPoolExecutor
                        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rlm_async")
                future = self._executor.submit(run_in_new_loop, coro)
                return future.result()
            except RuntimeError:
                # No running loop - safe to use asyncio.run()
                return asyncio.run(coro)

        results = run_coro(run_analysis())

        # Track token usage from results
        for r in results:
            usage = r.get("usage", {})
            self._tokens_used += usage.get("total_tokens", 0)
            # Estimate cost (rough approximation - actual pricing varies by model)
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            # Using generic pricing estimate ($0.50/$1.50 per 1M tokens)
            self._cost_spent += (input_tokens * 0.5 + output_tokens * 1.5) / 1_000_000

        # Collect partial answers
        partial_answers = []
        for r in results:
            if r["confidence"] != "none" and r["relevant_info"]:
                partial_answers.append(r["relevant_info"])
                trace.append(
                    {
                        "step": "analyze_chunk",
                        "chunk_index": r["index"],
                        "confidence": r["confidence"],
                        "latency_ms": r.get("latency_ms", 0),
                    }
                )

        if not partial_answers:
            return "No relevant information found in the context."

        # Aggregate
        final_answer = run_coro(run_aggregation(partial_answers))

        trace.append(
            {
                "step": "aggregate",
                "num_partial_answers": len(partial_answers),
            }
        )

        return final_answer

    def _process_map_reduce_threads(
        self,
        query: str,
        chunks: list[str],
        trace: list[dict[str, Any]],
    ) -> str:
        """Process using thread pool with DSPy - fallback method."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Map: analyze each chunk IN PARALLEL
        partial_answers: list[tuple[int, str, str]] = []  # (index, info, confidence)

        def analyze_chunk(i: int, chunk: str) -> tuple[int, str, str]:
            """Analyze a single chunk - runs in thread pool."""
            result = self.chunk_analyzer(
                query=query,
                chunk=chunk,
                chunk_index=i,
                total_chunks=len(chunks),
            )
            return (i, result.relevant_info, result.confidence)

        # Process chunks in parallel with thread pool
        with ThreadPoolExecutor(max_workers=self.config.parallel_chunks) as executor:
            futures = {executor.submit(analyze_chunk, i, chunk): i for i, chunk in enumerate(chunks)}

            for future in as_completed(futures):
                self._check_limits()
                i, info, confidence = future.result()
                if confidence != "none":
                    partial_answers.append((i, info, confidence))
                    trace.append(
                        {
                            "step": "analyze_chunk",
                            "chunk_index": i,
                            "confidence": confidence,
                        }
                    )

        # Sort by chunk index to maintain order
        partial_answers.sort(key=lambda x: x[0])

        # Reduce: aggregate answers
        if not partial_answers:
            return "No relevant information found in the context."

        self._check_limits()
        # Extract just the info strings from tuples for aggregation
        answer_texts = [info for _, info, _ in partial_answers]
        aggregated = self.aggregator(
            query=query,
            partial_answers=answer_texts,
        )

        trace.append(
            {
                "step": "aggregate",
                "num_partial_answers": len(partial_answers),
                "sources_used": aggregated.sources_used,
            }
        )

        return aggregated.final_answer

    def _process_iterative(
        self,
        query: str,
        context: str,
        chunk_size: int,
        trace: list[dict[str, Any]],
    ) -> str:
        """Process chunks iteratively, building up answer."""
        chunks = self._chunk_context(context, chunk_size)
        trace.append({"step": "chunk", "num_chunks": len(chunks)})

        buffer = ""
        for i, chunk in enumerate(chunks):
            self._check_limits()
            result = self.chunk_analyzer(
                query=f"{query}\n\nPrevious findings: {buffer}" if buffer else query,
                chunk=chunk,
                chunk_index=i,
                total_chunks=len(chunks),
            )
            if result.confidence != "none":
                buffer = f"{buffer}\n{result.relevant_info}" if buffer else result.relevant_info
                trace.append(
                    {
                        "step": "iterate",
                        "chunk_index": i,
                        "confidence": result.confidence,
                    }
                )

        return buffer or "No relevant information found."

    def _process_hierarchical(
        self,
        query: str,
        context: str,
        chunk_size: int,
        depth: int,
        trace: list[dict[str, Any]],
    ) -> str:
        """Process hierarchically with recursive sub-queries."""
        if depth >= self.config.max_depth:
            return self._process_map_reduce(query, context, chunk_size, trace)

        chunks = self._chunk_context(context, chunk_size)
        trace.append({"step": "hierarchical", "depth": depth, "num_chunks": len(chunks)})

        # Handle empty chunks edge case
        if not chunks:
            return "No content to analyze."

        # Create sub-RLM for each major section
        # Use REMAINING budget/timeout, not original, to prevent overspending
        remaining_budget = max(0.01, self.config.max_budget - self._cost_spent)
        remaining_timeout = max(10.0, self.config.max_timeout - (time.time() - (self._start_time or time.time())))
        sub_config = RLMConfig(
            model=self.config.sub_model,
            max_budget=remaining_budget / len(chunks),
            max_timeout=remaining_timeout / len(chunks),
            max_depth=self.config.max_depth,
        )

        partial_answers = []
        for i, chunk in enumerate(chunks):
            self._check_limits()
            if len(chunk) > chunk_size:
                # Recurse for large chunks
                with RLM(config=sub_config) as sub_rlm:
                    sub_result = sub_rlm.query(query, chunk, depth=depth + 1)
                    if sub_result.success:
                        partial_answers.append(sub_result.answer)
            else:
                # Process small chunks directly
                result = self.chunk_analyzer(
                    query=query,
                    chunk=chunk,
                    chunk_index=i,
                    total_chunks=len(chunks),
                )
                if result.confidence != "none":
                    partial_answers.append(result.relevant_info)

        # Aggregate
        if not partial_answers:
            return "No relevant information found."

        self._check_limits()
        aggregated = self.aggregator(
            query=query,
            partial_answers=partial_answers,
        )

        return aggregated.final_answer

    async def query_async(
        self,
        query: str,
        context: str,
    ) -> RLMResult:
        """Async version of query for concurrent processing."""
        return await asyncio.to_thread(self.query, query, context)

    def close(self) -> None:
        """Clean up resources (thread pool, etc.)."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __enter__(self) -> "RLM":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - clean up resources."""
        self.close()

    def __del__(self) -> None:
        """Destructor - clean up resources."""
        self.close()


# Custom exceptions
class BudgetExceededError(Exception):
    def __init__(self, spent: float, budget: float):
        self.spent = spent
        self.budget = budget
        super().__init__(f"Budget exceeded: ${spent:.4f} of ${budget:.4f}")


class TimeoutExceededError(Exception):
    def __init__(self, elapsed: float, timeout: float):
        self.elapsed = elapsed
        self.timeout = timeout
        self.partial_answer: str | None = None
        super().__init__(f"Timeout: {elapsed:.1f}s of {timeout:.1f}s")


class TokenLimitExceededError(Exception):
    def __init__(self, tokens: int, limit: int):
        self.tokens = tokens
        self.limit = limit
        self.partial_answer: str | None = None
        super().__init__(f"Token limit: {tokens:,} of {limit:,}")

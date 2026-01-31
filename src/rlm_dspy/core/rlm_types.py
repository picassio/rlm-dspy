"""RLM type definitions - config, result, and callbacks."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, TypeVar

_logger = logging.getLogger(__name__)

T = TypeVar("T", int, float, bool, str)


def _env(key: str, default: str) -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)


def _env_get(key: str, default: T, cast: type[T] | None = None) -> T:
    """Get environment variable with type casting and default."""
    val = os.environ.get(key)
    if val is None:
        return default

    target_type = cast or type(default)
    if target_type is bool:
        return val.lower() in ("true", "1", "yes", "on")  # type: ignore

    try:
        return target_type(val)  # type: ignore
    except ValueError:
        _logger.warning("Invalid %s for %s, using default", target_type.__name__, key)
        return default


def _get_user_config_default(key: str, default: Any) -> Any:
    """Get default from user config."""
    try:
        from .user_config import get_config_value
        return get_config_value(key, default)
    except ImportError:
        return default


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
    "ollama/": "",
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
        _logger.debug("user_config module not available")


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


@dataclass
class RLMConfig:
    """Configuration for RLM execution."""

    model: str = field(default_factory=lambda: _env(
        "RLM_MODEL", _get_user_config_default("model", "openai/gpt-4o-mini")))
    sub_model: str = field(default_factory=lambda: _env(
        "RLM_SUB_MODEL", _get_user_config_default("sub_model",
            _env("RLM_MODEL", _get_user_config_default("model", "openai/gpt-4o-mini")))))
    api_base: str | None = field(default_factory=lambda: os.environ.get("RLM_API_BASE"))
    api_key: str | None = field(default_factory=_resolve_api_key)

    max_iterations: int = field(default_factory=lambda: _env_get(
        "RLM_MAX_ITERATIONS", _get_user_config_default("max_iterations", 20)))
    max_llm_calls: int = field(default_factory=lambda: _env_get(
        "RLM_MAX_LLM_CALLS", _get_user_config_default("max_llm_calls", 50)))
    max_output_chars: int = field(default_factory=lambda: _env_get(
        "RLM_MAX_OUTPUT_CHARS", _get_user_config_default("max_output_chars", 100_000)))
    verbose: bool = field(default_factory=lambda: _env_get("RLM_VERBOSE", False))

    max_budget: float = field(default_factory=lambda: _env_get(
        "RLM_MAX_BUDGET", _get_user_config_default("max_budget", 1.0)))
    max_timeout: float = field(default_factory=lambda: _env_get(
        "RLM_MAX_TIMEOUT", _get_user_config_default("max_timeout", 300.0)))
    max_workers: int = field(default_factory=lambda: _env_get(
        "RLM_MAX_WORKERS", _get_user_config_default("max_workers", 8)))
    validate: bool = field(default_factory=lambda: _env_get(
        "RLM_VALIDATE", _get_user_config_default("validate", True)))
    enable_logging: bool = field(default_factory=lambda: _env_get(
        "RLM_ENABLE_LOGGING", _get_user_config_default("enable_logging", False)))
    enable_metrics: bool = field(default_factory=lambda: _env_get(
        "RLM_ENABLE_METRICS", _get_user_config_default("enable_metrics", False)))

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_iterations < 20 or self.max_iterations > 100:
            raise ValueError(f"max_iterations must be between 20 and 100, got {self.max_iterations}")
        if self.max_llm_calls < 1 or self.max_llm_calls > 500:
            raise ValueError(f"max_llm_calls must be between 1 and 500, got {self.max_llm_calls}")
        if self.max_timeout < 0 or self.max_timeout > 3600:
            raise ValueError(f"max_timeout must be between 0 and 3600, got {self.max_timeout}")
        if self.max_budget < 0 or self.max_budget > 100:
            raise ValueError(f"max_budget must be between 0 and 100, got {self.max_budget}")
        if self.max_workers < 1 or self.max_workers > 32:
            raise ValueError(f"max_workers must be between 1 and 32, got {self.max_workers}")

    def __repr__(self) -> str:
        key_display = "***" if self.api_key else None
        return f"RLMConfig(model={self.model!r}, api_key={key_display!r}, max_iterations={self.max_iterations}, ...)"


@dataclass
class RLMResult:
    """Result from RLM execution."""

    answer: str
    success: bool
    total_tokens: int = 0
    total_cost: float = 0.0
    elapsed_time: float = 0.0
    iterations: int = 0
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    final_reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    token_stats: Any = None
    outputs: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        """Allow accessing output fields as attributes."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        outputs = object.__getattribute__(self, "outputs")
        if name in outputs:
            return outputs[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class ProgressCallback:
    """Callback for RLM progress updates."""

    def on_start(self, query: str, context_tokens: int) -> None:
        pass

    def on_iteration(self, iteration: int, max_iterations: int) -> None:
        pass

    def on_lm_call(self, call_type: str, inputs: dict | None = None) -> None:
        pass

    def on_tool_use(self, tool_name: str, args: dict | None = None) -> None:
        pass

    def on_complete(self, result: RLMResult) -> None:
        pass

    def on_error(self, error: Exception) -> None:
        pass


class DspyProgressCallback:
    """DSPy-compatible callback that wraps our ProgressCallback."""

    def __init__(self, progress_callback: ProgressCallback | None = None):
        self.progress = progress_callback
        self.lm_calls = self.tool_calls = self.module_calls = 0

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self.lm_calls += 1
        if self.progress:
            self.progress.on_lm_call("lm", {"call_id": call_id, "count": self.lm_calls})

    def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        pass

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self.tool_calls += 1
        if self.progress:
            self.progress.on_lm_call("tool", {"tool": inputs.get("tool_name", "unknown")})

    def on_tool_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        pass

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self.module_calls += 1

    def on_module_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None):
        pass

    def on_adapter_format_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        pass

    def on_adapter_format_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        pass

    def on_adapter_parse_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        pass

    def on_adapter_parse_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        pass

    def get_stats(self) -> dict[str, int]:
        return {"lm_calls": self.lm_calls, "tool_calls": self.tool_calls, "module_calls": self.module_calls}


def sanitize_trajectory(trajectory: list, extra_secrets: list[str] | None = None) -> list:
    """Sanitize all strings in a trajectory list."""
    from .secrets import sanitize_value
    if not trajectory:
        return trajectory
    return [sanitize_value(item, extra_secrets) for item in trajectory]


def extract_trace_metadata(trajectory: list) -> dict[str, Any]:
    """Extract metadata from trajectory for trace collection."""
    reasoning_steps, code_blocks, outputs, tools_used = [], [], [], set()
    tool_patterns = {"read_file", "ripgrep", "find_files", "semantic_search", "run_shell_command"}

    for item in trajectory:
        if isinstance(item, dict):
            if reasoning := item.get("reasoning") or item.get("thought"):
                reasoning_steps.append(str(reasoning))
            if code := item.get("code") or item.get("action"):
                code_str = str(code)
                code_blocks.append(code_str)
                tools_used.update(t for t in tool_patterns if t in code_str)
            if output := item.get("output") or item.get("observation"):
                outputs.append(str(output))

    return {"reasoning_steps": reasoning_steps, "code_blocks": code_blocks,
            "outputs": outputs, "tools_used": sorted(tools_used)}


__all__ = [
    "RLMConfig", "RLMResult", "ProgressCallback", "DspyProgressCallback",
    "PROVIDER_API_KEYS", "get_provider_env_var",
    "sanitize_trajectory", "extract_trace_metadata",
]

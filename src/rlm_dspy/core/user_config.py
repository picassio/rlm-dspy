"""User configuration management for RLM-DSPy.

Provides persistent configuration via ~/.rlm/config.yaml
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config directory
CONFIG_DIR = Path.home() / ".rlm"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Default configuration with all supported options
DEFAULT_CONFIG = {
    # Model settings
    "model": "openai/gpt-4o-mini",
    "sub_model": None,  # Defaults to model if not set

    # Execution limits
    "max_iterations": 20,
    "max_llm_calls": 50,
    "max_output_chars": 100_000,

    # Parallelism
    "max_workers": 8,

    # Budget/safety limits
    "max_budget": 1.0,
    "max_timeout": 300,

    # Embedding settings (for semantic search)
    "embedding_model": "openai/text-embedding-3-small",
    "local_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_batch_size": 100,

    # Vector index settings
    "index_dir": "~/.rlm/indexes",
    "use_faiss": True,
    "faiss_threshold": 5000,
    "auto_update_index": True,
    "index_cache_ttl": 3600,

    # API key location
    "env_file": None,

    # Optimization settings
    "optimization": {
        "enabled": True,                # Enable auto-optimization
        "optimizer": "gepa",            # Optimizer type: gepa (recommended), simba (legacy)
        "model": None,                  # null = use default model
        "teacher_model": None,          # Teacher/reflection model for GEPA (null = use model)
        "min_new_traces": 50,           # Traces needed before optimizing
        "min_hours_between": 24,        # Minimum hours between optimizations
        "max_budget": 0.50,             # Max cost per optimization run
        "run_in_background": True,      # Run optimization in background thread
    },
}


# Optimization config defaults (for easy access)
DEFAULT_OPTIMIZATION_CONFIG = DEFAULT_CONFIG["optimization"]


@dataclass
class OptimizationConfig:
    """Configuration for auto-optimization."""

    enabled: bool = True
    optimizer: str = "gepa"  # "gepa" (recommended), "simba" (legacy)
    model: str | None = None  # None = use default model from config
    teacher_model: str | None = None  # Teacher/reflection model for GEPA (None = use model)
    min_new_traces: int = 50
    min_hours_between: int = 24
    max_budget: float = 0.50
    run_in_background: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationConfig":
        """Create from dictionary."""
        defaults = DEFAULT_OPTIMIZATION_CONFIG
        return cls(
            enabled=data.get("enabled", defaults["enabled"]),
            optimizer=data.get("optimizer", defaults["optimizer"]),
            model=data.get("model", defaults["model"]),
            teacher_model=data.get("teacher_model", defaults.get("teacher_model")),
            min_new_traces=data.get("min_new_traces", defaults["min_new_traces"]),
            min_hours_between=data.get("min_hours_between", defaults["min_hours_between"]),
            max_budget=data.get("max_budget", defaults["max_budget"]),
            run_in_background=data.get("run_in_background", defaults["run_in_background"]),
        )

    @classmethod
    def from_user_config(cls) -> "OptimizationConfig":
        """Load from user config file."""
        config = load_config()
        opt_data = config.get("optimization", {})
        return cls.from_dict(opt_data)

    def get_model(self, default_model: str) -> str:
        """Get the model to use for optimization.
        
        Args:
            default_model: The default model from main config
            
        Returns:
            The optimization model (self.model if set, else default_model)
        """
        return self.model if self.model else default_model

    def get_teacher_model(self, default_model: str) -> str:
        """Get the teacher/reflection model for GEPA.
        
        Args:
            default_model: The default model from main config
            
        Returns:
            The teacher model (self.teacher_model if set, else model, else default_model)
        """
        if self.teacher_model:
            return self.teacher_model
        return self.get_model(default_model)


# Template for config file with comments
CONFIG_TEMPLATE = """# RLM-DSPy Configuration
# Priority: CLI args > env vars > this file > defaults
# Docs: https://github.com/picassio/rlm-dspy

# ============================================================================
# Model Settings
# ============================================================================
# Model format: provider/model-name
# Examples: openai/gpt-4o, anthropic/claude-sonnet-4, deepseek/deepseek-chat
model: {model}

# Sub-model for llm_query() calls (defaults to main model if not set)
# Tip: Use same model as main to reduce hallucinations
sub_model: {sub_model}

# ============================================================================
# Embedding Settings (for semantic search)
# ============================================================================
# Embedding model (via litellm)
# Options: openai/text-embedding-3-small, openai/text-embedding-3-large,
#          cohere/embed-english-v3.0, voyage/voyage-3,
#          together_ai/togethercomputer/m2-bert-80M-8k-retrieval
#          or "local" to use sentence-transformers
embedding_model: {embedding_model}

# Local embedding model (used when embedding_model: local)
# Requires: pip install sentence-transformers
local_embedding_model: {local_embedding_model}

# Embedding batch size (default: 100)
embedding_batch_size: {embedding_batch_size}

# ============================================================================
# Vector Index Settings (for semantic search)
# ============================================================================
# Index storage directory
index_dir: {index_dir}

# Use FAISS for large indexes (requires: pip install faiss-cpu)
# If false, uses brute-force numpy search (slower but no dependencies)
use_faiss: {use_faiss}

# Threshold for switching to FAISS (number of documents)
faiss_threshold: {faiss_threshold}

# Auto-update index when files change
auto_update_index: {auto_update_index}

# Index cache TTL in seconds (0 = no expiry)
index_cache_ttl: {index_cache_ttl}

# ============================================================================
# Execution Limits
# ============================================================================
# Higher values = more thorough exploration, but slower and more expensive

# Max REPL iterations (default: 20)
# Tip: Use 25-30 for complex queries to avoid forced completion
max_iterations: {max_iterations}

# Max sub-LLM calls per query (default: 50)
max_llm_calls: {max_llm_calls}

# Max chars in REPL output (default: 100000)
max_output_chars: {max_output_chars}

# ============================================================================
# Parallelism
# ============================================================================
# Workers for batch operations (default: 8)
max_workers: {max_workers}

# ============================================================================
# Budget/Safety Limits
# ============================================================================
# Max cost per query in USD (default: 1.0)
max_budget: {max_budget}

# Max execution time in seconds (default: 300)
max_timeout: {max_timeout}

# ============================================================================
# API Key Location
# ============================================================================
# Path to .env file with API keys (optional)
# If not set, uses environment variables directly
env_file: {env_file}

# ============================================================================
# Auto-Optimization Settings
# ============================================================================
# RLM can automatically optimize itself using collected traces.
# When enough traces are collected, SIMBA runs in background to improve prompts.

optimization:
  # Enable/disable auto-optimization
  enabled: {opt_enabled}

  # Optimizer type: simba (future: mipro, copro, gepa)
  optimizer: {opt_optimizer}

  # Model for optimization (null = use default model above)
  # Tip: Use same model for best results, or cheaper model to save costs
  model: {opt_model}

  # Minimum new traces before triggering optimization
  min_new_traces: {opt_min_new_traces}

  # Minimum hours between optimization runs
  min_hours_between: {opt_min_hours_between}

  # Maximum budget per optimization run in USD
  max_budget: {opt_max_budget}

  # Run optimization in background (recommended)
  run_in_background: {opt_run_in_background}
"""


def ensure_config_dir() -> Path:
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def load_config() -> dict[str, Any]:
    """Load user configuration from file.

    Returns default config if file doesn't exist.
    """
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_FILE) as f:
            user_config = yaml.safe_load(f) or {}
        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config
    except FileNotFoundError:
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.warning("Failed to load config from %s: %s", CONFIG_FILE, e)
        return DEFAULT_CONFIG.copy()


def save_config(config: dict[str, Any], use_template: bool = True) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary
        use_template: If True, saves with full template and comments
    """
    ensure_config_dir()

    if use_template:
        # Merge with defaults to ensure all fields are present
        full_config = DEFAULT_CONFIG.copy()
        full_config.update(config)

        # Format values for YAML
        def fmt(val):
            if val is None:
                return "null"
            elif isinstance(val, bool):
                return "true" if val else "false"
            elif isinstance(val, str):
                return val
            else:
                return str(val)

        # Get optimization config
        opt_config = full_config.get("optimization", DEFAULT_OPTIMIZATION_CONFIG)

        content = CONFIG_TEMPLATE.format(
            model=fmt(full_config.get("model")),
            sub_model=fmt(full_config.get("sub_model")),
            # Embedding settings
            embedding_model=fmt(full_config.get("embedding_model")),
            local_embedding_model=fmt(full_config.get("local_embedding_model")),
            embedding_batch_size=fmt(full_config.get("embedding_batch_size")),
            # Vector index settings
            index_dir=fmt(full_config.get("index_dir")),
            use_faiss=fmt(full_config.get("use_faiss")),
            faiss_threshold=fmt(full_config.get("faiss_threshold")),
            auto_update_index=fmt(full_config.get("auto_update_index")),
            index_cache_ttl=fmt(full_config.get("index_cache_ttl")),
            # Execution limits
            max_iterations=fmt(full_config.get("max_iterations")),
            max_llm_calls=fmt(full_config.get("max_llm_calls")),
            max_output_chars=fmt(full_config.get("max_output_chars")),
            max_workers=fmt(full_config.get("max_workers")),
            max_budget=fmt(full_config.get("max_budget")),
            max_timeout=fmt(full_config.get("max_timeout")),
            env_file=fmt(full_config.get("env_file")),
            # Optimization settings
            opt_enabled=fmt(opt_config.get("enabled", True)),
            opt_optimizer=fmt(opt_config.get("optimizer", "gepa")),
            opt_model=fmt(opt_config.get("model")),
            opt_min_new_traces=fmt(opt_config.get("min_new_traces", 50)),
            opt_min_hours_between=fmt(opt_config.get("min_hours_between", 24)),
            opt_max_budget=fmt(opt_config.get("max_budget", 0.50)),
            opt_run_in_background=fmt(opt_config.get("run_in_background", True)),
        )

        _atomic_write(CONFIG_FILE, content)
    else:
        # Simple mode: only save non-default values
        to_save = {}
        for key, value in config.items():
            if key in DEFAULT_CONFIG and value != DEFAULT_CONFIG[key]:
                to_save[key] = value
            elif key not in DEFAULT_CONFIG:
                to_save[key] = value

        content = yaml.dump(to_save, default_flow_style=False, sort_keys=False)
        _atomic_write(CONFIG_FILE, content)


def _atomic_write(path: Path, content: str) -> None:
    """Write content to file atomically using temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=path.parent,
            suffix='.tmp',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(content)
            temp_path = Path(f.name)

        temp_path.replace(path)
    except Exception:
        if temp_path and temp_path.exists():
            temp_path.unlink()
        raise


def load_env_file(env_path: str | Path | None = None) -> dict[str, str]:
    """Load environment variables from .env file.

    Args:
        env_path: Path to .env file. If None, uses config's env_file setting.

    Returns:
        Dictionary of loaded environment variables.
    """
    if env_path is None:
        config = load_config()
        env_path = config.get("env_file")

    if not env_path:
        return {}

    env_path = Path(env_path).expanduser()
    if not env_path.exists():
        return {}

    loaded = {}
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)
                        loaded[key] = value
    except FileNotFoundError:
        logger.debug("Env file not found: %s", env_path)
    except Exception as e:
        logger.warning("Failed to load env file %s: %s", env_path, e)

    return loaded


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a single config value."""
    config = load_config()
    return config.get(key, default)


def set_config_value(key: str, value: Any) -> None:
    """Set a single config value."""
    config = load_config()
    config[key] = value
    save_config(config)


def get_api_key_for_model(model: str) -> str | None:
    """Get API key for a model, checking config's env_file first."""
    from .rlm import get_provider_env_var

    # Load env file if configured
    load_env_file()

    # Check RLM_API_KEY first
    if key := os.environ.get("RLM_API_KEY"):
        return key

    # Check provider-specific key
    if env_var := get_provider_env_var(model):
        if key := os.environ.get(env_var):
            return key

    return None


def is_configured() -> bool:
    """Check if RLM is configured with necessary settings."""
    config = load_config()
    model = config.get("model", "")

    # Load env file if configured
    load_env_file()

    # Check if we have an API key for the default model
    return get_api_key_for_model(model) is not None


def get_config_status() -> dict[str, Any]:
    """Get current configuration status for display."""
    config = load_config()
    model = config.get("model", "openai/gpt-4o-mini")

    # Load env file
    env_file = config.get("env_file")
    env_loaded = load_env_file(env_file) if env_file else {}

    # Check API key
    from .rlm import get_provider_env_var
    env_var = get_provider_env_var(model)
    api_key = get_api_key_for_model(model)

    return {
        "config_file": str(CONFIG_FILE) if CONFIG_FILE.exists() else None,
        "config": config,
        "env_file": env_file,
        "env_file_exists": Path(env_file).expanduser().exists() if env_file else False,
        "env_vars_loaded": list(env_loaded.keys()),
        "model": model,
        "api_key_env_var": env_var,
        "api_key_found": api_key is not None,
        "is_configured": is_configured(),
    }

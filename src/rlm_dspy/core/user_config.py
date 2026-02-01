"""User configuration management for RLM-DSPy.

Provides persistent configuration via ~/.rlm/config.yaml
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Thread lock for config file operations
_config_lock = threading.Lock()

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
        "fast": True,                   # Use fast proxy mode (50x faster, recommended)
        "threads": 2,                   # Parallel threads
        "min_new_traces": 50,           # Traces needed before optimizing
        "min_hours_between": 24,        # Minimum hours between optimizations
        "max_budget": 0.50,             # Max cost per optimization run
        "run_in_background": True,      # Run optimization in background thread
        
        # GEPA-specific settings
        "gepa": {
            "teacher_model": None,      # Teacher/reflection model (null = use optimization.model)
            "max_evals": None,          # Max evaluations (None = auto based on 'auto' preset)
            "auto": "light",            # Budget preset: light, medium, heavy
        },
        
        # SIMBA-specific settings
        "simba": {
            "steps": 1,                 # Optimization steps
            "candidates": 2,            # Candidates per step
            "batch_size": 8,            # Batch size
        },
    },
}


# Optimization config defaults (for easy access)
DEFAULT_OPTIMIZATION_CONFIG = DEFAULT_CONFIG["optimization"]


@dataclass
class GEPASettings:
    """GEPA-specific settings."""
    teacher_model: str | None = None  # Teacher/reflection model (None = use optimization.model)
    max_evals: int | None = None  # Max evaluations (None = auto)
    auto: str = "light"  # Budget preset: light, medium, heavy


@dataclass
class SIMBASettings:
    """SIMBA-specific settings."""
    steps: int = 1  # Optimization steps
    candidates: int = 2  # Candidates per step
    batch_size: int = 8  # Batch size


@dataclass
class OptimizationConfig:
    """Configuration for auto-optimization."""

    # General settings
    enabled: bool = True
    optimizer: str = "gepa"  # "gepa" (recommended), "simba" (legacy)
    model: str | None = None  # None = use default model from config
    fast: bool = True  # Use fast proxy mode (50x faster)
    threads: int = 2  # Parallel threads
    min_new_traces: int = 50
    min_hours_between: int = 24
    max_budget: float = 0.50
    run_in_background: bool = True
    
    # Optimizer-specific settings
    gepa: GEPASettings = None
    simba: SIMBASettings = None
    
    def __post_init__(self):
        if self.gepa is None:
            self.gepa = GEPASettings()
        if self.simba is None:
            self.simba = SIMBASettings()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationConfig":
        """Create from dictionary."""
        defaults = DEFAULT_OPTIMIZATION_CONFIG
        gepa_defaults = defaults.get("gepa", {})
        simba_defaults = defaults.get("simba", {})
        gepa_data = data.get("gepa", {})
        simba_data = data.get("simba", {})
        
        return cls(
            enabled=data.get("enabled", defaults["enabled"]),
            optimizer=data.get("optimizer", defaults["optimizer"]),
            model=data.get("model", defaults["model"]),
            fast=data.get("fast", defaults.get("fast", True)),
            threads=data.get("threads", defaults.get("threads", 2)),
            min_new_traces=data.get("min_new_traces", defaults["min_new_traces"]),
            min_hours_between=data.get("min_hours_between", defaults["min_hours_between"]),
            max_budget=data.get("max_budget", defaults["max_budget"]),
            run_in_background=data.get("run_in_background", defaults["run_in_background"]),
            gepa=GEPASettings(
                teacher_model=gepa_data.get("teacher_model", gepa_defaults.get("teacher_model")),
                max_evals=gepa_data.get("max_evals", gepa_defaults.get("max_evals")),
                auto=gepa_data.get("auto", gepa_defaults.get("auto", "light")),
            ),
            simba=SIMBASettings(
                steps=simba_data.get("steps", simba_defaults.get("steps", 1)),
                candidates=simba_data.get("candidates", simba_defaults.get("candidates", 2)),
                batch_size=simba_data.get("batch_size", simba_defaults.get("batch_size", 8)),
            ),
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
            The teacher model (gepa.teacher_model if set, else model, else default_model)
        """
        if self.gepa and self.gepa.teacher_model:
            return self.gepa.teacher_model
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
# GEPA (recommended) or SIMBA can run to improve prompts.

optimization:
  # Enable/disable auto-optimization
  enabled: {opt_enabled}

  # Optimizer type: gepa (recommended), simba (legacy)
  # GEPA evolves instruction text, SIMBA selects demos
  optimizer: {opt_optimizer}

  # Model for optimization (null = use default model above)
  # Tip: Use same model as main for best results
  model: {opt_model}

  # Fast proxy mode (50x faster, recommended)
  # Uses lightweight proxy instead of full RLM for evaluation
  fast: {opt_fast}

  # Parallel threads for optimization
  threads: {opt_threads}

  # Auto-optimization triggers
  min_new_traces: {opt_min_new_traces}
  min_hours_between: {opt_min_hours_between}
  max_budget: {opt_max_budget}

  # Run optimization in background (recommended)
  run_in_background: {opt_run_in_background}

  # GEPA-specific settings
  gepa:
    # Teacher/reflection model for analyzing failures and proposing improvements
    # Tip: Use a strong model like openai/gpt-4o or anthropic/claude-3-5-sonnet
    teacher_model: {opt_gepa_teacher_model}
    # Max evaluations (null = use 'auto' preset)
    max_evals: {opt_gepa_max_evals}
    # Budget preset: light (~50-100 evals), medium (~200-400), heavy (~500+)
    auto: {opt_gepa_auto}

  # SIMBA-specific settings
  simba:
    steps: {opt_simba_steps}           # Optimization iterations
    candidates: {opt_simba_candidates} # Candidates per step
    batch_size: {opt_simba_batch_size} # Batch size
  run_in_background: {opt_run_in_background}
"""


def ensure_config_dir() -> Path:
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitize configuration values.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Validated configuration with safe defaults for invalid values
    """
    validated = config.copy()
    
    # Validate numeric bounds
    if "max_iterations" in validated:
        val = validated["max_iterations"]
        if not isinstance(val, int) or val < 1 or val > 1000:
            logger.warning("Invalid max_iterations %s, using default", val)
            validated["max_iterations"] = DEFAULT_CONFIG.get("max_iterations", 30)
    
    if "max_budget" in validated:
        val = validated["max_budget"]
        if not isinstance(val, (int, float)) or val < 0 or val > 1000:
            logger.warning("Invalid max_budget %s, using default", val)
            validated["max_budget"] = DEFAULT_CONFIG.get("max_budget", 1.0)
    
    if "max_timeout" in validated:
        val = validated["max_timeout"]
        if not isinstance(val, (int, float)) or val < 1 or val > 86400:  # Max 24 hours
            logger.warning("Invalid max_timeout %s, using default", val)
            validated["max_timeout"] = DEFAULT_CONFIG.get("max_timeout", 300)
    
    if "max_workers" in validated:
        val = validated["max_workers"]
        if not isinstance(val, int) or val < 1 or val > 64:
            logger.warning("Invalid max_workers %s, using default", val)
            validated["max_workers"] = DEFAULT_CONFIG.get("max_workers", 8)
    
    # Validate optimization sub-config
    if "optimization" in validated and isinstance(validated["optimization"], dict):
        opt = validated["optimization"]
        if "threads" in opt:
            val = opt["threads"]
            if not isinstance(val, int) or val < 1 or val > 32:
                logger.warning("Invalid optimization.threads %s, using default", val)
                opt["threads"] = 2
        if "max_budget" in opt:
            val = opt["max_budget"]
            if not isinstance(val, (int, float)) or val < 0 or val > 100:
                logger.warning("Invalid optimization.max_budget %s, using default", val)
                opt["max_budget"] = 0.5
    
    return validated


def load_config() -> dict[str, Any]:
    """Load user configuration from file.

    Returns default config if file doesn't exist.
    Thread-safe via _config_lock.
    """
    with _config_lock:
        if not CONFIG_FILE.exists():
            return DEFAULT_CONFIG.copy()

        try:
            with open(CONFIG_FILE) as f:
                user_config = yaml.safe_load(f) or {}
            # Merge with defaults
            config = DEFAULT_CONFIG.copy()
            config.update(user_config)
            # Validate loaded config
            return _validate_config(config)
        except FileNotFoundError:
            return DEFAULT_CONFIG.copy()
    except (yaml.YAMLError, PermissionError, UnicodeDecodeError) as e:
        logger.warning("Failed to load config from %s: %s", CONFIG_FILE, e)
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.warning("Unexpected error loading config from %s: %s", CONFIG_FILE, e)
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
            opt_fast=fmt(opt_config.get("fast", True)),
            opt_threads=fmt(opt_config.get("threads", 2)),
            opt_min_new_traces=fmt(opt_config.get("min_new_traces", 50)),
            opt_min_hours_between=fmt(opt_config.get("min_hours_between", 24)),
            opt_max_budget=fmt(opt_config.get("max_budget", 0.50)),
            opt_run_in_background=fmt(opt_config.get("run_in_background", True)),
            # GEPA settings
            opt_gepa_teacher_model=fmt(opt_config.get("gepa", {}).get("teacher_model")),
            opt_gepa_max_evals=fmt(opt_config.get("gepa", {}).get("max_evals")),
            opt_gepa_auto=fmt(opt_config.get("gepa", {}).get("auto", "light")),
            # SIMBA settings
            opt_simba_steps=fmt(opt_config.get("simba", {}).get("steps", 1)),
            opt_simba_candidates=fmt(opt_config.get("simba", {}).get("candidates", 2)),
            opt_simba_batch_size=fmt(opt_config.get("simba", {}).get("batch_size", 8)),
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

    env_path = Path(env_path).expanduser().resolve()
    
    # Basic path safety check - must be under home or /etc
    home = Path.home().resolve()
    if not (str(env_path).startswith(str(home)) or str(env_path).startswith("/etc/")):
        logger.warning("Refusing to load env file outside home directory: %s", env_path)
        return {}
    
    if not env_path.exists():
        return {}

    loaded = {}
    try:
        with open(env_path, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Validate key is a valid env var name
                    if key and value and key.replace("_", "").isalnum():
                        os.environ.setdefault(key, value)
                        loaded[key] = value
                    elif key:
                        logger.debug("Skipping invalid env var name at line %d: %s", line_num, key)
    except FileNotFoundError:
        logger.debug("Env file not found: %s", env_path)
    except PermissionError:
        logger.warning("Permission denied reading env file: %s", env_path)
    except UnicodeDecodeError as e:
        logger.warning("Encoding error in env file %s: %s", env_path, e)
    except Exception as e:
        logger.warning("Failed to load env file %s: %s", env_path, e)

    return loaded


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a single config value. Thread-safe."""
    config = load_config()
    return config.get(key, default)


def set_config_value(key: str, value: Any) -> None:
    """Set a single config value. Thread-safe via _config_lock."""
    with _config_lock:
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

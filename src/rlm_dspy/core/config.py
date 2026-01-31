"""Unified configuration management for RLM-DSPy.

Provides a single source of truth for all configuration with clear precedence:
1. Environment variables (highest priority)
2. Config file (~/.rlm/config.yaml)
3. Defaults (lowest priority)

Example:
    from rlm_dspy.core.config import Config, get_config
    
    # Get the global config singleton
    config = get_config()
    
    # Access values with automatic precedence
    model = config.model              # From env RLM_MODEL, or file, or default
    iterations = config.max_iterations
    
    # Access nested configs
    index_config = config.index
    daemon_config = config.daemon
    
    # Reload from disk
    config.reload()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")

# =============================================================================
# Paths
# =============================================================================

CONFIG_DIR = Path.home() / ".rlm"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
ENV_FILE = CONFIG_DIR / ".env"


# =============================================================================
# Defaults
# =============================================================================

DEFAULTS = {
    # Model settings
    "model": "openai/gpt-4o-mini",
    "sub_model": None,
    "api_base": None,
    
    # Execution limits
    "max_iterations": 20,
    "max_llm_calls": 50,
    "max_output_chars": 100_000,
    "max_timeout": 300,
    "max_budget": 1.0,
    "max_workers": 8,
    
    # Behavior
    "verbose": False,
    "validate": True,
    "enable_logging": False,
    "enable_metrics": False,
    
    # Embedding
    "embedding_model": "openai/text-embedding-3-small",
    "local_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_batch_size": 100,
    
    # Index
    "index_dir": "~/.rlm/indexes",
    "use_faiss": True,
    "faiss_threshold": 5000,
    "auto_update_index": True,
    "index_cache_ttl": 3600,
    "max_snippet_chars": 18000,
    
    # Daemon
    "daemon_debounce": 5.0,
    "daemon_max_concurrent": 2,
    "daemon_idle_timeout": 0,
    
    # Project Registry
    "registry_file": "~/.rlm/projects.json",
    "max_projects": 50,
    "auto_register": True,
    
    # Traces
    "trace_storage": "~/.rlm/traces",
    "min_grounded_score": 0.8,
    "max_traces": 1000,
    
    # Optimization
    "optimization_enabled": True,
    "optimizer": "simba",
    "optimization_model": None,
    "min_new_traces": 50,
    "min_hours_between": 24,
    "optimization_max_budget": 0.50,
    "tip_refresh_interval": 50,
    
    # API
    "env_file": None,
}

# Environment variable mappings
ENV_MAPPINGS = {
    "model": "RLM_MODEL",
    "sub_model": "RLM_SUB_MODEL",
    "api_base": "RLM_API_BASE",
    "max_iterations": "RLM_MAX_ITERATIONS",
    "max_llm_calls": "RLM_MAX_LLM_CALLS",
    "max_output_chars": "RLM_MAX_OUTPUT_CHARS",
    "max_timeout": "RLM_MAX_TIMEOUT",
    "max_budget": "RLM_MAX_BUDGET",
    "max_workers": "RLM_MAX_WORKERS",
    "verbose": "RLM_VERBOSE",
    "validate": "RLM_VALIDATE",
    "enable_logging": "RLM_ENABLE_LOGGING",
    "enable_metrics": "RLM_ENABLE_METRICS",
    "embedding_model": "RLM_EMBEDDING_MODEL",
    "embedding_batch_size": "RLM_EMBEDDING_BATCH_SIZE",
    "index_dir": "RLM_INDEX_DIR",
}


# =============================================================================
# Helper Functions
# =============================================================================

def _cast_value(value: str, target_type: type) -> Any:
    """Cast string value to target type."""
    if target_type is bool:
        return value.lower() in ("true", "1", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


# Type hints for env var casting
ENV_TYPES: dict[str, type] = {
    "max_iterations": int,
    "max_llm_calls": int,
    "max_output_chars": int,
    "max_timeout": float,
    "max_budget": float,
    "max_workers": int,
    "verbose": bool,
    "validate": bool,
    "enable_logging": bool,
    "enable_metrics": bool,
    "embedding_batch_size": int,
}


def _get_env(key: str, default: T) -> T:
    """Get environment variable with type casting."""
    env_var = ENV_MAPPINGS.get(key)
    if not env_var:
        return default
    
    value = os.environ.get(env_var)
    if value is None:
        return default
    
    # Get target type from ENV_TYPES or from default
    target_type = ENV_TYPES.get(key)
    if target_type is None and default is not None:
        target_type = type(default)
    
    if target_type is None:
        return value  # Return as string if no type info
    
    try:
        return _cast_value(value, target_type)
    except (ValueError, TypeError):
        logger.warning("Invalid value for %s: %s", env_var, value)
        return default


def _expand_path(path: str | Path | None) -> Path | None:
    """Expand ~ and resolve path."""
    if path is None:
        return None
    return Path(path).expanduser().resolve()


# =============================================================================
# Config Dataclasses
# =============================================================================

@dataclass
class IndexConfig:
    """Index-specific configuration."""
    index_dir: Path = field(default_factory=lambda: Path.home() / ".rlm" / "indexes")
    use_faiss: bool = True
    faiss_threshold: int = 5000
    auto_update: bool = True
    cache_ttl: int = 3600
    max_snippet_chars: int = 18000


@dataclass
class DaemonConfig:
    """Daemon-specific configuration."""
    pid_file: Path = field(default_factory=lambda: Path.home() / ".rlm" / "daemon.pid")
    log_file: Path = field(default_factory=lambda: Path.home() / ".rlm" / "daemon.log")
    debounce_seconds: float = 5.0
    max_concurrent_indexes: int = 2
    idle_timeout: int = 0


@dataclass
class TraceConfig:
    """Trace collection configuration."""
    storage_path: Path = field(default_factory=lambda: Path.home() / ".rlm" / "traces")
    min_grounded_score: float = 0.8
    max_traces: int = 1000
    max_traces_per_type: int = 200
    enabled: bool = True


@dataclass 
class OptimizationConfig:
    """Optimization configuration."""
    enabled: bool = True
    optimizer: str = "simba"
    model: str | None = None
    min_new_traces: int = 50
    min_hours_between: int = 24
    max_budget: float = 0.50
    run_in_background: bool = True
    tip_refresh_interval: int = 50


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    model: str = "openai/text-embedding-3-small"
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 100
    api_key: str | None = None
    api_base: str | None = None


# =============================================================================
# Main Config Class
# =============================================================================

class Config:
    """Unified configuration with env > file > defaults precedence.
    
    Access configuration values as attributes:
        config = Config()
        config.model         # "kimi/k2p5"
        config.max_iterations  # 20
        config.index.use_faiss  # True
    """
    
    def __init__(self, config_file: Path | None = None):
        """Initialize configuration.
        
        Args:
            config_file: Path to config file (default: ~/.rlm/config.yaml)
        """
        self._config_file = config_file or CONFIG_FILE
        self._file_config: dict[str, Any] = {}
        self._load_file()
        self._load_env_file()
    
    def _load_file(self) -> None:
        """Load configuration from file."""
        if not self._config_file.exists():
            return
        
        try:
            content = self._config_file.read_text(encoding="utf-8")
            self._file_config = yaml.safe_load(content) or {}
        except Exception as e:
            logger.warning("Failed to load config file: %s", e)
            self._file_config = {}
    
    def _load_env_file(self) -> None:
        """Load environment variables from configured env file."""
        env_file = self._get("env_file")
        if not env_file:
            # Try default locations
            for path in [ENV_FILE, Path.home() / ".env"]:
                if path.exists():
                    env_file = str(path)
                    break
        
        if not env_file:
            return
        
        env_path = Path(env_file).expanduser()
        if not env_path.exists():
            return
        
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception as e:
            logger.debug("Failed to load env file: %s", e)
    
    def _get(self, key: str, default: Any = None) -> Any:
        """Get config value with precedence: env > file > default."""
        # Check environment first
        env_value = _get_env(key, None)
        if env_value is not None:
            return env_value
        
        # Check file config (handle nested keys)
        file_value = self._file_config.get(key)
        if file_value is not None:
            return file_value
        
        # Return default
        return default if default is not None else DEFAULTS.get(key)
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_file()
        self._load_env_file()
    
    # =========================================================================
    # Model Settings
    # =========================================================================
    
    @property
    def model(self) -> str:
        """Primary model ID."""
        return self._get("model", DEFAULTS["model"])
    
    @property
    def sub_model(self) -> str:
        """Sub-model for llm_query() calls."""
        return self._get("sub_model") or self.model
    
    @property
    def api_base(self) -> str | None:
        """Custom API base URL."""
        return self._get("api_base")
    
    @property
    def api_key(self) -> str | None:
        """API key (from env)."""
        # Try RLM_API_KEY first
        if key := os.environ.get("RLM_API_KEY"):
            return key
        
        # Try provider-specific key
        from .rlm_types import get_provider_env_var
        if env_var := get_provider_env_var(self.model):
            if key := os.environ.get(env_var):
                return key
        
        # Try common fallbacks
        for var in ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
            if key := os.environ.get(var):
                return key
        
        return None
    
    # =========================================================================
    # Execution Limits
    # =========================================================================
    
    @property
    def max_iterations(self) -> int:
        """Maximum REPL iterations."""
        return self._get("max_iterations", DEFAULTS["max_iterations"])
    
    @property
    def max_llm_calls(self) -> int:
        """Maximum sub-LLM calls."""
        return self._get("max_llm_calls", DEFAULTS["max_llm_calls"])
    
    @property
    def max_output_chars(self) -> int:
        """Maximum output characters per iteration."""
        return self._get("max_output_chars", DEFAULTS["max_output_chars"])
    
    @property
    def max_timeout(self) -> float:
        """Maximum execution timeout in seconds."""
        return self._get("max_timeout", DEFAULTS["max_timeout"])
    
    @property
    def max_budget(self) -> float:
        """Maximum cost budget in USD."""
        return self._get("max_budget", DEFAULTS["max_budget"])
    
    @property
    def max_workers(self) -> int:
        """Maximum parallel workers."""
        return self._get("max_workers", DEFAULTS["max_workers"])
    
    # =========================================================================
    # Behavior Flags
    # =========================================================================
    
    @property
    def verbose(self) -> bool:
        """Enable verbose output."""
        return self._get("verbose", DEFAULTS["verbose"])
    
    @property
    def validate(self) -> bool:
        """Enable output validation."""
        return self._get("validate", DEFAULTS["validate"])
    
    # =========================================================================
    # Nested Configs
    # =========================================================================
    
    @property
    def index(self) -> IndexConfig:
        """Index configuration."""
        return IndexConfig(
            index_dir=_expand_path(self._get("index_dir", DEFAULTS["index_dir"])),
            use_faiss=self._get("use_faiss", DEFAULTS["use_faiss"]),
            faiss_threshold=self._get("faiss_threshold", DEFAULTS["faiss_threshold"]),
            auto_update=self._get("auto_update_index", DEFAULTS["auto_update_index"]),
            cache_ttl=self._get("index_cache_ttl", DEFAULTS["index_cache_ttl"]),
            max_snippet_chars=self._get("max_snippet_chars", DEFAULTS["max_snippet_chars"]),
        )
    
    @property
    def daemon(self) -> DaemonConfig:
        """Daemon configuration."""
        daemon_config = self._file_config.get("daemon", {})
        return DaemonConfig(
            debounce_seconds=daemon_config.get("watch_debounce", DEFAULTS["daemon_debounce"]),
            max_concurrent_indexes=daemon_config.get("max_concurrent_indexes", DEFAULTS["daemon_max_concurrent"]),
            idle_timeout=daemon_config.get("idle_timeout", DEFAULTS["daemon_idle_timeout"]),
        )
    
    @property
    def traces(self) -> TraceConfig:
        """Trace collection configuration."""
        opt_config = self._file_config.get("optimization", {})
        return TraceConfig(
            storage_path=_expand_path(opt_config.get("trace_storage", DEFAULTS["trace_storage"])),
            min_grounded_score=opt_config.get("min_grounded_score", DEFAULTS["min_grounded_score"]),
            max_traces=opt_config.get("max_traces", DEFAULTS["max_traces"]),
            enabled=opt_config.get("enabled", True),
        )
    
    @property
    def optimization(self) -> OptimizationConfig:
        """Optimization configuration."""
        opt_config = self._file_config.get("optimization", {})
        return OptimizationConfig(
            enabled=opt_config.get("enabled", DEFAULTS["optimization_enabled"]),
            optimizer=opt_config.get("optimizer", DEFAULTS["optimizer"]),
            model=opt_config.get("model", DEFAULTS["optimization_model"]),
            min_new_traces=opt_config.get("min_new_traces", DEFAULTS["min_new_traces"]),
            min_hours_between=opt_config.get("min_hours_between", DEFAULTS["min_hours_between"]),
            max_budget=opt_config.get("max_budget", DEFAULTS["optimization_max_budget"]),
            tip_refresh_interval=opt_config.get("tip_refresh_interval", DEFAULTS["tip_refresh_interval"]),
        )
    
    @property
    def embedding(self) -> EmbeddingConfig:
        """Embedding configuration."""
        return EmbeddingConfig(
            model=self._get("embedding_model", DEFAULTS["embedding_model"]),
            local_model=self._get("local_embedding_model", DEFAULTS["local_embedding_model"]),
            batch_size=self._get("embedding_batch_size", DEFAULTS["embedding_batch_size"]),
            api_key=os.environ.get("RLM_EMBEDDING_API_KEY"),
            api_base=os.environ.get("RLM_EMBEDDING_API_BASE"),
        )
    
    def to_rlm_config(self) -> "RLMConfig":
        """Convert to RLMConfig for backward compatibility."""
        from .rlm_types import RLMConfig
        return RLMConfig(
            model=self.model,
            sub_model=self.sub_model,
            api_base=self.api_base,
            api_key=self.api_key,
            max_iterations=self.max_iterations,
            max_llm_calls=self.max_llm_calls,
            max_output_chars=self.max_output_chars,
            max_timeout=self.max_timeout,
            max_budget=self.max_budget,
            max_workers=self.max_workers,
            verbose=self.verbose,
            validate=self.validate,
        )


# =============================================================================
# Global Singleton
# =============================================================================

_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration singleton.
    
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Reload configuration from disk.
    
    Returns:
        Reloaded Config instance
    """
    global _config
    _config = Config()
    return _config

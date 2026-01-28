"""User configuration management for RLM-DSPy.

Provides persistent configuration via ~/.rlm/config.yaml
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config directory
CONFIG_DIR = Path.home() / ".rlm"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Default configuration
DEFAULT_CONFIG = {
    "model": "openai/gpt-4o-mini",
    "max_budget": 1.0,
    "max_timeout": 300,
    "chunk_size": 100000,
    "syntax_aware": True,
    "env_file": None,  # Optional path to .env file
}


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


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to file."""
    ensure_config_dir()

    # Only save non-default values
    to_save = {}
    for key, value in config.items():
        if key in DEFAULT_CONFIG and value != DEFAULT_CONFIG[key]:
            to_save[key] = value
        elif key not in DEFAULT_CONFIG:
            to_save[key] = value

    with open(CONFIG_FILE, "w") as f:
        yaml.dump(to_save, f, default_flow_style=False, sort_keys=False)


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

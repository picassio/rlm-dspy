#!/usr/bin/env python3
"""Generate model definitions from models.dev API.

Usage:
    python -m rlm_dspy.scripts.generate_models
    
Or directly:
    python src/rlm_dspy/scripts/generate_models.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime, UTC

import requests


MODELS_DEV_URL = "https://models.dev/api.json"

# Providers we support (order matters for ALL_MODELS list)
SUPPORTED_PROVIDERS = [
    "anthropic",
    "openai",
    "google",
    "deepseek",
    "groq",
    "mistral",
    "cohere",
    "xai",
    # Chinese providers
    "minimax",
    "minimax-cn",
    "kimi-for-coding",
    "zai",
    "zhipuai",
    "alibaba",
]


def fetch_models_dev() -> dict:
    """Fetch model data from models.dev API."""
    print(f"Fetching models from {MODELS_DEV_URL}...")
    resp = requests.get(MODELS_DEV_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def generate_model_info(provider: str, model_id: str, model: dict) -> str | None:
    """Generate ModelInfo code for a single model."""
    # Skip models without tool support
    if not model.get("tool_call"):
        return None
    
    name = model.get("name", model_id)
    reasoning = model.get("reasoning", False)
    
    # Input modalities
    input_types = ["text"]
    modalities = model.get("modalities", {})
    if "image" in modalities.get("input", []):
        input_types.append("image")
    
    # Limits
    limits = model.get("limit", {})
    context_window = limits.get("context", 128000)
    max_tokens = limits.get("output", 8192)
    
    # Cost (per million tokens)
    cost = model.get("cost", {})
    input_cost = cost.get("input", 0)
    output_cost = cost.get("output", 0)
    cache_read = cost.get("cache_read", 0)
    cache_write = cost.get("cache_write", 0)
    
    # API type
    api_map = {
        "anthropic": "anthropic",
        "openai": "openai",
        "google": "google",
        "deepseek": "openai-compatible",
        "groq": "openai-compatible",
        "mistral": "openai-compatible",
        "cohere": "openai-compatible",
        "xai": "openai-compatible",
        "minimax": "anthropic-compatible",
        "minimax-cn": "anthropic-compatible",
        "kimi-for-coding": "anthropic-compatible",
        "zai": "openai-compatible",
        "zhipuai": "openai-compatible",
        "alibaba": "openai-compatible",
    }
    api = api_map.get(provider, "openai-compatible")
    
    return f'''    ModelInfo(
        id="{provider}/{model_id}",
        name="{name}",
        provider="{provider}",
        api="{api}",
        reasoning={reasoning},
        input_types={input_types},
        context_window={context_window},
        max_tokens={max_tokens},
        cost=ModelCost(input={input_cost}, output={output_cost}, cache_read={cache_read}, cache_write={cache_write}),
    ),'''


def provider_to_var(provider: str) -> str:
    """Convert provider name to Python variable name."""
    return provider.upper().replace("-", "_") + "_MODELS"


def generate_models_code(data: dict) -> str:
    """Generate Python code for all models."""
    
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    sections = []
    provider_vars = []
    
    for provider in SUPPORTED_PROVIDERS:
        if provider not in data or "models" not in data[provider]:
            continue
        
        models = data[provider]["models"]
        model_entries = []
        
        for model_id, model_info in sorted(models.items()):
            entry = generate_model_info(provider, model_id, model_info)
            if entry:
                model_entries.append(entry)
        
        if model_entries:
            var_name = provider_to_var(provider)
            provider_vars.append(var_name)
            section = f'''{var_name}: list[ModelInfo] = [
{chr(10).join(model_entries)}
]'''
            sections.append(section)
    
    # Generate ALL_MODELS entries
    all_models_entries = ",\n".join(f"    *{var}" for var in provider_vars)
    
    # Generate the full file
    code = f'''"""Model definitions for RLM-DSPy.

Auto-generated from models.dev API on {timestamp}.
Do not edit manually - run: python -m rlm_dspy.scripts.generate_models

Provides model metadata including:
- Context window sizes
- Max output tokens  
- Pricing information
- Supported input types (text, image)
- Reasoning/thinking support
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelCost:
    """Cost per million tokens in USD."""
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass
class ModelInfo:
    """Model metadata."""
    id: str  # Full model ID (e.g., "anthropic/claude-sonnet-4-20250514")
    name: str  # Human-readable name
    provider: str  # Provider name (e.g., "anthropic", "openai")
    api: str  # API type (e.g., "anthropic", "openai", "openai-compatible")
    reasoning: bool = False  # Supports extended thinking
    input_types: list[str] = field(default_factory=lambda: ["text"])
    context_window: int = 128000
    max_tokens: int = 8192
    cost: ModelCost = field(default_factory=ModelCost)
    
    @property
    def supports_images(self) -> bool:
        return "image" in self.input_types


# ============================================================================
# Auto-generated Model Definitions
# ============================================================================

{chr(10).join(sections)}


# ============================================================================
# Model Registry
# ============================================================================

ALL_MODELS: list[ModelInfo] = [
{all_models_entries},
]

_MODEL_BY_ID: dict[str, ModelInfo] = {{m.id: m for m in ALL_MODELS}}


def find_model(model_id: str) -> ModelInfo | None:
    """Find model info by ID.
    
    Args:
        model_id: Full model ID (e.g., "anthropic/claude-sonnet-4-20250514")
        
    Returns:
        ModelInfo if found, None otherwise
    """
    return _MODEL_BY_ID.get(model_id)


def list_models(provider: str | None = None) -> list[ModelInfo]:
    """List all available models, optionally filtered by provider.
    
    Args:
        provider: Optional provider filter (e.g., "anthropic", "openai")
        
    Returns:
        List of ModelInfo objects
    """
    if provider:
        return [m for m in ALL_MODELS if m.provider == provider]
    return ALL_MODELS.copy()


def get_model_ids(provider: str | None = None) -> list[str]:
    """Get list of model IDs, optionally filtered by provider."""
    return [m.id for m in list_models(provider)]
'''
    
    return code


def main():
    # Fetch from API
    data = fetch_models_dev()
    
    # Generate code
    code = generate_models_code(data)
    
    # Find output path
    script_dir = Path(__file__).parent
    output_path = script_dir.parent / "core" / "models.py"
    
    # Write file
    output_path.write_text(code)
    print(f"Generated {output_path}")
    
    # Print summary
    for provider in SUPPORTED_PROVIDERS:
        if provider in data and "models" in data[provider]:
            models = data[provider]["models"]
            tool_models = sum(1 for m in models.values() if m.get("tool_call"))
            print(f"  {provider}: {tool_models} models with tool support")


if __name__ == "__main__":
    main()

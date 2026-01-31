"""Model registry and selection for rlm-dspy.

Provides:
- Built-in model definitions for common providers
- Model discovery based on available API keys/OAuth
- Fuzzy model search and selection
- Cost tracking per model
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from .oauth import get_oauth_token, is_anthropic_authenticated


@dataclass
class ModelCost:
    """Cost per million tokens."""
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass
class ModelInfo:
    """Information about a model."""
    
    id: str  # Full model ID (e.g., "anthropic/claude-sonnet-4-20250514")
    name: str  # Human-readable name
    provider: str  # Provider name (e.g., "anthropic", "openrouter")
    api: str  # API type (e.g., "anthropic", "openai-compatible")
    reasoning: bool = False  # Supports extended thinking
    input_types: list[str] = field(default_factory=lambda: ["text"])
    context_window: int = 128000
    max_tokens: int = 8192
    cost: ModelCost = field(default_factory=ModelCost)
    
    @property
    def supports_images(self) -> bool:
        return "image" in self.input_types


# ============================================================================
# Built-in Model Definitions
# ============================================================================

ANTHROPIC_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="anthropic/claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        provider="anthropic",
        api="anthropic",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200000,
        max_tokens=64000,
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
    ),
    ModelInfo(
        id="anthropic/claude-opus-4-20250514",
        name="Claude Opus 4",
        provider="anthropic",
        api="anthropic",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200000,
        max_tokens=64000,
        cost=ModelCost(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
    ),
    ModelInfo(
        id="anthropic/claude-haiku-4-20250514",
        name="Claude Haiku 4",
        provider="anthropic",
        api="anthropic",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200000,
        max_tokens=64000,
        cost=ModelCost(input=0.8, output=4.0, cache_read=0.08, cache_write=1.0),
    ),
    ModelInfo(
        id="anthropic/claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        api="anthropic",
        reasoning=False,
        input_types=["text", "image"],
        context_window=200000,
        max_tokens=8192,
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
    ),
]

OPENAI_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="openai/gpt-4o",
        name="GPT-4o",
        provider="openai",
        api="openai",
        reasoning=False,
        input_types=["text", "image"],
        context_window=128000,
        max_tokens=16384,
        cost=ModelCost(input=2.5, output=10.0),
    ),
    ModelInfo(
        id="openai/gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        api="openai",
        reasoning=False,
        input_types=["text", "image"],
        context_window=128000,
        max_tokens=16384,
        cost=ModelCost(input=0.15, output=0.6),
    ),
    ModelInfo(
        id="openai/o1",
        name="o1",
        provider="openai",
        api="openai",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200000,
        max_tokens=100000,
        cost=ModelCost(input=15.0, output=60.0),
    ),
    ModelInfo(
        id="openai/o3-mini",
        name="o3-mini",
        provider="openai",
        api="openai",
        reasoning=True,
        input_types=["text"],
        context_window=200000,
        max_tokens=100000,
        cost=ModelCost(input=1.1, output=4.4),
    ),
]

GOOGLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="google/gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider="google",
        api="google",
        reasoning=True,
        input_types=["text", "image"],
        context_window=1000000,
        max_tokens=8192,
        cost=ModelCost(input=0.1, output=0.4),
    ),
    ModelInfo(
        id="google/gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider="google",
        api="google",
        reasoning=True,
        input_types=["text", "image"],
        context_window=1000000,
        max_tokens=65536,
        cost=ModelCost(input=1.25, output=10.0),
    ),
]

OPENROUTER_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="openrouter/anthropic/claude-sonnet-4",
        name="Claude Sonnet 4 (OpenRouter)",
        provider="openrouter",
        api="openai-compatible",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200000,
        max_tokens=64000,
        cost=ModelCost(input=3.0, output=15.0),
    ),
    ModelInfo(
        id="openrouter/google/gemini-2.0-flash-001",
        name="Gemini 2.0 Flash (OpenRouter)",
        provider="openrouter",
        api="openai-compatible",
        reasoning=True,
        input_types=["text", "image"],
        context_window=1000000,
        max_tokens=8192,
        cost=ModelCost(input=0.1, output=0.4),
    ),
    ModelInfo(
        id="openrouter/google/gemini-3-flash-preview",
        name="Gemini 3 Flash Preview (OpenRouter)",
        provider="openrouter",
        api="openai-compatible",
        reasoning=True,
        input_types=["text", "image"],
        context_window=1000000,
        max_tokens=65536,
        cost=ModelCost(input=0.15, output=0.6),
    ),
    ModelInfo(
        id="openrouter/deepseek/deepseek-chat-v3-0324",
        name="DeepSeek V3 (OpenRouter)",
        provider="openrouter",
        api="openai-compatible",
        reasoning=True,
        input_types=["text"],
        context_window=131072,
        max_tokens=8192,
        cost=ModelCost(input=0.14, output=0.28),
    ),
]

# All built-in models
ALL_MODELS: list[ModelInfo] = (
    ANTHROPIC_MODELS + OPENAI_MODELS + GOOGLE_MODELS + OPENROUTER_MODELS
)


# ============================================================================
# Provider API Key Detection
# ============================================================================

PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN"],
    "openai": ["OPENAI_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "fireworks": ["FIREWORKS_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
}


def has_provider_auth(provider: str) -> bool:
    """Check if a provider has authentication configured."""
    # Check OAuth first
    if provider == "anthropic" and is_anthropic_authenticated():
        return True
    
    # Check environment variables
    env_vars = PROVIDER_ENV_VARS.get(provider, [])
    for var in env_vars:
        if os.environ.get(var):
            return True
    
    return False


def get_provider_api_key(provider: str) -> str | None:
    """Get API key for a provider."""
    # Check OAuth first
    if provider == "anthropic":
        token = get_oauth_token("anthropic")
        if token:
            return token
    
    # Check environment variables
    env_vars = PROVIDER_ENV_VARS.get(provider, [])
    for var in env_vars:
        if key := os.environ.get(var):
            return key
    
    return None


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """Registry of available models."""
    
    def __init__(self):
        self._models: list[ModelInfo] = list(ALL_MODELS)
        self._custom_models: list[ModelInfo] = []
    
    def get_all(self) -> list[ModelInfo]:
        """Get all registered models."""
        return self._models + self._custom_models
    
    def get_available(self) -> list[ModelInfo]:
        """Get models that have authentication configured."""
        return [m for m in self.get_all() if has_provider_auth(m.provider)]
    
    def get_by_provider(self, provider: str) -> list[ModelInfo]:
        """Get models for a specific provider."""
        return [m for m in self.get_all() if m.provider == provider]
    
    def find(self, model_id: str) -> ModelInfo | None:
        """Find a model by ID (exact match)."""
        for model in self.get_all():
            if model.id == model_id:
                return model
        return None
    
    def search(self, pattern: str) -> list[ModelInfo]:
        """Search models by fuzzy pattern."""
        pattern_lower = pattern.lower()
        results = []
        
        for model in self.get_all():
            # Check ID and name
            search_text = f"{model.id} {model.name} {model.provider}".lower()
            if pattern_lower in search_text:
                results.append(model)
                continue
            
            # Fuzzy match
            if self._fuzzy_match(pattern_lower, search_text):
                results.append(model)
        
        return results
    
    def _fuzzy_match(self, pattern: str, text: str) -> bool:
        """Simple fuzzy matching."""
        pattern_idx = 0
        for char in text:
            if pattern_idx < len(pattern) and char == pattern[pattern_idx]:
                pattern_idx += 1
        return pattern_idx == len(pattern)
    
    def register(self, model: ModelInfo) -> None:
        """Register a custom model."""
        # Remove existing model with same ID
        self._custom_models = [m for m in self._custom_models if m.id != model.id]
        self._custom_models.append(model)
    
    def get_providers(self) -> list[str]:
        """Get list of unique providers."""
        return sorted(set(m.provider for m in self.get_all()))
    
    def get_available_providers(self) -> list[str]:
        """Get providers that have authentication configured."""
        return sorted(set(m.provider for m in self.get_available()))


# Global registry instance
_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_available_models() -> list[ModelInfo]:
    """Get models that have authentication configured."""
    return get_model_registry().get_available()


def find_model(model_id: str) -> ModelInfo | None:
    """Find a model by ID."""
    return get_model_registry().find(model_id)


def search_models(pattern: str) -> list[ModelInfo]:
    """Search models by pattern."""
    return get_model_registry().search(pattern)

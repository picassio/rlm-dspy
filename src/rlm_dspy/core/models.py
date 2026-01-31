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

from .oauth import is_google_authenticated, is_antigravity_authenticated, get_credentials


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

ANTIGRAVITY_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="antigravity/gemini-3-flash",
        name="Gemini 3 Flash (Antigravity)",
        provider="antigravity",
        api="antigravity",
        reasoning=True,
        input_types=["text", "image"],
        context_window=1000000,
        max_tokens=65536,
        cost=ModelCost(input=0.15, output=0.6),
    ),
    ModelInfo(
        id="antigravity/gemini-3-pro-low",
        name="Gemini 3 Pro Low (Antigravity)",
        provider="antigravity",
        api="antigravity",
        reasoning=True,
        input_types=["text", "image"],
        context_window=1000000,
        max_tokens=65536,
        cost=ModelCost(input=1.25, output=10.0),
    ),
    ModelInfo(
        id="antigravity/gemini-3-pro-high",
        name="Gemini 3 Pro High (Antigravity)",
        provider="antigravity",
        api="antigravity",
        reasoning=True,
        input_types=["text", "image"],
        context_window=1000000,
        max_tokens=65536,
        cost=ModelCost(input=1.25, output=10.0),
    ),
    ModelInfo(
        id="antigravity/claude-sonnet-4-5",
        name="Claude Sonnet 4.5 (Antigravity)",
        provider="antigravity",
        api="antigravity",
        reasoning=False,
        input_types=["text", "image"],
        context_window=200000,
        max_tokens=64000,
        cost=ModelCost(input=3.0, output=15.0),
    ),
    ModelInfo(
        id="antigravity/claude-sonnet-4-5-thinking",
        name="Claude Sonnet 4.5 Thinking (Antigravity)",
        provider="antigravity",
        api="antigravity",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200000,
        max_tokens=64000,
        cost=ModelCost(input=3.0, output=15.0),
    ),
    ModelInfo(
        id="antigravity/claude-opus-4-5-thinking",
        name="Claude Opus 4.5 Thinking (Antigravity)",
        provider="antigravity",
        api="antigravity",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200000,
        max_tokens=64000,
        cost=ModelCost(input=5.0, output=25.0),
    ),
    ModelInfo(
        id="antigravity/gpt-oss-120b-medium",
        name="GPT-OSS 120B Medium (Antigravity)",
        provider="antigravity",
        api="antigravity",
        reasoning=True,
        input_types=["text"],
        context_window=128000,
        max_tokens=16384,
        cost=ModelCost(input=1.0, output=3.0),
    ),
]

KIMI_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="kimi/k2p5",
        name="Kimi K2.5",
        provider="kimi",
        api="anthropic-compatible",
        reasoning=True,
        input_types=["text", "image"],
        context_window=262144,
        max_tokens=32768,
        cost=ModelCost(input=0.0, output=0.0),  # Free tier
    ),
    ModelInfo(
        id="kimi/kimi-k2-thinking",
        name="Kimi K2 Thinking",
        provider="kimi",
        api="anthropic-compatible",
        reasoning=True,
        input_types=["text"],
        context_window=262144,
        max_tokens=32768,
        cost=ModelCost(input=0.0, output=0.0),  # Free tier
    ),
]

ZAI_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="zai/glm-4.5",
        name="GLM 4.5 (Z.AI)",
        provider="zai",
        api="openai-compatible",
        reasoning=True,
        input_types=["text"],
        context_window=131072,
        max_tokens=98304,
        cost=ModelCost(input=0.6, output=2.2),
    ),
    ModelInfo(
        id="zai/glm-4.5-air",
        name="GLM 4.5 Air (Z.AI)",
        provider="zai",
        api="openai-compatible",
        reasoning=True,
        input_types=["text"],
        context_window=131072,
        max_tokens=131072,
        cost=ModelCost(input=0.05, output=0.22),
    ),
    ModelInfo(
        id="zai/glm-4.6",
        name="GLM 4.6 (Z.AI)",
        provider="zai",
        api="openai-compatible",
        reasoning=True,
        input_types=["text"],
        context_window=204800,
        max_tokens=131072,
        cost=ModelCost(input=0.35, output=1.4),
    ),
    ModelInfo(
        id="zai/glm-4.7",
        name="GLM 4.7 (Z.AI)",
        provider="zai",
        api="openai-compatible",
        reasoning=True,
        input_types=["text"],
        context_window=204800,
        max_tokens=131072,
        cost=ModelCost(input=0.43, output=1.75),
    ),
]

OPENCODE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="opencode/glm-4.6",
        name="GLM 4.6 (OpenCode)",
        provider="opencode",
        api="openai-compatible",
        reasoning=True,
        input_types=["text"],
        context_window=204800,
        max_tokens=131072,
        cost=ModelCost(input=0.35, output=1.4),
    ),
    ModelInfo(
        id="opencode/glm-4.7",
        name="GLM 4.7 (OpenCode)",
        provider="opencode",
        api="openai-compatible",
        reasoning=True,
        input_types=["text"],
        context_window=204800,
        max_tokens=131072,
        cost=ModelCost(input=0.6, output=2.2),
    ),
    ModelInfo(
        id="opencode/glm-4.7-free",
        name="GLM 4.7 Free (OpenCode)",
        provider="opencode",
        api="openai-compatible",
        reasoning=True,
        input_types=["text"],
        context_window=204800,
        max_tokens=131072,
        cost=ModelCost(input=0.0, output=0.0),
    ),
]

MINIMAX_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="minimax/MiniMax-M2",
        name="MiniMax M2",
        provider="minimax",
        api="anthropic-compatible",
        reasoning=True,
        input_types=["text"],
        context_window=196608,
        max_tokens=128000,
        cost=ModelCost(input=0.3, output=1.2),
    ),
    ModelInfo(
        id="minimax/MiniMax-M2.1",
        name="MiniMax M2.1",
        provider="minimax",
        api="anthropic-compatible",
        reasoning=True,
        input_types=["text"],
        context_window=196608,
        max_tokens=128000,
        cost=ModelCost(input=0.3, output=1.2),
    ),
]

# All built-in models
ALL_MODELS: list[ModelInfo] = (
    ANTHROPIC_MODELS + OPENAI_MODELS + GOOGLE_MODELS + OPENROUTER_MODELS + ANTIGRAVITY_MODELS + KIMI_MODELS + MINIMAX_MODELS + OPENCODE_MODELS + ZAI_MODELS
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
    "minimax": ["MINIMAX_API_KEY"],
    "opencode": ["OPENCODE_API_KEY"],
    "zai": ["ZAI_API_KEY"],
    "kimi": ["KIMI_API_KEY"],
}


def has_provider_auth(provider: str) -> bool:
    """Check if a provider has authentication configured."""
    # Check OAuth first
    if provider in ("google", "google-gemini") and is_google_authenticated():
        return True
    
    if provider == "antigravity" and is_antigravity_authenticated():
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
    if provider in ("google", "google-gemini"):
        creds = get_credentials("google-gemini")
        if creds and not creds.is_expired:
            return creds.access_token
    
    if provider == "antigravity":
        creds = get_credentials("antigravity")
        if creds and not creds.is_expired:
            return creds.access_token
    
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

"""Embedding management for semantic search.

Provides unified interface for hosted and local embedding models.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models.

    Supports both hosted models (via litellm) and local models
    (via sentence-transformers).

    Attributes:
        model: Embedding model identifier. Use "local" for local models.
               Examples: "openai/text-embedding-3-small", "cohere/embed-english-v3.0"
        local_model: HuggingFace model ID for local embeddings.
        batch_size: Number of texts to embed in one batch.
        api_key: API key for hosted models (optional, uses env vars if not set).
        api_base: Custom API endpoint (optional).
        caching: Whether to cache embedding results.
    """
    model: str = "openai/text-embedding-3-small"
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 100
    api_key: str | None = None
    api_base: str | None = None
    caching: bool = True

    # Cached config instance
    _cached_config: "EmbeddingConfig | None" = None

    @classmethod
    def from_user_config(cls, use_cache: bool = True) -> "EmbeddingConfig":
        """Load embedding configuration from user config.

        Priority: env vars > config.yaml > defaults

        Args:
            use_cache: Use cached config if available (default True)
        """
        # Return cached config if available
        if use_cache and cls._cached_config is not None:
            return cls._cached_config

        from .user_config import load_config, load_env_file

        # Load env file first
        load_env_file()
        config = load_config()

        result = cls(
            model=os.environ.get(
                "RLM_EMBEDDING_MODEL",
                config.get("embedding_model", "openai/text-embedding-3-small")
            ),
            local_model=config.get(
                "local_embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            batch_size=int(os.environ.get(
                "RLM_EMBEDDING_BATCH_SIZE",
                config.get("embedding_batch_size", 100)
            )),
            api_key=_resolve_embedding_api_key(config),
            api_base=os.environ.get("RLM_EMBEDDING_API_BASE"),
            caching=config.get("embedding_caching", True),
        )

        # Cache for future calls
        if use_cache:
            cls._cached_config = result

        return result

    def __repr__(self) -> str:
        key_status = "***" if self.api_key else "not set"
        return (
            f"EmbeddingConfig(model={self.model!r}, "
            f"batch_size={self.batch_size}, api_key={key_status})"
        )


def _resolve_embedding_api_key(config: dict) -> str | None:
    """Resolve API key for embedding model.

    Checks in order:
    1. RLM_EMBEDDING_API_KEY env var
    2. Provider-specific key based on model prefix
    3. RLM_API_KEY as fallback
    """
    # Direct embedding key
    if key := os.environ.get("RLM_EMBEDDING_API_KEY"):
        return key

    # Get model to determine provider
    model = os.environ.get(
        "RLM_EMBEDDING_MODEL",
        config.get("embedding_model", "openai/text-embedding-3-small")
    )

    # Provider-specific keys
    provider_keys = {
        "openai/": "OPENAI_API_KEY",
        "cohere/": "COHERE_API_KEY",
        "voyage/": "VOYAGE_API_KEY",
        "together": "TOGETHER_API_KEY",
        "mistral/": "MISTRAL_API_KEY",
    }

    for prefix, env_var in provider_keys.items():
        if model.startswith(prefix) or prefix in model:
            if key := os.environ.get(env_var):
                return key

    # Fallback to RLM_API_KEY
    return os.environ.get("RLM_API_KEY")


# Cached embedder instance
_embedder_cache: dict[str, Any] = {}


def get_embedder(config: EmbeddingConfig | None = None) -> Any:
    """Get configured embedder instance.

    Returns a dspy.Embedder or compatible callable that takes
    a list of strings and returns embeddings.

    Args:
        config: Embedding configuration. Uses user config if not provided.

    Returns:
        Embedder instance (dspy.Embedder or SentenceTransformer wrapper)

    Raises:
        ImportError: If required dependencies are not installed.
        ValueError: If configuration is invalid.
    """

    config = config or EmbeddingConfig.from_user_config()

    # Check cache
    cache_key = f"{config.model}:{config.local_model}:{config.batch_size}"
    if cache_key in _embedder_cache:
        return _embedder_cache[cache_key]

    if config.model.lower() == "local":
        # Use local sentence-transformers model
        embedder = _create_local_embedder(config)
    else:
        # Use hosted model via litellm
        embedder = _create_hosted_embedder(config)

    _embedder_cache[cache_key] = embedder
    return embedder


def _create_local_embedder(config: EmbeddingConfig) -> Any:
    """Create embedder using local sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Local embeddings require sentence-transformers.\n"
            "Install with: pip install sentence-transformers\n"
            "Or use a hosted model by setting embedding_model in config."
        )

    logger.info("Loading local embedding model: %s", config.local_model)
    model = SentenceTransformer(config.local_model)

    # Wrap in dspy.Embedder for consistent interface
    import dspy
    return dspy.Embedder(model.encode, batch_size=config.batch_size)


def _create_hosted_embedder(config: EmbeddingConfig) -> Any:
    """Create embedder using hosted model via litellm."""
    import dspy

    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "caching": config.caching,
    }

    if config.api_key:
        kwargs["api_key"] = config.api_key

    if config.api_base:
        kwargs["api_base"] = config.api_base

    logger.info("Using hosted embedding model: %s", config.model)
    return dspy.Embedder(config.model, **kwargs)


def clear_embedder_cache() -> None:
    """Clear the embedder and config caches."""
    global _embedder_cache
    _embedder_cache.clear()
    EmbeddingConfig._cached_config = None
    logger.debug("Embedder and config caches cleared")


def embed_texts(
    texts: list[str],
    config: EmbeddingConfig | None = None,
) -> "np.ndarray":
    """Embed a list of texts.

    Convenience function that handles embedder creation.

    Args:
        texts: List of texts to embed
        config: Embedding configuration (optional)

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    embedder = get_embedder(config)
    return embedder(texts)


def get_embedding_dim(config: EmbeddingConfig | None = None) -> int:
    """Get the embedding dimension for the configured model.

    Args:
        config: Embedding configuration (optional)

    Returns:
        Embedding dimension (e.g., 1536 for text-embedding-3-small)
    """
    # Known dimensions for common models
    KNOWN_DIMS = {
        "openai/text-embedding-3-small": 1536,
        "openai/text-embedding-3-large": 3072,
        "openai/text-embedding-ada-002": 1536,
        "cohere/embed-english-v3.0": 1024,
        "cohere/embed-english-light-v3.0": 384,
        "voyage/voyage-3": 1024,
        "voyage/voyage-3-lite": 512,
    }

    config = config or EmbeddingConfig.from_user_config()

    if config.model in KNOWN_DIMS:
        return KNOWN_DIMS[config.model]

    # For local models, we need to check
    if config.model.lower() == "local":
        # Common sentence-transformer dimensions
        if "MiniLM" in config.local_model:
            return 384
        if "mpnet" in config.local_model.lower():
            return 768

    # Default fallback - embed a sample to find out
    logger.debug("Unknown embedding dim for %s, computing...", config.model)
    sample_embedding = embed_texts(["test"], config)
    return sample_embedding.shape[-1]


# Export
__all__ = [
    "EmbeddingConfig",
    "get_embedder",
    "embed_texts",
    "get_embedding_dim",
    "clear_embedder_cache",
]

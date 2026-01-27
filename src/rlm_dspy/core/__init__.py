"""Core RLM-DSPy modules."""

from .async_client import AsyncLLMClient, aggregate_answers_async, analyze_chunks_async
from .observability import (
    OperationMetrics,
    Span,
    SpanType,
    Tracker,
    enable_tracking,
    get_tracker,
    track,
)
from .programs import ChunkedProcessor, MapReduceProcessor, RecursiveAnalyzer
from .progress import BatchProgress, ProgressContext
from .registry import (
    Registry,
    RegistryError,
    builtin_processor,
    builtin_strategy,
    load_class,
    models,
    processors,
    strategies,
)
from .retry import retry_sync, retry_with_backoff
from .rlm import RLM, RLMConfig, RLMResult
from .secrets import (
    MissingSecretError,
    check_for_exposed_secrets,
    clean_secrets,
    get_api_key,
    inject_secrets,
    is_secret_key,
    mask_value,
)
from .signatures import AggregateAnswers, AnalyzeChunk, DecomposeTask, ExtractAnswer
from .types import BatchResult, ChunkResult, FailedChunk, ProcessingStats

__all__ = [
    # Core
    "RLM",
    "RLMConfig",
    "RLMResult",
    # Signatures
    "AnalyzeChunk",
    "AggregateAnswers",
    "DecomposeTask",
    "ExtractAnswer",
    # Programs
    "RecursiveAnalyzer",
    "ChunkedProcessor",
    "MapReduceProcessor",
    # Async
    "AsyncLLMClient",
    "analyze_chunks_async",
    "aggregate_answers_async",
    # Types (from modaic patterns)
    "FailedChunk",
    "ChunkResult",
    "BatchResult",
    "ProcessingStats",
    # Progress
    "BatchProgress",
    "ProgressContext",
    # Retry
    "retry_with_backoff",
    "retry_sync",
    # Secrets (from modaic patterns)
    "clean_secrets",
    "inject_secrets",
    "check_for_exposed_secrets",
    "is_secret_key",
    "mask_value",
    "get_api_key",
    "MissingSecretError",
    # Observability (from modaic patterns)
    "Tracker",
    "Span",
    "SpanType",
    "track",
    "get_tracker",
    "enable_tracking",
    "OperationMetrics",
    # Registry (from modaic patterns)
    "Registry",
    "RegistryError",
    "strategies",
    "processors",
    "models",
    "builtin_strategy",
    "builtin_processor",
    "load_class",
]

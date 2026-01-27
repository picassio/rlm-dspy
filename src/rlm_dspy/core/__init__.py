"""Core RLM-DSPy modules."""

from .async_client import AsyncLLMClient, aggregate_answers_async, analyze_chunks_async
from .programs import ChunkedProcessor, MapReduceProcessor, RecursiveAnalyzer
from .progress import BatchProgress, ProgressContext
from .retry import retry_sync, retry_with_backoff
from .rlm import RLM, RLMConfig, RLMResult
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
]

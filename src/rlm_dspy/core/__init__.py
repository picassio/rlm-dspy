"""Core RLM-DSPy modules."""

from .async_client import AsyncLLMClient, aggregate_answers_async, analyze_chunks_async
from .programs import ChunkedProcessor, MapReduceProcessor, RecursiveAnalyzer
from .rlm import RLM, RLMConfig, RLMResult
from .signatures import AggregateAnswers, AnalyzeChunk, DecomposeTask, ExtractAnswer

__all__ = [
    "RLM",
    "RLMConfig",
    "RLMResult",
    "AnalyzeChunk",
    "AggregateAnswers",
    "DecomposeTask",
    "ExtractAnswer",
    "RecursiveAnalyzer",
    "ChunkedProcessor",
    "MapReduceProcessor",
    "AsyncLLMClient",
    "analyze_chunks_async",
    "aggregate_answers_async",
]

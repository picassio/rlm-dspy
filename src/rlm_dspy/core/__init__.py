"""Core RLM-DSPy modules."""

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
]

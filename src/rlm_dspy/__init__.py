"""RLM-DSPy: Recursive Language Models with DSPy optimization."""

from .core.programs import (
    ChunkedProcessor,
    MapReduceProcessor,
    RecursiveAnalyzer,
)
from .core.rlm import RLM, RLMConfig, RLMResult
from .core.signatures import (
    AggregateAnswers,
    AnalyzeChunk,
    DecomposeTask,
    ExtractAnswer,
)

__version__ = "0.1.0"
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

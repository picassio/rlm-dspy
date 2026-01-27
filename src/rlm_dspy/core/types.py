"""Type definitions for RLM-DSPy.

Learned from modaic: proper error typing for partial failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FailedChunk:
    """
    Represents a failed chunk analysis that can be retried.

    Learned from modaic's FailedPrediction pattern:
    - Capture errors per-item instead of failing entire batch
    - Allow partial successes
    - Enable later retry of failed items
    """

    error: str
    index: int
    chunk_preview: str = ""  # First 100 chars for debugging
    retryable: bool = True

    def __str__(self) -> str:
        return f"FailedChunk(index={self.index}, error={self.error[:50]})"


@dataclass
class ChunkResult:
    """Result from analyzing a single chunk."""

    index: int
    relevant_info: str
    confidence: str  # high|medium|low|none
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def has_info(self) -> bool:
        return self.confidence != "none" and bool(self.relevant_info)


@dataclass
class BatchResult:
    """
    Result from batch processing multiple chunks.

    Supports partial success - some chunks may fail while others succeed.
    """

    results: list[ChunkResult] = field(default_factory=list)
    failed: list[FailedChunk] = field(default_factory=list)
    total_time: float = 0.0
    total_tokens: int = 0

    @property
    def success_count(self) -> int:
        return len([r for r in self.results if r.success])

    @property
    def failure_count(self) -> int:
        return len(self.failed)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def get_relevant_results(self) -> list[ChunkResult]:
        """Get only results with relevant information."""
        return [r for r in self.results if r.has_info]

    def get_answers(self) -> list[str]:
        """Get list of relevant info strings for aggregation."""
        return [r.relevant_info for r in self.get_relevant_results()]


@dataclass
class ProcessingStats:
    """Statistics from RLM processing."""

    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    elapsed_time: float = 0.0
    strategy_used: str = "auto"
    depth_reached: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "success_rate": self.processed_chunks / self.total_chunks if self.total_chunks > 0 else 0,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "elapsed_time": self.elapsed_time,
            "strategy": self.strategy_used,
            "depth": self.depth_reached,
        }

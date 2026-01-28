"""Token usage tracking for rlm-dspy.

Tracks token counts and cost estimation for RLM queries.
Uses tiktoken for accurate token counting (same tokenizer as GPT-4/Claude).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (as of 2024) - module level for efficiency
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}

# Lazy-loaded encoder (singleton)
_encoder = None
_tiktoken_warning_shown = False


def _get_encoder():
    """Get or create tiktoken encoder (lazy loading)."""
    global _encoder, _tiktoken_warning_shown
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            if not _tiktoken_warning_shown:
                logger.warning(
                    "tiktoken not available, using approximate token counting. "
                    "Install with: pip install tiktoken"
                )
                _tiktoken_warning_shown = True
            return None
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken.

    Uses cl100k_base encoding (compatible with GPT-4/Claude tokenization).

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens (0 if tiktoken not available)
    """
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is None:
        # Fallback: rough estimate of 4 chars per token
        return len(text) // 4
    return len(encoder.encode(text))


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4",
) -> float:
    """Estimate cost based on token counts.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name for pricing lookup

    Returns:
        Estimated cost in USD
    """
    # Find matching pricing from module-level constant
    model_lower = model.lower()
    rates = None
    for key, value in MODEL_PRICING.items():
        if key in model_lower:
            rates = value
            break

    if rates is None:
        # Default to GPT-4 pricing
        rates = MODEL_PRICING["gpt-4"]

    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]

    return input_cost + output_cost


@dataclass
class TokenStats:
    """Track token usage for a single RLM operation.

    Attributes:
        raw_context_tokens: Original context size in tokens
        llm_input_tokens: Total tokens sent to LLM (across iterations)
        llm_output_tokens: Total response tokens from LLM
        iterations: Number of REPL iterations
    """

    raw_context_tokens: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    iterations: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.llm_input_tokens + self.llm_output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "raw_context_tokens": self.raw_context_tokens,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "iterations": self.iterations,
            "total_tokens": self.total_tokens,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "Token Stats:",
            f"  Context: {self.raw_context_tokens:,} tokens",
            f"  LLM: {self.llm_input_tokens:,} in, {self.llm_output_tokens:,} out",
            f"  Iterations: {self.iterations}",
        ]
        return "\n".join(lines)


@dataclass
class SessionStats:
    """Aggregate stats across multiple RLM operations in a session."""

    session_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    operations: list[TokenStats] = field(default_factory=list)

    # Accumulated totals
    total_raw_tokens: int = 0
    total_llm_input: int = 0
    total_llm_output: int = 0
    total_iterations: int = 0

    def record(self, stats: TokenStats) -> None:
        """Record an operation's token stats."""
        self.operations.append(stats)
        self.total_raw_tokens += stats.raw_context_tokens
        self.total_llm_input += stats.llm_input_tokens
        self.total_llm_output += stats.llm_output_tokens
        self.total_iterations += stats.iterations

    @property
    def total_tokens(self) -> int:
        """Total tokens across all operations."""
        return self.total_llm_input + self.total_llm_output

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "operation_count": len(self.operations),
            "total_raw_tokens": self.total_raw_tokens,
            "total_llm_input": self.total_llm_input,
            "total_llm_output": self.total_llm_output,
            "total_tokens": self.total_tokens,
            "total_iterations": self.total_iterations,
        }

    def save(self, path: Path | str) -> None:
        """Save stats to JSONL file (append mode)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Session Stats ({self.session_id}):",
            f"  Operations: {len(self.operations)}",
            f"  Context: {self.total_raw_tokens:,} tokens",
            f"  LLM Total: {self.total_tokens:,} tokens",
            f"  Iterations: {self.total_iterations}",
        ]
        return "\n".join(lines)


# Global session for convenience
_current_session: SessionStats | None = None
# Use RLock to allow reentrant locking (record_operation calls get_session)
_session_lock = __import__("threading").RLock()


def get_session(session_id: str | None = None) -> SessionStats:
    """Get or create the current session stats (thread-safe)."""
    global _current_session
    with _session_lock:
        if _current_session is None or (
            session_id and _current_session.session_id != session_id
        ):
            _current_session = SessionStats(
                session_id=session_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            )
        return _current_session


def record_operation(stats: TokenStats) -> None:
    """Record an operation to the current session (thread-safe)."""
    with _session_lock:
        session = get_session()
        session.record(stats)


__all__ = [
    "count_tokens",
    "estimate_cost",
    "TokenStats",
    "SessionStats",
    "get_session",
    "record_operation",
]

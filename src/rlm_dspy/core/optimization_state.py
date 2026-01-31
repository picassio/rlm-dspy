"""Optimization state management for auto-optimization.

Tracks optimization state and manages saved programs.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Storage paths
OPTIMIZATION_DIR = Path.home() / ".rlm" / "optimization"
OPTIMIZED_PROGRAM_FILE = OPTIMIZATION_DIR / "optimized_program.json"
OPTIMIZATION_STATE_FILE = OPTIMIZATION_DIR / "state.json"

# Lock for background optimization
_optimization_lock = threading.Lock()
_optimization_running = False


@dataclass
class OptimizationResult:
    """Result of SIMBA optimization."""

    improved: bool
    baseline_score: float
    optimized_score: float
    improvement: float  # Percentage improvement
    num_steps: int
    num_candidates: int
    best_program_idx: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __str__(self) -> str:
        if self.improved:
            return (
                f"✓ Improved by {self.improvement:.1f}% "
                f"({self.baseline_score:.2f} → {self.optimized_score:.2f})"
            )
        return f"No improvement ({self.baseline_score:.2f})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "improved": self.improved,
            "baseline_score": self.baseline_score,
            "optimized_score": self.optimized_score,
            "improvement": self.improvement,
            "num_steps": self.num_steps,
            "num_candidates": self.num_candidates,
            "best_program_idx": self.best_program_idx,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationResult":
        """Create from dictionary."""
        return cls(
            improved=data.get("improved", False),
            baseline_score=data.get("baseline_score", 0),
            optimized_score=data.get("optimized_score", 0),
            improvement=data.get("improvement", 0),
            num_steps=data.get("num_steps", 0),
            num_candidates=data.get("num_candidates", 0),
            best_program_idx=data.get("best_program_idx", -1),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(UTC),
        )


@dataclass
class OptimizationState:
    """State of the optimization system."""

    last_optimization: datetime | None = None
    traces_at_last_optimization: int = 0
    last_result: OptimizationResult | None = None
    optimizer_type: str = "simba"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "traces_at_last_optimization": self.traces_at_last_optimization,
            "last_result": self.last_result.to_dict() if self.last_result else None,
            "optimizer_type": self.optimizer_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationState":
        """Create from dictionary."""
        last_opt = data.get("last_optimization")
        last_result_data = data.get("last_result")

        # Parse last_optimization and ensure it's timezone-aware
        last_optimization = None
        if last_opt:
            last_optimization = datetime.fromisoformat(last_opt)
            # If naive datetime, assume UTC
            if last_optimization.tzinfo is None:
                last_optimization = last_optimization.replace(tzinfo=UTC)

        return cls(
            last_optimization=last_optimization,
            traces_at_last_optimization=data.get("traces_at_last_optimization", 0),
            last_result=OptimizationResult.from_dict(last_result_data) if last_result_data else None,
            optimizer_type=data.get("optimizer_type", "simba"),
        )


@dataclass
class SavedOptimization:
    """A saved optimized program state."""

    demos: list[dict[str, Any]] = field(default_factory=list)
    instructions: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    optimizer_type: str = "simba"
    result: OptimizationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "demos": self.demos,
            "instructions": self.instructions,
            "timestamp": self.timestamp.isoformat(),
            "optimizer_type": self.optimizer_type,
            "result": self.result.to_dict() if self.result else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SavedOptimization":
        """Create from dictionary."""
        result_data = data.get("result")
        return cls(
            demos=data.get("demos", []),
            instructions=data.get("instructions", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(UTC),
            optimizer_type=data.get("optimizer_type", "simba"),
            result=OptimizationResult.from_dict(result_data) if result_data else None,
        )


# ============================================================================
# State Management Functions
# ============================================================================

def load_optimization_state() -> OptimizationState:
    """Load optimization state from disk."""
    if not OPTIMIZATION_STATE_FILE.exists():
        return OptimizationState()

    try:
        data = json.loads(OPTIMIZATION_STATE_FILE.read_text())
        return OptimizationState.from_dict(data)
    except Exception as e:
        logger.warning("Failed to load optimization state: %s", e)
        return OptimizationState()


def save_optimization_state(state: OptimizationState) -> None:
    """Save optimization state to disk atomically."""
    OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Write to temp file first, then atomic rename
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=OPTIMIZATION_DIR,
            suffix='.tmp',
            delete=False
        ) as f:
            json.dump(state.to_dict(), f, indent=2)
            temp_path = Path(f.name)
        # Atomic rename (works on POSIX, best-effort on Windows)
        temp_path.replace(OPTIMIZATION_STATE_FILE)
    except Exception as e:
        logger.warning("Failed to save optimization state: %s", e)
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            temp_path.unlink(missing_ok=True)


def load_optimized_program() -> SavedOptimization | None:
    """Load saved optimized program from disk."""
    if not OPTIMIZED_PROGRAM_FILE.exists():
        return None

    try:
        data = json.loads(OPTIMIZED_PROGRAM_FILE.read_text())
        return SavedOptimization.from_dict(data)
    except Exception as e:
        logger.warning("Failed to load optimized program: %s", e)
        return None


def save_optimized_program(program: Any, result: OptimizationResult, optimizer_type: str = "simba") -> None:
    """Save optimized program to disk.
    
    Args:
        program: The optimized DSPy program
        result: The optimization result
        optimizer_type: Type of optimizer used
    """
    OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)

    # Extract demos from program
    demos = []
    if hasattr(program, "demos"):
        demos = program.demos if isinstance(program.demos, list) else []

    # Extract instructions
    instructions = ""
    if hasattr(program, "signature") and hasattr(program.signature, "__doc__"):
        instructions = program.signature.__doc__ or ""

    saved = SavedOptimization(
        demos=demos,
        instructions=instructions,
        optimizer_type=optimizer_type,
        result=result,
    )

    try:
        # Write to temp file first, then atomic rename
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=OPTIMIZATION_DIR,
            suffix='.tmp',
            delete=False
        ) as f:
            json.dump(saved.to_dict(), f, indent=2)
            temp_path = Path(f.name)
        temp_path.replace(OPTIMIZED_PROGRAM_FILE)
        logger.info("Saved optimized program to %s", OPTIMIZED_PROGRAM_FILE)
    except Exception as e:
        logger.warning("Failed to save optimized program: %s", e)
        if 'temp_path' in locals():
            temp_path.unlink(missing_ok=True)


def clear_optimization() -> bool:
    """Clear all saved optimization data.
    
    Returns:
        True if anything was cleared
    """
    cleared = False

    if OPTIMIZED_PROGRAM_FILE.exists():
        OPTIMIZED_PROGRAM_FILE.unlink()
        cleared = True

    if OPTIMIZATION_STATE_FILE.exists():
        OPTIMIZATION_STATE_FILE.unlink()
        cleared = True

    return cleared


def get_trace_count() -> int:
    """Get current trace count."""
    try:
        from .trace_collector import get_trace_collector
        collector = get_trace_collector()
        return len(collector.traces)
    except Exception:
        return 0


def should_optimize(config: Any = None) -> bool:
    """Check if we should run background optimization.
    
    Args:
        config: OptimizationConfig (optional, loads from user config if not provided)
        
    Returns:
        True if optimization should run
    """
    if config is None:
        from .user_config import OptimizationConfig
        config = OptimizationConfig.from_user_config()

    if not config.enabled:
        return False

    state = load_optimization_state()
    current_traces = get_trace_count()

    # Need enough new traces
    new_traces = current_traces - state.traces_at_last_optimization
    if new_traces < config.min_new_traces:
        logger.debug("Not enough new traces: %d < %d", new_traces, config.min_new_traces)
        return False

    # Check time since last optimization
    if state.last_optimization:
        hours_since = (datetime.now(UTC) - state.last_optimization).total_seconds() / 3600
        if hours_since < config.min_hours_between:
            logger.debug("Too soon since last optimization: %.1f < %d hours", hours_since, config.min_hours_between)
            return False

    return True


def is_optimization_running() -> bool:
    """Check if background optimization is currently running."""
    return _optimization_running


def set_optimization_running(running: bool) -> bool:
    """Set the optimization running flag (thread-safe).
    
    Returns:
        True if the flag was set, False if already in the requested state
    """
    global _optimization_running

    with _optimization_lock:
        if _optimization_running == running:
            return False
        _optimization_running = running
        return True

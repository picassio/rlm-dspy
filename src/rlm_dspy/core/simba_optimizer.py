"""
SIMBA (Stochastic Introspective Mini-Batch Ascent) integration for RLM.

SIMBA is a DSPy optimizer that uses the LLM to analyze its own performance
and generate improvement rules. This module provides:

1. SIMBAOptimizer - Wrapper for DSPy's SIMBA with RLM-specific defaults
2. Metric functions for code analysis tasks
3. Training data collection from traces
4. Automatic optimization workflow
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


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


def grounded_metric(example: Any, prediction: dict[str, Any]) -> float:
    """
    Metric for evaluating grounded code analysis.

    Scores based on:
    - Whether the answer is grounded in tool outputs (citations)
    - Completeness of the answer
    - Proper use of tools

    Args:
        example: The input example with expected outputs
        prediction: The model's prediction

    Returns:
        Score from 0.0 to 1.0
    """
    score = 0.0

    # Check if answer exists
    answer = prediction.get("answer", "") or prediction.get("output", "")
    if not answer:
        return 0.0

    # Base score for having an answer
    score += 0.3

    # Check for citations/evidence
    citations = prediction.get("citations", [])
    if citations:
        score += 0.3
    elif "[" in answer and "]" in answer:
        # Inline citations like [file.py:10]
        score += 0.2

    # Check for tool usage (if available)
    tool_calls = prediction.get("tool_calls", []) or prediction.get("iterations", 0)
    if tool_calls:
        score += 0.2

    # Check against expected answer (if provided)
    expected = getattr(example, "expected_answer", None) or getattr(example, "answer", None)
    if expected:
        # Simple substring match - could be more sophisticated
        if expected.lower() in answer.lower():
            score += 0.2
        elif any(word in answer.lower() for word in expected.lower().split()[:5]):
            score += 0.1

    return min(score, 1.0)


def accuracy_metric(example: Any, prediction: dict[str, Any]) -> float:
    """
    Simple accuracy metric for exact or partial match.

    Args:
        example: The input example
        prediction: The model's prediction

    Returns:
        Score from 0.0 to 1.0
    """
    answer = prediction.get("answer", "") or prediction.get("output", "")
    expected = getattr(example, "expected_answer", None) or getattr(example, "answer", None)

    if not expected:
        # No expected answer - just check if we got something
        return 1.0 if answer else 0.0

    answer_lower = answer.lower().strip()
    expected_lower = expected.lower().strip()

    # Exact match
    if answer_lower == expected_lower:
        return 1.0

    # Contains expected
    if expected_lower in answer_lower:
        return 0.8

    # Partial word overlap
    answer_words = set(answer_lower.split())
    expected_words = set(expected_lower.split())
    overlap = len(answer_words & expected_words)
    if overlap > 0:
        return min(0.6, overlap / len(expected_words))

    return 0.0


class SIMBAOptimizer:
    """
    SIMBA optimizer wrapper for RLM code analysis.

    Provides simplified interface for optimizing RLM programs using
    SIMBA's self-improving mini-batch approach.
    """

    def __init__(
        self,
        metric: Callable[[Any, dict[str, Any]], float] | None = None,
        batch_size: int = 16,
        num_candidates: int = 4,
        max_steps: int = 4,
        max_demos: int = 3,
        num_threads: int | None = None,
    ):
        """
        Initialize SIMBA optimizer.

        Args:
            metric: Evaluation metric function. Defaults to grounded_metric.
            batch_size: Mini-batch size for optimization. Defaults to 16.
            num_candidates: Number of candidate programs per step. Defaults to 4.
            max_steps: Maximum optimization steps. Defaults to 4.
            max_demos: Maximum demos per predictor. Defaults to 3.
            num_threads: Parallel execution threads. Defaults to None (auto).
        """
        self.metric = metric or grounded_metric
        self.batch_size = batch_size
        self.num_candidates = num_candidates
        self.max_steps = max_steps
        self.max_demos = max_demos
        self.num_threads = num_threads

    def optimize(
        self,
        program: Any,
        trainset: list[Any],
        seed: int = 42,
    ) -> tuple[Any, OptimizationResult]:
        """
        Optimize a program using SIMBA.

        Args:
            program: DSPy module to optimize
            trainset: Training examples (list of dspy.Example)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (optimized_program, optimization_result)
        """
        # Lazy import to avoid startup cost
        try:
            from dspy.teleprompt import SIMBA
        except ImportError:
            logger.error("DSPy SIMBA not available")
            raise ImportError("DSPy SIMBA optimizer not available. Install dspy>=2.5")

        # Validate trainset size
        if len(trainset) < self.batch_size:
            logger.warning(
                "Trainset size %d < batch_size %d, reducing batch_size",
                len(trainset),
                self.batch_size,
            )
            batch_size = max(4, len(trainset) // 2)
        else:
            batch_size = self.batch_size

        # Create SIMBA optimizer
        optimizer = SIMBA(
            metric=self.metric,
            bsize=batch_size,
            num_candidates=self.num_candidates,
            max_steps=self.max_steps,
            max_demos=self.max_demos,
            num_threads=self.num_threads,
        )

        # Get baseline score
        baseline_scores = []
        for example in trainset[:batch_size]:
            try:
                pred = program(example)
                pred_dict = pred.toDict() if hasattr(pred, "toDict") else {"answer": str(pred)}
                score = self.metric(example, pred_dict)
                baseline_scores.append(score)
            except Exception as e:
                logger.debug("Baseline evaluation failed: %s", e)
                baseline_scores.append(0.0)

        baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0

        # Run optimization
        logger.info(
            "Starting SIMBA optimization: %d examples, %d steps, %d candidates",
            len(trainset),
            self.max_steps,
            self.num_candidates,
        )

        optimized = optimizer.compile(program, trainset=trainset, seed=seed)

        # Get optimized score from candidate_programs
        if hasattr(optimized, "candidate_programs") and optimized.candidate_programs:
            optimized_score = optimized.candidate_programs[0]["score"]
            best_idx = 0
        else:
            optimized_score = baseline_score
            best_idx = -1

        # Calculate improvement
        improvement = ((optimized_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0

        result = OptimizationResult(
            improved=optimized_score > baseline_score,
            baseline_score=baseline_score,
            optimized_score=optimized_score,
            improvement=improvement,
            num_steps=self.max_steps,
            num_candidates=self.num_candidates,
            best_program_idx=best_idx,
        )

        logger.info("SIMBA optimization complete: %s", result)

        return optimized, result

    def optimize_from_traces(
        self,
        program: Any,
        traces_dir: Path | str | None = None,
        min_score: float = 0.7,
        max_examples: int = 100,
    ) -> tuple[Any, OptimizationResult]:
        """
        Optimize using collected traces as training data.

        Args:
            program: DSPy module to optimize
            traces_dir: Directory containing trace files. Defaults to ~/.rlm/traces
            min_score: Minimum score to include trace. Defaults to 0.7.
            max_examples: Maximum examples to use. Defaults to 100.

        Returns:
            Tuple of (optimized_program, optimization_result)
        """
        # Lazy import
        import dspy

        if traces_dir is None:
            traces_dir = Path.home() / ".rlm" / "traces"
        else:
            traces_dir = Path(traces_dir)

        if not traces_dir.exists():
            raise ValueError(f"Traces directory not found: {traces_dir}")

        # Load traces from traces.json (new format) or individual files (legacy)
        all_traces = []
        traces_file = traces_dir / "traces.json"
        
        if traces_file.exists():
            try:
                data = json.loads(traces_file.read_text())
                all_traces = data.get("traces", [])
            except Exception as e:
                logger.debug("Failed to load traces.json: %s", e)
        
        # Also check for individual trace files (legacy format)
        for trace_file in sorted(traces_dir.glob("trace_*.json"))[:max_examples * 2]:
            try:
                all_traces.append(json.loads(trace_file.read_text()))
            except Exception as e:
                logger.debug("Failed to load trace %s: %s", trace_file, e)

        # Convert traces to examples
        examples = []
        for trace in all_traces:
            try:
                # Support both grounded_score (new) and validation_score (old)
                score = trace.get("grounded_score", trace.get("validation_score", trace.get("score", 0)))
                if not isinstance(score, (int, float)) or score < min_score:
                    continue

                # Create example
                example = dspy.Example(
                    query=trace.get("query", ""),
                    context=trace.get("context", ""),
                ).with_inputs("query", "context")

                # Add expected output if available
                answer = trace.get("final_answer", trace.get("answer", ""))
                if answer:
                    example.answer = answer
                if "citations" in trace:
                    example.citations = trace["citations"]

                examples.append(example)

                if len(examples) >= max_examples:
                    break

            except Exception as e:
                logger.debug("Failed to process trace: %s", e)

        if len(examples) < self.batch_size:
            raise ValueError(
                f"Not enough traces ({len(examples)}) for optimization. "
                f"Need at least {self.batch_size}. Lower min_score or collect more traces."
            )

        logger.info("Loaded %d traces for optimization", len(examples))

        return self.optimize(program, examples)


def create_training_example(
    query: str,
    answer: str,
    context: str = "",
    citations: list[str] | None = None,
) -> Any:
    """
    Create a training example for SIMBA optimization.

    Args:
        query: The input query
        answer: Expected answer
        context: Optional context (e.g., file path)
        citations: Optional list of citations

    Returns:
        dspy.Example object
    """
    import dspy

    example = dspy.Example(
        query=query,
        context=context,
    ).with_inputs("query", "context")

    example.answer = answer
    if citations:
        example.citations = citations

    return example


# Singleton optimizer instance
_optimizer: SIMBAOptimizer | None = None


def get_simba_optimizer(**kwargs) -> SIMBAOptimizer:
    """Get or create the global SIMBA optimizer."""
    global _optimizer
    if _optimizer is None or kwargs:
        _optimizer = SIMBAOptimizer(**kwargs)
    return _optimizer

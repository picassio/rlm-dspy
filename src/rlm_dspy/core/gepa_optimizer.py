"""
GEPA (Reflective Prompt Evolution) optimizer for RLM.

GEPA is an evolutionary optimizer that uses reflection to evolve text components.
It captures full execution traces and uses textual feedback to guide optimization.

Key advantages for RLM:
- Uses execution traces with textual feedback
- Reflection-based instruction evolution
- Better for complex multi-step agents
- Joint optimization of tools and instructions (optional)

Reference: https://arxiv.org/abs/2507.19457
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any, Callable

import dspy

from .optimization_state import OptimizationResult

logger = logging.getLogger(__name__)


def extract_gepa_instructions(optimized_program: dspy.Module) -> dict[str, str]:
    """
    Extract evolved instructions from a GEPA-optimized program.
    
    GEPA evolves the instruction text in DSPy predictors. This function
    extracts those evolved instructions so they can be saved and reapplied.
    
    Args:
        optimized_program: The GEPA-optimized DSPy module
    
    Returns:
        Dict mapping predictor name to evolved instruction text
    """
    instructions = {}
    
    try:
        # Iterate through named predictors
        for name, predictor in optimized_program.named_predictors():
            # Check if predictor has a signature with instructions
            if hasattr(predictor, 'signature'):
                sig = predictor.signature
                # Get instruction from signature
                if hasattr(sig, 'instructions') and sig.instructions:
                    instructions[name] = sig.instructions
                elif hasattr(sig, '__doc__') and sig.__doc__:
                    instructions[name] = sig.__doc__
            
            # Also check for extended_signature (used by some optimizers)
            if hasattr(predictor, 'extended_signature'):
                ext_sig = predictor.extended_signature
                if hasattr(ext_sig, 'instructions') and ext_sig.instructions:
                    instructions[f"{name}_extended"] = ext_sig.instructions
        
        # For RLM specifically, check for generate_action and extract
        if hasattr(optimized_program, 'generate_action'):
            gen = optimized_program.generate_action
            if hasattr(gen, 'signature') and hasattr(gen.signature, 'instructions'):
                instructions['generate_action'] = gen.signature.instructions
        
        if hasattr(optimized_program, 'extract'):
            ext = optimized_program.extract
            if hasattr(ext, 'signature') and hasattr(ext.signature, 'instructions'):
                instructions['extract'] = ext.signature.instructions
                
    except Exception as e:
        logger.warning("Failed to extract GEPA instructions: %s", e)
    
    return instructions


def apply_gepa_instructions(program: dspy.Module, instructions: dict[str, str]) -> None:
    """
    Apply saved GEPA instructions to a program.
    
    Args:
        program: DSPy module to update
        instructions: Dict mapping predictor name to instruction text
    """
    try:
        for name, predictor in program.named_predictors():
            if name in instructions:
                if hasattr(predictor, 'signature'):
                    # Create new signature with updated instructions
                    predictor.signature = predictor.signature.with_instructions(instructions[name])
                    logger.debug("Applied GEPA instruction to %s", name)
    except Exception as e:
        logger.warning("Failed to apply GEPA instructions: %s", e)


@dataclass
class GEPAConfig:
    """Configuration for GEPA optimizer."""
    
    # Budget (exactly one must be set)
    auto: str | None = "light"  # "light", "medium", "heavy"
    max_full_evals: int | None = None
    max_metric_calls: int | None = None
    
    # Reflection settings
    reflection_minibatch_size: int = 3
    skip_perfect_score: bool = True
    
    # Merge settings
    use_merge: bool = True
    max_merge_invocations: int = 5
    
    # Parallelism
    num_threads: int = 2
    
    # Score range
    failure_score: float = 0.0
    perfect_score: float = 1.0
    
    # Tool optimization (for ReAct-style modules)
    enable_tool_optimization: bool = False
    
    # Seed for reproducibility
    seed: int = 42


def create_rlm_feedback_metric(
    base_metric: Callable[[Any, Any], float] | None = None,
) -> Callable:
    """
    Create a GEPA feedback metric for RLM optimization.
    
    GEPA expects metrics that return either:
    - A float score
    - A ScoreWithFeedback (Prediction with score and feedback fields)
    
    The feedback helps GEPA understand WHY a prediction is good or bad,
    which guides the reflection process for evolving instructions.
    
    Args:
        base_metric: Optional base metric function (defaults to grounded_metric)
    
    Returns:
        A GEPA-compatible feedback metric function
    """
    if base_metric is None:
        from .simba_optimizer import grounded_metric
        base_metric = grounded_metric
    
    def feedback_metric(
        gold: dspy.Example,
        pred: dspy.Prediction | None,
        trace: list | None = None,
        pred_name: str | None = None,
        pred_trace: list | None = None,
    ) -> dspy.Prediction:
        """
        GEPA feedback metric that provides textual feedback along with scores.
        
        Args:
            gold: The expected example (with query, answer fields)
            pred: The model's prediction
            trace: Full execution trace [(predictor, inputs, output), ...]
            pred_name: Name of predictor being evaluated (e.g., 'generate_action')
            pred_trace: Sub-trace for just this predictor
            
        Returns:
            ScoreWithFeedback prediction with score and feedback fields
        """
        # Handle None prediction (failed execution)
        if pred is None:
            return dspy.Prediction(
                score=0.0,
                feedback="Prediction failed - no output generated. Check for API errors or timeouts."
            )
        
        # Get base score
        score = base_metric(gold, pred)
        
        # Extract answer for analysis
        if hasattr(pred, 'toDict'):
            pred_dict = pred.toDict()
        elif hasattr(pred, '__dict__'):
            pred_dict = {k: v for k, v in pred.__dict__.items() if not k.startswith('_')}
        elif isinstance(pred, dict):
            pred_dict = pred
        else:
            pred_dict = {"answer": str(pred) if pred else ""}
        
        answer = (
            pred_dict.get("answer", "") or
            pred_dict.get("output", "") or
            pred_dict.get("response", "") or
            ""
        )
        
        # Get expected answer
        expected = getattr(gold, "expected_answer", None) or getattr(gold, "answer", None)
        
        # Build feedback based on score and context
        feedback_parts = []
        
        if score >= 0.9:
            feedback_parts.append("Excellent response that matches the expected answer well.")
        elif score >= 0.7:
            feedback_parts.append("Good response with minor issues.")
            if expected and expected.lower() not in answer.lower():
                feedback_parts.append("Consider including key terms from expected answer.")
        elif score >= 0.5:
            feedback_parts.append("Partial match - answer is related but incomplete.")
            if expected:
                exp_preview = expected[:100] + "..." if len(expected) > 100 else expected
                feedback_parts.append(f"Expected answer contains: '{exp_preview}'")
        elif score >= 0.3:
            feedback_parts.append("Poor match - significant information missing.")
            if not answer:
                feedback_parts.append("No answer was generated.")
            elif expected:
                feedback_parts.append("Answer doesn't match expected output.")
        else:
            feedback_parts.append("Failed to generate a useful response.")
            if not answer:
                feedback_parts.append("Empty answer - ensure tools are being used correctly.")
        
        # Add trace analysis for predictor-specific feedback
        if pred_name and pred_trace:
            feedback_parts.append(f"[Analyzing predictor: {pred_name}]")
            
            # Analyze trace for specific predictor
            if pred_name == "generate_action":
                # Check if too many iterations
                if trace and len(trace) > 10:
                    feedback_parts.append("Many iterations used - consider more direct tool usage.")
                # Check if SUBMIT was called
                if "SUBMIT" not in answer and "SUBMIT" not in str(pred_dict):
                    feedback_parts.append("Ensure SUBMIT() is called with final answer.")
            elif pred_name == "extract":
                if not answer:
                    feedback_parts.append("Extract failed to produce output from trajectory.")
        
        # Add trace length info if available
        if trace:
            feedback_parts.append(f"Trace length: {len(trace)} steps.")
        
        feedback = " ".join(feedback_parts)
        
        return dspy.Prediction(score=score, feedback=feedback)
    
    return feedback_metric


class GEPAOptimizer:
    """
    GEPA optimizer wrapper for RLM.
    
    GEPA uses reflection to evolve prompts based on execution traces and feedback.
    It's particularly effective for complex multi-step agents like RLM.
    
    Example:
        optimizer = GEPAOptimizer(config=GEPAConfig(auto="light"))
        optimized, result = optimizer.optimize(rlm._rlm, trainset)
    """
    
    def __init__(
        self,
        config: GEPAConfig | None = None,
        metric: Callable | None = None,
        reflection_lm: dspy.LM | None = None,
    ):
        """
        Initialize GEPA optimizer.
        
        Args:
            config: GEPA configuration
            metric: Feedback metric function (defaults to RLM feedback metric)
            reflection_lm: LM to use for reflection (defaults to current dspy.settings.lm)
        """
        self.config = config or GEPAConfig()
        self.metric = metric or create_rlm_feedback_metric()
        self.reflection_lm = reflection_lm
        self._patched = False
    
    def _patch_dspy_parallelizer(self):
        """
        Monkey-patch DSPy's ParallelExecutor to return proper failure values.
        
        DSPy bug: When an exception occurs, the parallelizer returns the exception
        object. But Evaluate only handles `r is None`, causing "too many values 
        to unpack" errors when trying to unpack exception as (prediction, score).
        
        This patch makes the parallelizer return None for exceptions, which 
        Evaluate handles correctly.
        """
        if self._patched:
            return
            
        try:
            from dspy.utils import parallelizer
            
            original_wrap = parallelizer.ParallelExecutor._wrap_function
            
            def patched_wrap(self, user_function):
                original_safe_func = original_wrap(self, user_function)
                
                def safer_func(item):
                    result = original_safe_func(item)
                    # If result is an exception, convert to None
                    # so Evaluate handles it with failure_score
                    if isinstance(result, Exception):
                        return None
                    return result
                
                return safer_func
            
            parallelizer.ParallelExecutor._wrap_function = patched_wrap
            self._patched = True
            logger.debug("Applied DSPy parallelizer patch for exception handling")
            
        except Exception as e:
            logger.warning("Failed to patch DSPy parallelizer: %s", e)
    
    def optimize(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        valset: list[dspy.Example] | None = None,
        lm: dspy.LM | None = None,
    ) -> tuple[dspy.Module, OptimizationResult]:
        """
        Optimize a program using GEPA.
        
        Args:
            program: DSPy module to optimize (e.g., rlm._rlm)
            trainset: Training examples
            valset: Optional validation set (defaults to trainset)
            lm: Optional LM to use (defaults to dspy.settings.lm)
        
        Returns:
            Tuple of (optimized_program, optimization_result)
        """
        try:
            from dspy.teleprompt import GEPA
        except ImportError:
            raise ImportError("GEPA requires dspy>=3.0. Install with: pip install 'dspy>=3.0'")
        
        # Monkey-patch DSPy's parallelizer to handle exceptions properly
        # DSPy bug: parallelizer returns exceptions as results, but Evaluate
        # only handles None, causing "too many values to unpack" errors
        self._patch_dspy_parallelizer()
        
        # Configure LM if provided
        if lm is not None:
            dspy.configure(lm=lm)
        
        # Use reflection LM or default to current LM
        # GEPA requires a reflection_lm - it cannot be None
        reflection_lm = self.reflection_lm or dspy.settings.lm
        if reflection_lm is None:
            raise ValueError(
                "GEPA requires a reflection language model. "
                "Set via: --teacher CLI option, optimization.teacher_model in config, "
                "or pass lm= to optimize()"
            )
        
        # Validate budget configuration
        budget_set = sum([
            self.config.auto is not None,
            self.config.max_full_evals is not None,
            self.config.max_metric_calls is not None,
        ])
        if budget_set != 1:
            logger.warning("Exactly one of auto/max_full_evals/max_metric_calls should be set. Using auto='light'")
            self.config.auto = "light"
            self.config.max_full_evals = None
            self.config.max_metric_calls = None
        
        # Build GEPA kwargs
        gepa_kwargs = {
            "metric": self.metric,
            "reflection_lm": reflection_lm,
            "reflection_minibatch_size": self.config.reflection_minibatch_size,
            "skip_perfect_score": self.config.skip_perfect_score,
            "use_merge": self.config.use_merge,
            "max_merge_invocations": self.config.max_merge_invocations,
            "num_threads": self.config.num_threads,
            "failure_score": self.config.failure_score,
            "perfect_score": self.config.perfect_score,
            "enable_tool_optimization": self.config.enable_tool_optimization,
            "seed": self.config.seed,
            "track_stats": True,  # Always track for results
        }
        
        # Add budget
        if self.config.auto:
            gepa_kwargs["auto"] = self.config.auto
        elif self.config.max_full_evals:
            gepa_kwargs["max_full_evals"] = self.config.max_full_evals
        elif self.config.max_metric_calls:
            gepa_kwargs["max_metric_calls"] = self.config.max_metric_calls
        
        logger.info(
            "Starting GEPA optimization: %d train, %d val, auto=%s",
            len(trainset),
            len(valset) if valset else len(trainset),
            self.config.auto,
        )
        
        # Calculate baseline
        baseline_scores = []
        for example in trainset[:min(5, len(trainset))]:
            try:
                pred = program(**example.inputs())
                result = self.metric(example, pred)
                # Handle both dict and Prediction return types
                if hasattr(result, 'score'):
                    score = result.score
                elif isinstance(result, dict):
                    score = result.get("score", 0.0)
                else:
                    score = float(result)
                baseline_scores.append(score)
            except Exception as e:
                logger.debug("Baseline evaluation failed: %s", e)
                baseline_scores.append(0.0)
        
        baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        
        # Create and run GEPA optimizer
        # Note: The parallelizer patch handles exceptions at the eval level
        optimizer = GEPA(**gepa_kwargs)
        
        try:
            optimized = optimizer.compile(
                student=program,
                trainset=trainset,
                valset=valset,
            )
        except Exception as e:
            logger.error("GEPA optimization failed: %s", e)
            return program, OptimizationResult(
                improved=False,
                baseline_score=baseline_score,
                optimized_score=baseline_score,
                improvement=0.0,
                num_steps=0,
                num_candidates=0,
                best_program_idx=-1,
            )
        
        # Get optimized score from GEPA results if available
        optimized_score = baseline_score
        if hasattr(optimized, 'detailed_results') and optimized.detailed_results:
            # Use GEPA's tracked validation scores
            results = optimized.detailed_results
            if hasattr(results, 'val_aggregate_scores') and results.val_aggregate_scores:
                optimized_score = max(results.val_aggregate_scores)
                logger.info("GEPA best validation score: %.2f", optimized_score)
        else:
            # Fallback: manually evaluate optimized program
            optimized_scores = []
            for example in trainset[:min(5, len(trainset))]:
                try:
                    pred = optimized(**example.inputs())
                    result = self.metric(example, pred)
                    if hasattr(result, 'score'):
                        score = result.score
                    elif isinstance(result, dict):
                        score = result.get("score", 0.0)
                    else:
                        score = float(result)
                    optimized_scores.append(score)
                except Exception as e:
                    logger.debug("Optimized evaluation failed: %s", e)
                    optimized_scores.append(0.0)
            
            if optimized_scores:
                optimized_score = sum(optimized_scores) / len(optimized_scores)
        
        improvement = ((optimized_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0.0
        
        result = OptimizationResult(
            improved=optimized_score > baseline_score,
            baseline_score=baseline_score,
            optimized_score=optimized_score,
            improvement=improvement,
            num_steps=1,  # GEPA doesn't have discrete steps
            num_candidates=1,  # Not applicable for GEPA
            best_program_idx=0,
        )
        
        logger.info(
            "GEPA optimization complete: %.2f -> %.2f (%.1f%% improvement)",
            baseline_score, optimized_score, improvement,
        )
        
        # Extract evolved instructions for saving
        evolved_instructions = extract_gepa_instructions(optimized)
        if evolved_instructions:
            logger.info("Extracted %d evolved instructions from GEPA", len(evolved_instructions))
        
        return optimized, result
    
    def optimize_from_traces(
        self,
        program: dspy.Module,
        min_score: float = 0.7,
        max_examples: int = 50,
        lm: dspy.LM | None = None,
    ) -> tuple[dspy.Module, OptimizationResult]:
        """
        Optimize using traces from TraceCollector.
        
        Args:
            program: DSPy module to optimize
            min_score: Minimum grounded_score for training examples
            max_examples: Maximum number of examples to use
            lm: Optional LM to use
        
        Returns:
            Tuple of (optimized_program, optimization_result)
        """
        from .trace_collector import get_trace_collector
        from .simba_optimizer import create_training_example
        
        collector = get_trace_collector()
        traces = collector.get_successes(min_score)[:max_examples]
        
        if len(traces) < 4:
            logger.warning("Not enough traces for GEPA optimization (%d < 4)", len(traces))
            return program, OptimizationResult(
                improved=False,
                baseline_score=0.0,
                optimized_score=0.0,
                improvement=0.0,
                num_steps=0,
                num_candidates=0,
                best_program_idx=-1,
            )
        
        # Convert traces to examples
        examples = []
        for trace in traces:
            example = create_training_example(
                query=trace.query,
                answer=trace.final_answer,
                context="",
            )
            if example:
                examples.append(example)
        
        return self.optimize(program, examples, lm=lm)


# Singleton instance
_gepa_optimizer: GEPAOptimizer | None = None


def get_gepa_optimizer(**kwargs) -> GEPAOptimizer:
    """Get or create the global GEPA optimizer."""
    global _gepa_optimizer
    if _gepa_optimizer is None or kwargs:
        config = GEPAConfig(**{k: v for k, v in kwargs.items() if hasattr(GEPAConfig, k)})
        _gepa_optimizer = GEPAOptimizer(config=config)
    return _gepa_optimizer

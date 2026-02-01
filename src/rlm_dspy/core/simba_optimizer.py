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
import threading
from pathlib import Path
from typing import Any, Callable

from datetime import datetime, UTC

# Re-export from optimization_state for backwards compatibility
from .optimization_state import (
    OptimizationResult,
    OptimizationState,
    SavedOptimization,
    OPTIMIZATION_DIR,
    OPTIMIZED_PROGRAM_FILE,
    OPTIMIZATION_STATE_FILE,
    load_optimization_state,
    save_optimization_state,
    load_optimized_program,
    save_optimized_program,
    clear_optimization,
    get_trace_count,
    should_optimize,
    is_optimization_running,
    set_optimization_running,
)

logger = logging.getLogger(__name__)


def run_background_optimization(config: Any = None, model: str | None = None) -> None:
    """Run optimization in background thread.
    
    Supports both GEPA and SIMBA optimizers based on config.
    Uses fast proxy mode by default for 50x faster optimization.
    
    Args:
        config: OptimizationConfig (optional)
        model: Model to use for optimization (optional)
    """
    # Prevent multiple simultaneous optimizations
    if not set_optimization_running(True):
        logger.debug("Background optimization already running")
        return

    if config is None:
        from .user_config import OptimizationConfig
        config = OptimizationConfig.from_user_config()

    def _optimize():
        try:
            if config.optimizer == "simba":
                logger.info(
                    "Starting background SIMBA optimization (fast=%s, steps=%d, candidates=%d)...",
                    config.fast, config.simba.steps, config.simba.candidates
                )
            else:
                logger.info(
                    "Starting background GEPA optimization (fast=%s, auto=%s)...",
                    config.fast, config.gepa.auto
                )

            # Load env file for API keys
            from .user_config import load_env_file, load_config
            load_env_file()

            # Get the RLM program
            from .rlm import RLM
            rlm = RLM()
            
            # Get model from config
            user_cfg = load_config()
            default_model = user_cfg.get("model", "openai/gpt-4o-mini")
            model_name = config.get_model(default_model)
            
            # Configure LM (handle custom providers)
            import dspy
            lm = _create_lm(model_name)
            dspy.configure(lm=lm)

            optimized_program = None
            result = None
            instructions = {}

            if config.optimizer == "gepa":
                # Run GEPA optimization
                if config.fast:
                    # Fast proxy mode
                    from .gepa_proxy import RLMProxy, create_proxy_metric, extract_proxy_instructions
                    from .gepa_optimizer import GEPAConfig, apply_gepa_instructions
                    
                    proxy = RLMProxy.from_rlm(rlm._rlm)
                    
                    # Get teacher model
                    teacher_model_name = config.get_teacher_model(default_model)
                    teacher_lm = _create_lm(teacher_model_name)
                    
                    # Configure GEPA from nested config
                    gepa_config = GEPAConfig(
                        auto=config.gepa.auto if config.gepa.max_evals is None else None,
                        max_full_evals=config.gepa.max_evals,
                        num_threads=config.threads,
                    )
                    
                    # Get training examples
                    from .trace_collector import get_trace_collector
                    collector = get_trace_collector()
                    traces = [t for t in collector.traces if t.grounded_score >= 0.7][:50]
                    
                    if len(traces) < 4:
                        logger.info("Not enough traces for GEPA optimization: %d < 4", len(traces))
                        return  # finally block handles cleanup
                    
                    examples = [
                        dspy.Example(query=t.query, answer=t.final_answer).with_inputs("query")
                        for t in traces
                    ]
                    
                    # Run GEPA on proxy
                    from dspy.teleprompt import GEPA
                    optimizer = GEPA(
                        metric=create_proxy_metric(),
                        reflection_lm=teacher_lm,
                        **gepa_config.__dict__,
                    )
                    optimized_proxy = optimizer.compile(proxy, trainset=examples)
                    
                    # Extract instructions
                    instructions = extract_proxy_instructions(optimized_proxy)
                    
                    # Apply to real RLM
                    if instructions:
                        apply_gepa_instructions(rlm._rlm, instructions)
                    
                    optimized_program = optimized_proxy
                    
                    # Create result
                    from .optimization_state import OptimizationResult
                    result = OptimizationResult(
                        improved=bool(instructions),
                        baseline_score=0.0,
                        optimized_score=0.0,
                        improvement=0.0,
                        num_steps=1,
                        num_candidates=1,
                        best_program_idx=0,
                    )
                else:
                    # Full RLM mode (slow)
                    logger.warning("Background GEPA without fast mode is very slow, consider enabling fast mode")
                    # Skip for now - too slow for background
                    return  # finally block handles cleanup

            elif config.optimizer == "simba":
                # Run SIMBA optimization
                if config.fast:
                    # Fast proxy mode
                    from .gepa_proxy import RLMProxy, create_proxy_metric, extract_proxy_instructions
                    from dspy.teleprompt import SIMBA
                    
                    proxy = RLMProxy.from_rlm(rlm._rlm)
                    
                    # Get training examples
                    from .trace_collector import get_trace_collector
                    collector = get_trace_collector()
                    traces = [t for t in collector.traces if t.grounded_score >= 0.7][:50]
                    
                    if len(traces) < 4:
                        logger.info("Not enough traces for SIMBA optimization: %d < 4", len(traces))
                        return  # finally block handles cleanup
                    
                    examples = [
                        create_training_example(t.query, t.final_answer, "")
                        for t in traces if create_training_example(t.query, t.final_answer, "")
                    ]
                    
                    if not examples:
                        logger.info("No valid training examples for SIMBA")
                        return  # finally block handles cleanup
                    
                    # Adjust batch size using SIMBA-specific config
                    batch_size = min(config.simba.batch_size, len(examples))
                    
                    # Create metric
                    def simba_proxy_metric(example, pred):
                        r = create_proxy_metric()(example, pred)
                        return r.score if hasattr(r, 'score') else float(r)
                    
                    optimizer = SIMBA(
                        metric=simba_proxy_metric,
                        bsize=batch_size,
                        num_candidates=config.simba.candidates,
                        max_steps=config.simba.steps,
                        num_threads=config.threads,
                    )
                    
                    optimized_proxy = optimizer.compile(proxy, trainset=examples)
                    
                    # Extract instructions
                    instructions = extract_proxy_instructions(optimized_proxy)
                    optimized_program = optimized_proxy
                    
                    # Create result
                    from .optimization_state import OptimizationResult
                    result = OptimizationResult(
                        improved=bool(instructions),
                        baseline_score=0.0,
                        optimized_score=0.0,
                        improvement=0.0,
                        num_steps=config.simba.steps,
                        num_candidates=config.simba.candidates,
                        best_program_idx=0,
                    )
                else:
                    # Full RLM mode
                    optimizer = get_simba_optimizer(
                        batch_size=config.simba.batch_size,
                        num_candidates=config.simba.candidates,
                        max_steps=config.simba.steps,
                        num_threads=config.threads,
                    )
                    
                    optimized_program, result = optimizer.optimize_from_traces(
                        program=rlm._rlm,
                        min_score=0.7,
                        max_examples=50,
                    )

            # Generate tips from failure patterns
            tips = _generate_optimized_tips()
            
            # Extract any rules from optimization
            rules = _extract_simba_rules(optimized_program) if optimized_program else []

            if (result and result.improved) or tips or rules or instructions:
                # Save the optimized program with all components
                save_optimized_program(
                    optimized_program,
                    result,
                    config.optimizer,
                    tips=tips,
                    rules=rules,
                    instructions=instructions,
                )

                # Update state
                state = OptimizationState(
                    last_optimization=datetime.now(UTC),
                    traces_at_last_optimization=get_trace_count(),
                    last_result=result,
                    optimizer_type=config.optimizer,
                )
                save_optimization_state(state)

                logger.info(
                    "Background optimization complete: %s, %d tips, %d rules, %d instructions",
                    f"+{result.improvement:.1f}%" if result and result.improved else "no score improvement",
                    len(tips),
                    len(rules),
                    len(instructions),
                )
            else:
                # Still update state to prevent re-running
                state = OptimizationState(
                    last_optimization=datetime.now(UTC),
                    traces_at_last_optimization=get_trace_count(),
                    last_result=result,
                    optimizer_type=config.optimizer,
                )
                save_optimization_state(state)
                logger.info("Background optimization complete: no improvement")

        except Exception as e:
            logger.warning("Background optimization failed: %s", e)
            import traceback
            traceback.print_exc()
        finally:
            set_optimization_running(False)

    if config.run_in_background:
        thread = threading.Thread(target=_optimize, daemon=True, name="background-optimizer")
        thread.start()
        logger.debug("Started background optimization thread")
    else:
        _optimize()


def _create_lm(model_name: str):
    """Create LM instance for the given model name, handling custom providers."""
    import dspy
    
    if model_name.startswith("zai/"):
        from .zai_lm import ZaiLM
        return ZaiLM(model_name.replace("zai/", ""))
    elif model_name.startswith("kimi/"):
        from .kimi_lm import KimiLM
        model_id = model_name.replace("kimi/", "")
        if model_id == "k2p5":
            model_id = "k2-0130-8k"
        return KimiLM(model_id)
    elif model_name.startswith("anthropic/"):
        from .anthropic_oauth_lm import get_anthropic_api_key, is_oauth_token, AnthropicOAuthLM
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY") or get_anthropic_api_key()
        if api_key and is_oauth_token(api_key):
            return AnthropicOAuthLM(
                model=model_name.replace("anthropic/", ""),
                api_key=api_key,
            )
        else:
            return dspy.LM(model_name, api_key=api_key)
    elif model_name.startswith("google/"):
        from .oauth import get_google_token
        from .google_oauth_lm import GoogleOAuthLM
        token, project_id = get_google_token()
        if token:
            return GoogleOAuthLM(
                model=model_name.replace("google/", ""),
                access_token=token,
                project_id=project_id,
            )
        else:
            return dspy.LM(model_name)
    else:
        return dspy.LM(model_name)


def _extract_simba_rules(program: Any) -> list[str]:
    """Extract rules generated by SIMBA from the optimized program.
    
    SIMBA can generate self-reflective rules that improve performance.
    These are stored in the program's predictors.
    """
    rules = []
    
    try:
        # Check for rules attribute on program
        if hasattr(program, "rules"):
            if isinstance(program.rules, list):
                rules.extend(program.rules)
        
        # Check predictors for rules
        if hasattr(program, "predictors"):
            for predictor in program.predictors():
                if hasattr(predictor, "rules") and isinstance(predictor.rules, list):
                    rules.extend(predictor.rules)
        
        # Check for extended_signature with rules
        if hasattr(program, "extended_signature"):
            sig = program.extended_signature
            if hasattr(sig, "__doc__") and sig.__doc__:
                # Look for SIMBA-added rules in signature doc
                doc = sig.__doc__
                if "RULE:" in doc or "Rule:" in doc:
                    # Extract rule lines
                    for line in doc.split("\n"):
                        line = line.strip()
                        if line.startswith("RULE:") or line.startswith("Rule:"):
                            rules.append(line.split(":", 1)[1].strip())
        
        # Deduplicate while preserving order
        seen = set()
        unique_rules = []
        for rule in rules:
            if rule and rule not in seen:
                seen.add(rule)
                unique_rules.append(rule)
        
        return unique_rules[:10]  # Limit to 10 rules
        
    except Exception as e:
        logger.debug("Failed to extract SIMBA rules: %s", e)
        return []


def _generate_optimized_tips() -> list[str]:
    """Generate tips from failure patterns in traces.
    
    Uses trace data to identify common failure patterns and generate
    actionable tips to prevent them.
    """
    try:
        from .trace_collector import get_trace_collector
        
        collector = get_trace_collector()
        failure_patterns = collector.to_failure_patterns(max_score=0.6, max_patterns=30)
        success_patterns = collector.to_success_patterns(min_score=0.85, max_patterns=20)
        
        if not failure_patterns:
            logger.debug("No failure patterns to generate tips from")
            return []
        
        # Generate tips using LLM
        import dspy
        from .user_config import load_config
        
        config = load_config()
        model = config.get("model", "openrouter/google/gemini-2.0-flash-001")
        
        # Configure LM
        lm = dspy.LM(model)
        
        class GenerateTips(dspy.Signature):
            """Generate actionable tips from failure patterns to prevent future failures.
            
            Analyze the failure patterns and success patterns to identify:
            1. What tools should be used but weren't
            2. What verification steps were skipped
            3. What patterns lead to success vs failure
            
            Generate specific, actionable tips that would prevent the failures.
            """
            failure_patterns: str = dspy.InputField(desc="JSON of failure patterns with queries, tools used, and reasons")
            success_patterns: str = dspy.InputField(desc="JSON of success patterns showing what works")
            tips: list[str] = dspy.OutputField(desc="List of 5-10 specific actionable tips")
        
        with dspy.context(lm=lm):
            generator = dspy.ChainOfThought(GenerateTips)
            result = generator(
                failure_patterns=json.dumps(failure_patterns, indent=2),
                success_patterns=json.dumps(success_patterns, indent=2),
            )
        
        tips = result.tips if isinstance(result.tips, list) else []
        logger.info("Generated %d tips from %d failure patterns", len(tips), len(failure_patterns))
        
        return tips[:10]  # Limit to 10 tips
        
    except Exception as e:
        logger.warning("Failed to generate tips: %s", e)
        return []


def grounded_metric(example: Any, prediction: Any) -> float:
    """
    Metric for evaluating grounded code analysis.

    Scores based on similarity to expected answer (from trace).
    Uses a simple word overlap metric for variance in scores.

    Args:
        example: The input example with expected outputs
        prediction: The model's prediction (DSPy Prediction object or dict)

    Returns:
        Score from 0.0 to 1.0
    """
    # Handle None prediction (from failed runs)
    if prediction is None:
        return 0.0
    
    # Convert prediction to dict if it's a DSPy Prediction object
    if hasattr(prediction, 'toDict'):
        pred_dict = prediction.toDict()
    elif hasattr(prediction, '__dict__'):
        pred_dict = prediction.__dict__
    elif isinstance(prediction, dict):
        pred_dict = prediction
    else:
        # Try to get answer attribute directly
        pred_dict = {"answer": str(prediction) if prediction else ""}

    # Check if answer exists - try multiple possible keys
    answer = (
        pred_dict.get("answer", "") or 
        pred_dict.get("output", "") or 
        pred_dict.get("response", "") or
        pred_dict.get("result", "") or
        ""
    )
    
    if not answer:
        return 0.0

    # Get expected answer from example
    expected = getattr(example, "expected_answer", None) or getattr(example, "answer", None)
    
    if not expected:
        # No expected answer - give base score for having something
        return 0.5 if len(answer) > 10 else 0.3
    
    # Calculate word overlap score for more variance
    answer_lower = answer.lower()
    expected_lower = expected.lower()
    
    # Exact match
    if expected_lower.strip() == answer_lower.strip():
        return 1.0
    
    # Expected contained in answer
    if expected_lower in answer_lower:
        return 0.9
    
    # Word overlap scoring
    expected_words = set(expected_lower.split())
    answer_words = set(answer_lower.split())
    
    if not expected_words:
        return 0.5
    
    # Calculate Jaccard-like overlap
    overlap = len(expected_words & answer_words)
    total = len(expected_words | answer_words)
    
    if total == 0:
        return 0.3
    
    # Scale to 0.3-0.8 range based on overlap
    overlap_score = overlap / len(expected_words)  # Recall-focused
    return 0.3 + (0.5 * overlap_score)


def accuracy_metric(example: Any, prediction: Any) -> float:
    """
    Simple accuracy metric for exact or partial match.

    Args:
        example: The input example
        prediction: The model's prediction (DSPy Prediction object or dict)

    Returns:
        Score from 0.0 to 1.0
    """
    # Handle None prediction
    if prediction is None:
        return 0.0
    
    # Convert prediction to dict if needed
    if hasattr(prediction, 'toDict'):
        pred_dict = prediction.toDict()
    elif hasattr(prediction, '__dict__'):
        pred_dict = prediction.__dict__
    elif isinstance(prediction, dict):
        pred_dict = prediction
    else:
        pred_dict = {"answer": str(prediction) if prediction else ""}
    
    answer = (
        pred_dict.get("answer", "") or 
        pred_dict.get("output", "") or 
        pred_dict.get("response", "") or
        ""
    )
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
        lm: Any = None,
    ) -> tuple[Any, OptimizationResult]:
        """
        Optimize a program using SIMBA.

        Args:
            program: DSPy module to optimize
            trainset: Training examples (list of dspy.Example)
            seed: Random seed for reproducibility
            lm: Optional language model (will create default if not provided)

        Returns:
            Tuple of (optimized_program, optimization_result)
        """
        import dspy

        # Lazy import to avoid startup cost
        try:
            from dspy.teleprompt import SIMBA
        except ImportError:
            logger.error("DSPy SIMBA not available")
            raise ImportError("DSPy SIMBA optimizer not available. Install dspy>=2.5")

        # Configure LM if not already set
        if lm is None and dspy.settings.lm is None:
            from .user_config import load_config
            config = load_config()
            model = config.get("model", "openai/gpt-4o-mini")
            lm = dspy.LM(model)
            logger.info("Created LM for SIMBA: %s", model)

        if lm is not None:
            dspy.configure(lm=lm)

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
        # Use configured num_threads, defaulting to 4 for parallelism
        # dspy.LM is pickleable so threading works fine
        # Default to 2 threads to avoid rate limiting on most APIs
        effective_threads = self.num_threads if self.num_threads is not None else 2
        logger.info("SIMBA using %d threads for parallel evaluation", effective_threads)
        
        optimizer = SIMBA(
            metric=self.metric,
            bsize=batch_size,
            num_candidates=self.num_candidates,
            max_steps=self.max_steps,
            max_demos=self.max_demos,
            num_threads=effective_threads,
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

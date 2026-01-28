"""Hallucination guards using LLM-as-judge validation.

This module provides semantic validation tools to detect LLM hallucinations
by using DSPy's built-in evaluation capabilities.

Example:
    ```python
    from rlm_dspy import RLM, validate_groundedness

    result = rlm.query("Find bugs", context)
    
    # Check if output is grounded in context
    validation = validate_groundedness(result.answer, context, query)
    if not validation.is_grounded:
        print(f"Warning: {validation.score:.0%} grounded")
        print(f"Issues: {validation.discussion}")
    ```
"""

from dataclasses import dataclass


@dataclass
class GroundednessResult:
    """Result of LLM-based groundedness validation."""
    
    score: float
    """Groundedness score (0-1), fraction of claims supported by context."""
    
    claims: str
    """Enumeration of claims found in the output."""
    
    discussion: str
    """Discussion of how well claims are supported."""
    
    is_grounded: bool
    """Whether the output passes the groundedness threshold."""
    
    threshold: float = 0.66
    """Threshold used for is_grounded determination."""


def _get_configured_lm(model: str | None = None):
    """Get a configured DSPy LM using centralized RLMConfig.
    
    Uses the same model/API key resolution as the main RLM class:
    1. Explicit model parameter
    2. Environment variables (RLM_MODEL, RLM_API_KEY)
    3. User config (~/.rlm/config.yaml)
    4. Provider-specific API keys
    """
    import dspy
    from .core.rlm import RLMConfig
    
    # Create config - it handles all resolution automatically
    config = RLMConfig(model=model) if model else RLMConfig()
    
    return dspy.LM(config.model, api_key=config.api_key)


def validate_groundedness(
    output: str,
    context: str,
    query: str,
    threshold: float = 0.66,
    model: str | None = None,
) -> GroundednessResult:
    """Validate output groundedness using DSPy's LLM-as-judge.
    
    Uses DSPy's AnswerGroundedness signature to check if claims
    in the output are supported by the context.
    
    Uses the same model/API key configuration as the main RLM class
    (env vars, ~/.rlm/config.yaml, provider-specific keys).
    
    Args:
        output: LLM output to validate
        context: Source context the output should be grounded in
        query: Original question asked
        threshold: Minimum groundedness score (0-1) to pass (default: 0.66)
        model: Model to use (default: from RLMConfig)
        
    Returns:
        GroundednessResult with score, claims, and discussion
        
    Example:
        ```python
        result = validate_groundedness(
            output="The bug is in process_data() on line 42",
            context="def add(a, b):\\n    return a + b",
            query="Find bugs in this code",
        )
        if not result.is_grounded:
            print(f"Hallucination detected: {result.discussion}")
        ```
    """
    import dspy
    from dspy.evaluate.auto_evaluation import AnswerGroundedness
    from dspy.predict.chain_of_thought import ChainOfThought
    
    lm = _get_configured_lm(model)
    
    # Run with configured LM
    with dspy.settings.context(lm=lm):
        checker = ChainOfThought(AnswerGroundedness)
        result = checker(
            question=query,
            retrieved_context=context,
            system_response=output,
        )
    
    return GroundednessResult(
        score=float(result.groundedness),
        claims=result.system_response_claims,
        discussion=result.discussion,
        is_grounded=float(result.groundedness) >= threshold,
        threshold=threshold,
    )


def validate_completeness(
    output: str,
    expected: str,
    query: str,
    model: str | None = None,
) -> float:
    """Check if output covers expected content using LLM-as-judge.
    
    Uses DSPy's AnswerCompleteness to measure what fraction of
    expected key ideas are present in the output.
    
    Uses the same model/API key configuration as the main RLM class.
    
    Args:
        output: LLM output to validate
        expected: Expected/ground truth response
        query: Original question
        model: Model to use (default: from RLMConfig)
        
    Returns:
        Completeness score (0-1)
    """
    import dspy
    from dspy.evaluate.auto_evaluation import AnswerCompleteness
    from dspy.predict.chain_of_thought import ChainOfThought
    
    lm = _get_configured_lm(model)
    
    with dspy.settings.context(lm=lm):
        checker = ChainOfThought(AnswerCompleteness)
        result = checker(
            question=query,
            ground_truth=expected,
            system_response=output,
        )
    
    return float(result.completeness)


def semantic_f1(
    output: str,
    expected: str,
    query: str,
    decompositional: bool = False,
    model: str | None = None,
) -> float:
    """Calculate semantic F1 score between output and expected.
    
    Uses DSPy's SemanticF1 which measures both precision (output
    claims supported by expected) and recall (expected ideas
    covered by output).
    
    Uses the same model/API key configuration as the main RLM class.
    
    Args:
        output: LLM output to evaluate
        expected: Ground truth/expected response
        query: Original question
        decompositional: Use detailed key-idea decomposition
        model: Model to use (default: from RLMConfig)
        
    Returns:
        F1 score (0-1)
    """
    import dspy
    
    lm = _get_configured_lm(model)
    
    # Create example and prediction objects
    class Example:
        def __init__(self, question, response):
            self.question = question
            self.response = response
    
    class Prediction:
        def __init__(self, response):
            self.response = response
    
    with dspy.settings.context(lm=lm):
        evaluator = dspy.evaluate.SemanticF1(decompositional=decompositional)
        return evaluator(Example(query, expected), Prediction(output))

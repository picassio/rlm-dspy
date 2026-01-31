"""Hallucination guards for RLM outputs.

Provides multiple validation strategies:

1. **Trajectory-based validation** (fast, deterministic):
   - Checks if answer terms appear in code execution outputs
   - Validates that the answer is grounded in what the code found
   - No LLM call required

2. **LLM-as-judge validation** (slower, semantic):
   - Uses DSPy's AnswerGroundedness signature
   - Checks if claims can be deduced from context
   - Good for semantic validation but has LLM overhead

Example:
    ```python
    from rlm_dspy import RLM
    from rlm_dspy.guards import validate_trajectory, validate_groundedness

    result = rlm.query("Find bugs", context)

    # Fast: Check if answer is grounded in trajectory
    traj_result = validate_trajectory(result)
    
    # Slower: LLM-based semantic validation
    llm_result = validate_groundedness(result.answer, context, query)
    ```
"""

import re
from dataclasses import dataclass
from typing import Any


# Threshold for when to use snippet-based validation (in characters)
LARGE_CONTEXT_THRESHOLD = 50_000


def _extract_keywords(text: str, min_length: int = 3) -> set[str]:
    """Extract significant keywords from text for searching."""
    # Extract words, numbers, and identifiers
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
    # Filter short words and common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                  'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                  'during', 'before', 'after', 'above', 'below', 'between',
                  'under', 'again', 'further', 'then', 'once', 'here', 'there',
                  'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
                  'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                  'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and',
                  'but', 'if', 'or', 'because', 'until', 'while', 'this', 'that',
                  'these', 'those', 'what', 'which', 'who', 'whom', 'its', 'it'}
    return {w.lower() for w in words if len(w) >= min_length and w.lower() not in stop_words}


def _extract_specific_terms(text: str) -> set[str]:
    """Extract specific terms like filenames, identifiers, numbers from text."""
    terms = set()
    
    # Extract filenames (word.ext patterns)
    filenames = re.findall(r'\b[\w-]+\.\w+\b', text)
    terms.update(f.lower() for f in filenames)
    
    # Extract identifiers with underscores (function_name, class_name)
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]+\b', text)
    terms.update(i.lower() for i in identifiers if len(i) >= 4)
    
    # Extract quoted strings
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    terms.update(q.lower() for q in quoted if len(q) >= 3)
    
    # Extract numbers that might be significant (line numbers, versions)
    numbers = re.findall(r'\b\d+(?:\.\d+)*\b', text)
    terms.update(numbers)
    
    return terms


def _find_relevant_snippets(
    context: str, 
    output: str, 
    query: str,
    max_snippets: int = 15,
    snippet_size: int = 2000,
    overlap: int = 500,
) -> str:
    """Find context snippets relevant to the output using smart matching.
    
    Uses multiple strategies:
    1. Exact term matching (filenames, identifiers, numbers)
    2. Keyword matching for semantic relevance
    
    This enables validation of large contexts by finding the relevant portions.
    """
    # Extract specific terms (filenames, identifiers) from output
    output_terms = _extract_specific_terms(output)
    query_terms = _extract_specific_terms(query)
    
    # Also get general keywords
    output_keywords = _extract_keywords(output)
    query_keywords = _extract_keywords(query)
    
    # Combine all search terms
    all_terms = output_terms | query_terms | output_keywords | query_keywords
    
    if not all_terms:
        # Fallback: return start of context
        return context[:snippet_size * max_snippets]
    
    # Split context into overlapping chunks
    chunks = []
    pos = 0
    while pos < len(context):
        chunk_end = min(pos + snippet_size, len(context))
        chunks.append((pos, context[pos:chunk_end]))
        pos += snippet_size - overlap
        if pos >= len(context):
            break
    
    # Score each chunk - weight exact term matches higher
    scored_chunks = []
    for start_pos, chunk in chunks:
        chunk_lower = chunk.lower()
        
        # Exact term matches (higher weight)
        exact_score = sum(3 for term in (output_terms | query_terms) if term in chunk_lower)
        
        # Keyword matches (lower weight)
        keyword_score = sum(1 for kw in (output_keywords | query_keywords) if kw in chunk_lower)
        
        total_score = exact_score + keyword_score
        if total_score > 0:
            scored_chunks.append((total_score, start_pos, chunk))
    
    # Sort by score and take top chunks
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    top_chunks = scored_chunks[:max_snippets]
    
    # Sort by position to maintain context order
    top_chunks.sort(key=lambda x: x[1])
    
    if not top_chunks:
        # No matches found, return start of context
        return context[:snippet_size * max_snippets]
    
    # Combine snippets with position markers
    snippets = []
    for score, start_pos, chunk in top_chunks:
        snippets.append(f"[... context at position {start_pos:,} ...]\n{chunk}")
    
    return "\n\n".join(snippets)


# =============================================================================
# Trajectory-based Validation (Fast, Deterministic)
# =============================================================================

@dataclass
class TrajectoryValidationResult:
    """Result of trajectory-based validation."""
    
    score: float
    """Groundedness score (0-1), fraction of answer terms found in outputs."""
    
    found_terms: list[str]
    """Terms from answer that were found in trajectory outputs."""
    
    missing_terms: list[str]
    """Terms from answer that were NOT found in trajectory outputs."""
    
    is_grounded: bool
    """Whether the answer passes the groundedness threshold."""
    
    threshold: float = 0.5
    """Threshold used for is_grounded determination."""


def validate_trajectory(
    result: Any,  # RLMResult
    threshold: float = 0.5,
) -> TrajectoryValidationResult:
    """Validate RLM output against its execution trajectory.
    
    This is a fast, deterministic validation that checks if the answer
    contains terms that appeared in the code execution outputs.
    
    The intuition: if RLM's answer mentions something, that something
    should have appeared in the output of the code it executed.
    
    Args:
        result: RLMResult with trajectory
        threshold: Minimum fraction of terms found (default: 0.5)
        
    Returns:
        TrajectoryValidationResult with score and details
    """
    if not hasattr(result, 'trajectory') or not result.trajectory:
        # No trajectory - can't validate
        return TrajectoryValidationResult(
            score=1.0,
            found_terms=[],
            missing_terms=[],
            is_grounded=True,
            threshold=threshold,
        )
    
    # Collect all outputs from trajectory
    all_outputs = []
    for entry in result.trajectory:
        if isinstance(entry, dict) and 'output' in entry:
            all_outputs.append(str(entry['output']))
    
    combined_output = "\n".join(all_outputs).lower()
    
    # Extract specific terms from the answer
    answer = result.answer if hasattr(result, 'answer') else str(result)
    answer_terms = _extract_specific_terms(answer)
    
    if not answer_terms:
        # No specific terms to validate
        return TrajectoryValidationResult(
            score=1.0,
            found_terms=[],
            missing_terms=[],
            is_grounded=True,
            threshold=threshold,
        )
    
    # Check which terms appear in outputs
    found = []
    missing = []
    for term in answer_terms:
        if term.lower() in combined_output:
            found.append(term)
        else:
            missing.append(term)
    
    score = len(found) / len(answer_terms) if answer_terms else 1.0
    
    return TrajectoryValidationResult(
        score=score,
        found_terms=found,
        missing_terms=missing,
        is_grounded=score >= threshold,
        threshold=threshold,
    )


# =============================================================================
# LLM-as-Judge Validation (Slower, Semantic)
# =============================================================================

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
    
    For Anthropic models with OAuth, uses custom AnthropicOAuthLM.
    """
    import dspy
    from .core.rlm import RLMConfig
    from .core.models import find_model

    # Create config - it handles all resolution automatically
    config = RLMConfig(model=model) if model else RLMConfig()
    
    # Get model info for max_tokens
    model_info = find_model(config.model)
    lm_kwargs = {"api_key": config.api_key}
    if model_info:
        lm_kwargs["max_tokens"] = model_info.max_tokens

    # Check for Anthropic models - may need OAuth
    if config.model.startswith("anthropic/"):
        from .core.anthropic_oauth_lm import get_anthropic_api_key, is_oauth_token, AnthropicOAuthLM
        
        # Check if the configured api_key is actually an Anthropic key
        api_key = config.api_key
        is_anthropic_key = api_key and (
            api_key.startswith("sk-ant-api") or 
            api_key.startswith("sk-ant-oat")
        )
        
        if not is_anthropic_key:
            api_key = get_anthropic_api_key()
        
        if api_key and is_oauth_token(api_key):
            model_id = config.model.split("/", 1)[1]
            return AnthropicOAuthLM(model_id, auth_token=api_key)
    
    # Check for Google models - may need OAuth
    if config.model.startswith("google/"):
        from .core.oauth import get_google_token
        from .core.google_oauth_lm import GoogleOAuthLM
        
        # Check if we have Google OAuth credentials
        google_creds = get_google_token()
        if google_creds:
            model_id = config.model.split("/", 1)[1]
            token, project_id = google_creds
            return GoogleOAuthLM(model_id, auth_token=token, project_id=project_id)

    return dspy.LM(config.model, **lm_kwargs)


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

    For large contexts (>50K chars), uses smart snippet extraction:
    1. Extract keywords from output and query
    2. Find context chunks with highest keyword overlap
    3. Validate against relevant snippets (not truncated context)

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
    
    # For large contexts, find relevant snippets instead of truncating
    if len(context) > LARGE_CONTEXT_THRESHOLD:
        validation_context = _find_relevant_snippets(context, output, query)
    else:
        validation_context = context

    # Run with configured LM
    with dspy.settings.context(lm=lm):
        checker = ChainOfThought(AnswerGroundedness)
        result = checker(
            question=query,
            retrieved_context=validation_context,
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

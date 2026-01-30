"""
Grounded tool wrappers for hallucination prevention.

Wraps tools that make LLM calls (like llm_query) with automatic
grounding validation to catch hallucinations during the REPL
exploration phase, not just at the end.

Usage:
    from rlm_dspy.core.grounded_tools import create_grounded_tools
    
    grounded = create_grounded_tools(tools, context)
    # Use grounded tools in RLM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

_logger = logging.getLogger(__name__)


@dataclass
class GroundedToolConfig:
    """Configuration for grounded tool wrappers."""
    
    enabled: bool = True
    """Whether grounding validation is enabled."""
    
    min_score: float = 0.5
    """Minimum grounding score to accept without warning."""
    
    retry_on_low_score: bool = True
    """Whether to retry with refined prompt on low grounding."""
    
    max_retries: int = 1
    """Maximum retry attempts for low-grounding responses."""
    
    annotate_low_confidence: bool = True
    """Whether to annotate low-confidence responses with warnings."""
    
    cache_validations: bool = True
    """Whether to cache validation results for identical prompts."""
    
    # Internal cache
    _validation_cache: dict[str, float] = field(default_factory=dict, repr=False)


def grounded_llm_query(
    original_func: Callable[[str], str],
    context: str,
    config: GroundedToolConfig | None = None,
) -> Callable[[str], str]:
    """
    Wrap llm_query to validate responses against context.
    
    This catches hallucinations during the REPL exploration phase
    by checking if sub-LLM responses are grounded in the context.
    
    Args:
        original_func: The original llm_query function
        context: The source context for grounding validation
        config: Configuration options
    
    Returns:
        Wrapped function with grounding validation
    """
    config = config or GroundedToolConfig()
    
    @wraps(original_func)
    def wrapped(prompt: str) -> str:
        if not config.enabled:
            return original_func(prompt)
        
        # Check cache
        cache_key = prompt[:500]  # Truncate for reasonable cache keys
        if config.cache_validations and cache_key in config._validation_cache:
            cached_score = config._validation_cache[cache_key]
            _logger.debug("Using cached validation score: %.2f", cached_score)
        
        # Call original function
        response = original_func(prompt)
        
        # Validate grounding
        try:
            from ..guards import validate_groundedness
            
            validation = validate_groundedness(
                output=response,
                context=context,
                query=prompt,
                threshold=config.min_score,
            )
            
            # Cache the score
            if config.cache_validations:
                config._validation_cache[cache_key] = validation.score
            
            if validation.score < config.min_score:
                _logger.warning(
                    "Low grounding (%.0f%%) for llm_query: %s...",
                    validation.score * 100,
                    prompt[:50],
                )
                
                if config.retry_on_low_score and config.max_retries > 0:
                    # Retry with grounding instruction
                    refined_prompt = (
                        f"{prompt}\n\n"
                        "IMPORTANT: Only state facts that can be verified "
                        "from the provided context. Quote relevant code or text "
                        "to support your claims."
                    )
                    
                    _logger.debug("Retrying with refined prompt")
                    response = original_func(refined_prompt)
                    
                    # Re-validate
                    retry_validation = validate_groundedness(
                        output=response,
                        context=context,
                        query=refined_prompt,
                        threshold=config.min_score,
                    )
                    
                    if retry_validation.score >= config.min_score:
                        _logger.debug(
                            "Retry improved grounding: %.0f%% -> %.0f%%",
                            validation.score * 100,
                            retry_validation.score * 100,
                        )
                        validation = retry_validation
                    else:
                        _logger.debug(
                            "Retry did not improve grounding: %.0f%%",
                            retry_validation.score * 100,
                        )
                
                # Annotate if still low confidence
                if config.annotate_low_confidence and validation.score < config.min_score:
                    response = (
                        f"⚠️ [Low confidence - {validation.score:.0%} grounded]\n"
                        f"{response}"
                    )
                    
        except Exception as e:
            _logger.debug("Grounding validation failed: %s", e)
            # Don't fail the call, just log and continue
        
        return response
    
    return wrapped


def grounded_llm_query_batched(
    original_func: Callable[[list[str]], list[str]],
    context: str,
    config: GroundedToolConfig | None = None,
) -> Callable[[list[str]], list[str]]:
    """
    Wrap llm_query_batched to validate responses against context.
    
    Similar to grounded_llm_query but for batched calls.
    
    Args:
        original_func: The original llm_query_batched function
        context: The source context for grounding validation
        config: Configuration options
    
    Returns:
        Wrapped function with grounding validation
    """
    config = config or GroundedToolConfig()
    
    # Create a single-query wrapper to reuse logic
    single_wrapper = grounded_llm_query(
        lambda p: original_func([p])[0],
        context,
        config,
    )
    
    @wraps(original_func)
    def wrapped(prompts: list[str]) -> list[str]:
        if not config.enabled:
            return original_func(prompts)
        
        # Process each prompt individually with validation
        # This is less efficient but ensures each response is validated
        results = []
        for prompt in prompts:
            result = single_wrapper(prompt)
            results.append(result)
        
        return results
    
    return wrapped


def create_grounded_tools(
    tools: dict[str, Callable[..., Any]],
    context: str,
    config: GroundedToolConfig | None = None,
) -> dict[str, Callable[..., Any]]:
    """
    Wrap tools that make LLM calls with grounding validation.
    
    Currently wraps:
    - llm_query: Single LLM sub-query
    - llm_query_batched: Batched LLM sub-queries
    
    Other tools pass through unchanged.
    
    Args:
        tools: Original tool dictionary
        context: Source context for grounding validation
        config: Configuration options
    
    Returns:
        Tools dictionary with grounded wrappers for LLM-calling tools
    
    Example:
        ```python
        from rlm_dspy.core.grounded_tools import create_grounded_tools
        
        config = GroundedToolConfig(min_score=0.6, retry_on_low_score=True)
        grounded_tools = create_grounded_tools(original_tools, context, config)
        ```
    """
    config = config or GroundedToolConfig()
    
    if not config.enabled:
        return tools
    
    wrapped = tools.copy()
    
    # Wrap llm_query
    if "llm_query" in wrapped:
        wrapped["llm_query"] = grounded_llm_query(
            wrapped["llm_query"],
            context,
            config,
        )
        _logger.debug("Wrapped llm_query with grounding validation")
    
    # Wrap llm_query_batched
    if "llm_query_batched" in wrapped:
        wrapped["llm_query_batched"] = grounded_llm_query_batched(
            wrapped["llm_query_batched"],
            context,
            config,
        )
        _logger.debug("Wrapped llm_query_batched with grounding validation")
    
    return wrapped


# Tracking for grounding statistics
@dataclass
class GroundingStats:
    """Statistics about grounding validation during a session."""
    
    total_validations: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    retries_attempted: int = 0
    retries_succeeded: int = 0
    avg_score: float = 0.0
    _scores: list[float] = field(default_factory=list, repr=False)
    
    def record(self, score: float, passed: bool, retried: bool = False, retry_succeeded: bool = False) -> None:
        """Record a validation result."""
        self.total_validations += 1
        self._scores.append(score)
        self.avg_score = sum(self._scores) / len(self._scores)
        
        if passed:
            self.passed_validations += 1
        else:
            self.failed_validations += 1
        
        if retried:
            self.retries_attempted += 1
            if retry_succeeded:
                self.retries_succeeded += 1
    
    def summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "total": self.total_validations,
            "passed": self.passed_validations,
            "failed": self.failed_validations,
            "pass_rate": self.passed_validations / max(1, self.total_validations),
            "avg_score": self.avg_score,
            "retries": self.retries_attempted,
            "retry_success_rate": self.retries_succeeded / max(1, self.retries_attempted),
        }


# Global stats instance (can be replaced per-query)
_grounding_stats: GroundingStats | None = None


def get_grounding_stats() -> GroundingStats:
    """Get or create grounding statistics tracker."""
    global _grounding_stats
    if _grounding_stats is None:
        _grounding_stats = GroundingStats()
    return _grounding_stats


def reset_grounding_stats() -> None:
    """Reset grounding statistics."""
    global _grounding_stats
    _grounding_stats = GroundingStats()


# =============================================================================
# GroundedRLM - Subclass with built-in grounding validation
# =============================================================================

class GroundedRLM:
    """
    A wrapper around dspy.RLM that adds grounding validation to llm_query calls.
    
    This intercepts the internal llm_query and llm_query_batched functions
    to validate that sub-LLM responses are grounded in the provided context,
    catching hallucinations during the exploration phase.
    
    Usage:
        ```python
        from rlm_dspy.core.grounded_tools import GroundedRLM
        
        # Create grounded RLM
        grounded_rlm = GroundedRLM(
            signature="context, query -> answer",
            context=my_context,  # Context for grounding validation
            grounding_config=GroundedToolConfig(min_score=0.5),
        )
        
        # Use like regular dspy.RLM
        result = grounded_rlm(context=my_context, query="Find bugs")
        ```
    """
    
    def __init__(
        self,
        signature: str | type,
        context: str,
        grounding_config: GroundedToolConfig | None = None,
        **rlm_kwargs,
    ):
        """
        Initialize GroundedRLM.
        
        Args:
            signature: DSPy signature for the RLM
            context: The context string used for grounding validation
            grounding_config: Configuration for grounding validation
            **rlm_kwargs: Additional kwargs passed to dspy.RLM
        """
        import dspy
        
        self._context = context
        self._config = grounding_config or GroundedToolConfig()
        self._rlm_kwargs = rlm_kwargs
        self._signature = signature
        self._base_rlm: dspy.RLM | None = None
        self._stats = GroundingStats()
        
    def _create_grounded_llm_tools(self, original_tools: dict[str, Callable]) -> dict[str, Callable]:
        """Wrap the LLM tools with grounding validation."""
        wrapped = original_tools.copy()
        
        if "llm_query" in wrapped and self._config.enabled:
            original_llm_query = wrapped["llm_query"]
            
            def grounded_query(prompt: str) -> str:
                response = original_llm_query(prompt)
                
                # Validate grounding
                if self._config.enabled:
                    try:
                        from ..guards import validate_groundedness
                        
                        validation = validate_groundedness(
                            output=response,
                            context=self._context,
                            query=prompt,
                            threshold=self._config.min_score,
                        )
                        
                        self._stats.record(
                            validation.score, 
                            validation.is_grounded,
                        )
                        
                        if validation.score < self._config.min_score:
                            _logger.debug(
                                "Low grounding (%.0f%%) for llm_query",
                                validation.score * 100,
                            )
                            
                            if self._config.retry_on_low_score:
                                # Retry with grounding instruction
                                refined = original_llm_query(
                                    f"{prompt}\n\nIMPORTANT: Only state facts verifiable from the context."
                                )
                                retry_val = validate_groundedness(
                                    output=refined,
                                    context=self._context,
                                    query=prompt,
                                    threshold=self._config.min_score,
                                )
                                if retry_val.score > validation.score:
                                    self._stats.record(retry_val.score, retry_val.is_grounded, retried=True, retry_succeeded=True)
                                    return refined
                                self._stats.record(retry_val.score, retry_val.is_grounded, retried=True, retry_succeeded=False)
                            
                            if self._config.annotate_low_confidence:
                                return f"⚠️ [{validation.score:.0%} grounded]\n{response}"
                                
                    except Exception as e:
                        _logger.debug("Grounding validation failed: %s", e)
                
                return response
            
            wrapped["llm_query"] = grounded_query
            _logger.debug("Wrapped llm_query with grounding validation")
        
        return wrapped
    
    def __call__(self, **kwargs) -> Any:
        """Execute the RLM with grounded tools."""
        import dspy
        
        # Create fresh RLM for this call
        # Note: We create this INSIDE the dspy.settings.context() call in rlm.py
        # so it inherits the configured LM
        self._base_rlm = dspy.RLM(
            signature=self._signature,
            **self._rlm_kwargs,
        )
        
        # Ensure sub_lm is set - inherit from dspy.settings if not explicitly provided
        if self._base_rlm.sub_lm is None and dspy.settings.lm is not None:
            self._base_rlm.sub_lm = dspy.settings.lm
            _logger.debug("Inherited sub_lm from dspy.settings.lm")
        
        # Store original _make_llm_tools
        original_make_tools = self._base_rlm._make_llm_tools
        
        # Monkey-patch to wrap with grounding
        # Note: _make_llm_tools signature is (self, max_workers=8) -> dict
        def grounded_make_tools(max_workers: int = 8):
            tools = original_make_tools(max_workers)
            return self._create_grounded_llm_tools(tools)
        
        self._base_rlm._make_llm_tools = grounded_make_tools
        
        # Execute
        return self._base_rlm(**kwargs)
    
    def get_grounding_stats(self) -> GroundingStats:
        """Get grounding statistics from this session."""
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset grounding statistics."""
        self._stats = GroundingStats()

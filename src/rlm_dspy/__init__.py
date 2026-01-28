"""RLM-DSPy: Recursive Language Models with DSPy optimization."""

from .core.rlm import RLM, RLMConfig, RLMResult
from .signatures import (
    ArchitectureAnalysis,
    BugFinder,
    CodeReview,
    DiffReview,
    PerformanceAnalysis,
    SecurityAudit,
    SIGNATURES,
    get_signature,
    list_signatures,
)
from .guards import (
    GroundednessResult,
    validate_groundedness,
    validate_completeness,
    semantic_f1,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "RLM",
    "RLMConfig",
    "RLMResult",
    # Signatures
    "SecurityAudit",
    "CodeReview",
    "BugFinder",
    "ArchitectureAnalysis",
    "PerformanceAnalysis",
    "DiffReview",
    "SIGNATURES",
    "get_signature",
    "list_signatures",
    # Hallucination Guards (LLM-as-judge)
    "GroundednessResult",
    "validate_groundedness",
    "validate_completeness",
    "semantic_f1",
]

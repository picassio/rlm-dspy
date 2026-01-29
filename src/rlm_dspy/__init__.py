"""RLM-DSPy: Recursive Language Models with DSPy optimization."""

from .core.rlm import RLM, RLMConfig, RLMResult, ProgressCallback
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
from .tools import (
    BUILTIN_TOOLS,
    SAFE_TOOLS,
    ripgrep,
    grep_context,
    find_files,
    read_file,
    file_stats,
    index_code,
    find_definitions,
    find_classes,
    find_functions,
    find_methods,
    find_imports,
    find_calls,
    semantic_search,
    get_tool_descriptions,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "RLM",
    "RLMConfig",
    "RLMResult",
    "ProgressCallback",
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
    # Tools
    "BUILTIN_TOOLS",
    "SAFE_TOOLS",
    "ripgrep",
    "grep_context",
    "find_files",
    "read_file",
    "file_stats",
    "index_code",
    "find_definitions",
    "find_classes",
    "find_functions",
    "find_methods",
    "find_imports",
    "find_calls",
    "semantic_search",
    "get_tool_descriptions",
]

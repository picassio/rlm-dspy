"""RLM-DSPy: Recursive Language Models with DSPy optimization.

This module uses lazy imports to avoid the ~3s DSPy startup cost when
only using lightweight features like CLI commands that don't need the LLM.
"""


def __getattr__(name: str):
    """Lazy import heavy modules (DSPy) only when accessed."""
    # Core RLM (imports DSPy)
    if name in ("RLM", "RLMConfig", "RLMResult", "ProgressCallback"):
        from .core.rlm import RLM, RLMConfig, RLMResult, ProgressCallback
        return {"RLM": RLM, "RLMConfig": RLMConfig, "RLMResult": RLMResult, "ProgressCallback": ProgressCallback}[name]

    # Signatures (imports DSPy)
    if name in ("ArchitectureAnalysis", "BugFinder", "CodeReview", "DiffReview",
                "PerformanceAnalysis", "SecurityAudit", "CitedAnalysis",
                "CitedSecurityAudit", "CitedBugFinder", "CitedCodeReview",
                "SIGNATURES", "get_signature", "list_signatures"):
        from .signatures import (
            ArchitectureAnalysis, BugFinder, CodeReview, DiffReview,
            PerformanceAnalysis, SecurityAudit, CitedAnalysis,
            CitedSecurityAudit, CitedBugFinder, CitedCodeReview,
            SIGNATURES, get_signature, list_signatures,
        )
        _sigs = {
            "ArchitectureAnalysis": ArchitectureAnalysis,
            "BugFinder": BugFinder,
            "CodeReview": CodeReview,
            "DiffReview": DiffReview,
            "PerformanceAnalysis": PerformanceAnalysis,
            "SecurityAudit": SecurityAudit,
            "CitedAnalysis": CitedAnalysis,
            "CitedSecurityAudit": CitedSecurityAudit,
            "CitedBugFinder": CitedBugFinder,
            "CitedCodeReview": CitedCodeReview,
            "SIGNATURES": SIGNATURES,
            "get_signature": get_signature,
            "list_signatures": list_signatures,
        }
        return _sigs[name]

    # Citations (imports DSPy)
    if name in ("SourceLocation", "CitedFinding", "CitedAnalysisResult",
                "code_to_document", "files_to_documents",
                "citations_to_locations", "parse_findings_from_text"):
        from .core.citations import (
            SourceLocation, CitedFinding, CitedAnalysisResult,
            code_to_document, files_to_documents,
            citations_to_locations, parse_findings_from_text,
        )
        _cites = {
            "SourceLocation": SourceLocation,
            "CitedFinding": CitedFinding,
            "CitedAnalysisResult": CitedAnalysisResult,
            "code_to_document": code_to_document,
            "files_to_documents": files_to_documents,
            "citations_to_locations": citations_to_locations,
            "parse_findings_from_text": parse_findings_from_text,
        }
        return _cites[name]

    # Guards (imports DSPy)
    if name in ("GroundednessResult", "validate_groundedness",
                "validate_completeness", "semantic_f1"):
        from .guards import (
            GroundednessResult, validate_groundedness,
            validate_completeness, semantic_f1,
        )
        _guards = {
            "GroundednessResult": GroundednessResult,
            "validate_groundedness": validate_groundedness,
            "validate_completeness": validate_completeness,
            "semantic_f1": semantic_f1,
        }
        return _guards[name]

    # Tools (mostly lightweight, but imports some DSPy)
    if name in ("BUILTIN_TOOLS", "SAFE_TOOLS", "ripgrep", "grep_context",
                "find_files", "read_file", "file_stats", "index_code",
                "find_definitions", "find_classes", "find_functions",
                "find_methods", "find_imports", "find_calls",
                "semantic_search", "get_tool_descriptions"):
        from .tools import (
            BUILTIN_TOOLS, SAFE_TOOLS, ripgrep, grep_context,
            find_files, read_file, file_stats, index_code,
            find_definitions, find_classes, find_functions,
            find_methods, find_imports, find_calls,
            semantic_search, get_tool_descriptions,
        )
        _tools = {
            "BUILTIN_TOOLS": BUILTIN_TOOLS,
            "SAFE_TOOLS": SAFE_TOOLS,
            "ripgrep": ripgrep,
            "grep_context": grep_context,
            "find_files": find_files,
            "read_file": read_file,
            "file_stats": file_stats,
            "index_code": index_code,
            "find_definitions": find_definitions,
            "find_classes": find_classes,
            "find_functions": find_functions,
            "find_methods": find_methods,
            "find_imports": find_imports,
            "find_calls": find_calls,
            "semantic_search": semantic_search,
            "get_tool_descriptions": get_tool_descriptions,
        }
        return _tools[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # Cited signatures
    "CitedAnalysis",
    "CitedSecurityAudit",
    "CitedBugFinder",
    "CitedCodeReview",
    "SIGNATURES",
    "get_signature",
    "list_signatures",
    # Citations utilities
    "SourceLocation",
    "CitedFinding",
    "CitedAnalysisResult",
    "code_to_document",
    "files_to_documents",
    "citations_to_locations",
    "parse_findings_from_text",
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
    # Cited signatures
    "CitedAnalysis",
    "CitedSecurityAudit",
    "CitedBugFinder",
    "CitedCodeReview",
    "SIGNATURES",
    "get_signature",
    "list_signatures",
    # Citations utilities
    "SourceLocation",
    "CitedFinding",
    "CitedAnalysisResult",
    "code_to_document",
    "files_to_documents",
    "citations_to_locations",
    "parse_findings_from_text",
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

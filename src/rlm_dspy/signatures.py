"""Predefined signatures for common analysis tasks.

These signatures provide structured output instead of free-form text,
making it easier to programmatically process results.

Example:
    ```python
    from rlm_dspy import RLM, RLMConfig
    from rlm_dspy.signatures import SecurityAudit

    rlm = RLM(config=config, signature=SecurityAudit)
    result = rlm.query("Audit this code", context)

    print(result.vulnerabilities)  # list[str]
    print(result.is_secure)        # bool
    print(result.severity)         # str
    ```

Adding custom signatures:
    ```python
    from rlm_dspy.signatures import register_signature
    import dspy

    @register_signature("my_analysis", aliases=["myalias"])
    class MyAnalysis(dspy.Signature):
        '''My custom analysis.'''
        context: str = dspy.InputField()
        query: str = dspy.InputField()
        result: str = dspy.OutputField()
    ```
"""

from __future__ import annotations

from typing import Callable, TypeVar

import dspy

# Internal registry
_SIGNATURES: dict[str, type[dspy.Signature]] = {}

T = TypeVar("T", bound=type[dspy.Signature])


def register_signature(
    name: str,
    aliases: list[str] | None = None,
) -> "Callable[[T], T]":
    """Decorator to register a signature in the registry.

    Args:
        name: Primary name for the signature
        aliases: Optional list of alias names

    Returns:
        Decorator function

    Example:
        @register_signature("security", aliases=["audit"])
        class SecurityAudit(dspy.Signature):
            ...
    """

    def decorator(cls: T) -> T:
        _SIGNATURES[name.lower()] = cls
        if aliases:
            for alias in aliases:
                _SIGNATURES[alias.lower()] = cls
        return cls

    return decorator


@register_signature("security", aliases=["audit"])
class SecurityAudit(dspy.Signature):
    """Analyze code for security vulnerabilities.

    Returns structured security findings with severity levels.
    """
    context: str = dspy.InputField(desc="Code to analyze")
    query: str = dspy.InputField(desc="Specific security concern or 'general audit'")

    vulnerabilities: list[str] = dspy.OutputField(
        desc="List of security issues found (empty if none)"
    )
    severity: str = dspy.OutputField(
        desc="Overall severity: none, low, medium, high, or critical"
    )
    is_secure: bool = dspy.OutputField(
        desc="True if no significant vulnerabilities found"
    )
    recommendations: list[str] = dspy.OutputField(
        desc="Security recommendations and fixes"
    )


@register_signature("review")
class CodeReview(dspy.Signature):
    """Comprehensive code review with quality scoring.

    Returns structured review with issues, suggestions, and quality score.
    """
    context: str = dspy.InputField(desc="Code to review")
    query: str = dspy.InputField(desc="Review focus or 'general review'")

    summary: str = dspy.OutputField(
        desc="Brief summary of the code (1-2 sentences)"
    )
    issues: list[str] = dspy.OutputField(
        desc="Problems found: bugs, code smells, anti-patterns"
    )
    suggestions: list[str] = dspy.OutputField(
        desc="Improvement suggestions"
    )
    quality_score: int = dspy.OutputField(
        desc="Quality score from 1 (poor) to 10 (excellent)"
    )


@register_signature("bugs")
class BugFinder(dspy.Signature):
    """Find bugs and potential issues in code.

    Returns structured bug report with severity classification.
    """
    context: str = dspy.InputField(desc="Code to analyze")
    query: str = dspy.InputField(desc="Specific bug type to look for or 'all bugs'")

    bugs: list[str] = dspy.OutputField(
        desc="List of bugs found with descriptions"
    )
    has_critical: bool = dspy.OutputField(
        desc="True if any critical/blocking bugs found"
    )
    affected_functions: list[str] = dspy.OutputField(
        desc="Names of functions/classes with bugs"
    )
    fix_suggestions: list[str] = dspy.OutputField(
        desc="Suggested fixes for each bug"
    )


@register_signature("architecture", aliases=["arch"])
class ArchitectureAnalysis(dspy.Signature):
    """Analyze code architecture and structure.

    Returns structured overview of codebase organization.
    """
    context: str = dspy.InputField(desc="Code to analyze")
    query: str = dspy.InputField(desc="Architecture aspect to focus on or 'overview'")

    summary: str = dspy.OutputField(
        desc="High-level architecture summary"
    )
    components: list[str] = dspy.OutputField(
        desc="Main components/modules and their purposes"
    )
    dependencies: list[str] = dspy.OutputField(
        desc="Key dependencies between components"
    )
    patterns: list[str] = dspy.OutputField(
        desc="Design patterns identified"
    )


@register_signature("performance", aliases=["perf"])
class PerformanceAnalysis(dspy.Signature):
    """Analyze code for performance issues.

    Returns structured performance findings.
    """
    context: str = dspy.InputField(desc="Code to analyze")
    query: str = dspy.InputField(desc="Performance aspect to focus on or 'general'")

    issues: list[str] = dspy.OutputField(
        desc="Performance issues found"
    )
    hotspots: list[str] = dspy.OutputField(
        desc="Functions/areas likely to be slow"
    )
    optimizations: list[str] = dspy.OutputField(
        desc="Suggested optimizations"
    )
    complexity_concerns: list[str] = dspy.OutputField(
        desc="O(n²) or worse algorithms found"
    )


@register_signature("diff")
class DiffReview(dspy.Signature):
    """Review a code diff/patch.

    Returns structured diff analysis.
    """
    context: str = dspy.InputField(desc="Git diff or patch")
    query: str = dspy.InputField(desc="What to check for or 'general review'")

    summary: str = dspy.OutputField(
        desc="What the diff does (1-2 sentences)"
    )
    change_type: str = dspy.OutputField(
        desc="Type: feature, bugfix, refactor, docs, test, or other"
    )
    is_breaking: bool = dspy.OutputField(
        desc="True if this is a breaking change"
    )
    risks: list[str] = dspy.OutputField(
        desc="Potential risks or issues with the change"
    )
    suggestions: list[str] = dspy.OutputField(
        desc="Suggestions for improving the change"
    )


# =============================================================================
# Cited Signatures (with file:line source references)
# =============================================================================

@register_signature("cited", aliases=["cite"])
class CitedAnalysis(dspy.Signature):
    """Analyze code and provide findings with source citations.

    For each finding, include the exact file and line number.
    Format references as: filename.py:line_number

    Example output:
    - Found SQL injection at db/query.py:45
    - Unused import at utils.py:3
    """
    context: str = dspy.InputField(desc="Source code with line numbers")
    query: str = dspy.InputField(desc="What to analyze or look for")

    summary: str = dspy.OutputField(
        desc="Brief summary of findings"
    )
    findings: list[str] = dspy.OutputField(
        desc="Each finding with file:line citation"
    )
    locations: list[str] = dspy.OutputField(
        desc="List of file:line locations referenced"
    )


@register_signature("cited-security", aliases=["cited-audit"])
class CitedSecurityAudit(dspy.Signature):
    """Security audit with precise source citations.

    Identify security vulnerabilities and reference exact file:line locations.
    Format: [SEVERITY] Description - filename.py:line

    Example:
    - [CRITICAL] SQL injection - db/query.py:45
    - [HIGH] Hardcoded password - config.py:12
    """
    context: str = dspy.InputField(desc="Source code with line numbers")
    query: str = dspy.InputField(desc="Security aspect to focus on or 'full audit'")

    vulnerabilities: list[str] = dspy.OutputField(
        desc="Each vulnerability with [SEVERITY] and file:line citation"
    )
    risk_level: str = dspy.OutputField(
        desc="Overall risk: safe, low, medium, high, or critical"
    )
    locations: list[str] = dspy.OutputField(
        desc="All file:line locations with security issues"
    )
    remediation: list[str] = dspy.OutputField(
        desc="How to fix each vulnerability"
    )


@register_signature("cited-bugs")
class CitedBugFinder(dspy.Signature):
    """Find bugs with precise source citations.

    Identify potential bugs and reference exact file:line locations.
    Format: [TYPE] Description - filename.py:line

    Example:
    - [NULL] Possible null dereference - parser.py:89
    - [LOGIC] Off-by-one error - utils.py:156
    """
    context: str = dspy.InputField(desc="Source code with line numbers")
    query: str = dspy.InputField(desc="Bug type to focus on or 'all bugs'")

    bugs: list[str] = dspy.OutputField(
        desc="Each bug with [TYPE] and file:line citation"
    )
    severity: str = dspy.OutputField(
        desc="Overall severity: none, low, medium, high, or critical"
    )
    locations: list[str] = dspy.OutputField(
        desc="All file:line locations with bugs"
    )
    fixes: list[str] = dspy.OutputField(
        desc="Suggested fix for each bug"
    )


@register_signature("cited-review")
class CitedCodeReview(dspy.Signature):
    """Code review with precise source citations.

    Review code quality and reference exact file:line locations.
    Format: [CATEGORY] Description - filename.py:line

    Example:
    - [STYLE] Missing docstring - api.py:23
    - [SMELL] God class - models.py:1
    """
    context: str = dspy.InputField(desc="Source code with line numbers")
    query: str = dspy.InputField(desc="Review focus or 'general review'")

    issues: list[str] = dspy.OutputField(
        desc="Each issue with [CATEGORY] and file:line citation"
    )
    quality_score: int = dspy.OutputField(
        desc="Quality score from 1 (poor) to 10 (excellent)"
    )
    locations: list[str] = dspy.OutputField(
        desc="All file:line locations with issues"
    )
    improvements: list[str] = dspy.OutputField(
        desc="Suggested improvements for each issue"
    )


# Public alias for the registry (for backwards compatibility)
SIGNATURES = _SIGNATURES


def get_signature(name: str) -> type[dspy.Signature] | None:
    """Get a signature by name.

    Args:
        name: Signature name (e.g., 'security', 'bugs', 'review')

    Returns:
        Signature class or None if not found
    """
    return _SIGNATURES.get(name.lower())


def list_signatures() -> list[str]:
    """List available signature names (excluding aliases)."""
    # Return unique names (not aliases)
    seen = set()
    names = []
    for name, sig in _SIGNATURES.items():
        if sig not in seen:
            names.append(name)
            seen.add(sig)
    return sorted(names)


def get_all_signatures() -> dict[str, type[dspy.Signature]]:
    """Get all registered signatures including aliases.

    Returns:
        Dictionary of name -> signature class
    """
    return dict(_SIGNATURES)


__all__ = [
    # Signature classes
    "SecurityAudit",
    "CodeReview",
    "BugFinder",
    "ArchitectureAnalysis",
    "PerformanceAnalysis",
    "DiffReview",
    # Cited signatures (with file:line references)
    "CitedAnalysis",
    "CitedSecurityAudit",
    "CitedBugFinder",
    "CitedCodeReview",
    # Registry
    "SIGNATURES",
    "register_signature",
    "get_signature",
    "list_signatures",
    "get_all_signatures",
]

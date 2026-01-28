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
"""

import dspy


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


# Signature registry for CLI lookup
SIGNATURES = {
    "security": SecurityAudit,
    "review": CodeReview,
    "bugs": BugFinder,
    "architecture": ArchitectureAnalysis,
    "performance": PerformanceAnalysis,
    "diff": DiffReview,
    # Aliases
    "audit": SecurityAudit,
    "perf": PerformanceAnalysis,
    "arch": ArchitectureAnalysis,
}


def get_signature(name: str) -> type[dspy.Signature] | None:
    """Get a signature by name.
    
    Args:
        name: Signature name (e.g., 'security', 'bugs', 'review')
        
    Returns:
        Signature class or None if not found
    """
    return SIGNATURES.get(name.lower())


def list_signatures() -> list[str]:
    """List available signature names."""
    # Return unique names (not aliases)
    seen = set()
    names = []
    for name, sig in SIGNATURES.items():
        if sig not in seen:
            names.append(name)
            seen.add(sig)
    return sorted(names)


__all__ = [
    "SecurityAudit",
    "CodeReview", 
    "BugFinder",
    "ArchitectureAnalysis",
    "PerformanceAnalysis",
    "DiffReview",
    "SIGNATURES",
    "get_signature",
    "list_signatures",
]

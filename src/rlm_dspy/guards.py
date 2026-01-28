"""Hallucination guards and validation utilities.

This module provides tools to detect and prevent common LLM hallucinations
by validating outputs against the source context.

Two types of validation are available:
1. **Fast regex-based**: Immediate, no LLM calls (validate_line_numbers, etc.)
2. **LLM-as-judge**: Uses DSPy's semantic evaluation (validate_groundedness)

Example:
    ```python
    from rlm_dspy import RLM, RLMConfig
    from rlm_dspy.guards import validate_all, validate_groundedness

    result = rlm.query("Find bugs", context)
    
    # Fast regex validation
    issues = validate_all(result.answer, context)
    
    # LLM-as-judge validation (more accurate, slower)
    groundedness = validate_groundedness(result.answer, context, query)
    print(f"Groundedness score: {groundedness.score:.0%}")
    ```
"""

import re
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of hallucination validation."""
    
    is_valid: bool
    """Whether the output passed validation."""
    
    issues: list[str] = field(default_factory=list)
    """List of potential hallucination issues found."""
    
    confidence: float = 1.0
    """Confidence score (0-1) that output is grounded in context."""
    
    def __bool__(self) -> bool:
        return self.is_valid


def validate_line_numbers(output: str, context: str) -> ValidationResult:
    """Check if line numbers mentioned in output exist in context.
    
    Detects hallucinated line references like "line 500" when the file
    only has 100 lines.
    
    Args:
        output: LLM output text
        context: Original source context
        
    Returns:
        ValidationResult with any invalid line references
    """
    issues = []
    
    # Count actual lines in context
    context_lines = context.count('\n') + 1
    
    # Find line number references in output
    # Patterns: "line 42", "Line 42", "L42", "at line 42", "on line 42"
    patterns = [
        r'[Ll]ine\s*(\d+)',
        r'[Ll]ines?\s*(\d+)\s*[-–]\s*(\d+)',  # line ranges
        r'L(\d+)',
        r':(\d+):',  # file:line:col format
    ]
    
    mentioned_lines = set()
    for pattern in patterns:
        for match in re.finditer(pattern, output):
            for group in match.groups():
                if group and group.isdigit():
                    mentioned_lines.add(int(group))
    
    # Check for invalid references
    for line_num in mentioned_lines:
        if line_num > context_lines:
            issues.append(
                f"Line {line_num} referenced but context only has {context_lines} lines"
            )
        elif line_num < 1:
            issues.append(f"Invalid line number: {line_num}")
    
    confidence = 1.0 - (len(issues) / max(len(mentioned_lines), 1))
    
    return ValidationResult(
        is_valid=len(issues) == 0,
        issues=issues,
        confidence=max(0.0, confidence),
    )


def validate_references(
    output: str,
    context: str,
    check_functions: bool = True,
    check_classes: bool = True,
    check_files: bool = True,
) -> ValidationResult:
    """Check if identifiers mentioned in output exist in context.
    
    Detects hallucinated function names, class names, or file paths
    that don't appear in the source context.
    
    Args:
        output: LLM output text
        context: Original source context
        check_functions: Validate function name references
        check_classes: Validate class name references
        check_files: Validate file path references
        
    Returns:
        ValidationResult with any ungrounded references
    """
    issues = []
    context_lower = context.lower()
    
    if check_functions:
        # Find function references: func(), function_name(), etc.
        func_pattern = r'\b([a-z_][a-z0-9_]*)\s*\('
        mentioned_funcs = set(re.findall(func_pattern, output.lower()))
        
        # Common false positives to ignore
        ignore_funcs = {
            'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set',
            'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'sum',
            'min', 'max', 'abs', 'round', 'open', 'type', 'isinstance',
            'hasattr', 'getattr', 'setattr', 'input', 'format', 'repr',
            'if', 'for', 'while', 'def', 'class', 'return', 'import',
        }
        
        for func in mentioned_funcs - ignore_funcs:
            # Check if function exists in context (as definition or call)
            if f'def {func}' not in context_lower and f'{func}(' not in context_lower:
                # Only flag if it looks like a specific reference
                if len(func) > 3 and '_' in func:
                    issues.append(f"Function '{func}' not found in context")
    
    if check_classes:
        # Find class references: ClassName, MyClass, etc.
        class_pattern = r'\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*)\b'
        mentioned_classes = set(re.findall(class_pattern, output))
        
        # Common false positives
        ignore_classes = {
            'None', 'True', 'False', 'Exception', 'Error', 'Warning',
            'String', 'Integer', 'Boolean', 'List', 'Dict', 'Set',
            'Type', 'Any', 'Optional', 'Union', 'Callable',
            'API', 'URL', 'HTTP', 'JSON', 'XML', 'HTML', 'CSS', 'SQL',
            'ID', 'UUID', 'ASCII', 'UTF', 'IO', 'OS',
        }
        
        for cls in mentioned_classes - ignore_classes:
            if f'class {cls}' not in context and cls not in context:
                # Only flag multi-word class names (likely specific references)
                if len(cls) > 5 and any(c.islower() for c in cls):
                    issues.append(f"Class '{cls}' not found in context")
    
    if check_files:
        # Find file references: file.py, path/to/file.js, etc.
        file_pattern = r'[\w/\\.-]+\.(?:py|js|ts|java|cpp|c|h|go|rs|rb|php|sh|yaml|yml|json|md|txt)'
        mentioned_files = set(re.findall(file_pattern, output))
        
        for filepath in mentioned_files:
            # Check if file is mentioned in context
            filename = filepath.split('/')[-1].split('\\')[-1]
            if filename not in context and filepath not in context:
                issues.append(f"File '{filepath}' not found in context")
    
    # Calculate confidence based on ratio of valid references
    total_checks = len(issues) + 10  # Assume ~10 valid refs baseline
    confidence = 1.0 - (len(issues) / total_checks)
    
    return ValidationResult(
        is_valid=len(issues) == 0,
        issues=issues,
        confidence=max(0.0, min(1.0, confidence)),
    )


def validate_code_blocks(output: str, context: str) -> ValidationResult:
    """Check if code snippets in output match context.
    
    Detects when LLM invents code that doesn't exist in the source.
    
    Args:
        output: LLM output text
        context: Original source context
        
    Returns:
        ValidationResult with any fabricated code blocks
    """
    issues = []
    
    # Extract code blocks from output
    code_block_pattern = r'```(?:\w+)?\n(.*?)```'
    code_blocks = re.findall(code_block_pattern, output, re.DOTALL)
    
    # Also check inline code
    inline_pattern = r'`([^`]+)`'
    inline_codes = re.findall(inline_pattern, output)
    
    for block in code_blocks:
        # Normalize whitespace for comparison
        block_normalized = ' '.join(block.split())
        context_normalized = ' '.join(context.split())
        
        # Check if significant portions exist in context
        # Use sliding window to find matches
        block_lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if len(block_lines) > 2:
            # For multi-line blocks, check if at least 50% of lines exist
            matches = sum(1 for line in block_lines if line in context)
            if matches < len(block_lines) * 0.5:
                preview = block_lines[0][:50] + '...' if len(block_lines[0]) > 50 else block_lines[0]
                issues.append(f"Code block may be fabricated: '{preview}'")
    
    for code in inline_codes:
        # Skip short inline code (likely formatting)
        if len(code) > 20 and '(' in code and code not in context:
            # Looks like a function call or statement
            if re.match(r'^\w+\(.*\)$', code.strip()):
                issues.append(f"Inline code may be fabricated: '{code[:40]}...'")
    
    confidence = 1.0 - (len(issues) * 0.2)  # Each issue reduces confidence by 20%
    
    return ValidationResult(
        is_valid=len(issues) == 0,
        issues=issues,
        confidence=max(0.0, confidence),
    )


def validate_all(output: str, context: str) -> ValidationResult:
    """Run all fast (regex-based) hallucination checks.
    
    Combines line number, reference, and code block validation.
    This is fast and doesn't require LLM calls.
    
    For deeper semantic validation, use validate_groundedness().
    
    Args:
        output: LLM output text
        context: Original source context
        
    Returns:
        Combined ValidationResult from all checks
    """
    results = [
        validate_line_numbers(output, context),
        validate_references(output, context),
        validate_code_blocks(output, context),
    ]
    
    all_issues = []
    for r in results:
        all_issues.extend(r.issues)
    
    avg_confidence = sum(r.confidence for r in results) / len(results)
    
    return ValidationResult(
        is_valid=len(all_issues) == 0,
        issues=all_issues,
        confidence=avg_confidence,
    )


# =============================================================================
# LLM-as-Judge Validation (uses DSPy's built-in evaluation)
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


def validate_groundedness(
    output: str,
    context: str,
    query: str,
    threshold: float = 0.66,
    model: str | None = None,
) -> GroundednessResult:
    """Validate output groundedness using DSPy's LLM-as-judge.
    
    This uses DSPy's AnswerGroundedness signature to check if claims
    in the output are supported by the context. More accurate than
    regex-based validation but requires an LLM call.
    
    Args:
        output: LLM output to validate
        context: Source context the output should be grounded in
        query: Original question asked
        threshold: Minimum groundedness score (0-1) to pass
        model: Model to use for validation (default: from env or gpt-4o-mini)
        
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
            print(f"Output may be hallucinated: {result.discussion}")
        ```
    """
    import os
    import dspy
    from dspy.evaluate.auto_evaluation import AnswerGroundedness
    from dspy.predict.chain_of_thought import ChainOfThought
    
    # Configure LM - use same resolution as RLM class
    if model is None:
        model = os.environ.get("RLM_MODEL")
        if not model:
            try:
                from .core.user_config import get_config_value
                model = get_config_value("model")
            except Exception:
                pass
        model = model or "openai/gpt-4o-mini"
    
    # Get API key for the model
    from .core.rlm import get_provider_env_var
    api_key = None
    if env_var := get_provider_env_var(model):
        api_key = os.environ.get(env_var)
    if not api_key:
        api_key = os.environ.get("RLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    
    lm = dspy.LM(model, api_key=api_key)
    
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
    threshold: float = 0.66,
    model: str | None = None,
) -> float:
    """Check if output covers expected content using LLM-as-judge.
    
    Uses DSPy's AnswerCompleteness to measure what fraction of
    expected key ideas are present in the output.
    
    Args:
        output: LLM output to validate
        expected: Expected/ground truth response
        query: Original question
        threshold: Minimum completeness to pass
        model: Model to use for validation (default: from env or gpt-4o-mini)
        
    Returns:
        Completeness score (0-1)
    """
    import os
    import dspy
    from dspy.evaluate.auto_evaluation import AnswerCompleteness
    from dspy.predict.chain_of_thought import ChainOfThought
    
    # Configure LM - use same resolution as RLM class
    if model is None:
        model = os.environ.get("RLM_MODEL")
        if not model:
            try:
                from .core.user_config import get_config_value
                model = get_config_value("model")
            except Exception:
                pass
        model = model or "openai/gpt-4o-mini"
    
    from .core.rlm import get_provider_env_var
    api_key = None
    if env_var := get_provider_env_var(model):
        api_key = os.environ.get(env_var)
    if not api_key:
        api_key = os.environ.get("RLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    
    lm = dspy.LM(model, api_key=api_key)
    
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
    
    Args:
        output: LLM output to evaluate
        expected: Ground truth/expected response
        query: Original question
        decompositional: Use detailed key-idea decomposition
        model: Model to use for evaluation (default: from env or gpt-4o-mini)
        
    Returns:
        F1 score (0-1)
    """
    import os
    import dspy
    
    # Configure LM - use same resolution as RLM class
    if model is None:
        model = os.environ.get("RLM_MODEL")
        if not model:
            try:
                from .core.user_config import get_config_value
                model = get_config_value("model")
            except Exception:
                pass
        model = model or "openai/gpt-4o-mini"
    
    from .core.rlm import get_provider_env_var
    api_key = None
    if env_var := get_provider_env_var(model):
        api_key = os.environ.get(env_var)
    if not api_key:
        api_key = os.environ.get("RLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    
    lm = dspy.LM(model, api_key=api_key)
    
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

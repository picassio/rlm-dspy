# Implementation Plan: Advanced dspy.RLM Features

## Overview

Add three major features to rlm-dspy:
1. **Custom Interpreter Support** - Allow pluggable code execution backends
2. **Batch Processing** - Process multiple queries in parallel
3. **Custom Signatures** - Structured output schemas

---

## Phase 1: Custom Interpreter Support

### Goal
Allow users to provide custom code execution environments (E2B, Modal, Docker, etc.)

### API Design

```python
from rlm_dspy import RLM, RLMConfig

# Default: uses dspy's PythonInterpreter (Deno/Pyodide)
rlm = RLM(config=config)

# Custom interpreter
from e2b_code_interpreter import CodeInterpreter as E2BInterpreter
rlm = RLM(config=config, interpreter=E2BInterpreter())

# Or via config
config = RLMConfig(
    model="openai/gpt-4o",
    interpreter="e2b",  # shorthand
)
```

### Implementation

1. **Add to RLMConfig**:
```python
@dataclass
class RLMConfig:
    # ... existing fields ...
    interpreter: str | None = None  # "local", "e2b", "modal", or None (auto)
```

2. **Add to RLM.__init__**:
```python
def __init__(
    self,
    config: RLMConfig | None = None,
    tools: dict[str, Callable[..., str]] | None = None,
    signature: str = "context, query -> answer",
    interpreter: CodeInterpreter | None = None,  # NEW
):
```

3. **Pass to dspy.RLM**:
```python
def _create_rlm(self) -> dspy.RLM:
    return dspy.RLM(
        signature=self._signature,
        # ... existing params ...
        interpreter=self._interpreter,  # NEW
    )
```

### CLI Integration
```bash
# Use E2B sandbox
rlm-dspy ask "Find bugs" src/ --interpreter e2b

# Use local (default)
rlm-dspy ask "Find bugs" src/ --interpreter local
```

### Files to Modify
- `src/rlm_dspy/core/rlm.py` - Add interpreter parameter
- `src/rlm_dspy/cli.py` - Add --interpreter option
- `pyproject.toml` - Add optional e2b dependency

---

## Phase 2: Batch Processing

### Goal
Process multiple queries in parallel for faster analysis.

### Use Cases

1. **analyze command** - 3 independent queries → run in parallel
2. **Multi-file analysis** - Analyze each file independently
3. **Multi-query** - Ask multiple questions about same context

### API Design

```python
from rlm_dspy import RLM, RLMConfig

rlm = RLM(config=config)
context = rlm.load_context(["src/"])

# Single query (existing)
result = rlm.query("Find bugs", context)

# Batch queries (NEW)
results = rlm.batch([
    {"query": "Summarize the architecture"},
    {"query": "Find security issues"},
    {"query": "Find performance bottlenecks"},
], context=context, num_threads=3)

# Or with different contexts
results = rlm.batch([
    {"context": file1_content, "query": "Find bugs"},
    {"context": file2_content, "query": "Find bugs"},
], num_threads=4)
```

### Implementation

1. **Add batch() method to RLM class**:
```python
def batch(
    self,
    queries: list[dict[str, str]],
    context: str | None = None,
    num_threads: int = 4,
    max_errors: int | None = None,
) -> list[RLMResult]:
    """
    Process multiple queries in parallel.
    
    Args:
        queries: List of {"query": str} or {"context": str, "query": str}
        context: Shared context (used if not in query dict)
        num_threads: Parallel threads
        max_errors: Max failures before stopping
    
    Returns:
        List of RLMResult in same order as queries
    """
    import dspy
    
    examples = []
    for q in queries:
        ctx = q.get("context", context)
        if ctx is None:
            raise ValueError("Context required")
        examples.append(dspy.Example(
            context=ctx,
            query=q["query"],
        ).with_inputs("context", "query"))
    
    with dspy.settings.context(lm=self._lm):
        results = self._rlm.batch(
            examples,
            num_threads=num_threads,
            max_errors=max_errors,
        )
    
    return [self._wrap_result(r) for r in results]
```

2. **Update analyze command**:
```python
@app.command()
def analyze(paths, ...):
    rlm = RLM(config=config)
    context = rlm.load_context([str(p) for p in paths])
    
    # OLD: Sequential
    # structure = rlm.query("List files...", context)
    # components = rlm.query("Identify components...", context)
    # issues = rlm.query("Find issues...", context)
    
    # NEW: Parallel
    results = rlm.batch([
        {"query": "List all files and their purposes"},
        {"query": "Identify main components, classes, functions"},
        {"query": "Find potential bugs, code smells, improvements"},
    ], context=context, num_threads=3)
    
    structure, components, issues = results
```

### CLI Integration
```bash
# Analyze with parallel queries (automatic)
rlm-dspy analyze src/

# Multi-file batch
rlm-dspy ask "Find bugs" file1.py file2.py file3.py --batch
```

### Files to Modify
- `src/rlm_dspy/core/rlm.py` - Add batch() method
- `src/rlm_dspy/cli.py` - Update analyze, add --batch flag

---

## Phase 3: Custom Signatures

### Goal
Allow structured output schemas instead of just string answers.

### Use Cases

1. **Security audit** → `code -> vulnerabilities: list[str], severity: str, is_secure: bool`
2. **Code review** → `code -> summary: str, issues: list[Issue], suggestions: list[str]`
3. **Bug finding** → `code -> bugs: list[Bug], has_critical: bool`

### API Design

```python
from rlm_dspy import RLM, RLMConfig

# Default signature (string answer)
rlm = RLM(config=config)
result = rlm.query("Find bugs", context)
print(result.answer)  # str

# Custom signature (structured output)
rlm = RLM(
    config=config,
    signature="context, query -> summary: str, bugs: list[str], severity: str"
)
result = rlm.query("Find bugs", context)
print(result.summary)      # str
print(result.bugs)         # list[str]
print(result.severity)     # str

# Predefined signatures
from rlm_dspy.signatures import SecurityAudit, CodeReview, BugReport

rlm = RLM(config=config, signature=SecurityAudit)
result = rlm.query("Audit this code", context)
print(result.vulnerabilities)  # list[str]
print(result.is_secure)        # bool
```

### Predefined Signatures

```python
# src/rlm_dspy/signatures.py

import dspy

class SecurityAudit(dspy.Signature):
    """Analyze code for security vulnerabilities."""
    context: str = dspy.InputField()
    query: str = dspy.InputField()
    
    vulnerabilities: list[str] = dspy.OutputField(desc="List of security issues")
    severity: str = dspy.OutputField(desc="Overall severity: low, medium, high, critical")
    is_secure: bool = dspy.OutputField(desc="Whether the code is secure")
    recommendations: list[str] = dspy.OutputField(desc="Security recommendations")

class CodeReview(dspy.Signature):
    """Comprehensive code review."""
    context: str = dspy.InputField()
    query: str = dspy.InputField()
    
    summary: str = dspy.OutputField(desc="Brief summary of the code")
    issues: list[str] = dspy.OutputField(desc="Problems found")
    suggestions: list[str] = dspy.OutputField(desc="Improvement suggestions")
    quality_score: int = dspy.OutputField(desc="Quality score 1-10")

class BugReport(dspy.Signature):
    """Find bugs in code."""
    context: str = dspy.InputField()
    query: str = dspy.InputField()
    
    bugs: list[str] = dspy.OutputField(desc="List of bugs found")
    has_critical: bool = dspy.OutputField(desc="Whether any bugs are critical")
    affected_functions: list[str] = dspy.OutputField(desc="Functions with bugs")
```

### Implementation

1. **Update RLM class**:
```python
def __init__(
    self,
    config: RLMConfig | None = None,
    tools: dict[str, Callable[..., str]] | None = None,
    signature: str | type[dspy.Signature] = "context, query -> answer",  # Allow class
    interpreter: CodeInterpreter | None = None,
):
    self._signature = signature
```

2. **Update RLMResult to handle structured output**:
```python
@dataclass
class RLMResult:
    answer: str  # Always present (stringified)
    success: bool
    
    # Structured output fields (if custom signature)
    outputs: dict[str, Any] = field(default_factory=dict)
    
    def __getattr__(self, name: str) -> Any:
        """Allow accessing output fields as attributes."""
        if name in self.outputs:
            return self.outputs[name]
        raise AttributeError(f"No output field '{name}'")
```

3. **Create signatures module**:
```
src/rlm_dspy/signatures.py  # NEW
```

### CLI Integration
```bash
# Use predefined signature
rlm-dspy ask "Audit this" src/ --signature security

# Inline signature
rlm-dspy ask "Review" src/ --signature "context, query -> bugs: list[str], score: int"

# JSON output for structured results
rlm-dspy ask "Find bugs" src/ --signature bugs --format json
```

### Files to Modify
- `src/rlm_dspy/core/rlm.py` - Handle signature classes
- `src/rlm_dspy/signatures.py` - NEW: Predefined signatures
- `src/rlm_dspy/cli.py` - Add --signature option
- `src/rlm_dspy/__init__.py` - Export signatures

---

## Implementation Order

### Priority 1: Batch Processing (High Impact, Low Risk)
- Immediate benefit for `analyze` command (3x faster)
- No breaking changes
- Simple implementation

### Priority 2: Custom Signatures (High Impact, Medium Risk)
- Enables structured output
- Need to update RLMResult
- Create signatures module

### Priority 3: Custom Interpreter (Medium Impact, Low Risk)
- Enables advanced sandboxing
- Optional feature
- Requires optional dependencies

---

## Testing Plan

### Unit Tests
- `test_batch.py` - Batch processing
- `test_signatures.py` - Custom signatures
- `test_interpreter.py` - Interpreter options

### Integration Tests
```bash
# Batch
rlm-dspy analyze src/rlm_dspy/core/rlm.py

# Signatures
rlm-dspy ask "Find bugs" src/ --signature bugs --format json

# Interpreter
rlm-dspy ask "Test" src/ --interpreter local
```

---

## Timeline

| Phase | Feature | Effort | Files |
|-------|---------|--------|-------|
| 1 | Batch Processing | 2 hours | rlm.py, cli.py |
| 2 | Custom Signatures | 3 hours | rlm.py, cli.py, signatures.py |
| 3 | Custom Interpreter | 1 hour | rlm.py, cli.py |
| 4 | Tests + Docs | 2 hours | tests/, README.md |

**Total: ~8 hours**

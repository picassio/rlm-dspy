# Implementation Plan: Hallucination Reduction in rlm-dspy

## Overview

This plan addresses the hallucination issues identified in the analysis:
1. Silent interpreter failures causing 100% hallucination
2. Post-hoc only validation
3. Unprotected intermediate llm_query() calls
4. Manual-only optimization

**Estimated Total Effort:** 3-4 days
**Priority Order:** P0 (critical) → P1 (high) → P2 (medium)

---

## Phase 1: Fail-Fast on Interpreter Failure (P0 - Critical)

**Goal:** Prevent catastrophic hallucination when Deno/Pyodide is unavailable

### Task 1.1: Add Interpreter Health Check

**File:** `src/rlm_dspy/core/rlm.py`

**Location:** `_get_or_create_interpreter()` method (lines 862-880)

**Changes:**
```python
def _get_or_create_interpreter(self):
    """Get or create a persistent interpreter for reuse."""
    if self._interpreter is not None:
        return self._interpreter

    if self._persistent_interpreter is None:
        try:
            from dspy.primitives.python_interpreter import PythonInterpreter
            self._persistent_interpreter = PythonInterpreter()
            
            # NEW: Verify interpreter actually works
            self._verify_interpreter(self._persistent_interpreter)
            
            _logger.debug("Created persistent interpreter for reuse")
        except Exception as e:
            _logger.error("Failed to create interpreter: %s", e)
            raise InterpreterError(
                "Code interpreter unavailable. Tools will not work.\n"
                "Install Deno: curl -fsSL https://deno.land/install.sh | sh\n"
                "Add to PATH: export PATH=\"$HOME/.deno/bin:$PATH\""
            ) from e

    return self._persistent_interpreter

def _verify_interpreter(self, interpreter) -> None:
    """Verify interpreter can execute code."""
    try:
        # Simple test execution
        result = interpreter.execute("x = 1 + 1", {"__builtins__": {}})
        if isinstance(result, dict) and result.get("error"):
            raise InterpreterError(f"Interpreter test failed: {result['error']}")
    except Exception as e:
        raise InterpreterError(f"Interpreter verification failed: {e}") from e
```

**New Exception Class:**
```python
class InterpreterError(Exception):
    """Raised when code interpreter is unavailable or broken."""
    pass
```

**Effort:** 2 hours

### Task 1.2: Add Preflight Interpreter Check

**File:** `src/rlm_dspy/core/validation.py`

**Changes:** Add new validation function

```python
def check_interpreter() -> ValidationResult:
    """Check if code interpreter (Deno/Pyodide) is available."""
    import shutil
    
    # Check for Deno in PATH
    deno_path = shutil.which("deno")
    if deno_path:
        return ValidationResult(
            name="Interpreter",
            passed=True,
            message=f"Deno found: {deno_path}",
        )
    
    # Check ~/.deno/bin
    deno_home = Path.home() / ".deno" / "bin" / "deno"
    if deno_home.exists():
        return ValidationResult(
            name="Interpreter",
            passed=False,
            message="Deno installed but not in PATH",
            severity="error",
            suggestion='Add to PATH: export PATH="$HOME/.deno/bin:$PATH"',
        )
    
    return ValidationResult(
        name="Interpreter",
        passed=False,
        message="Deno not installed",
        severity="error",
        suggestion="Install: curl -fsSL https://deno.land/install.sh | sh",
    )
```

**Update `preflight_check()`** to include interpreter check by default.

**Effort:** 1 hour

### Task 1.3: Update CLI Preflight

**File:** `src/rlm_dspy/cli.py`

**Changes:** Add interpreter check to `preflight` command and `ask` command

```python
# In ask command, before running query
if not dry_run:
    from .core.validation import check_interpreter
    interp_check = check_interpreter()
    if not interp_check.passed:
        console.print(f"[red]✗ {interp_check.message}[/red]")
        console.print(f"[dim]{interp_check.suggestion}[/dim]")
        raise typer.Exit(1)
```

**Effort:** 30 minutes

---

## Phase 2: Default Validation in Core API (P1 - High)

**Goal:** Ensure all API users get hallucination detection by default

### Task 2.1: Add Validation to RLM.query()

**File:** `src/rlm_dspy/core/rlm.py`

**Location:** `query()` method (around line 1150)

**Changes:**
```python
def query(
    self,
    query: str,
    context: str,
    validate: bool | None = None,  # NEW parameter
) -> RLMResult:
    """
    Query the RLM with automatic validation.
    
    Args:
        query: The question to answer
        context: The context to explore
        validate: Whether to validate output for hallucinations.
                  Default: True (from config.validate)
    """
    # ... existing code ...
    
    result = self._build_result(prediction, elapsed)
    
    # NEW: Automatic validation
    should_validate = validate if validate is not None else self.config.validate
    if should_validate and result.success:
        validation = self._validate_result(result, context, query)
        result.metadata["validation"] = {
            "score": validation.score,
            "is_grounded": validation.is_grounded,
            "claims": validation.claims,
        }
        if not validation.is_grounded:
            _logger.warning(
                "Low grounding score: %.0f%% - %s",
                validation.score * 100,
                validation.discussion[:100],
            )
    
    return result

def _validate_result(
    self,
    result: RLMResult,
    context: str,
    query: str,
) -> "GroundednessResult":
    """Validate result using LLM-as-judge."""
    from ..guards import validate_groundedness
    return validate_groundedness(result.answer, context, query)
```

**Effort:** 1.5 hours

### Task 2.2: Add Validation Config Options

**File:** `src/rlm_dspy/core/rlm.py`

**Update RLMConfig:**
```python
@dataclass
class RLMConfig:
    # ... existing fields ...
    
    # Validation settings
    validate: bool = field(
        default_factory=lambda: _env_get(
            "RLM_VALIDATE", _get_user_config_default("validate", True)
        )
    )
    validation_threshold: float = field(
        default_factory=lambda: _env_get(
            "RLM_VALIDATION_THRESHOLD",
            _get_user_config_default("validation_threshold", 0.66)
        )
    )
```

**Effort:** 30 minutes

---

## Phase 3: Real-Time Grounding in REPL Loop (P1 - High)

**Goal:** Catch hallucinations during exploration, not just at the end

### Task 3.1: Create Grounded Tool Wrapper

**File:** `src/rlm_dspy/core/grounded_tools.py` (NEW)

```python
"""
Wrappers for tools that add grounding validation.
"""

import logging
from typing import Callable
from functools import wraps

from ..guards import validate_groundedness

logger = logging.getLogger(__name__)


class GroundedToolConfig:
    """Configuration for grounded tool wrappers."""
    
    enabled: bool = True
    min_score: float = 0.5
    retry_on_low_score: bool = True
    max_retries: int = 1


def grounded_llm_query(
    original_func: Callable,
    context: str,
    config: GroundedToolConfig | None = None,
) -> Callable:
    """
    Wrap llm_query to validate responses against context.
    
    Args:
        original_func: The original llm_query function
        context: The source context for grounding
        config: Configuration options
    
    Returns:
        Wrapped function with grounding validation
    """
    config = config or GroundedToolConfig()
    
    @wraps(original_func)
    def wrapped(prompt: str) -> str:
        if not config.enabled:
            return original_func(prompt)
        
        response = original_func(prompt)
        
        # Validate grounding
        try:
            validation = validate_groundedness(response, context, prompt)
            
            if validation.score < config.min_score:
                logger.warning(
                    "Low grounding (%.0f%%) for llm_query: %s",
                    validation.score * 100,
                    prompt[:50],
                )
                
                if config.retry_on_low_score:
                    # Retry with grounding instruction
                    refined_prompt = (
                        f"{prompt}\n\n"
                        "IMPORTANT: Only state facts that can be verified "
                        "from the provided context. Quote relevant code."
                    )
                    response = original_func(refined_prompt)
                else:
                    # Annotate response with warning
                    response = (
                        f"⚠️ [Low confidence - {validation.score:.0%} grounded]\n"
                        f"{response}"
                    )
        except Exception as e:
            logger.debug("Grounding validation failed: %s", e)
        
        return response
    
    return wrapped


def create_grounded_tools(
    tools: dict[str, Callable],
    context: str,
    config: GroundedToolConfig | None = None,
) -> dict[str, Callable]:
    """
    Wrap tools that make LLM calls with grounding validation.
    
    Args:
        tools: Original tool dictionary
        context: Source context for grounding
        config: Configuration options
    
    Returns:
        Tools dictionary with grounded wrappers
    """
    config = config or GroundedToolConfig()
    wrapped = tools.copy()
    
    # Wrap LLM-calling tools
    llm_tools = ["llm_query", "llm_query_batched"]
    for name in llm_tools:
        if name in wrapped:
            wrapped[name] = grounded_llm_query(wrapped[name], context, config)
    
    return wrapped
```

**Effort:** 2 hours

### Task 3.2: Integrate Grounded Tools into RLM

**File:** `src/rlm_dspy/core/rlm.py`

**Location:** `query()` method

**Changes:**
```python
def query(self, query: str, context: str, validate: bool | None = None) -> RLMResult:
    # ... existing setup ...
    
    # NEW: Wrap tools with grounding validation if enabled
    if self.config.validate:
        from .grounded_tools import create_grounded_tools, GroundedToolConfig
        grounded_config = GroundedToolConfig(
            enabled=True,
            min_score=self.config.validation_threshold,
        )
        # Note: This requires modifying how tools are passed to dspy.RLM
        # May need to rebuild RLM with grounded tools for this query
```

**Note:** This is complex because dspy.RLM's llm_query is internal. May need to:
1. Fork/extend dspy.RLM, OR
2. Use a callback mechanism to intercept llm_query calls

**Effort:** 3-4 hours (depends on dspy.RLM architecture)

---

## Phase 4: Auto-Triggered Optimization (P2 - Medium)

**Goal:** System improves itself when quality degrades

### Task 4.1: Add Failure Rate Tracking

**File:** `src/rlm_dspy/core/grounded_proposer.py`

**Add method:**
```python
def get_recent_failure_rate(self, window: int = 20) -> float:
    """Get failure rate from recent queries.
    
    Args:
        window: Number of recent queries to consider
    
    Returns:
        Failure rate (0.0 to 1.0)
    """
    recent_failures = self.failures[-window:] if self.failures else []
    recent_successes = self.successes[-window:] if self.successes else []
    
    total = len(recent_failures) + len(recent_successes)
    if total == 0:
        return 0.0
    
    return len(recent_failures) / total

def should_suggest_optimization(self, threshold: float = 0.3) -> bool:
    """Check if optimization should be suggested.
    
    Args:
        threshold: Failure rate threshold to trigger suggestion
    
    Returns:
        True if optimization is recommended
    """
    return self.get_recent_failure_rate() > threshold
```

**Effort:** 1 hour

### Task 4.2: Add Auto-Optimization Suggestion to CLI

**File:** `src/rlm_dspy/cli.py`

**Location:** After validation in `ask` command

```python
# After validation
if not validation.is_grounded:
    # ... existing failure recording ...
    
    # NEW: Check if optimization should be suggested
    if proposer.should_suggest_optimization():
        failure_rate = proposer.get_recent_failure_rate()
        console.print(
            f"\n[yellow]⚠ High failure rate ({failure_rate:.0%}) - "
            f"consider running optimization:[/yellow]"
        )
        console.print("  [dim]rlm-dspy optimize simba[/dim]")
```

**Effort:** 30 minutes

### Task 4.3: Add Auto-Optimization Mode

**File:** `src/rlm_dspy/cli.py`

**New command option:**
```python
@optimize_app.command("auto")
def optimize_auto(
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Failure rate threshold"),
    ] = 0.3,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Run even if threshold not met"),
    ] = False,
) -> None:
    """Auto-run optimization if failure rate is high.
    
    Checks recent failure rate and runs SIMBA if above threshold.
    
    Examples:
        rlm-dspy optimize auto              # Run if failure rate > 30%
        rlm-dspy optimize auto -t 0.2       # Lower threshold
        rlm-dspy optimize auto --force      # Always run
    """
    from .core.grounded_proposer import get_grounded_proposer
    
    proposer = get_grounded_proposer()
    failure_rate = proposer.get_recent_failure_rate()
    
    console.print(f"[bold]Auto-Optimization Check[/bold]")
    console.print(f"  Recent failure rate: {failure_rate:.0%}")
    console.print(f"  Threshold: {threshold:.0%}")
    
    if failure_rate > threshold or force:
        console.print("\n[green]→ Running SIMBA optimization...[/green]")
        # Call existing simba command logic
        ctx = typer.Context(optimize_simba)
        ctx.invoke(optimize_simba)
    else:
        console.print("\n[dim]Failure rate below threshold - no optimization needed[/dim]")
```

**Effort:** 1 hour

---

## Phase 5: BootstrapFewShot Integration (P2 - Medium)

**Goal:** Use collected traces as few-shot examples

### Task 5.1: Add Bootstrap Mode to Trace Collector

**File:** `src/rlm_dspy/core/trace_collector.py`

**Add method:**
```python
def get_demos_for_query_type(
    self,
    query_type: str,
    max_demos: int = 3,
) -> list[str]:
    """Get formatted demos for a query type.
    
    Args:
        query_type: Type of query (bugs, security, review, etc.)
        max_demos: Maximum demos to return
    
    Returns:
        List of formatted demo strings
    """
    # Filter by type and sort by score
    matching = [
        t for t in self.traces
        if t.query_type == query_type and t.grounded_score >= 0.9
    ]
    matching.sort(key=lambda t: t.grounded_score, reverse=True)
    
    return [t.format_as_demo() for t in matching[:max_demos]]

def to_dspy_examples(
    self,
    min_score: float = 0.8,
    max_examples: int = 100,
) -> list:
    """Convert traces to DSPy Example objects for bootstrapping.
    
    Args:
        min_score: Minimum grounded score
        max_examples: Maximum examples to return
    
    Returns:
        List of dspy.Example objects
    """
    import dspy
    
    examples = []
    for trace in self.traces:
        if trace.grounded_score < min_score:
            continue
        
        example = dspy.Example(
            query=trace.query,
            context="",  # Context is typically large, omit
        ).with_inputs("query", "context")
        
        example.answer = trace.final_answer
        example.tools_used = trace.tools_used
        
        examples.append(example)
        
        if len(examples) >= max_examples:
            break
    
    return examples
```

**Effort:** 1.5 hours

### Task 5.2: Add Bootstrap Command

**File:** `src/rlm_dspy/cli.py`

```python
@optimize_app.command("bootstrap")
def optimize_bootstrap(
    min_score: Annotated[
        float,
        typer.Option("--min-score", "-s", help="Minimum trace score"),
    ] = 0.9,
    max_demos: Annotated[
        int,
        typer.Option("--max-demos", "-n", help="Maximum demos per predictor"),
    ] = 3,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Save bootstrapped program"),
    ] = None,
) -> None:
    """Bootstrap few-shot examples from collected traces.
    
    Uses DSPy's BootstrapFewShot to create optimized demos.
    
    Examples:
        rlm-dspy optimize bootstrap
        rlm-dspy optimize bootstrap --min-score 0.95
        rlm-dspy optimize bootstrap -o optimized_rlm.json
    """
    from dspy.teleprompt import BootstrapFewShot
    from .core.trace_collector import get_trace_collector
    from .core.simba_optimizer import grounded_metric
    from .core import RLM, RLMConfig
    
    collector = get_trace_collector()
    examples = collector.to_dspy_examples(min_score=min_score)
    
    if len(examples) < 10:
        console.print(f"[yellow]Only {len(examples)} qualifying traces[/yellow]")
        console.print("[dim]Collect more high-quality traces first[/dim]")
        raise typer.Exit(1)
    
    console.print(f"[bold]BootstrapFewShot[/bold]")
    console.print(f"  Traces: {len(examples)}")
    console.print(f"  Max demos: {max_demos}")
    
    # Create base RLM
    rlm = RLM(config=RLMConfig())
    
    # Bootstrap
    teleprompter = BootstrapFewShot(
        metric=grounded_metric,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=0,
    )
    
    optimized = teleprompter.compile(rlm._rlm, trainset=examples)
    
    console.print(f"\n[green]✓ Bootstrapped {max_demos} demos[/green]")
    
    if output:
        # Save optimized program
        optimized.save(str(output))
        console.print(f"[dim]Saved to {output}[/dim]")
```

**Effort:** 2 hours

---

## Implementation Schedule

### Day 1: Critical Fixes (Phase 1)
- [ ] Task 1.1: Interpreter health check (2h)
- [ ] Task 1.2: Preflight interpreter check (1h)
- [ ] Task 1.3: CLI preflight update (0.5h)
- [ ] Testing and verification (1h)

### Day 2: Core API Validation (Phase 2)
- [ ] Task 2.1: Add validation to RLM.query() (1.5h)
- [ ] Task 2.2: Add validation config options (0.5h)
- [ ] Testing and verification (1h)
- [ ] Start Phase 3 research (1h)

### Day 3: Real-Time Grounding (Phase 3)
- [ ] Task 3.1: Create grounded tool wrapper (2h)
- [ ] Task 3.2: Integrate into RLM (3-4h)
- [ ] Testing (1h)

### Day 4: Auto-Optimization & Bootstrap (Phase 4-5)
- [ ] Task 4.1: Failure rate tracking (1h)
- [ ] Task 4.2: Auto-optimization suggestion (0.5h)
- [ ] Task 4.3: Auto-optimization mode (1h)
- [ ] Task 5.1: Bootstrap mode for traces (1.5h)
- [ ] Task 5.2: Bootstrap command (2h)
- [ ] Final testing and documentation (1h)

---

## Testing Plan

### Unit Tests

**File:** `tests/test_hallucination_fixes.py`

```python
import pytest
from unittest.mock import Mock, patch

class TestInterpreterHealthCheck:
    def test_verify_interpreter_success(self):
        """Test interpreter verification passes with working interpreter."""
        pass
    
    def test_verify_interpreter_failure(self):
        """Test interpreter verification fails without Deno."""
        pass
    
    def test_preflight_detects_missing_deno(self):
        """Test preflight check catches missing Deno."""
        pass

class TestValidation:
    def test_query_validates_by_default(self):
        """Test that RLM.query() validates output by default."""
        pass
    
    def test_query_validation_can_be_disabled(self):
        """Test validation can be disabled via parameter."""
        pass
    
    def test_low_grounding_logged(self):
        """Test that low grounding scores are logged."""
        pass

class TestGroundedTools:
    def test_llm_query_wrapper_validates(self):
        """Test grounded llm_query wrapper validates responses."""
        pass
    
    def test_low_score_triggers_retry(self):
        """Test low grounding triggers retry with refined prompt."""
        pass

class TestAutoOptimization:
    def test_failure_rate_calculation(self):
        """Test failure rate is calculated correctly."""
        pass
    
    def test_optimization_suggested_at_threshold(self):
        """Test optimization is suggested when threshold exceeded."""
        pass
```

### Integration Tests

```python
class TestEndToEnd:
    def test_query_with_working_interpreter(self):
        """Test full query flow with working interpreter."""
        pass
    
    def test_query_fails_fast_without_interpreter(self):
        """Test query fails immediately without Deno."""
        pass
    
    def test_hallucination_detected_and_logged(self):
        """Test hallucinations are caught by validation."""
        pass
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Hallucination rate with broken interpreter | 100% | 0% (fail-fast) |
| Queries with validation | ~50% (CLI only) | 100% (default) |
| Average grounding score | Unknown | Track and improve |
| Auto-optimization triggers | 0 | When failure rate > 30% |

---

## Rollback Plan

All changes are backward compatible:
1. New config options have sensible defaults
2. Validation can be disabled via `validate=False`
3. New CLI commands are additive
4. Existing traces remain compatible

If issues arise:
1. Set `RLM_VALIDATE=false` to disable validation
2. Use `--no-validate` flag in CLI
3. Revert specific commits as needed

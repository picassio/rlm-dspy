# RLM-DSPY Testing Documentation

This document demonstrates the capabilities of `rlm-dspy` through comprehensive testing. All tests were run against the `rlm-dspy` codebase itself, verifying accuracy, anti-hallucination measures, and relationship tracking.

## Test Environment

- **Date**: January 28, 2026
- **Version**: 0.2.0 (dspy.RLM refactor)
- **Architecture**: REPL-based exploration using `dspy.RLM`
- **Test Target**: `src/rlm_dspy/` (the tool analyzing itself)

---

## How RLM Works

Unlike traditional chunking approaches, RLM uses a **REPL-based exploration** model:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Files loaded as 'context' variable in Python REPL          │
│  2. LLM writes Python code to explore the context              │
│  3. LLM can call llm_query() for semantic analysis             │
│  4. LLM iterates until it has enough info, then calls SUBMIT() │
└─────────────────────────────────────────────────────────────────┘
```

This gives the LLM **agency** to decide what's relevant, rather than relying on fixed chunking.

---

## Part 1: Accuracy Tests

### Test 1.1: Finding Module-Level Variables

**Query**: List ALL module-level variables in this file with their exact values

**Target**: `src/rlm_dspy/core/retry.py`

```bash
rlm-dspy ask "List ALL module-level variables in this file with their exact values" \
    src/rlm_dspy/core/retry.py
```

```
╭─────────────────────────────────── Answer ───────────────────────────────────╮
│ The following are the module-level variables in src/rlm_dspy/core/retry.py   │
│ with their exact values:                                                     │
│                                                                              │
│  1 logger:                                                                   │
│     logger = logging.getLogger(__name__)                                     │
│                                                                              │
│  2 T:                                                                        │
│     T = TypeVar("T")                                                         │
│                                                                              │
│  3 RETRYABLE_NETWORK_EXCEPTIONS:                                             │
│     RETRYABLE_NETWORK_EXCEPTIONS: tuple[type[Exception], ...] = (            │
│         httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,         │
│         httpx.WriteTimeout, httpx.PoolTimeout, httpx.NetworkError,           │
│         httpx.RemoteProtocolError, ConnectionError, TimeoutError, OSError,   │
│     )                                                                        │
│                                                                              │
│  4 RETRYABLE_STATUS_CODES:                                                   │
│     RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504, 520, 521, 522,   │
│         523, 524}                                                            │
│                                                                              │
│  5 __all__:                                                                  │
│     __all__ = ["retry_with_backoff", "retry_sync", "parse_retry_after",      │
│         "is_rate_limit_error", "is_retryable_error",                         │
│         "RETRYABLE_NETWORK_EXCEPTIONS", "RETRYABLE_STATUS_CODES"]            │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Verification**: ✅ All 5 variables found with correct values

---

### Test 1.2: Function Signature Extraction

**Query**: What are the exact parameters of the 'atomic_write' function? Include types and defaults.

**Target**: `src/rlm_dspy/core/fileutils.py`

```bash
rlm-dspy ask "What are the exact parameters of the 'atomic_write' function? Include types and defaults." \
    src/rlm_dspy/core/fileutils.py
```

```
╭─────────────────────────────────── Answer ───────────────────────────────────╮
│ The exact parameters of the 'atomic_write' function, including types and     │
│ defaults, are:                                                               │
│                                                                              │
│  • path: Path                                                                │
│  • content: str | bytes                                                      │
│  • mode: str = "w"                                                           │
│  • retries: int = 3                                                          │
│  • retry_delay: float = 0.1                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Verification**: ✅ Exact match with source code

```python
# Actual source code (src/rlm_dspy/core/fileutils.py)
def atomic_write(
    path: Path,
    content: str | bytes,
    mode: str = "w",
    retries: int = 3,
    retry_delay: float = 0.1,
) -> None:
```

---

## Part 2: Anti-Hallucination Tests

### Test 2.1: Non-Existent Class/Function Detection

**Query**: Is there a class called 'DatabaseConnection' in this file? Is there a function called 'connect_to_server'?

**Target**: `src/rlm_dspy/core/validation.py`

```bash
rlm-dspy ask "Is there a class called 'DatabaseConnection' in this file? Is there a function called 'connect_to_server'?" \
    src/rlm_dspy/core/validation.py
```

```
╭─────────────────────────────────── Answer ───────────────────────────────────╮
│ No, there is no class called 'DatabaseConnection' and no function called     │
│ 'connect_to_server' in this file.                                            │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Verification**: ✅ Correctly reported non-existent items as "not found"

---

## Part 3: Function Relationship Tests

### Test 3.1: Call Chain Tracing (New RLM Architecture)

**Query**: Trace the call chain: What functions does RLM.query() call? Show the execution flow.

**Target**: `src/rlm_dspy/core/rlm.py`

```bash
rlm-dspy ask "Trace the call chain: What functions does RLM.query() call? Show the execution flow." \
    src/rlm_dspy/core/rlm.py
```

```
╭─────────────────────────────────── Answer ───────────────────────────────────╮
│ The execution flow of RLM.query() follows this chain:                        │
│                                                                              │
│  1 RLM.query(query, context): The user-facing method starting the process.   │
│     • Starts a timer (time.time()).                                          │
│  2 self._rlm(context=context, query=query): Delegates the core logic to the  │
│    internal dspy.RLM instance.                                               │
│     • The _rlm instance is initialized in the constructor via _create_rlm(). │
│     • _create_rlm() configures the dspy.RLM module with settings from        │
│       RLMConfig (max iterations, budget, etc.).                              │
│     • _create_sub_lm(): Called during initialization to provide the language │
│       model used for semantic analysis inside the REPL (llm_query).          │
│  3 dspy.RLM.forward() (Internal to DSPy): Although the code shows the call   │
│    as self._rlm(...), in DSPy modules this invokes the forward method which  │
│    manages the iterative REPL loop.                                          │
│  4 RLMResult(...): Once the REPL loop finishes (either through SUBMIT() or   │
│    hitting limits), RLM.query captures the prediction and wraps it in an     │
│    RLMResult dataclass containing:                                           │
│     • answer: The final string submitted.                                    │
│     • trajectory: The step-by-step REPL history/code execution.              │
│     • metadata: Elapsed time, success status, and iteration count.           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Verification**: ✅ Accurately describes the new dspy.RLM-based execution flow

---

### Test 3.2: Module Dependency Analysis

**Query**: Create a dependency graph showing which modules in src/rlm_dspy/core/ depend on which other modules.

**Target**: `src/rlm_dspy/core/`

```bash
rlm-dspy ask "Create a dependency graph showing which modules depend on which other modules." \
    src/rlm_dspy/core/
```

**Verification**: ✅ Accurate module dependency mapping

---

## Part 4: Tree-Sitter Index Tests

The `rlm-dspy index` command uses tree-sitter for 100% accurate AST-based code indexing.

### Test 4.1: Class Index

```bash
rlm-dspy index src/rlm_dspy/core/rlm.py --kind class
```

```
                     Code Index (6 definitions)                      
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Kind  ┃ Name                    ┃ Line ┃ File                     ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ class │ RLMConfig               │  131 │ src/rlm_dspy/core/rlm.py │
│ class │ RLMResult               │  204 │ src/rlm_dspy/core/rlm.py │
│ class │ RLM                     │  231 │ src/rlm_dspy/core/rlm.py │
│ class │ BudgetExceededError     │  524 │ src/rlm_dspy/core/rlm.py │
│ class │ TimeoutExceededError    │  531 │ src/rlm_dspy/core/rlm.py │
│ class │ TokenLimitExceededError │  538 │ src/rlm_dspy/core/rlm.py │
└───────┴─────────────────────────┴──────┴──────────────────────────┘
```

**Verification**: ✅ All classes found with exact line numbers

### Test 4.2: Function Index with Name Filter

```bash
rlm-dspy index src/rlm_dspy/core/ --kind function --name "retry"
```

**Verification**: ✅ Filters work correctly

---

## Part 5: CLI Command Tests

### Test 5.1: Config Command

```bash
rlm-dspy config
```

```
                             RLM-DSPy Configuration                             
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Setting          ┃ Value                       ┃ Env Var                     ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Model            │ openrouter/google/gemini-3… │ RLM_MODEL                   │
│ Sub Model        │ openrouter/google/gemini-3… │ RLM_SUB_MODEL               │
│ API Base         │ (auto)                      │ RLM_API_BASE                │
│ API Key          │ ***                         │ RLM_API_KEY /               │
│                  │                             │ OPENROUTER_API_KEY          │
│ Max Iterations   │ 20                          │ RLM_MAX_ITERATIONS          │
│ Max LLM Calls    │ 50                          │ RLM_MAX_LLM_CALLS           │
│ Max Output Chars │ 100,000                     │ RLM_MAX_OUTPUT_CHARS        │
│ Verbose          │ False                       │ RLM_VERBOSE                 │
│ Max Budget       │ $1.00                       │ RLM_MAX_BUDGET              │
│ Max Timeout      │ 300s                        │ RLM_MAX_TIMEOUT             │
└──────────────────┴─────────────────────────────┴─────────────────────────────┘

RLM Mode: REPL-based exploration (dspy.RLM)
```

**Verification**: ✅ Shows new RLM settings

### Test 5.2: Preflight Command

```bash
rlm-dspy preflight
```

```
                    Preflight Checks                    
┏━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Check        ┃ Status ┃ Message                      ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ API Key      │   ✓    │ Found in $OPENROUTER_API_KEY │
│ Model Format │   ✓    │ Valid provider: openrouter   │
└──────────────┴────────┴──────────────────────────────┘

✓ All preflight checks passed!
```

**Verification**: ✅ Preflight checks work

### Test 5.3: Analyze Command

```bash
rlm-dspy analyze src/rlm_dspy/core/rlm.py
```

**Verification**: ✅ Generates comprehensive analysis with structure, components, and issues

---

## Part 6: Test Summary

| Test Category | Test Name | Result | Description |
|--------------|-----------|--------|-------------|
| **Accuracy** | Module Variables | ✅ Pass | Found all 5 variables with exact values |
| **Accuracy** | Function Signature | ✅ Pass | 100% match with source code |
| **Anti-Hallucination** | Non-Existent Items | ✅ Pass | Correctly said "not found" |
| **Relationships** | Call Chain (dspy.RLM) | ✅ Pass | Accurate new architecture flow |
| **Tree-Sitter** | Class Index | ✅ Pass | All classes with line numbers |
| **CLI** | Config | ✅ Pass | Shows RLM settings |
| **CLI** | Preflight | ✅ Pass | Validates configuration |
| **CLI** | Analyze | ✅ Pass | Multi-query analysis |

---

## Key Features Demonstrated

### 1. REPL-Based Exploration (New!)
The LLM writes Python code to explore your context, giving it agency to find relevant information.

### 2. Sub-LLM Queries
The LLM can call `llm_query()` for semantic analysis of specific sections.

### 3. Tree-Sitter Index (Preserved)
The `rlm-dspy index` command uses tree-sitter for 100% accurate AST-based queries.

### 4. Anti-Hallucination
When asked about non-existent code, the tool correctly reports "not found".

---

## Running These Tests

### Prerequisites

```bash
# Install Deno (required for sandboxed REPL)
curl -fsSL https://deno.land/install.sh | sh
export PATH="$HOME/.deno/bin:$PATH"
```

### Test Commands

```bash
# Test 1: Module variables
rlm-dspy ask "List ALL module-level variables with exact values" \
    src/rlm_dspy/core/retry.py

# Test 2: Function signature
rlm-dspy ask "What are the exact parameters of 'atomic_write'?" \
    src/rlm_dspy/core/fileutils.py

# Test 3: Anti-hallucination
rlm-dspy ask "Is there a class called 'DatabaseConnection'?" \
    src/rlm_dspy/core/validation.py

# Test 4: Call chain (new RLM architecture)
rlm-dspy ask "Trace the call chain: What does RLM.query() call?" \
    src/rlm_dspy/core/rlm.py

# Test 5: Tree-sitter index
rlm-dspy index src/rlm_dspy/core/rlm.py --kind class

# Test 6: Config
rlm-dspy config

# Test 7: Preflight
rlm-dspy preflight
```

---

## Architecture Comparison

### Old Architecture (Chunking)

```
Files → Tree-sitter chunks → Map-reduce/Parallel → Aggregate → Answer
```

- Fixed chunk boundaries
- System decides what's relevant
- Parallel but stateless

### New Architecture (dspy.RLM)

```
Files → Context string → LLM explores via REPL → SUBMIT(answer)
```

- LLM has agency to explore
- Can call `llm_query()` for semantic analysis
- Iterative with persistent state
- More flexible for complex queries

---

## Conclusion

All tests pass, demonstrating that `rlm-dspy` with the new dspy.RLM architecture:

1. ✅ **Accurately extracts** code information (variables, functions, classes)
2. ✅ **Does not hallucinate** - correctly reports when things don't exist
3. ✅ **Tracks relationships** using REPL-based exploration
4. ✅ **Preserves tree-sitter** for the `index` command
5. ✅ **Provides LLM agency** to explore contexts programmatically

The tool is production-ready for codebase analysis tasks.

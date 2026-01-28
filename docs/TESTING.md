# RLM-DSPY Testing Documentation

This document demonstrates the capabilities of `rlm-dspy` through comprehensive testing. All tests were run against the `rlm-dspy` codebase itself, verifying accuracy and anti-hallucination measures.

## Test Environment

- **Date**: January 28, 2026
- **Version**: 0.2.0 (dspy.RLM refactor)
- **Architecture**: REPL-based exploration using `dspy.RLM`
- **Test Target**: `src/rlm_dspy/` (the tool analyzing itself)

---

## How RLM Works

RLM uses a **REPL-based exploration** model where the LLM has agency to explore data:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Files loaded as 'context' variable in Python REPL          │
│  2. LLM writes Python code to explore the context              │
│  3. LLM can call llm_query() for semantic analysis             │
│  4. LLM iterates until it has enough info, then calls SUBMIT() │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Accuracy Tests

### Test 1.1: Finding Module-Level Variables

**Query**: List ALL module-level variables in this file with their exact values

**Target**: `src/rlm_dspy/core/retry.py`

```bash
rlm-dspy ask "List ALL module-level variables in this file with their exact values" \
    src/rlm_dspy/core/retry.py
```

**Result**: ✅ All 5 variables found with correct values:
- `logger`
- `T` (TypeVar)
- `RETRYABLE_NETWORK_EXCEPTIONS`
- `RETRYABLE_STATUS_CODES`
- `__all__`

---

### Test 1.2: Function Signature Extraction

**Query**: What are the exact parameters of the 'atomic_write' function?

**Target**: `src/rlm_dspy/core/fileutils.py`

**Result**: ✅ Exact match with source code:
```python
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

**Query**: Is there a class called 'DatabaseConnection' in this file?

**Target**: `src/rlm_dspy/core/validation.py`

**Result**: ✅ Correctly reported: "No, there is no class called 'DatabaseConnection'"

---

## Part 3: Call Chain Tests

### Test 3.1: RLM Architecture Flow

**Query**: Trace the call chain: What functions does RLM.query() call?

**Target**: `src/rlm_dspy/core/rlm.py`

**Result**: ✅ Accurately describes the dspy.RLM execution flow:
1. `RLM.query()` starts timer
2. `self._rlm(context, query)` delegates to dspy.RLM
3. dspy.RLM manages REPL loop
4. Returns `RLMResult` with answer, trajectory, metadata

---

## Part 4: CLI Command Tests

### Test 4.1: Config Command

```bash
rlm-dspy config
```

**Result**: ✅ Shows all RLM settings with env vars

### Test 4.2: Preflight Command

```bash
rlm-dspy preflight
```

**Result**: ✅ Validates API key and model format

### Test 4.3: Analyze Command

```bash
rlm-dspy analyze src/rlm_dspy/core/rlm.py
```

**Result**: ✅ Generates comprehensive analysis

---

## Test Summary

| Test Category | Test Name | Result |
|--------------|-----------|--------|
| **Accuracy** | Module Variables | ✅ Pass |
| **Accuracy** | Function Signature | ✅ Pass |
| **Anti-Hallucination** | Non-Existent Items | ✅ Pass |
| **Relationships** | Call Chain (dspy.RLM) | ✅ Pass |
| **CLI** | Config | ✅ Pass |
| **CLI** | Preflight | ✅ Pass |
| **CLI** | Analyze | ✅ Pass |

---

## Running Tests

### Prerequisites

```bash
# Install Deno (required for sandboxed REPL)
curl -fsSL https://deno.land/install.sh | sh
export PATH="$HOME/.deno/bin:$PATH"
```

### Quick Test

```bash
# Functional test
echo "def hello(): return 'world'" | rlm-dspy ask "What does this do?" --stdin

# Config check
rlm-dspy preflight

# Run unit tests
python -m pytest tests/ -v
```

---

## Conclusion

All tests pass, demonstrating that `rlm-dspy` with dspy.RLM:

1. ✅ **Accurately extracts** code information
2. ✅ **Does not hallucinate** - correctly reports when things don't exist
3. ✅ **Tracks relationships** using REPL-based exploration
4. ✅ **Provides LLM agency** to explore contexts programmatically

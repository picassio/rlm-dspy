# Hallucination Analysis for rlm-dspy

## Summary

This analysis used rlm-dspy to investigate its own hallucination vulnerabilities. 
The first run **without Deno in PATH** produced 100% hallucinated results (inventing 
fake files like `src/rlm_dspy/repl/agent.py`). The second run **with Deno properly configured** 
produced 100% grounded results verified by LLM-as-judge.

## Root Causes of Hallucination

### 1. Interpreter Dependency Failure (Critical)

**Location**: `src/rlm_dspy/core/rlm.py` (lines 862-880, 882-907)

When Deno/Pyodide isn't available:
- Code execution silently fails
- Tools (`read_file`, `ripgrep`, `index_code`) return errors
- The LLM can't verify anything against actual code
- It resorts to "educated guesses" - inventing plausible-sounding but fake paths

**Evidence**: First analysis run hallucinated:
- `src/rlm_dspy/repl/agent.py` (doesn't exist)
- `src/rlm_dspy/llm.py` (doesn't exist)
- `src/rlm_dspy/retrieval/grounding.py` (doesn't exist)
- `src/rlm_dspy/utils/parser.py` (doesn't exist)

### 2. Post-Hoc Only Validation

**Location**: `src/rlm_dspy/cli.py` (line 603)

```python
if validate and result.success:
    from .guards import validate_groundedness
    validation = validate_groundedness(result.answer, context, query)
```

- Validation is opt-in (controlled by `--validate` flag)
- Only checks the **final answer**, not intermediate reasoning
- Hallucinations during the REPL exploration phase go undetected

### 3. Unprotected Intermediate llm_query() Calls

**Location**: Native to `dspy.RLM` (enabled via `sub_lm` parameter in `_create_rlm`, line 896)

```python
kwargs = {
    ...
    "sub_lm": self._create_sub_lm(),  # Enables llm_query()
}
```

- `llm_query()` is NOT defined in rlm-dspy source
- It's a native feature of `dspy.RLM` primitive
- No grounding checks on sub-LLM responses
- Hallucinations in sub-queries can cascade to final answer

### 4. No Automatic Grounding in Core API

**Location**: `src/rlm_dspy/core/rlm.py` (lines 1043-1181)

The `RLM.query()` method:
1. Executes the REPL loop via `dspy.RLM`
2. Builds the result via `_build_result()`
3. Returns immediately without validation

Users of the Python API must manually call `validate_groundedness()`.

## Hallucination Prevention Gaps

| Gap | File | Lines | Impact |
|-----|------|-------|--------|
| Silent interpreter failure | `core/rlm.py` | 862-880 | Total hallucination when tools don't work |
| No core API validation | `core/rlm.py` | 1043-1181 | Direct API users get no protection |
| Opt-in CLI validation | `cli.py` | 603 | Can be disabled with `--no-validate` |
| Unvalidated sub-queries | `dspy.RLM` native | - | Discovery phase hallucinations cascade |

## Existing Mitigations

1. **LLM-as-judge validation** (`guards.py:61-118`)
   - Uses DSPy's `AnswerGroundedness` 
   - Compares final answer claims against source context
   - Reports grounding percentage and specific issues

2. **Grounded Proposer** (`core/grounded_proposer.py`)
   - Learns from past failures
   - Proposes improved prompts based on failure patterns
   - Updates tips dynamically

3. **Instruction Optimizer** (`core/instruction_optimizer.py`)
   - Optimizes tool instructions based on outcomes
   - Tracks grounded_score for each query type

## Recommendations

### Immediate Fixes

1. **Fail-fast on interpreter issues**: Detect Deno/Pyodide unavailability and raise clear error
2. **Default validation ON**: Make `validate=True` the default in `RLM.query()`
3. **Log tool failures prominently**: When tools fail, surface this to the user

### Architectural Improvements

1. **Intermediate grounding**: Validate `llm_query()` responses before using them
2. **Confidence scoring**: Track tool success rate during REPL to detect degraded state
3. **Cascading validation**: Check claims at each iteration, not just at the end

## Test Commands

```bash
# With Deno in PATH (should work correctly)
export PATH="$HOME/.deno/bin:$PATH"
rlm-dspy ask "What files exist in src/rlm_dspy/?" src/rlm_dspy/ -v

# Without Deno (will likely hallucinate)
# unset PATH to simulate missing Deno
PATH=/usr/bin:/bin rlm-dspy ask "What files exist?" src/rlm_dspy/ -v
```

## Conclusion

The primary hallucination risk in rlm-dspy is **tool failure leading to blind guessing**. 
When the code interpreter (Deno/Pyodide) is unavailable, the LLM cannot verify its claims 
against actual code, leading to coherent-sounding but completely fabricated responses.

The secondary risk is that validation is post-hoc and optional, allowing hallucinations 
during the exploration phase to influence the final answer even when it's eventually validated.

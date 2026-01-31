# Hallucination Fixes - Task Checklist

## Phase 1: Fail-Fast on Interpreter Failure (P0) ✅ COMPLETE

### Task 1.1: Interpreter Health Check ✅
- [x] Add `InterpreterError` exception class to `rlm.py`
- [x] Add `TimeoutExceededError` exception class to `rlm.py`
- [x] Add `_verify_interpreter()` method to `RLM` class
- [x] Update `_get_or_create_interpreter()` to verify on creation

**Files:** `src/rlm_dspy/core/rlm.py`

### Task 1.2: Preflight Interpreter Check ✅
- [x] Add `check_interpreter()` function to `validation.py`
- [x] Check Deno in PATH and ~/.deno/bin
- [x] Update `preflight_check()` to include interpreter check

**Files:** `src/rlm_dspy/core/validation.py`

### Task 1.3: CLI Preflight Update ✅
- [x] Preflight now includes interpreter check by default
- [x] Shows clear error message with install instructions

**Files:** `src/rlm_dspy/cli.py` (automatic via `preflight_check()`)

---

## Phase 2: Default Validation in Core API (P1) ✅ COMPLETE

### Task 2.1: Add Validation to RLM.query() ✅
- [x] Add `validate` parameter to `query()` method
- [x] Add `_validate_result()` helper method
- [x] Store validation results in `result.metadata["validation"]`
- [x] Log warning for low grounding scores

**Files:** `src/rlm_dspy/core/rlm.py`

### Task 2.2: Validation Config Options ✅
- [x] Add `validation_threshold` field to `RLMConfig` (default: 0.66)
- [x] Support `RLM_VALIDATION_THRESHOLD` env var
- [x] Existing `validate` field already present (default: True)

**Files:** `src/rlm_dspy/core/rlm.py`

---

## Phase 3: Real-Time Grounding in REPL Loop (P1) ✅ COMPLETE

### Task 3.1: Grounded Tool Wrapper ✅
- [x] Create `src/rlm_dspy/core/grounded_tools.py`
- [x] Implement `GroundedToolConfig` class
- [x] Implement `grounded_llm_query()` wrapper
- [x] Implement `grounded_llm_query_batched()` wrapper
- [x] Implement `create_grounded_tools()` factory
- [x] Add `GroundingStats` for tracking
- [x] Export from `core/__init__.py`

**Files:** `src/rlm_dspy/core/grounded_tools.py` (NEW)

### Task 3.2: Full Integration into RLM Query Flow ✅
- [x] Created `GroundedRLM` class that wraps dspy.RLM
- [x] Monkey-patches `_make_llm_tools()` to add grounding validation
- [x] Added `validate_intermediate` config option (default: False)
- [x] Added `validate_intermediate` parameter to `RLM.query()`
- [x] Added `--validate-intermediate` CLI flag
- [x] Logs intermediate grounding stats when verbose

**Files:** `src/rlm_dspy/core/grounded_tools.py`, `src/rlm_dspy/core/rlm.py`, `src/rlm_dspy/cli.py`

---

## Phase 4: Auto-Triggered Optimization (P2) ✅ COMPLETE

### Task 4.1: Failure Rate Tracking ✅
- [x] Add `get_recent_failure_rate()` to `GroundedProposer`
- [x] Add `get_average_grounded_score()` method
- [x] Add `should_suggest_optimization()` method
- [x] Add `get_optimization_recommendation()` method

**Files:** `src/rlm_dspy/core/grounded_proposer.py`

### Task 4.2: Auto-Optimization Suggestion ✅
- [x] Check failure rate after validation in CLI
- [x] Show suggestion when threshold exceeded
- [x] Include command to run optimization

**Files:** `src/rlm_dspy/cli.py`

### Task 4.3: Auto-Optimization Command ✅
- [x] Add `rlm-dspy optimize auto` command
- [x] Check failure rate against threshold
- [x] Option to force run with `--force`
- [x] Dry run mode with `--dry-run`

**Files:** `src/rlm_dspy/cli.py`

---

## Phase 5: BootstrapFewShot Integration (P2) ✅ COMPLETE

### Task 5.1: Bootstrap Mode for Traces ✅
- [x] Add `get_demos_for_query_type()` to `TraceCollector`
- [x] Add `to_dspy_examples()` method
- [x] Add `get_high_quality_traces()` method

**Files:** `src/rlm_dspy/core/trace_collector.py`

### Task 5.2: Bootstrap Command ✅
- [x] Add `rlm-dspy optimize bootstrap` command
- [x] Show trace statistics
- [x] Format and display demos
- [x] Option to save to file
- [x] Filter by query type

**Files:** `src/rlm_dspy/cli.py`

---

## Testing ✅ VERIFIED

### Automated Tests
- [x] All existing tests pass (52 validation/proposer tests)
- [x] Module imports work correctly
- [x] CLI commands functional

### Manual Tests
- [x] `rlm-dspy preflight` shows interpreter status
- [x] `rlm-dspy optimize auto --dry-run` shows recommendation
- [x] `rlm-dspy optimize bootstrap` shows trace stats
- [x] End-to-end query with validation works

---

## Summary

| Phase | Priority | Status | Changes |
|-------|----------|--------|---------|
| 1. Fail-Fast | P0 | ✅ Complete | `rlm.py`, `validation.py` |
| 2. Default Validation | P1 | ✅ Complete | `rlm.py` |
| 3. Grounded Tools | P1 | ✅ Complete | `grounded_tools.py` (NEW) |
| 4. Auto-Optimization | P2 | ✅ Complete | `grounded_proposer.py`, `cli.py` |
| 5. Bootstrap | P2 | ✅ Complete | `trace_collector.py`, `cli.py` |

### New Files Created
- `src/rlm_dspy/core/grounded_tools.py`

### Files Modified
- `src/rlm_dspy/core/rlm.py`
- `src/rlm_dspy/core/validation.py`
- `src/rlm_dspy/core/grounded_proposer.py`
- `src/rlm_dspy/core/trace_collector.py`
- `src/rlm_dspy/core/__init__.py`
- `src/rlm_dspy/cli.py`

### New CLI Commands
- `rlm-dspy optimize auto` - Auto-run optimization if failure rate high
- `rlm-dspy optimize bootstrap` - Bootstrap few-shot from traces

### New API Features
- `RLM.query(validate=True/False)` - Control validation per-query
- `RLMConfig.validation_threshold` - Set grounding threshold
- `check_interpreter()` - Verify Deno availability
- `GroundedProposer.get_recent_failure_rate()` - Track failures
- `GroundedProposer.should_suggest_optimization()` - Auto-suggest
- `TraceCollector.to_dspy_examples()` - Bootstrap integration
- `create_grounded_tools()` - Wrap tools with validation

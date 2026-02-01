# Optimizer Integration Plan

## Status: Phases 1-5 COMPLETE ✅

Implementation completed on 2026-01-31.

---

## What Was Implemented

### Phase 1: Unified Data Sources ✅

All optimizers now use `TraceCollector.traces` as single source:

```python
# trace_collector.py - New methods:
- get_failures(max_score) -> list[REPLTrace]
- get_successes(min_score) -> list[REPLTrace]
- to_failure_patterns() -> list[dict]  # For tip generation
- to_success_patterns() -> list[dict]  # For tip generation
```

### Phase 2: SIMBA Rules Integration ✅

SIMBA now extracts and saves rules:

```python
# simba_optimizer.py - New functions:
- _extract_simba_rules(program) -> list[str]
- _generate_optimized_tips() -> list[str]

# instruction_optimizer.py - New methods:
- set_instruction(key, text)
- add_rules(rules)  # Merges rules into tool_instructions
- get_all_instructions() -> dict[str, str]
```

### Phase 3: Tips Integration ✅

Tips are now generated during optimization and saved:

```python
# grounded_proposer.py - New methods:
- set_optimized_tips(tips)  # Merges with existing
- set_tips(tips)  # Replaces all
```

### Phase 4: Unified CLI ✅

New unified command:

```bash
rlm-dspy optimize run            # Full optimization (demos + tips + rules)
rlm-dspy optimize run --fast     # Fast mode (~5-15 min)
rlm-dspy optimize run --target tips  # Only tips
rlm-dspy optimize run --dry-run  # Preview
```

### Updated SavedOptimization Schema ✅

```python
@dataclass
class SavedOptimization:
    demos: list[dict[str, Any]]       # Few-shot examples
    instructions: dict[str, str]      # Keyed instructions (was: str)
    tips: list[str]                   # NEW: Learned tips
    rules: list[str]                  # NEW: SIMBA rules
    timestamp: datetime
    optimizer_type: str
    result: OptimizationResult | None
```

### Updated RLM._load_and_apply_optimization() ✅

Now applies all components:
- Demos → `self._rlm.demos`
- Tips → `GroundedProposer.set_optimized_tips()`
- Rules → `InstructionOptimizer.add_rules()`
- Instructions → `InstructionOptimizer.set_instruction()`

---

## Architecture After Implementation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATED OPTIMIZATION SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        TraceCollector                                │   │
│  │                    (Single Source of Truth)                          │   │
│  │  - traces: list[REPLTrace]                                          │   │
│  │  - get_failures(), get_successes()                                  │   │
│  │  - to_failure_patterns(), to_success_patterns()                     │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 rlm-dspy optimize run                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │  │ Step 1:     │  │ Step 2:     │  │ Step 3:     │                  │   │
│  │  │ SIMBA       │─▶│ Generate    │─▶│ Extract     │                  │   │
│  │  │ (demos)     │  │ Tips        │  │ Rules       │                  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SavedOptimization                                 │   │
│  │  - demos: list[dict]           (few-shot examples)                   │   │
│  │  - instructions: dict[str,str] (keyed instruction texts)             │   │
│  │  - tips: list[str]             (learned tips)                        │   │
│  │  - rules: list[str]            (SIMBA rules)                         │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           RLM.py                                     │   │
│  │  _load_and_apply_optimization():                                     │   │
│  │    ✓ Apply demos to self._rlm.demos                                  │   │
│  │    ✓ Apply tips via GroundedProposer.set_optimized_tips()           │   │
│  │    ✓ Apply rules via InstructionOptimizer.add_rules()               │   │
│  │    ✓ Apply instructions via InstructionOptimizer.set_instruction()  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Files Changed

| File | Changes |
|------|---------|
| `optimization_state.py` | Updated `SavedOptimization` schema (instructions→dict, +tips, +rules) |
| `simba_optimizer.py` | Added `_extract_simba_rules()`, `_generate_optimized_tips()`, updated background opt |
| `instruction_optimizer.py` | Added `set_instruction()`, `add_rules()`, `get_all_instructions()` |
| `grounded_proposer.py` | Added `set_optimized_tips()`, `set_tips()` |
| `trace_collector.py` | Added `get_failures()`, `get_successes()`, `to_failure_patterns()`, `to_success_patterns()` |
| `rlm.py` | Updated `_load_and_apply_optimization()` to apply all components |
| `cli_optimize.py` | Added `optimize run` command, updated `status` to show tips/rules |
| `README.md` | Documented new unified optimization |

---

## Phase 5: GEPA Support ✅

Implementation completed on 2026-01-31.

Added:
1. `gepa_optimizer.py` - GEPA wrapper with feedback metric
2. `rlm-dspy optimize gepa` CLI command
3. `rlm-dspy optimize run -o gepa` option
4. `GEPAConfig` and `GEPAOptimizer` classes
5. Teacher model support via `--teacher` CLI and `optimization.teacher_model` config
6. `extract_gepa_instructions()` and `apply_gepa_instructions()` functions

GEPA advantages for RLM:
- Uses execution traces with textual feedback
- Reflection-based instruction evolution
- Better for complex multi-step agents
- Optional joint tool optimization

```bash
# CLI commands
rlm-dspy optimize gepa                           # Light budget
rlm-dspy optimize gepa --auto medium             # Medium budget
rlm-dspy optimize gepa --teacher openai/gpt-4o   # Use specific teacher
rlm-dspy optimize gepa --tools                   # Enable tool optimization
rlm-dspy optimize run -o gepa --fast             # Use in unified run
```

Config:
```yaml
# ~/.rlm/config.yaml
optimization:
  teacher_model: openai/gpt-4o  # Strong model for GEPA reflection
```

---

## Usage

```bash
# Full optimization (recommended)
rlm-dspy optimize run --fast

# Check status
rlm-dspy optimize status

# View what's saved
# Output shows: Demos, Tips, Rules, Instructions
```

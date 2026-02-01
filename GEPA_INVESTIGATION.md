# GEPA Performance Investigation Summary

## Problem Statement
When running GEPA optimization via the CLI (`rlm-dspy optimize gepa`), it was extremely slow (hours) compared to simple GEPA usage in other projects like [qmd](https://github.com/tobi/qmd) which runs in minutes.

## Root Cause Analysis

After cloning and analyzing qmd's GEPA implementation, the fundamental difference is **program complexity**:

| Project | Program Type | LLM Calls per Eval | Time per Eval |
|---------|--------------|-------------------|---------------|
| qmd | `dspy.Predict` (simple) | 1 | 1-5 seconds |
| rlm-dspy | `dspy.RLM` (complex agent) | 10-50 | 30-120 seconds |

GEPA evaluates the full program for each metric call. With `auto="light"` budget (~760 metric calls) and 10-50 LLM calls per evaluation, **full RLM mode would take 6-20 hours**.

### Key Differences from qmd:
1. **qmd uses simple `dspy.Predict`** - single LLM call per evaluation
2. **rlm-dspy uses `dspy.RLM`** - full interpreter loop with:
   - 20+ iterations per query
   - Deno subprocess for code execution
   - 10+ tool invocations per evaluation

## Solution: Fast Proxy Mode (`--fast`)

Created a **lightweight proxy module** that:
1. Uses `dspy.Predict` instead of full RLM interpreter loop
2. Shares the same signature/instructions as RLM
3. Runs in 1 LLM call per eval (**50x faster**)
4. Transfers optimized instructions back to RLM after optimization

### Usage
```bash
# Fast proxy mode (recommended) - 2-10 minutes
rlm-dspy optimize gepa --fast

# With more budget
rlm-dspy optimize gepa --fast --auto medium

# Full RLM mode (slow but more accurate) - 1-6 hours
rlm-dspy optimize gepa
```

### Performance Comparison
| Mode | Time for 2 full evals (5 examples) | Estimated for light budget |
|------|-------------------------------------|---------------------------|
| Fast Proxy | ~2.5 minutes | ~15-30 minutes |
| Full RLM | ~30-60 minutes | ~6-20 hours |

## Files Changed

### New Files
- `src/rlm_dspy/core/gepa_proxy.py` - Lightweight proxy module for fast GEPA

### Modified Files
- `src/rlm_dspy/cli_optimize.py` - Added `--fast` flag for proxy mode
- `src/rlm_dspy/core/gepa_optimizer.py` - Fixed OptimizationResult field names

## Requirements
- **Deno** must be installed for RLM code execution:
  ```bash
  curl -fsSL https://deno.land/install.sh | sh
  export PATH="$HOME/.deno/bin:$PATH"
  ```

## Technical Details

### RLMProxy Class
The proxy creates a simple `dspy.Predict` module that:
- Extracts the signature and instructions from the real RLM
- Runs predictions in a single LLM call
- Allows GEPA to optimize the instructions efficiently

### Instruction Extraction
After GEPA optimization:
1. Extract evolved instructions from the optimized proxy
2. Filter out invalid instructions (e.g., Python's `str.__doc__`)
3. Save to `~/.rlm/optimization/optimized_program.json`
4. Auto-load on next RLM query

### Trade-offs
- **Fast proxy mode**: Optimizes instruction text only, doesn't test actual tool usage
- **Full RLM mode**: Tests complete agent behavior but is much slower

For most use cases, the fast proxy mode provides a good balance of speed and optimization quality.

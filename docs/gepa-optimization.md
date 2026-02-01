# GEPA Optimization for RLM

This document explains GEPA optimization, the fast proxy mode, and why it's necessary.

## Background: The Performance Problem

GEPA (Reflective Prompt Evolution) is DSPy's optimizer that evolves prompt instructions
based on execution traces and textual feedback. It works great for simple `dspy.Predict`
modules but becomes extremely slow for complex agents like RLM.

### Root Cause

GEPA evaluates the program hundreds of times during optimization:
- With `auto="light"`, GEPA runs ~760 metric calls
- Each metric call evaluates the full program

For simple modules:
```
1 LLM call per eval × 760 evals × 2 seconds = ~25 minutes
```

For RLM (complex agent with interpreter loop):
```
30 LLM calls per eval × 760 evals × 60 seconds = ~12+ hours
```

### Comparison with Other Projects

We investigated [qmd](https://github.com/tobi/qmd), which uses GEPA effectively:

| Project | Program Type | LLM Calls/Eval | Time/Eval |
|---------|--------------|----------------|-----------|
| qmd | `dspy.Predict` | 1 | 1-5 seconds |
| rlm-dspy (full) | `dspy.RLM` | 10-50 | 30-120 seconds |

## Solution: Fast Proxy Mode

We created a lightweight proxy that shares RLM's signature but runs in 1 LLM call.

### How It Works

1. **Create Proxy**: Extract signature/instructions from RLM, create `dspy.Predict` module
2. **Run GEPA**: Optimize the proxy (fast - 1 LLM call per eval)
3. **Extract Instructions**: Get evolved instructions from optimized proxy
4. **Apply to RLM**: Save instructions, auto-load on RLM initialization

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GEPA Fast Proxy Mode                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RLM (Complex)                    RLMProxy (Lightweight)            │
│  ─────────────                    ─────────────────────             │
│  ┌─────────────────┐              ┌─────────────────┐               │
│  │ generate_action │─ extract ───▶│ dspy.Predict    │               │
│  │ (10-50 LLM calls)│  signature  │ (1 LLM call)    │               │
│  ├─────────────────┤              ├─────────────────┤               │
│  │ interpreter loop│              │ Same signature  │               │
│  │ tool execution  │              │ Same instruction│               │
│  │ extract         │              └────────┬────────┘               │
│  └─────────────────┘                       │                        │
│                                            ▼                        │
│                                    ┌───────────────┐                │
│                                    │    GEPA       │                │
│                                    │  Optimizer    │                │
│                                    │ (760 evals)   │                │
│                                    └───────┬───────┘                │
│                                            │                        │
│                                            ▼                        │
│                                    ┌───────────────┐                │
│                                    │  Optimized    │                │
│                                    │ Instructions  │                │
│                                    └───────┬───────┘                │
│                                            │                        │
│  ┌─────────────────┐                       │                        │
│  │ RLM with        │◀── apply ────────────┘                        │
│  │ new instructions│                                                │
│  └─────────────────┘                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Performance Comparison

| Mode | Per-Eval Time | Light Budget | Medium Budget |
|------|---------------|--------------|---------------|
| Full RLM | 30-120 sec | 6-20+ hours | 12-40+ hours |
| Fast Proxy | 1-5 sec | 5-30 min | 15-60 min |
| **Speedup** | **50x** | **50x** | **50x** |

## Usage

```bash
# Fast proxy mode (recommended)
rlm-dspy optimize gepa --fast

# With more budget
rlm-dspy optimize gepa --fast --auto medium

# Explicit budget control
rlm-dspy optimize gepa --fast --max-evals 3

# Full RLM mode (only if needed)
rlm-dspy optimize gepa
```

## Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--fast` | `-f` | false | Use fast proxy mode (50x faster) |
| `--auto` | `-a` | `light` | Budget preset: `light`, `medium`, `heavy` |
| `--max-evals` | `-e` | (auto) | Explicit full evaluations count (overrides `--auto`) |
| `--max-examples` | `-n` | 50 | Max training examples from traces |
| `--min-score` | `-s` | 0.7 | Min `grounded_score` for trace inclusion |
| `--threads` | `-t` | 2 | Parallel threads (keep low for rate limits) |
| `--teacher` | | (config) | Teacher model for reflection |
| `--tools` | | false | Enable tool optimization (experimental) |
| `--dry-run` | | false | Preview without running |

### Budget Control

**Using presets:**
```bash
rlm-dspy optimize gepa --fast --auto light   # ~2-5 min
rlm-dspy optimize gepa --fast --auto medium  # ~5-15 min
rlm-dspy optimize gepa --fast --auto heavy   # ~15-30 min
```

**Using explicit evaluations:**
```bash
rlm-dspy optimize gepa --fast --max-evals 1  # Minimal, ~1-2 min
rlm-dspy optimize gepa --fast --max-evals 5  # Moderate, ~5-10 min
rlm-dspy optimize gepa --fast --max-evals 10 # Thorough, ~10-20 min
```

### Examples

```bash
# Quick test
rlm-dspy optimize gepa --fast --max-evals 1 --max-examples 4

# Standard optimization
rlm-dspy optimize gepa --fast

# Thorough with custom teacher
rlm-dspy optimize gepa --fast --auto medium --teacher openai/gpt-4o

# Preview settings
rlm-dspy optimize gepa --fast --dry-run
```

## Trade-offs

### Advantages of Fast Proxy Mode
- ✅ 50x faster optimization
- ✅ Same instruction quality (GEPA evolves text, not tool behavior)
- ✅ Instructions transfer correctly to real RLM
- ✅ Practical for iterative experimentation

### Limitations of Fast Proxy Mode
- ⚠️ Doesn't test actual tool execution during optimization
- ⚠️ Metric scores may differ from real RLM performance
- ⚠️ Can't optimize tool selection strategies

### When to Use Full RLM Mode
- Testing how instructions affect tool usage patterns
- Debugging tool-related issues
- Final validation after proxy optimization
- When you have several hours to spare

## Implementation Details

### RLMProxy Class (`gepa_proxy.py`)

```python
class RLMProxy(dspy.Module):
    """Lightweight proxy for fast GEPA optimization."""
    
    def __init__(self, signature, instructions):
        self.predict = dspy.Predict(signature)
        # Apply instructions to signature
        
    @classmethod
    def from_rlm(cls, rlm):
        """Extract signature and instructions from RLM."""
        
    def forward(self, context, query):
        return self.predict(context=context, query=query)
```

### Instruction Extraction

After GEPA optimization, we extract instructions from the optimized proxy:

```python
def extract_proxy_instructions(proxy):
    """Extract evolved instructions from optimized proxy."""
    instructions = {}
    sig = proxy.predict.signature
    doc = getattr(sig, 'instructions', None) or getattr(sig, '__doc__', None)
    if doc:
        instructions['generate_action'] = doc
        instructions['predict'] = doc
    return instructions
```

### Instruction Application

On RLM initialization, saved instructions are loaded and applied:

```python
def _load_and_apply_optimization(self):
    saved = load_optimized_program()
    if saved and saved.instructions:
        from .gepa_optimizer import apply_gepa_instructions
        apply_gepa_instructions(self._rlm, saved.instructions)
```

## Verification

After optimization, verify instructions are loaded correctly:

```bash
# Check saved optimization
cat ~/.rlm/optimization/optimized_program.json

# Test instruction loading
python -c "
from rlm_dspy.core.rlm import RLM
rlm = RLM()
print('Signature instructions:', len(rlm._signature.__doc__), 'chars')
"
```

## Requirements

- DSPy >= 3.0 (for GEPA)
- At least 4 traces with `grounded_score >= 0.7`
- Deno runtime for RLM code execution:
  ```bash
  curl -fsSL https://deno.land/install.sh | sh
  export PATH="$HOME/.deno/bin:$PATH"
  ```

## See Also

- [DSPy GEPA Documentation](https://dspy.ai/api/optimizers/GEPA)
- [RLM-DSPy Optimization Guide](../README.md#self-optimization-system)
- [qmd GEPA Implementation](https://github.com/tobi/qmd/tree/main/finetune/gepa)

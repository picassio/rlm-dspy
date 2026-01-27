# RLM-DSPy

> Recursive Language Models with DSPy optimization - combining RLM's recursive decomposition with DSPy's compiled prompts.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why RLM-DSPy?

| Feature | Traditional RLM | DSPy | **RLM-DSPy** |
|---------|----------------|------|--------------|
| Large context handling | ✅ Recursive decomposition | ❌ Single pass | ✅ Recursive + typed |
| Prompt optimization | ❌ Hand-crafted | ✅ Auto-compiled | ✅ Compiled recursive |
| Typed I/O | ❌ Unstructured | ✅ Signatures | ✅ Full typing |
| Budget/timeout controls | ✅ Built-in | ❌ Manual | ✅ Built-in |
| Parallel processing | ✅ Batched queries | ⚠️ Limited | ✅ Async + batched |

## Installation

```bash
pip install rlm-dspy
```

Or with uv:
```bash
uv pip install rlm-dspy
```

## Quick Start

### CLI Usage

```bash
# Ask a question about code
rlm-dspy ask "What does the main function do?" src/

# Analyze from stdin
cat large_file.txt | rlm-dspy ask "Summarize this" --stdin

# Analyze a git diff
git diff | rlm-dspy diff "Are there any breaking changes?"

# Full codebase analysis
rlm-dspy analyze src/ --output analysis.md
```

### Python API

```python
from rlm_dspy import RLM, RLMConfig

# Configure
config = RLMConfig(
    model="openrouter/anthropic/claude-sonnet-4",
    max_budget=1.0,  # USD
    max_timeout=300,  # seconds
)

# Create RLM instance
rlm = RLM(config=config)

# Load context from files
context = rlm.load_context(["src/", "docs/"])

# Query
result = rlm.query("What are the main components?", context)
print(result.answer)
print(f"Cost: ${result.total_cost:.4f}")
```

### Using DSPy Programs Directly

```python
from rlm_dspy import RecursiveAnalyzer, ChunkedProcessor
import dspy

# Configure DSPy
lm = dspy.LM("openrouter/anthropic/claude-sonnet-4")
dspy.configure(lm=lm)

# Use recursive analyzer
analyzer = RecursiveAnalyzer(max_depth=3)
result = analyzer(
    query="Find all API endpoints",
    context=large_codebase,
    chunk_size=100_000,
)
print(result.answer)

# Or use map-reduce pattern
processor = ChunkedProcessor()
result = processor(
    query="Summarize each module",
    context=documentation,
)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_MODEL` | `openrouter/anthropic/claude-sonnet-4` | Model to use |
| `OPENROUTER_API_KEY` | - | OpenRouter API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key (fallback) |
| `RLM_MAX_BUDGET` | `1.0` | Maximum cost in USD |
| `RLM_MAX_TIMEOUT` | `300` | Maximum time in seconds |
| `RLM_CHUNK_SIZE` | `100000` | Default chunk size |

### Programmatic Configuration

```python
from rlm_dspy import RLMConfig

config = RLMConfig(
    # Model settings
    model="openrouter/anthropic/claude-sonnet-4",
    sub_model="openrouter/anthropic/claude-sonnet-4",  # For sub-queries
    api_key="sk-...",
    
    # Execution limits
    max_budget=2.0,
    max_timeout=600,
    max_tokens=1_000_000,
    max_iterations=50,
    max_depth=5,
    
    # Chunking
    default_chunk_size=150_000,
    overlap=1000,
    
    # Processing
    strategy="auto",  # auto|map_reduce|iterative|hierarchical
    parallel_chunks=10,
    
    # DSPy optimization
    use_compiled_prompts=True,
    prompt_cache_dir=Path("~/.rlm-dspy/compiled"),
)
```

## Processing Strategies

### Auto (Default)
RLM-DSPy automatically selects the best strategy based on:
- Context size
- Context type (code, docs, mixed)
- Query complexity

### Map-Reduce
Best for: Large contexts with independent chunks
```
[Chunk 1] → [Analyze] ─┐
[Chunk 2] → [Analyze] ─┼─→ [Aggregate] → Answer
[Chunk 3] → [Analyze] ─┘
```

### Iterative
Best for: Sequential content where order matters
```
[Chunk 1] → [Analyze] → Buffer
[Chunk 2] → [Analyze + Buffer] → Buffer
[Chunk 3] → [Analyze + Buffer] → Answer
```

### Hierarchical
Best for: Very large contexts requiring multiple levels
```
[Section 1] → [Sub-RLM] → Summary 1 ─┐
[Section 2] → [Sub-RLM] → Summary 2 ─┼─→ [Aggregate] → Answer
[Section 3] → [Sub-RLM] → Summary 3 ─┘
```

## Compiling Optimized Prompts

DSPy can automatically optimize prompts from examples:

```bash
# Create training data
cat > training.json << 'EOF'
[
  {
    "query": "What does this function do?",
    "context": "def add(a, b): return a + b",
    "answer": "The add function takes two parameters and returns their sum."
  }
]
EOF

# Compile
rlm-dspy compile training.json --output ~/.rlm-dspy/compiled/
```

Then use compiled prompts:
```python
config = RLMConfig(
    use_compiled_prompts=True,
    prompt_cache_dir=Path("~/.rlm-dspy/compiled"),
)
```

## DSPy Signatures

RLM-DSPy uses typed DSPy signatures for structured I/O:

```python
from rlm_dspy import AnalyzeChunk, AggregateAnswers

# AnalyzeChunk signature
class AnalyzeChunk(dspy.Signature):
    query: str = dspy.InputField()
    chunk: str = dspy.InputField()
    chunk_index: int = dspy.InputField()
    total_chunks: int = dspy.InputField()
    
    relevant_info: str = dspy.OutputField()
    confidence: Literal["high", "medium", "low", "none"] = dspy.OutputField()
```

## Comparison with Original RLM

| Aspect | Original RLM | RLM-DSPy |
|--------|-------------|----------|
| Prompts | Hand-crafted system prompt | DSPy signatures + optimization |
| REPL | Python REPL execution | Optional (signatures preferred) |
| Output | Unstructured text | Typed Pydantic models |
| Optimization | None | BootstrapFewShot, MIPRO, etc. |
| Caching | None | Compiled prompt caching |

## License

MIT

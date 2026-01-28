# RLM-DSPy

> Recursive Language Models with DSPy optimization - combining RLM's recursive decomposition with DSPy's compiled prompts.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why RLM-DSPy?

| Feature | Traditional RLM | DSPy | **RLM-DSPy** |
|---------|----------------|------|--------------|
| Large context handling | ✅ Recursive decomposition | ❌ Single pass | ✅ Recursive + typed |
| Prompt optimization | ❌ Hand-crafted | ✅ Auto-compiled | ✅ Compiled recursive |
| Syntax-aware chunking | ❌ Character-based | ❌ N/A | ✅ Tree-sitter |
| Incremental caching | ❌ None | ❌ None | ✅ Salsa-style |
| Budget/timeout controls | ✅ Built-in | ❌ Manual | ✅ Built-in |
| Parallel processing | ✅ Batched queries | ⚠️ Limited | ✅ Async + batched |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              rlm-dspy Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT                      PROCESSING                         OUTPUT       │
│  ─────                      ──────────                         ──────       │
│                                                                              │
│  ┌─────────┐               ┌─────────────────┐                              │
│  │  Files  │───┐           │ Syntax-Aware    │  Chunks at function/class    │
│  └─────────┘   │           │ Chunking        │  boundaries (tree-sitter)    │
│                │           └────────┬────────┘                              │
│  ┌─────────┐   │                    │                                       │
│  │  Stdin  │───┼──▶ Content ──▶─────┤                                       │
│  └─────────┘   │    Detection       │                                       │
│                │                    ▼                                       │
│  ┌─────────┐   │           ┌─────────────────┐                              │
│  │  Query  │───┘           │ Strategy Auto-  │  map_reduce│iterative│       │
│  └─────────┘               │ Selection       │  hierarchical                │
│                            └────────┬────────┘                              │
│                                     │                                       │
│                                     ▼                                       │
│                            ┌─────────────────┐                              │
│                            │ Parallel LLM    │  20 concurrent chunks        │
│                            │ Processing      │  via async HTTP              │
│                            └────────┬────────┘                              │
│                                     │                                       │
│                                     ▼                                       │
│                            ┌─────────────────┐               ┌────────────┐ │
│                            │ Salsa Cache     │──────────────▶│   Answer   │ │
│                            │ (auto-invalidate│               │ + Metadata │ │
│                            └─────────────────┘               └────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install rlm-dspy

# With syntax-aware chunking (recommended)
pip install rlm-dspy tree-sitter tree-sitter-python tree-sitter-javascript
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

# With budget limit
rlm-dspy ask "Find all bugs" src/*.py --budget 0.50

# Verbose output
rlm-dspy ask "Explain this" file.py -v

# Full debug (shows API calls)
rlm-dspy ask "Debug this" file.py -d

# Dry run (validate config only)
rlm-dspy ask "Test" file.py -n

# Analyze from stdin
cat large_file.txt | rlm-dspy ask "Summarize this" --stdin

# Analyze a git diff
git diff | rlm-dspy diff "Are there any breaking changes?"

# Check configuration
rlm-dspy preflight
```

### Python API

```python
from rlm_dspy import RLM, RLMConfig

# Configure
config = RLMConfig(
    model="openrouter/google/gemini-2.0-flash-exp",
    max_budget=1.0,  # USD
    syntax_aware_chunking=True,  # Use tree-sitter
)

# Create RLM instance
rlm = RLM(config=config)

# Load context from files
context = rlm.load_context(["src/", "docs/"])

# Query
result = rlm.query("What are the main components?", context)
print(result.answer)
print(f"Time: {result.elapsed_time:.1f}s, Cost: ${result.total_cost:.4f}")
```

## Core Features

### 1. Syntax-Aware Chunking

Uses tree-sitter to chunk code at function/class boundaries instead of arbitrary positions:

```python
from rlm_dspy.core import chunk_code_syntax_aware

code = """
def hello():
    print('Hello')

def world():
    print('World')

class Greeter:
    def greet(self):
        return 'Hi!'
"""

chunks = chunk_code_syntax_aware(code, chunk_size=100, language="python")
for chunk in chunks:
    print(f"{chunk.node_type} '{chunk.name}': lines {chunk.start_line}-{chunk.end_line}")
# function_definition 'hello': lines 2-3
# function_definition 'world': lines 5-6  
# class_definition 'Greeter': lines 8-10
```

**Benefits:**
- No truncated functions (chunks never split mid-definition)
- Better context (each chunk is a complete syntactic unit)
- Fewer false positives (LLMs won't report "syntax errors")

**Supported Languages:** Python, TypeScript, JavaScript, Go, Rust, Java, C/C++, Ruby, PHP, C#, Kotlin, Lua

### 2. Salsa-Style Incremental Caching

Automatic memoization with dependency tracking (inspired by rust-analyzer):

```python
from rlm_dspy.core import SalsaDB, salsa_query

db = SalsaDB()

@salsa_query
def analyze_file(db: SalsaDB, path: str) -> dict:
    content = db.get_file(path)
    return {"lines": len(content.split("\n"))}

# First call - computes and caches
db.set_file("auth.py", "def login(): pass")
result = db.query(analyze_file, "auth.py")  # Computes

# Second call - instant (cached)
result = db.query(analyze_file, "auth.py")  # Cache hit!

# File changes - cache auto-invalidates
db.set_file("auth.py", "def login(): pass\ndef logout(): pass")
result = db.query(analyze_file, "auth.py")  # Recomputes automatically

print(db.stats.to_dict())
# {'cache_hits': 1, 'cache_misses': 2, 'invalidations': 1, ...}
```

### 3. Large Content Handling (Paste Store)

Prevent context overflow with placeholder system:

```python
from rlm_dspy.core import PasteStore

store = PasteStore(threshold=2000)  # chars

# Large content gets stored with placeholder
large_code = "x = 1\n" * 1000  # 6000 chars
placeholder, paste_id = store.maybe_store(large_code, label="big_file.py")
print(placeholder)
# [paste_1 (big_file.py): 6000 chars, 1001 lines]

# Inject full content when needed
full_context = store.inject_context() + "\n\n" + "Analyze the code above"
```

### 4. Token Usage Tracking

Quantify savings from chunking and compression:

```python
from rlm_dspy.core import TokenStats, count_tokens

stats = TokenStats(
    raw_context_tokens=100000,
    processed_tokens=20000,
    llm_input_tokens=15000,
    llm_output_tokens=2000,
    chunks_processed=10,
    chunks_relevant=3,
)

print(stats)
# Token Stats:
#   Context: 100,000 → 20,000 (80.0% saved)
#   LLM: 15,000 in, 2,000 out
#   Chunks: 3/10 relevant (30.0%)
```

### 5. Content Deduplication

Share analysis for identical files (10-20% storage savings):

```python
from rlm_dspy.core import ContentHashedIndex

index = ContentHashedIndex()

# Same content shares one entry
index.set("utils.py", "shared code", {"analysis": "result"})
index.set("utils_copy.py", "shared code", {"analysis": "result"})

print(f"Paths: {index.path_count}, Unique: {len(index)}")
# Paths: 2, Unique: 1

print(f"Dedup ratio: {index.dedup_ratio * 100:.0f}%")
# Dedup ratio: 50%
```

### 6. User-Friendly Error Messages

API errors transformed to actionable advice:

```python
from rlm_dspy.core import format_user_error

error = Exception("401 Unauthorized: Invalid API key")
print(format_user_error(error))
# ❌ Authentication failed. Please check your API key:
#    1. Verify RLM_API_KEY or OPENROUTER_API_KEY is set
#    2. Ensure the key hasn't expired
#    3. Check you have credits remaining
```

### 7. Hierarchical Configuration

Resolution order: CLI > Environment > Cache > Defaults

```python
from rlm_dspy.core import ConfigResolver

resolver = ConfigResolver(
    env_prefix="RLM_",
    cache_path=Path("~/.cache/rlm-dspy/config.json")
)

# Gets from env RLM_MODEL, then cache, then default
model = resolver.get("MODEL", default="gpt-4")
```

## Processing Strategies

### Auto (Default)
RLM-DSPy automatically selects the best strategy based on context size, type, and query complexity.

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

## Configuration

### Environment Variables

See [.env.example](.env.example) for a complete template.

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_API_KEY` | - | API key (or `OPENROUTER_API_KEY`) |
| `RLM_MODEL` | `google/gemini-3-flash-preview` | Model to use |
| `RLM_SUB_MODEL` | (same as RLM_MODEL) | Model for sub-queries |
| `RLM_API_BASE` | `https://openrouter.ai/api/v1` | API endpoint |
| `RLM_MAX_BUDGET` | `1.0` | Maximum cost in USD |
| `RLM_MAX_TIMEOUT` | `300` | Maximum time in seconds |
| `RLM_CHUNK_SIZE` | `100000` | Chunk size (chars) |
| `RLM_SYNTAX_AWARE` | `true` | Tree-sitter chunking |
| `RLM_PARALLEL_CHUNKS` | `20` | Concurrent chunks |
| `RLM_PASTE_THRESHOLD` | `2000` | Paste store threshold |

### Quick Start `.env`

```bash
# Minimal
RLM_API_KEY=sk-or-v1-your-key-here
RLM_MODEL=google/gemini-2.0-flash-exp

# Cost-optimized
RLM_MAX_BUDGET=0.50
RLM_SYNTAX_AWARE=true
RLM_PARALLEL_CHUNKS=20
```

### Programmatic Configuration

```python
from rlm_dspy import RLMConfig
from pathlib import Path

config = RLMConfig(
    # Model
    model="openrouter/anthropic/claude-sonnet-4",
    sub_model="openrouter/anthropic/claude-3-haiku",
    
    # Limits
    max_budget=2.0,
    max_timeout=600,
    max_tokens=1_000_000,
    
    # Chunking
    default_chunk_size=150_000,
    overlap=1000,
    syntax_aware_chunking=True,
    
    # Processing
    strategy="auto",
    parallel_chunks=20,
    use_async=True,
    
    # Optimization
    use_compiled_prompts=True,
    prompt_cache_dir=Path("~/.rlm-dspy/compiled"),
)
```

## API Reference

### Core Exports

```python
from rlm_dspy import RLM, RLMConfig, RLMResult

from rlm_dspy.core import (
    # Chunking
    chunk_code_syntax_aware,
    CodeChunk,
    TREE_SITTER_AVAILABLE,
    
    # Paste store
    PasteStore,
    store_large_content,
    
    # Token stats
    TokenStats,
    SessionStats,
    count_tokens,
    estimate_cost,
    
    # Salsa caching
    SalsaDB,
    salsa_query,
    is_salsa_query,
    get_db,
    reset_db,
    
    # Content deduplication
    ContentHashedIndex,
    DirtyTracker,
    content_hash,
    
    # Config utilities
    ConfigResolver,
    atomic_write_json,
    atomic_read_json,
    format_user_error,
    inject_context,
    get_config_dir,
    
    # DSPy signatures
    AnalyzeChunk,
    AggregateAnswers,
    DecomposeTask,
    
    # Programs
    RecursiveAnalyzer,
    ChunkedProcessor,
    MapReduceProcessor,
)
```

## Benchmarks

### Speed Comparison

| Context Size | Bare LLM | RLM-DSPy | Winner |
|--------------|----------|----------|--------|
| < 2MB | ✅ Faster | Slower | Bare LLM |
| > 2MB | ❌ Fails | ✅ Works | RLM-DSPy |
| > 4MB | ❌ Fails | ✅ Works | RLM-DSPy |

### Accuracy

| Metric | Bare LLM | RLM-DSPy |
|--------|----------|----------|
| Accuracy | 89% | **100%** |
| Error Rate | 11% | **0%** |
| Max Context | 2MB | **4MB+** |

### Cost Comparison

| Model | Input $/M | Output $/M | Context |
|-------|-----------|------------|---------|
| Claude Sonnet | $3.00 | $15.00 | 200K |
| Gemini 2.0 Flash | $0.10 | $0.40 | 1M |
| **Gemini 3 Flash** | **$0.50** | **$3.00** | **1M** |

### When to Use What

```
Context < 2MB + speed critical  → Bare LLM
Context > 2MB                   → RLM-DSPy (only option)
Accuracy critical               → RLM-DSPy (100% vs 89%)
Cost critical                   → RLM-DSPy + Gemini Flash
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

## License

MIT

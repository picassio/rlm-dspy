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
| Response caching | ❌ None | ❌ None | ✅ DSPy disk cache |
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
│                            │ DSPy Cache      │──────────────▶│   Answer   │ │
│                            │ (disk-persisted)│               │ + Metadata │ │
│                            └─────────────────┘               └────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install rlm-dspy
```

Or with uv:
```bash
uv pip install rlm-dspy
```

All supported languages for syntax-aware chunking are included by default:
Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, C#.

## Quick Start

### 1. Setup (One-Time)

```bash
# Interactive setup wizard
rlm-dspy setup

# Or configure directly
rlm-dspy setup --env-file ~/.env --model openai/gpt-4o --budget 1.0
```

This creates `~/.rlm/config.yaml` with your preferences and links to your API keys.

### 2. CLI Usage

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

# Show current config
rlm-dspy config
```

### 3. Python API

```python
from rlm_dspy import RLM, RLMConfig

# Uses settings from ~/.rlm/config.yaml automatically
rlm = RLM()

# Or configure explicitly
config = RLMConfig(
    model="openai/gpt-4o",
    max_budget=1.0,  # USD
    syntax_aware_chunking=True,  # Use tree-sitter
)
rlm = RLM(config=config)

# Load context from files
context = rlm.load_context(["src/", "docs/"])

# Query
result = rlm.query("What are the main components?", context)
print(result.answer)
print(f"Time: {result.elapsed_time:.1f}s, Cost: ${result.total_cost:.4f}")
```

## Supported Providers

RLM-DSPY supports **100+ LLM providers** via LiteLLM:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
rlm-dspy ask "..." ./src --model openai/gpt-4o

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
rlm-dspy ask "..." ./src --model anthropic/claude-sonnet-4-20250514

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."
rlm-dspy ask "..." ./src --model deepseek/deepseek-chat

# Moonshot (Kimi)
export MOONSHOT_API_KEY="sk-..."
rlm-dspy ask "..." ./src --model moonshot/kimi-latest

# MiniMax
export MINIMAX_API_KEY="..."
rlm-dspy ask "..." ./src --model minimax/MiniMax-M2.1

# Ollama (local)
rlm-dspy ask "..." ./src --model ollama/llama3.2
```

**Full provider list**: See [docs/PROVIDERS.md](docs/PROVIDERS.md) for OpenAI, Anthropic, Google, DeepSeek, Kimi, MiniMax, Qwen, GLM, AWS Bedrock, Azure, Ollama, and more.

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

### 2. Smart Response Caching

DSPy's built-in disk cache provides automatic memoization:

```python
# Same query + same code = instant cached response
rlm-dspy ask "What does foo do?" src/foo.py  # First call: ~5s (API call)
rlm-dspy ask "What does foo do?" src/foo.py  # Second call: ~1s (cached!)

# Code changes = fresh response (cache key includes content)
echo "def foo(): return 42" > src/foo.py
rlm-dspy ask "What does foo do?" src/foo.py  # Fresh call (content changed)
```

Cache is content-aware: `cache_key = hash(model + prompt + code_content)`

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

## Limitations & Solutions

RLM-DSPy provides **two modes** for different needs:

| Use Case | Command | Accuracy |
|----------|---------|----------|
| "What does function X do?" | `rlm-dspy ask` | ✅ Excellent (semantic) |
| "Find all bugs in this code" | `rlm-dspy ask` | ✅ Excellent (semantic) |
| "Find all classes with line numbers" | `rlm-dspy index --kind class` | ✅ 100% (AST) |
| "Where is function foo defined?" | `rlm-dspy index --name foo` | ✅ 100% (AST) |
| "Find exact string 'foo_bar'" | `grep` | ✅ 100% (text) |

### Semantic Search (`ask`) vs Structural Search (`index`)

```bash
# Semantic understanding (LLM-powered)
rlm-dspy ask "What does the RLM class do?" src/

# Structural lookup (tree-sitter AST, 100% accurate)
rlm-dspy index src/ --kind class              # All classes
rlm-dspy index src/ --name "Error"            # Find by name
rlm-dspy index src/ --kind method --json      # JSON output
```

The `index` command uses tree-sitter for **zero hallucination** on structural queries.

## Suggested Prompts by Use Case

### 🔍 Code Understanding

```bash
# High-level understanding
rlm-dspy ask "What does this codebase do? Summarize the main components." src/

# Specific function/class
rlm-dspy ask "Explain what the RLM class does and how to use it" src/rlm.py

# Architecture
rlm-dspy ask "How is the code organized? What are the main modules and their responsibilities?" src/

# Data flow
rlm-dspy ask "Trace the flow: what happens when a user calls query()?" src/
```

### 🐛 Bug Finding

```bash
# General bug hunt
rlm-dspy ask "Find potential bugs, edge cases, or error conditions that aren't handled" src/

# Specific categories
rlm-dspy ask "Check for: 1) Division by zero 2) Null pointer dereferences 3) Unhandled exceptions 4) Race conditions" src/

# Input validation
rlm-dspy ask "Find places where user input isn't validated or sanitized" src/

# Resource leaks
rlm-dspy ask "Find resource leaks: unclosed files, connections, or memory that isn't freed" src/

# Data flow tracing (reduces false positives)
rlm-dspy ask "In function X, trace where variable Y comes from, what produces it, and verify if Y['key'] can raise KeyError" src/

# Evidence-based (quote exact lines to prevent hallucination)
rlm-dspy ask "For each exception handler, quote the exact logging line. If no logging exists, say 'no logging found'" src/
```

### 🗑️ Dead Code Detection

```bash
# Find unused modules (the "Better Review Prompt")
rlm-dspy ask "For each module in this directory, check if it's actually imported and used by the main code. List any modules that are exported but never used in the main code flow." src/

# Find unused functions
rlm-dspy ask "List all functions that are defined but never called anywhere in the codebase" src/

# Find unused exports
rlm-dspy ask "Check __init__.py exports - which ones are never imported by external code?" src/

# Trace callers (verify if function is actually used)
rlm-dspy ask "For function X, trace all callers. Show the call chain from entry points (main, CLI, public API) to this function." src/
```

### 🔒 Security Review

```bash
# General security
rlm-dspy ask "Find security vulnerabilities: injection, hardcoded secrets, unsafe deserialization, path traversal" src/

# Authentication/Authorization
rlm-dspy ask "Review authentication and authorization. Are there any bypass vulnerabilities?" src/

# Secrets handling
rlm-dspy ask "Find any hardcoded API keys, passwords, or secrets. Check if secrets are properly masked in logs." src/

# Input sanitization
rlm-dspy ask "Find places where external input reaches dangerous functions without sanitization" src/

# Taint tracing (trace user input to dangerous sinks)
rlm-dspy ask "Trace user input from CLI/API entry through all functions until it reaches a dangerous sink (exec, SQL, file ops). Show full path." src/
```

### 📝 Code Review

```bash
# Comprehensive review
rlm-dspy ask "Review this code for: 1) Bugs 2) Performance issues 3) Security problems 4) Code smells 5) Missing error handling" src/

# PR review style
rlm-dspy ask "Review this like a senior engineer. What would you flag in a code review?" src/

# Best practices
rlm-dspy ask "Does this code follow best practices? What improvements would you suggest?" src/
```

### ⚡ Performance

```bash
# Find bottlenecks
rlm-dspy ask "Identify performance bottlenecks: O(n²) algorithms, repeated computations, unnecessary allocations" src/

# Database/IO
rlm-dspy ask "Find N+1 queries, missing indexes, or inefficient database access patterns" src/

# Memory
rlm-dspy ask "Find memory issues: large allocations, memory leaks, objects kept alive unnecessarily" src/

# Hot path analysis
rlm-dspy ask "For the hot path starting at main(), trace each function call, count loop iterations, and identify the most expensive operations." src/
```

### 🏗️ Structural Queries (use `index` for 100% accuracy)

```bash
# List all classes
rlm-dspy index src/ --kind class

# Find specific function
rlm-dspy index src/ --name "process" --kind function

# All methods in a file
rlm-dspy index src/main.py --kind method

# Export as JSON for scripting
rlm-dspy index src/ --kind class --json | jq '.[] | .name'

# Find all error classes
rlm-dspy index src/ --name "Error" --kind class
```

### 📊 Codebase Analysis

```bash
# Dependencies
rlm-dspy ask "What external libraries does this code depend on? Are there any outdated or risky dependencies?" src/

# Complexity
rlm-dspy ask "Which functions are most complex and might need refactoring?" src/

# Test coverage gaps
rlm-dspy ask "What code paths are not covered by tests? What edge cases are missing?" src/ tests/

# Documentation gaps
rlm-dspy ask "Which public functions are missing docstrings or have inadequate documentation?" src/
```

### 🔄 Refactoring

```bash
# Find duplicates
rlm-dspy ask "Find duplicate or very similar code that could be refactored into shared functions" src/

# Code smells
rlm-dspy ask "Find code smells: long functions, deep nesting, god classes, feature envy" src/

# Simplification
rlm-dspy ask "What code could be simplified? Find overly complex implementations." src/

# Dependency analysis (safe to refactor?)
rlm-dspy ask "For class X, list all its dependencies (imports, calls) and all dependents (who uses it). Can it be safely refactored?" src/
```

### 💡 Pro Tips

1. **Be specific** - "Find bugs" is less effective than "Find null pointer dereferences in error handling paths"

2. **Trace data flow** - "Trace where X comes from" reduces false positives vs generic "find bugs"

3. **Combine with `index`** - Use `index` for structural queries, `ask` for semantic understanding

4. **Use budget wisely** - Start with `--budget 0.10` for exploration, increase for thorough analysis

5. **Multi-file context** - Include related files: `rlm-dspy ask "..." src/main.py src/utils.py tests/`

6. **Iterate** - If the first answer is shallow, follow up: "Go deeper on point #2"

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

### Setup Command (Recommended)

The easiest way to configure RLM-DSPy:

```bash
# Interactive wizard
rlm-dspy setup

# Or with options
rlm-dspy setup --env-file ~/.env --model deepseek/deepseek-chat --budget 0.50
```

This creates `~/.rlm/config.yaml`:

```yaml
model: deepseek/deepseek-chat
max_budget: 0.5
env_file: /path/to/.env
```

### Config File Settings

All supported settings for `~/.rlm/config.yaml`:

```yaml
# Model to use (provider/model-name format)
model: openai/gpt-4o-mini

# Maximum cost in USD per query
max_budget: 1.0

# Maximum time in seconds per query
max_timeout: 300

# Chunk size in characters
chunk_size: 100000

# Use tree-sitter for syntax-aware chunking
syntax_aware: true

# Path to .env file with API keys (auto-loaded on startup)
env_file: ~/.env
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `model` | string | `openai/gpt-4o-mini` | LLM model to use |
| `max_budget` | float | `1.0` | Max cost in USD |
| `max_timeout` | int | `300` | Max time in seconds |
| `chunk_size` | int | `100000` | Chunk size (chars) |
| `syntax_aware` | bool | `true` | Tree-sitter chunking |
| `env_file` | string | `null` | Path to .env file |

### Configuration Priority

Settings are resolved in this order (highest to lowest):

| Priority | Source | Example |
|----------|--------|---------|
| 1 | CLI arguments | `--model openai/gpt-4o` |
| 2 | Environment variables | `RLM_MODEL=openai/gpt-4o` |
| 3 | User config | `~/.rlm/config.yaml` |
| 4 | Built-in defaults | `openai/gpt-4o-mini` |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_API_KEY` | - | API key (or use provider-specific) |
| `RLM_MODEL` | `openai/gpt-4o-mini` | Model to use |
| `RLM_API_BASE` | - | Custom API endpoint (optional) |
| `RLM_MAX_BUDGET` | `1.0` | Maximum cost in USD |
| `RLM_MAX_TIMEOUT` | `300` | Maximum time in seconds |
| `RLM_CHUNK_SIZE` | `100000` | Chunk size (chars) |
| `RLM_SYNTAX_AWARE` | `true` | Tree-sitter chunking |
| `RLM_PARALLEL_CHUNKS` | `20` | Concurrent chunks |

### Provider-Specific API Keys

RLM-DSPy auto-detects API keys based on model prefix:

| Model Prefix | Environment Variable |
|--------------|---------------------|
| `openai/` | `OPENAI_API_KEY` |
| `anthropic/` | `ANTHROPIC_API_KEY` |
| `deepseek/` | `DEEPSEEK_API_KEY` |
| `moonshot/` | `MOONSHOT_API_KEY` |
| `minimax/` | `MINIMAX_API_KEY` |
| `gemini/` | `GEMINI_API_KEY` |
| `groq/` | `GROQ_API_KEY` |
| `openrouter/` | `OPENROUTER_API_KEY` |

### Quick Start

```bash
# Option 1: Use setup wizard (links to existing .env)
rlm-dspy setup --env-file ~/.env

# Option 2: Set environment variable directly
export OPENAI_API_KEY="sk-..."
rlm-dspy ask "What does this do?" ./src

# Option 3: Provider-specific key
export DEEPSEEK_API_KEY="sk-..."
rlm-dspy ask "..." ./src --model deepseek/deepseek-chat
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

## Documentation

- **[Provider Guide](docs/PROVIDERS.md)** - Full list of 100+ supported LLM providers
- **[Testing Results](docs/TESTING.md)** - Accuracy tests, anti-hallucination checks, and benchmarks

## License

MIT

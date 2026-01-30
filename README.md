# RLM-DSPy

> Recursive Language Models powered by DSPy - treating large contexts as environments for LLMs to explore programmatically.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is RLM?

**Recursive Language Models (RLMs)** are a fundamentally different approach to handling large contexts. Instead of chunking text and processing it passively, RLMs give the LLM **agency** to explore the data programmatically through a Python REPL.

> Reference: ["Recursive Language Models"](https://arxiv.org/abs/2501.xxxxx) (Zhang, Kraska, Khattab, 2025)

## Why RLM-DSPy?

| Approach | How it works | Limitation |
|----------|--------------|------------|
| **Chunking/RAG** | Split text → Process each chunk → Aggregate | Fixed boundaries, misses cross-chunk context |
| **Long-context models** | Feed everything to model | Expensive, still has limits |
| **RLM (this library)** | LLM writes code to explore data | LLM decides what's relevant |

### Key Insight

The LLM has **agency**. It writes Python code to:
1. Navigate and slice the context
2. Call `llm_query()` for semantic analysis of specific sections
3. Build up an answer iteratively
4. Call `SUBMIT()` when it has enough information

## When to Use RLM

RLM is designed for **large contexts** that don't fit in a single LLM context window:

| Context Size | Recommendation |
|--------------|----------------|
| Small (<50KB) | Direct LLM query may be faster |
| Medium (50-200KB) | RLM helps with navigation and accuracy |
| Large (>200KB) | **RLM essential** - can't fit in standard context |

### Best Use Cases

✅ **RLM excels at:**
- Analyzing entire codebases (multiple files, 1000s of lines)
- Cross-file dependency analysis ("How does data flow from A to B?")
- Architecture overviews ("Describe how these components interact")
- Finding patterns across many files ("Find all error handling issues")

⚠️ **Consider direct LLM for:**
- Single small file analysis
- Simple questions with obvious answers
- When speed is more important than thoroughness

### Accuracy with Tools

When analyzing large codebases, RLM automatically uses built-in tools for **100% accurate** structural queries:

| Query Type | Without Tools | With Tools (default) |
|------------|---------------|---------------------|
| "List all classes" | ~70% accurate | **100% accurate** |
| "Find methods in class X" | Line numbers often wrong | **Exact line numbers** |
| "Architecture overview" | Good | **100% grounded** |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RLM-DSPy Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT                      REPL ENVIRONMENT                    OUTPUT      │
│  ─────                      ────────────────                    ──────      │
│                                                                              │
│  ┌─────────┐               ┌─────────────────────────────────┐              │
│  │  Files  │──┐            │  Python REPL (Sandboxed)        │              │
│  └─────────┘  │            │                                 │              │
│               │            │  Variables:                     │              │
│  ┌─────────┐  │            │    • context = "your files..."  │              │
│  │  Query  │──┼──────────▶ │    • query = "your question"    │              │
│  └─────────┘  │            │                                 │              │
│               │            │  Tools:                         │              │
│  ┌─────────┐  │            │    • llm_query(prompt)          │──────┐      │
│  │ Config  │──┘            │    • llm_query_batched(prompts) │      │      │
│  └─────────┘               │    • print()                    │      │      │
│                            │    • SUBMIT(answer)             │      │      │
│                            └─────────────┬───────────────────┘      │      │
│                                          │                          │      │
│                            ┌─────────────▼───────────────────┐      │      │
│                            │  LLM writes Python code to:     │      │      │
│                            │  1. Explore context             │◀─────┘      │
│                            │  2. Call sub-LLMs               │              │
│                            │  3. Build answer iteratively    │──────────┐  │
│                            │  4. SUBMIT() when ready         │          │  │
│                            └─────────────────────────────────┘          │  │
│                                                                         │  │
│                            ┌────────────────────────────────────────────▼──┤
│                            │  RLMResult                                    │
│                            │    • answer: str                              │
│                            │    • trajectory: list[{code, output, ...}]   │
│                            │    • iterations: int                          │
│                            └───────────────────────────────────────────────┘
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

**Deno** is required for the sandboxed Python REPL:
```bash
# Install Deno
curl -fsSL https://deno.land/install.sh | sh
# Add to PATH (add to your shell profile for persistence)
export PATH="$HOME/.deno/bin:$PATH"
```

> ⚠️ **Important**: Without Deno in PATH, code execution fails and LLM outputs hallucinated results.

### Install RLM-DSPy

```bash
pip install rlm-dspy
```

Or with uv:
```bash
uv pip install rlm-dspy
```

### Optional Dependencies

```bash
# Local embeddings (no API key required)
pip install rlm-dspy[local]


# All optional features
pip install rlm-dspy[all]
```

### External Tools (Optional)

**ripgrep** - Fast regex search (highly recommended):
```bash
# macOS
brew install ripgrep

# Ubuntu/Debian
sudo apt install ripgrep

# Or via cargo
cargo install ripgrep
```

```

Tree-sitter is included by default for AST-based code analysis.

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

# Verbose output (shows REPL iterations)
rlm-dspy ask "Explain this" file.py -v

# Full debug (shows API calls)
rlm-dspy ask "Debug this" file.py -d

# Dry run (validate config only)
rlm-dspy ask "Test" file.py -n

# Output formats
rlm-dspy ask "Explain" file.py                    # Rich text (default)
rlm-dspy ask "Explain" file.py --format json      # JSON output
rlm-dspy ask "Explain" file.py --format markdown  # Markdown output
rlm-dspy ask "Explain" file.py -j                 # JSON shorthand

# Save to file
rlm-dspy ask "Analyze" src/ -o report.md              # Text to file
rlm-dspy ask "Analyze" src/ -f markdown -o report.md  # Markdown to file
rlm-dspy ask "Analyze" src/ -j -o data.json           # JSON to file

# Analyze from stdin
cat large_file.txt | rlm-dspy ask "Summarize this" --stdin

# Analyze a git diff
git diff | rlm-dspy diff "Are there any breaking changes?"

# Multi-analysis (runs 3 queries in parallel)
rlm-dspy analyze src/

# Check configuration
rlm-dspy preflight

# Show current config
rlm-dspy config

# Example prompts
rlm-dspy example
```

### 3. Python API

```python
from rlm_dspy import RLM, RLMConfig

# Basic usage - loads settings from environment
rlm = RLM()

# Load context from files
context = rlm.load_context(["src/", "docs/README.md"])

# Query - the LLM explores context via REPL
result = rlm.query("What does the main function do?", context)

print(result.answer)         # The final answer
print(result.trajectory)     # See how LLM explored the context
print(result.iterations)     # Number of REPL iterations

# Custom configuration
config = RLMConfig(
    model="openai/gpt-4o",           # Primary model for reasoning
    sub_model="openai/gpt-4o-mini",  # Cheaper model for llm_query()
    max_iterations=20,               # Max REPL iterations
    max_llm_calls=50,                # Max sub-LLM calls
    verbose=True,                    # Show iteration logs
)
rlm = RLM(config=config)
```

## How RLM Works (Example Trajectory)

When you ask a question, the LLM **writes and executes Python code** to explore your context:

```
Query: "What does the main function do?"
Context: [large codebase...]

─── Iteration 1 ───
Reasoning: I need to explore the context to find the main function.
Code:
    print(f"Context length: {len(context)}")
    print(context[:2000])  # Preview first 2000 chars
Output:
    Context length: 45000
    === FILE: src/main.py ===
    1 | import sys
    2 | from app import Application
    ...

─── Iteration 2 ───
Reasoning: Found main.py. Let me search for the main function.
Code:
    for i, line in enumerate(context.split('\n')):
        if 'def main' in line:
            print(f"Found at line {i}: {line}")
Output:
    Found at line 45: def main(args):

─── Iteration 3 ───
Reasoning: Found it at line 45. Let me extract and analyze it.
Code:
    lines = context.split('\n')
    main_func = '\n'.join(lines[44:70])
    analysis = llm_query(f"Explain this function:\n{main_func}")
    print(analysis)
Output:
    The main function initializes the Application with CLI args,
    sets up logging, and calls app.run()...

─── Iteration 4 ───
Reasoning: I have enough information to answer.
Code:
    SUBMIT("The main function initializes the Application...")

→ Answer: "The main function initializes the Application..."
```

## Supported Providers

RLM-DSPy supports **100+ LLM providers** via LiteLLM:

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

# Google Gemini
export GEMINI_API_KEY="..."
rlm-dspy ask "..." ./src --model gemini/gemini-2.0-flash

# OpenRouter (any model)
export OPENROUTER_API_KEY="sk-or-..."
rlm-dspy ask "..." ./src --model openrouter/google/gemini-3-flash-preview

# Ollama (local)
rlm-dspy ask "..." ./src --model ollama/llama3.2
```

**Full provider list**: See [docs/PROVIDERS.md](docs/PROVIDERS.md)

## Configuration

### Config File (`~/.rlm/config.yaml`)

```yaml
# Model settings
model: openrouter/google/gemini-3-flash-preview
sub_model: openrouter/google/gemini-3-flash-preview  # Use same model to reduce hallucination

# Execution limits (higher = fewer hallucinations from forced completion)
max_iterations: 30      # Max REPL iterations (min: 20, max: 100)
max_llm_calls: 100      # Max sub-LLM calls (default: 50, max: 500)

# Parallelism settings
max_workers: 8          # Workers for batch operations (default: 8, max: 32)

# Budget/safety limits
max_budget: 2.0         # Max cost in USD (default: 1.0, max: 100)
max_timeout: 600        # Max time in seconds (default: 300, max: 3600)

# Embedding settings (for semantic search)
embedding_model: openrouter/openai/text-embedding-3-small
embedding_batch_size: 100  # Embeddings per API call

# Vector index settings
index_dir: ~/.rlm/indexes  # Where indexes are stored
use_faiss: true            # Use FAISS for large indexes
faiss_threshold: 20000     # Switch to FAISS above this snippet count
auto_update_index: true    # Auto-rebuild when files change
index_cache_ttl: 3600      # Index cache TTL in seconds

# Path to .env file with API keys
env_file: ~/.env
```

**Tip**: To reduce hallucinations, use the same high-quality model for both `model` and `sub_model`, and increase `max_iterations`. Minimum 20 iterations enforced to prevent early termination.

### Environment Variables

**Core Settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_API_KEY` | - | API key (or use provider-specific) |
| `RLM_MODEL` | `openai/gpt-4o-mini` | Model to use |
| `RLM_SUB_MODEL` | same as RLM_MODEL | Model for sub-LLM calls |
| `RLM_API_BASE` | - | Custom API endpoint (optional) |
| `RLM_MAX_ITERATIONS` | `20` | Max REPL iterations (min: 20) |
| `RLM_MAX_LLM_CALLS` | `50` | Max sub-LLM calls per query |
| `RLM_MAX_OUTPUT_CHARS` | `100000` | Max chars in REPL output |
| `RLM_MAX_WORKERS` | `8` | Parallel workers for batch ops |
| `RLM_MAX_BUDGET` | `1.0` | Maximum cost in USD |
| `RLM_MAX_TIMEOUT` | `300` | Maximum time in seconds |

**Embedding Settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `RLM_EMBEDDING_API_KEY` | - | Embedding API key (if different) |
| `RLM_EMBEDDING_API_BASE` | - | Custom embedding endpoint |
| `RLM_EMBEDDING_BATCH_SIZE` | `100` | Embeddings per API call |
| `RLM_INDEX_DIR` | `~/.rlm/indexes` | Index storage directory |

**Debug/Feature Flags:**

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_DEBUG` | `false` | Enable debug output |
| `RLM_VERBOSE` | `false` | Enable verbose output |
| `RLM_QUIET` | `false` | Suppress non-essential output |
| `RLM_ALLOW_SHELL` | `false` | Enable shell tool (security risk) |

### Provider-Specific API Keys

| Model Prefix | Environment Variable |
|--------------|---------------------|
| `openai/` | `OPENAI_API_KEY` |
| `anthropic/` | `ANTHROPIC_API_KEY` |
| `deepseek/` | `DEEPSEEK_API_KEY` |
| `gemini/` | `GEMINI_API_KEY` |
| `groq/` | `GROQ_API_KEY` |
| `openrouter/` | `OPENROUTER_API_KEY` |

## Example Prompts

```bash
# Code understanding
rlm-dspy ask "What does this codebase do?" src/
rlm-dspy ask "Explain the RLM class" src/rlm.py

# Bug finding
rlm-dspy ask "Find potential bugs and edge cases" src/
rlm-dspy ask "Check for security vulnerabilities" src/

# Code review
rlm-dspy ask "Review this code for issues" src/

# Architecture analysis
rlm-dspy ask "How is the code organized?" src/

# See more examples
rlm-dspy example
```

## CLI Reference

### Command Groups

| Group | Purpose |
|-------|---------|
| `rlm-dspy ask` | Query files/directories |
| `rlm-dspy analyze` | Parallel multi-analysis |
| `rlm-dspy diff` | Analyze git diffs |
| `rlm-dspy index` | Manage semantic search indexes |
| `rlm-dspy project` | Multi-project management |
| `rlm-dspy daemon` | Background index monitoring |
| `rlm-dspy traces` | Manage execution traces |
| `rlm-dspy optimize` | Self-optimization controls |
| `rlm-dspy setup` | Configuration wizard |
| `rlm-dspy config` | View/modify settings |
| `rlm-dspy preflight` | Environment checks |

### Basic Usage

```bash
rlm-dspy ask "Your question" src/           # Analyze files/directories
rlm-dspy ask "Question" --stdin < file.txt  # From stdin
rlm-dspy analyze src/                       # Parallel batch analysis
rlm-dspy diff HEAD~1                        # Review git diff
```

### Key Options

| Option | Short | Description |
|--------|-------|-------------|
| `--signature` | `-S` | Output format: security, bugs, review, architecture, performance, diff |
| `--max-iterations` | `-i` | Max REPL iterations (default: 20, min: 20) |
| `--max-tokens` | `-T` | Truncate context to token limit |
| `--max-workers` | `-w` | Parallel workers for batch ops |
| `--no-tools` | | Disable built-in tools (ripgrep, AST) |
| `--no-cache` | | Disable context caching |
| `--validate` | `-V` | Check output for hallucinations (default: on) |
| `--json` | `-j` | JSON output format |
| `--verbose` | `-v` | Show detailed progress |
| `--debug` | `-d` | Full debug output |

### Examples with Options

```bash
# Structured security audit
rlm-dspy ask "Find vulnerabilities" src/ -S security -j

# Truncate large codebase
rlm-dspy ask "Overview" src/ -T 50000

# More thorough analysis
rlm-dspy ask "Find bugs" src/ -S bugs -i 30

# Parallel batch (8 workers)
rlm-dspy ask "Analyze" src/ -w 8

# Fresh load (no cache)
rlm-dspy ask "Check" src/ --no-cache

# Disable hallucination check (faster)
rlm-dspy ask "Quick check" src/ --no-validate
```

### Index Commands

```bash
rlm-dspy index build .              # Build semantic index
rlm-dspy index status .             # Check index status
rlm-dspy index search "query" -k 10 # Search code semantically
rlm-dspy index compress .           # Compress index (~2x savings)
rlm-dspy index clear .              # Remove index
```

### Project Commands

```bash
rlm-dspy project add myproject /path/to/project
rlm-dspy project list
rlm-dspy project default myproject
rlm-dspy project tag myproject python backend
rlm-dspy project remove myproject
```

### Daemon Commands

```bash
rlm-dspy daemon start               # Start background indexer
rlm-dspy daemon status              # Check daemon status
rlm-dspy daemon watch /path/to/dir  # Add watched directory
rlm-dspy daemon list                # List watched paths
rlm-dspy daemon stop                # Stop daemon
```

## Python API Reference

```python
from rlm_dspy import RLM, RLMConfig, RLMResult, ProgressCallback

# Configuration
config = RLMConfig(
    model="openai/gpt-4o",
    sub_model="openai/gpt-4o-mini",
    max_iterations=20,
    max_llm_calls=50,
    max_output_chars=100_000,
    max_workers=8,             # Parallel workers for batch
    max_budget=1.0,
    max_timeout=300,
    verbose=False,
)

# Create RLM instance
rlm = RLM(config=config)

# Load context with options
context = rlm.load_context(
    ["src/", "docs/"],
    max_tokens=100_000,   # Truncate if too large
    use_cache=True,       # Use context caching
)

# Query
result = rlm.query("What does this do?", context)

# Result fields
result.answer          # str: The final answer
result.success         # bool: Whether query succeeded
result.trajectory      # list: REPL iteration history
result.iterations      # int: Number of iterations
result.elapsed_time    # float: Total time in seconds
result.error           # str | None: Error message if failed
result.outputs         # dict: Structured outputs (for custom signatures)

# Batch processing (parallel)
results = rlm.batch([
    {"query": "What is this?"},
    {"query": "Find bugs"},
], context, num_threads=4)

# Progress callbacks
class MyProgress(ProgressCallback):
    def on_start(self, query, tokens):
        print(f"Starting with {tokens} tokens")
    def on_complete(self, result):
        print(f"Done in {result.elapsed_time:.1f}s")

rlm = RLM(config=config, progress_callback=MyProgress())

# Add custom tools
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Results for: {query}"

rlm.add_tool("search_docs", search_docs)
```

## Advanced Features

### Batch Processing

Process multiple queries in parallel for faster analysis:

```python
# Same context, different queries (3x faster than sequential)
results = rlm.batch([
    {"query": "Summarize the architecture"},
    {"query": "Find security issues"},
    {"query": "Find performance bottlenecks"},
], context=context, num_threads=3)

# Different contexts
results = rlm.batch([
    {"context": file1, "query": "Find bugs"},
    {"context": file2, "query": "Find bugs"},
], num_threads=2)
```

The `analyze` command uses batch processing by default:
```bash
rlm-dspy analyze src/  # Runs 3 queries in parallel
```

### Structured Output (Custom Signatures)

Get structured JSON output instead of free-form text:

```python
from rlm_dspy import RLM, BugFinder, SecurityAudit

# Use predefined signature
rlm = RLM(config=config, signature=BugFinder)
result = rlm.query("Find all bugs", context)

# Access structured fields
print(result.bugs)              # list[str]
print(result.has_critical)      # bool
print(result.affected_functions) # list[str]
print(result.fix_suggestions)   # list[str]

# Or via dict
print(result.outputs)  # {"bugs": [...], "has_critical": True, ...}
```

**Standard Signatures:**

| Signature | Output Fields |
|-----------|---------------|
| `SecurityAudit` | `vulnerabilities`, `severity` (none/low/medium/high/critical), `is_secure`, `recommendations` |
| `CodeReview` | `summary`, `issues`, `suggestions`, `quality_score` (1-10) |
| `BugFinder` | `bugs`, `has_critical`, `affected_functions`, `fix_suggestions` |
| `ArchitectureAnalysis` | `summary`, `components`, `dependencies`, `patterns` |
| `PerformanceAnalysis` | `issues`, `hotspots`, `optimizations`, `complexity_concerns` |
| `DiffReview` | `summary`, `change_type`, `is_breaking`, `risks`, `suggestions` |

**Cited Signatures** (include `file:line` references):

| Signature | Output Fields |
|-----------|---------------|
| `CitedAnalysis` | `summary`, `findings`, `locations` |
| `CitedSecurityAudit` | `vulnerabilities`, `risk_level`, `locations`, `remediation` |
| `CitedBugFinder` | `bugs`, `severity`, `locations`, `fixes` |
| `CitedCodeReview` | `issues`, `quality_score`, `locations`, `improvements` |

CLI usage:
```bash
# Structured security audit (JSON)
rlm-dspy ask "Audit this code" src/ --signature security --format json

# Structured bug report (JSON shorthand)
rlm-dspy ask "Find bugs" src/ -S bugs -j

# Structured output as Markdown
rlm-dspy ask "Find bugs" src/ -S bugs --format markdown -o report.md
```

### Hallucination Detection

Validate outputs using LLM-as-judge (DSPy's semantic evaluation):

```bash
# CLI: Validate output is grounded in context
rlm-dspy ask "Find bugs with line numbers" src/ --validate
```

```python
from rlm_dspy import RLM, validate_groundedness, semantic_f1

result = rlm.query("Find bugs", context)

# Check if claims are supported by context
validation = validate_groundedness(
    output=result.answer,
    context=context,
    query="Find bugs in this code",
)
print(f"Groundedness: {validation.score:.0%}")
if not validation.is_grounded:
    print(f"Warning: Potential hallucination!")
    print(f"Claims: {validation.claims}")
    print(f"Discussion: {validation.discussion}")

# Compare against expected output
f1 = semantic_f1(
    output=result.answer,
    expected="Expected response here",
    query="Find bugs",
)
print(f"Semantic F1: {f1:.0%}")
```

Available validators:
- **validate_groundedness()** - Check if claims are supported by context
- **validate_completeness()** - Check coverage of expected content
- **semantic_f1()** - Precision/recall of semantic content

### Custom Interpreter

Use a custom code execution environment:

```python
# Default: dspy's PythonInterpreter (Deno/Pyodide WASM)
rlm = RLM(config=config)

# Custom interpreter (e.g., E2B cloud sandbox)
from e2b_code_interpreter import CodeInterpreter
rlm = RLM(config=config, interpreter=CodeInterpreter())
```

The interpreter must implement the CodeInterpreter protocol:
- `tools` property - available functions
- `start()` - initialize environment  
- `execute(code, variables)` - run code
- `shutdown()` - cleanup

### Code Analysis Tools

Powerful code analysis tools are **enabled by default**. They run on the host (not in sandbox):

```bash
# Tools are enabled by default - LLM can use ripgrep, tree-sitter, etc.
rlm-dspy ask "Find all functions that call 'execute'" src/
rlm-dspy ask "Use index_code to find classes, then analyze them for bugs" src/

# Disable tools if needed
rlm-dspy ask "Simple question" src/ --no-tools
```

```python
from rlm_dspy import RLM

# Tools enabled by default
rlm = RLM(config=config)

# Enable all tools including shell (requires RLM_ALLOW_SHELL=1)
rlm = RLM(config=config, use_tools="all")

# Disable tools
rlm = RLM(config=config, use_tools=False)

result = rlm.query(
    "Use index_code to find all classes, then analyze them for bugs",
    context
)
```

**Available tools:**

| Category | Tool | Description |
|----------|------|-------------|
| **Text Search** | `ripgrep(pattern, path, flags)` | Fast regex search via `rg` |
| | `grep_context(pattern, path, lines)` | Search with surrounding context |
| **File Operations** | `find_files(pattern, path, type)` | Find files by glob pattern |
| | `read_file(path, start, end)` | Read file with line numbers |
| | `file_stats(path)` | Get file/directory statistics (JSON) |
| **AST Analysis** | `index_code(path, kind, name)` | **AST index with EXACT line numbers** (10+ languages) |
| | `find_definitions(path, name)` | Find all definitions (classes, functions, methods) |
| | `find_classes(path, name)` | Find class definitions |
| | `find_functions(path, name)` | Find top-level functions |
| | `find_methods(path, name)` | Find methods (shows parent class) |
| | `find_imports(path)` | Find all import statements |
| | `find_calls(path, func_name)` | Find function/method call sites |
| **Semantic Search** | `semantic_search(query, path, k)` | Search by concept similarity |
| | `go_to_definition(file, line, col)` | Jump to definition |
| | `get_type_info(file, line, col)` | Get type signatures (hover) |
| | `get_symbol_hierarchy(file)` | Get file's symbol tree |
| **Shell** | `shell(cmd, timeout)` | Run shell commands (disabled by default) |

**Shell Security:**
- Disabled by default - requires `RLM_ALLOW_SHELL=1` environment variable
- Only allowlisted commands: `ls`, `cat`, `grep`, `find`, `git`, `python`, etc.
- Dangerous patterns blocked: `rm -rf`, `curl | sh`, `eval`, backticks, etc.
- Uses `shell=False` with parsed arguments (no shell injection)
- **Note:** `python` and `git` in the allowlist can execute arbitrary code via flags. Only enable shell in trusted environments.

**Supported languages for AST tools:** Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, C#

**Prerequisites:**

```bash
# ripgrep (optional but recommended for fast regex search)
# macOS: brew install ripgrep
# Ubuntu: sudo apt install ripgrep

# Tree-sitter is included by default (10+ languages supported)
```

**How it works:**
1. Tools are documented in the LLM's prompt automatically
2. LLM is instructed to use tools FIRST for accurate results
3. Tools run on the host (not in sandbox) and return results to the LLM
4. LLM combines tool outputs with semantic analysis

**Automatic tool selection:**
- For structural queries (classes, functions, line numbers): Uses AST tools
- For pattern search: Uses ripgrep (faster than regex on large contexts)
- For semantic analysis: Uses llm_query() on relevant sections
- The LLM decides the best approach based on the query

### Semantic Search

Find code by conceptual similarity using embeddings:

```bash
# CLI commands
rlm-dspy index build src/                    # Build vector index
rlm-dspy index status src/                   # Check index status
rlm-dspy index search "auth logic" -p src/   # Search semantically
rlm-dspy index clear                         # Clear all indexes
```

```python
from rlm_dspy import semantic_search
from rlm_dspy.core import CodeIndex, get_index_manager

# Quick search (auto-builds index if needed)
results = semantic_search("authentication logic", path="src/", k=5)
for r in results:
    print(f"{r['file']}:{r['line']} - {r['name']}")

# Full control
index = CodeIndex()
index.build("src/")                          # Build index
results = index.search("src/", "error handling", k=10)
```

**Features:**
- **Incremental updates**: Only re-indexes changed files
- **Auto-skip**: Ignores .venv, node_modules, __pycache__
- **Multi-language**: Python, JS, TS, Go, Rust, Java, C/C++, Ruby
- **Fast**: FAISS for large codebases (>5000 snippets)
- **Cached**: TTL-based in-memory + persistent disk storage

**Configuration** (`~/.rlm/config.yaml`):
```yaml
# Embedding settings
embedding_model: openai/text-embedding-3-small  # or local
local_embedding_model: sentence-transformers/all-MiniLM-L6-v2
embedding_batch_size: 100

# Index settings
index_dir: ~/.rlm/indexes
use_faiss: true
faiss_threshold: 5000
auto_update_index: true  # See note below
```

**Index Auto-Update Behavior:**

When `auto_update_index: true` (default):
- On each search, RLM checks file modification times against the index manifest
- Changed/new files are incrementally re-embedded (only changed files, not full rebuild)
- Deleted files are removed from the index
- This ensures search results are always up-to-date

When `auto_update_index: false`:
- The index is built once and cached
- Faster searches (no file mtime checks)
- **Stale results possible** if files changed since last indexing
- Useful for: read-only codebases, CI environments, or when you explicitly rebuild with `rlm-dspy index build`

To force a fresh index regardless of setting:
```bash
rlm-dspy index build src/ --force
```

### Cited Analysis

Get analysis with precise file:line source references:

```bash
# Use cited signatures for grounded analysis
rlm-dspy ask "audit security" src/ -S cited-security -j
rlm-dspy ask "find bugs" src/ -S cited-bugs -j
rlm-dspy ask "review code" src/ -S cited-review -j
```

```python
from rlm_dspy import RLM, CitedSecurityAudit

# Use cited signature
rlm = RLM(signature=CitedSecurityAudit)
result = rlm.query("audit", context)

# Access citations
for vuln in result.vulnerabilities:
    print(vuln)  # "[CRITICAL] SQL injection - db.py:45"
for loc in result.locations:
    print(loc)   # "db.py:45"
```

**Cited Signatures:**

| Signature | Alias | Output Fields |
|-----------|-------|---------------|
| `CitedAnalysis` | `cited` | summary, findings, locations |
| `CitedSecurityAudit` | `cited-security` | vulnerabilities, risk_level, locations, remediation |
| `CitedBugFinder` | `cited-bugs` | bugs, severity, locations, fixes |
| `CitedCodeReview` | `cited-review` | issues, quality_score, locations, improvements |

**Citation Utilities:**
```python
from rlm_dspy import (
    code_to_document,       # Add line numbers for citation
    files_to_documents,     # Batch convert files
    citations_to_locations, # Extract file:line from text
    parse_findings_from_text, # Auto-extract findings
)

# Convert code to document with line numbers
doc = code_to_document("src/api.py")
# "   1 | def hello():\n   2 |     pass"

# Parse findings from analysis text
findings = parse_findings_from_text(analysis_text, documents)
for f in findings:
    print(f.format())  # "[CRITICAL:security] SQL injection → db.py:45"
```

### Project Management

Manage multiple indexed codebases with named projects:

```bash
# Register a project
rlm-dspy project add my-app ~/projects/my-app
rlm-dspy project add backend ./backend --tags python,api

# List all projects
rlm-dspy project list
# ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┓
# ┃ Name      ┃ Path           ┃ Snippets ┃ Files ┃ Tags      ┃ Indexed    ┃
# ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━┩
# │ * my-app  │ ~/projects/... │      450 │    32 │ -         │ 2025-01-28 │
# │ backend   │ ./backend      │      120 │    15 │ python,api│ 2025-01-27 │
# └───────────┴────────────────┴──────────┴───────┴───────────┴────────────┘

# Set default project
rlm-dspy project default my-app

# Add tags for organization
rlm-dspy project tag my-app python,web,frontend

# Remove a project
rlm-dspy project remove old-project
rlm-dspy project remove old-project --delete-index  # Also delete index

# Migrate legacy hash-based indexes
rlm-dspy project migrate

# Cleanup orphaned index directories
rlm-dspy project cleanup
```

```python
from rlm_dspy.core import get_project_registry

# Access registry
registry = get_project_registry()

# List projects
for project in registry.list(tags=["python"]):
    print(f"{project.name}: {project.snippet_count} snippets")

# Get project by name or alias
project = registry.get("my-app")
```

**Features:**
- **Named projects**: Clear names instead of hash directories
- **Auto-register**: Projects registered on first `index build`
- **Tags**: Organize projects by category
- **Default project**: Set default for searches
- **Migration**: Auto-migrate legacy hash-based indexes
- **Cleanup**: Remove orphaned index directories

**Storage Structure:**
```
~/.rlm/
├── config.yaml      # Global configuration
├── projects.json    # Project registry
└── indexes/
    ├── my-app/      # Named project indexes
    └── backend/
```

### Index Daemon (Auto-Indexing)

Run a background daemon that automatically re-indexes projects when files change:

```bash
# Start daemon
rlm-dspy daemon start              # Background (daemonize)
rlm-dspy daemon start --foreground # Foreground (Ctrl+C to stop)

# Stop daemon
rlm-dspy daemon stop

# Check status
rlm-dspy daemon status
# ● Daemon running (PID: 12345)
#   Watching 2 project(s):
#     - my-app
#     - backend

# Add project to watch list
rlm-dspy daemon watch my-app

# Remove from watch list
rlm-dspy daemon unwatch my-app

# List watched projects
rlm-dspy daemon list

# View daemon logs
rlm-dspy daemon log              # Show last 50 lines
rlm-dspy daemon log -n 100       # Show last 100 lines  
rlm-dspy daemon log -f           # Follow log (tail -f)
rlm-dspy daemon log --clear      # Clear log file
```

**Example log output:**
```
2026-01-29 09:12:42 - INFO - Daemon started (PID: 12345)
2026-01-29 09:12:43 - INFO - Watching project: my-app
2026-01-29 09:12:45 - INFO - [my-app] MODIFIED: api.py
2026-01-29 09:12:50 - INFO - Re-indexing project: my-app
2026-01-29 09:12:51 - INFO - Incremental update: 1 new/modified, 0 deleted
2026-01-29 09:12:52 - INFO - Incremental update complete: 450 snippets
```

```python
from rlm_dspy.core import IndexDaemon

# Start daemon programmatically
daemon = IndexDaemon()
daemon.start()

# Watch a project
daemon.watch("my-app")

# Stop daemon
daemon.stop()
```

**Features:**
- **File watching**: Uses `watchdog` for cross-platform file monitoring
- **Debouncing**: Waits 5s after last change before re-indexing
- **Incremental indexing**: Only re-embeds changed files (91% API cost reduction)
- **Auto-watch**: Projects with `auto_watch=True` are watched on daemon start
- **Logging**: Full activity log with `daemon log` command
- **Resource limits**: Configurable max concurrent index builds
- **Idle timeout**: Optional auto-stop after period of inactivity

**Configuration** (`~/.rlm/config.yaml`):
```yaml
daemon:
  watch_debounce: 5        # Seconds to wait after file change
  max_concurrent_indexes: 2 # Max parallel index builds
  idle_timeout: 0          # Auto-stop after N seconds (0 = never)
```

## Self-Optimization (DSPy Patterns)

RLM-DSPy automatically improves over time using DSPy-inspired optimization patterns:

### Automatic Features (No Manual Action Required)

| Feature | What It Does | When It Triggers |
|---------|--------------|------------------|
| **Trace Collection** | Records successful query trajectories | After every validated query |
| **Failure Analysis** | Tracks why queries fail validation | After every failed validation |
| **Tip Injection** | Adds learned tips to prompts | On every RLM initialization |
| **Tip Refresh** | Regenerates tips from patterns | Every 50 queries (if ≥5 failures) |

### How It Works

```
Query → Validation → Record Success/Failure → Update Stats
                            ↓
              (every 50 queries with ≥5 failures)
                            ↓
                   Auto-regenerate tips via LLM
                            ↓
                   Next query uses improved tips
```

### CLI Commands

```bash
# View optimization statistics
rlm-dspy optimize stats

# View current tips (learned from failures)
rlm-dspy optimize tips

# Force regenerate tips now (uses LLM)
rlm-dspy optimize tips --regenerate

# Reset to default tips
rlm-dspy optimize tips --reset

# View/modify tool instructions
rlm-dspy optimize instructions
rlm-dspy optimize instructions tool_instructions

# Run SIMBA optimization (requires collected traces)
rlm-dspy optimize simba --dry-run     # Preview
rlm-dspy optimize simba --batch-size 8  # Run with smaller batch
```

### Trace Management

```bash
# View collected traces
rlm-dspy traces list
rlm-dspy traces stats

# Show specific trace
rlm-dspy traces show <trace_id>

# Export/import traces
rlm-dspy traces export backup.json
rlm-dspy traces import backup.json

# Clear all traces
rlm-dspy traces clear
```

### Grounded Proposer (MIPROv2 Pattern)

The Grounded Proposer analyzes failure patterns to generate actionable tips:

```python
# Example generated tips (from actual failures):
# - "Use ripgrep to search for specific decorators to ensure accurate counts"
# - "Verify the existence of CLI commands in source before confirming"
# - "Execute read_file to inspect specific lines identified by search tools"
```

These tips are automatically injected into the RLM signature, helping the model avoid past mistakes.

### SIMBA Optimizer

SIMBA (Stochastic Introspective Mini-Batch Ascent) uses collected traces to optimize prompts:

```bash
# Check if you have enough traces
rlm-dspy optimize simba --dry-run

# Output:
# SIMBA Optimization
#   Traces found: 28
#   Qualifying (score >= 0.7): 28
#   Batch size: 16
#   Steps: 4
#   Candidates/step: 4
```

Requirements:
- At least `batch_size` traces (default: 16)
- Traces must have `grounded_score >= min_score` (default: 0.7)

## Index Compression

Reduce index disk usage with compression:

```bash
# Compress all project indexes (~2x compression ratio)
rlm-dspy index compress

# Compress specific project
rlm-dspy index compress .

# Decompress for debugging
rlm-dspy index compress . --decompress
```

**Compression strategies:**
- **Vector embeddings**: Float16 quantization + ZIP compression (~4x savings)
- **JSON metadata**: Compact format + Gzip
- **Large text files**: Gzip (files >10KB)

Indexes are automatically decompressed when loaded, so compression is transparent.

## Callback System

Hook into the RLM execution lifecycle for monitoring, logging, or custom behavior:

```python
from rlm_dspy.core.callbacks import (
    Callback, CallbackManager, MetricsCallback, 
    LoggingCallback, ProgressCallback, get_callback_manager
)

# Create custom callback
class MyCallback(Callback):
    def on_query_start(self, ctx):
        print(f"Starting query: {ctx.data.get('query', '')[:50]}")
    
    def on_iteration_end(self, ctx):
        print(f"Iteration complete in {ctx.elapsed:.2f}s")
    
    def on_error(self, ctx):
        print(f"Error: {ctx.data.get('error')}")

# Register callback
manager = get_callback_manager()
manager.add(MyCallback())
manager.add(MetricsCallback())  # Collect timing stats

# After queries, get metrics
metrics = manager.callbacks[1]  # MetricsCallback
print(metrics.get_summary())
```

### Built-in Callbacks

| Callback | Purpose |
|----------|---------|
| `LoggingCallback` | Logs all events to Python logger |
| `MetricsCallback` | Collects counts, timing, error stats |
| `ProgressCallback` | Reports progress percentage |

### Lifecycle Events

- `query.start` / `query.end`
- `iteration.start` / `iteration.end`
- `tool_call`
- `llm_call`
- `validation`
- `error`

## Background Services

### Index Daemon (Optional)

The Index Daemon provides automatic index updates when files change. It's **optional** - indexes work fine without it, just need manual rebuilds.

```bash
# Start daemon (runs in background)
rlm-dspy daemon start

# Check status
rlm-dspy daemon status

# Stop daemon
rlm-dspy daemon stop
```

See [Index Daemon](#index-daemon-auto-indexing) section for full details.

### Summary: What Needs Manual Start

| Service | Required? | When Needed |
|---------|-----------|-------------|
| **Deno** | ✅ Yes | Always - for sandboxed REPL |
| **Index Daemon** | ❌ No | Only if you want auto-indexing |
| **ripgrep** | ❌ No | Only for `ripgrep` tool (fallback: grep) |

## Documentation

- **[Provider Guide](docs/PROVIDERS.md)** - Supported LLM providers
- **[Testing](docs/TESTING.md)** - Test results and benchmarks
- **[Enhancement Plan](docs/ENHANCEMENT_PLAN.md)** - Development roadmap

## License

MIT

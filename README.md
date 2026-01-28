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
# Add to PATH
export PATH="$HOME/.deno/bin:$PATH"
```

### Install RLM-DSPy

```bash
pip install rlm-dspy
```

Or with uv:
```bash
uv pip install rlm-dspy
```

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

# Analyze from stdin
cat large_file.txt | rlm-dspy ask "Summarize this" --stdin

# Analyze a git diff
git diff | rlm-dspy diff "Are there any breaking changes?"

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
# Model to use (provider/model-name format)
model: openai/gpt-4o-mini

# Maximum cost in USD per query
max_budget: 1.0

# Maximum time in seconds per query  
max_timeout: 300

# Path to .env file with API keys (auto-loaded on startup)
env_file: ~/.env
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_API_KEY` | - | API key (or use provider-specific) |
| `RLM_MODEL` | `openai/gpt-4o-mini` | Model to use |
| `RLM_SUB_MODEL` | same as RLM_MODEL | Model for sub-LLM calls |
| `RLM_API_BASE` | - | Custom API endpoint (optional) |
| `RLM_MAX_ITERATIONS` | `20` | Max REPL iterations |
| `RLM_MAX_LLM_CALLS` | `50` | Max sub-LLM calls per query |
| `RLM_MAX_OUTPUT_CHARS` | `100000` | Max chars in REPL output |
| `RLM_MAX_BUDGET` | `1.0` | Maximum cost in USD |
| `RLM_MAX_TIMEOUT` | `300` | Maximum time in seconds |

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

## Python API Reference

```python
from rlm_dspy import RLM, RLMConfig, RLMResult

# Configuration
config = RLMConfig(
    model="openai/gpt-4o",
    sub_model="openai/gpt-4o-mini",
    max_iterations=20,
    max_llm_calls=50,
    max_output_chars=100_000,
    max_budget=1.0,
    max_timeout=300,
    verbose=False,
)

# Create RLM instance
rlm = RLM(config=config)

# Load context
context = rlm.load_context(["src/", "docs/"])

# Query
result = rlm.query("What does this do?", context)

# Result fields
result.answer          # str: The final answer
result.success         # bool: Whether query succeeded
result.trajectory      # list: REPL iteration history
result.iterations      # int: Number of iterations
result.elapsed_time    # float: Total time in seconds
result.error           # str | None: Error message if failed

# Add custom tools
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Results for: {query}"

rlm.add_tool("search_docs", search_docs)
```

## Documentation

- **[Provider Guide](docs/PROVIDERS.md)** - Supported LLM providers
- **[Testing](docs/TESTING.md)** - Test results and benchmarks

## License

MIT

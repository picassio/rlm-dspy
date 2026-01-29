# PR: Major Refactor - DSPy RLM Integration & Enhancement

## Summary

This PR represents a complete refactoring of rlm-dspy to use DSPy's native `dspy.RLM` module for REPL-based exploration. The changes include extensive new features, comprehensive test coverage, security hardening, and performance optimizations.

**Branch:** `refactor/use-dspy-rlm`  
**Commits:** 95  
**Lines Changed:** +16,781 / -6,540  

## Key Changes

### 1. Core Architecture Rewrite

- **DSPy RLM Integration**: Rewrote `src/rlm_dspy/core/rlm.py` to wrap `dspy.RLM` for sandboxed Python REPL execution
- **Centralized Configuration**: CLI args > env vars > `~/.rlm/config.yaml` > defaults
- **Hybrid Tool Architecture**:
  - **Fast Path (Tree-sitter)**: AST-based tools for 15 languages - instant, 100% accurate
  - **Precise Path (LSP)**: IDE-quality tools via language servers - 60+ languages
  - **Semantic Path (Embeddings)**: FAISS-powered conceptual search

### 2. New Features

| Feature | Description |
|---------|-------------|
| **Semantic Search** | Embedding-based code search with FAISS (<1ms for 1000s of vectors) |
| **Citations** | Source-backed answers with file:line references |
| **Index Daemon** | Background file watcher with debounced auto-indexing |
| **Multi-Project** | Named projects with registry, tags, cross-project search |
| **Incremental Indexing** | Only re-embed changed files |
| **LSP Integration** | `find_references()`, `go_to_definition()`, `get_type_info()`, `get_symbol_hierarchy()` |
| **15 Languages** | Python, JS, TS, Go, Rust, Java, C, C++, Ruby, C#, Kotlin, Scala, PHP, Lua, Bash, Haskell |
| **Batch Processing** | Parallel query execution with configurable workers |
| **Custom Signatures** | Define custom DSPy signatures for specialized analysis |

### 3. Test Coverage

| Before | After | Increase |
|--------|-------|----------|
| ~50 tests | **478 tests** | +856% |

**New Test Files:**
- `test_ast_index.py` (37 tests) - Tree-sitter parsing
- `test_lsp.py` (22 tests) - LSP integration
- `test_fileutils.py` (57 tests) - File operations
- `test_validation.py` (26 tests) - Input validation
- `test_secrets.py` (25 tests) - Secret handling
- `test_retry.py` (37 tests) - Retry logic with backoff
- `test_debug.py` (36 tests) - Debug utilities
- `test_user_config.py` (26 tests) - Config management
- `test_rlm.py` (22 tests) - Core RLM functionality
- `test_vector_index.py` (13 tests) - Semantic search
- `test_daemon.py` (12 tests) - Index daemon
- `test_citations.py` (28 tests) - Citation support
- `test_embeddings.py` (15 tests) - Embedding manager
- `test_project_registry.py` (20 tests) - Multi-project
- `test_tools.py` (35 tests) - Built-in tools
- `test_guards.py` (15 tests) - Input guards
- `test_token_stats.py` (16 tests) - Token tracking

### 4. Security Hardening

- **Path Traversal Protection**: `_is_safe_path()` with symlink resolution
- **Shell Command Safety**: Allowlist + dangerous pattern blocking + `shell=False`
- **Secret Sanitization**: Pattern-based redaction in trajectory and outputs
- **Input Validation**: Bounds checking on all config values
- **No eval/exec**: Sandboxed execution via DSPy RLM

### 5. Performance Optimizations

- **Regex Caching**: Pre-compiled patterns for secret detection
- **Hash Indexes**: O(1) lookup for AST definitions
- **FAISS**: Sub-millisecond vector search
- **Incremental Updates**: Only re-embed changed files
- **Lazy Initialization**: Language servers started on-demand
- **Context Caching**: Avoid re-reading unchanged files

### 6. Bug Fixes (54+ fixes across 4 review rounds)

- Thread safety with RLock on ProjectRegistry
- Atomic file writes with fsync
- Cross-process locking with fcntl.flock
- File descriptor leak prevention
- Embedding chunk size limits (18K chars max)
- Skip minified files (.min.js)
- Path resolution relative to project root

## New Files

```
src/rlm_dspy/core/
├── lsp.py              # LSP integration via solidlsp
├── vector_index.py     # FAISS-powered semantic search
├── embeddings.py       # Embedding manager (litellm/local)
├── citations.py        # Citation utilities
├── daemon.py           # Index file watcher
├── project_registry.py # Multi-project management
├── ast_index.py        # Tree-sitter AST indexing
├── treesitter.py       # Language parsers
├── syntax_chunker.py   # Semantic code chunking
├── fileutils.py        # File operations
├── validation.py       # Input validation
├── secrets.py          # Secret masking
├── retry.py            # Retry with backoff
├── debug.py            # Debug utilities
└── user_config.py      # Config management
```

## Configuration

```yaml
# ~/.rlm/config.yaml
model: openrouter/google/gemini-3-flash-preview
embedding_model: openrouter/openai/text-embedding-3-small

# Execution limits (with bounds)
max_iterations: 30      # 1-100
max_llm_calls: 100      # 1-500
max_timeout: 600        # 0-3600
max_budget: 2.0         # 0-100
max_workers: 8          # 1-32

# Index settings
index_dir: ~/.rlm/indexes
use_faiss: true
auto_update_index: true
```

## CLI Examples

```bash
# Basic query
rlm-dspy ask "explain the architecture" src/

# Semantic search
rlm-dspy index search "authentication" -k 10

# With citations
rlm-dspy ask "audit security" src/ -S cited-security -j

# Multi-project
rlm-dspy project add myapp ./src --tags python,web
rlm-dspy project list
rlm-dspy index search "error handling" --all-projects

# Daemon
rlm-dspy daemon start
rlm-dspy daemon watch ./src
```

## Breaking Changes

1. **Config Location**: Now uses `~/.rlm/config.yaml` instead of inline config
2. **Tool Names**: Some tool names changed for consistency
3. **Return Types**: `RLMResult` dataclass instead of dict

## Dependencies Added

```toml
[project.dependencies]
dspy>=2.7.0
tree-sitter>=0.20.0
tree-sitter-python>=0.20.0
# ... 14 more tree-sitter languages
faiss-cpu>=1.7.0
watchdog>=3.0.0
pathspec>=0.11.0

[project.optional-dependencies]
lsp = ["serena-agent"]  # For LSP tools
```

## Migration Guide

1. Create `~/.rlm/config.yaml` with your model settings
2. Set API keys in environment or `~/.claude/.env`
3. Update any custom tool integrations to use new function signatures
4. Run `rlm-dspy index build src/` to create initial semantic index

## Test Results

```
================== 478 passed, 2 skipped in 83.60s ===================
```

The 2 skipped tests are for features requiring external services (LSP integration tests when solidlsp not installed).

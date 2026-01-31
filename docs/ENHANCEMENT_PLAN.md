# RLM-DSPy Enhancement Plan

## Overview

This document outlines the plan to add advanced features to rlm-dspy:
1. Citations Support - Source-backed answers
2. Semantic Code Search - Embedding-based retrieval
3. MCP Tool Integration - External tool support
4. SIMBA - Self-improving optimization

All features will use centralized configuration via `~/.rlm/config.yaml` and `.env`.

---

## 1. Configuration Updates

### Updated `~/.rlm/config.yaml`

```yaml
# RLM-DSPy Configuration
# Priority: CLI args > env vars > this file > defaults

# ============================================================================
# Model Settings
# ============================================================================
model: openrouter/google/gemini-2.0-flash-001
sub_model: openrouter/google/gemini-2.0-flash-001

# ============================================================================
# Embedding Settings (for semantic search)
# ============================================================================
# Embedding model (via litellm)
# Options: openai/text-embedding-3-small, openai/text-embedding-3-large,
#          cohere/embed-english-v3.0, voyage/voyage-3, 
#          together_ai/togethercomputer/m2-bert-80M-8k-retrieval
#          or "local" to use sentence-transformers
embedding_model: openai/text-embedding-3-small

# Local embedding model (if embedding_model: local)
# Requires: pip install sentence-transformers
local_embedding_model: sentence-transformers/all-MiniLM-L6-v2

# Embedding batch size
embedding_batch_size: 100

# ============================================================================
# Vector Index Settings (for semantic search)
# ============================================================================
# Index storage directory
index_dir: ~/.rlm/indexes

# Use FAISS for large indexes (requires: pip install faiss-cpu)
# If false, uses brute-force numpy search (slower but no dependencies)
use_faiss: true

# Threshold for switching to FAISS (number of documents)
faiss_threshold: 5000

# Auto-update index when files change
auto_update_index: true

# Index cache TTL in seconds (0 = no expiry)
index_cache_ttl: 3600

# ============================================================================
# Execution Limits
# ============================================================================
max_iterations: 30
max_llm_calls: 100
max_output_chars: 100000
max_workers: 8

# ============================================================================
# Budget/Safety Limits
# ============================================================================
max_budget: 2.0
max_timeout: 600

# ============================================================================
# MCP Settings (Model Context Protocol)
# ============================================================================
# List of MCP servers to auto-connect
# mcp_servers:
#   - name: filesystem
#     command: npx @modelcontextprotocol/server-filesystem /path/to/allowed/dir
#   - name: github
#     command: npx @modelcontextprotocol/server-github

# ============================================================================
# API Key Location
# ============================================================================
env_file: ~/.env
```

### Updated `.env.example`

```bash
# ============================================================================
# API Keys for LLM Providers
# ============================================================================
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...
DEEPSEEK_API_KEY=sk-...

# ============================================================================
# Embedding API Keys (may be same as LLM keys)
# ============================================================================
# OpenAI embeddings use OPENAI_API_KEY
# Cohere embeddings:
# COHERE_API_KEY=...
# Voyage embeddings:
# VOYAGE_API_KEY=...

# ============================================================================
# RLM Settings (override config.yaml)
# ============================================================================
RLM_MODEL=openrouter/google/gemini-2.0-flash-001
RLM_EMBEDDING_MODEL=openai/text-embedding-3-small
RLM_INDEX_DIR=~/.rlm/indexes
RLM_MAX_ITERATIONS=30
```

---

## 2. Implementation Details

### 2.1 Embedding Manager

```python
# src/rlm_dspy/core/embeddings.py

from dataclasses import dataclass
from pathlib import Path
import dspy

@dataclass 
class EmbeddingConfig:
    """Embedding configuration."""
    model: str = "openai/text-embedding-3-small"
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 100
    api_key: str | None = None
    
    @classmethod
    def from_user_config(cls) -> "EmbeddingConfig":
        """Load from user config."""
        from .user_config import load_config, load_env_file
        load_env_file()
        config = load_config()
        return cls(
            model=os.environ.get("RLM_EMBEDDING_MODEL", config.get("embedding_model", "openai/text-embedding-3-small")),
            local_model=config.get("local_embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=config.get("embedding_batch_size", 100),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )


def get_embedder(config: EmbeddingConfig | None = None) -> dspy.Embedder:
    """Get configured embedder instance."""
    config = config or EmbeddingConfig.from_user_config()
    
    if config.model == "local":
        # Use local sentence-transformers model
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.local_model)
            return dspy.Embedder(model.encode, batch_size=config.batch_size)
        except ImportError:
            raise ImportError("pip install sentence-transformers for local embeddings")
    else:
        # Use hosted model via litellm
        kwargs = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return dspy.Embedder(config.model, batch_size=config.batch_size, **kwargs)
```

### 2.2 Vector Index Manager

```python
# src/rlm_dspy/core/vector_index.py

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dspy.retrievers import Embeddings

@dataclass
class IndexConfig:
    """Vector index configuration."""
    index_dir: Path = Path.home() / ".rlm" / "indexes"
    use_faiss: bool = True
    faiss_threshold: int = 5000
    auto_update: bool = True
    cache_ttl: int = 3600
    
    @classmethod
    def from_user_config(cls) -> "IndexConfig":
        from .user_config import load_config
        config = load_config()
        return cls(
            index_dir=Path(os.environ.get("RLM_INDEX_DIR", config.get("index_dir", "~/.rlm/indexes"))).expanduser(),
            use_faiss=config.get("use_faiss", True),
            faiss_threshold=config.get("faiss_threshold", 5000),
            auto_update=config.get("auto_update_index", True),
            cache_ttl=config.get("index_cache_ttl", 3600),
        )


class CodeIndex:
    """Manages vector indexes for code repositories."""
    
    def __init__(self, config: IndexConfig | None = None):
        self.config = config or IndexConfig.from_user_config()
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = None
        self._indexes: dict[str, Embeddings] = {}
    
    @property
    def embedder(self):
        if self._embedder is None:
            from .embeddings import get_embedder
            self._embedder = get_embedder()
        return self._embedder
    
    def _get_index_path(self, repo_path: Path) -> Path:
        """Get index storage path for a repository."""
        # Use hash of absolute path as index name
        path_hash = hashlib.md5(str(repo_path.resolve()).encode()).hexdigest()[:12]
        return self.config.index_dir / path_hash
    
    def _get_manifest_path(self, index_path: Path) -> Path:
        return index_path / "manifest.json"
    
    def _load_manifest(self, index_path: Path) -> dict:
        manifest_path = self._get_manifest_path(index_path)
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())
        return {"files": {}, "created": 0, "updated": 0}
    
    def _save_manifest(self, index_path: Path, manifest: dict):
        manifest_path = self._get_manifest_path(index_path)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    
    def _extract_snippets(self, repo_path: Path) -> list[tuple[str, str, dict]]:
        """Extract code snippets from repository.
        
        Returns list of (id, text, metadata) tuples.
        """
        from .ast_index import index_directory
        
        snippets = []
        
        # Get all indexed items (functions, classes, etc.)
        for file_path in repo_path.rglob("*.py"):
            try:
                items = index_directory(str(file_path))
                for item in items:
                    snippet_id = f"{file_path}:{item['name']}:{item['line']}"
                    # Include context around the definition
                    text = f"File: {file_path}\nType: {item['type']}\nName: {item['name']}\n\n{item.get('text', '')}"
                    metadata = {
                        "file": str(file_path),
                        "line": item["line"],
                        "type": item["type"],
                        "name": item["name"],
                    }
                    snippets.append((snippet_id, text, metadata))
            except Exception as e:
                continue
        
        return snippets
    
    def _needs_update(self, repo_path: Path, manifest: dict) -> tuple[bool, list[Path], list[str]]:
        """Check if index needs updating.
        
        Returns (needs_update, new_files, deleted_files).
        """
        current_files = {}
        for f in repo_path.rglob("*.py"):
            current_files[str(f)] = f.stat().st_mtime
        
        old_files = manifest.get("files", {})
        
        new_or_modified = []
        for path, mtime in current_files.items():
            if path not in old_files or old_files[path] < mtime:
                new_or_modified.append(Path(path))
        
        deleted = [p for p in old_files if p not in current_files]
        
        return bool(new_or_modified or deleted), new_or_modified, deleted
    
    def index(self, repo_path: Path, force: bool = False) -> Embeddings:
        """Index a repository, with incremental updates."""
        repo_path = Path(repo_path).resolve()
        index_path = self._get_index_path(repo_path)
        
        # Check if we have a cached index
        cache_key = str(repo_path)
        if cache_key in self._indexes and not force:
            return self._indexes[cache_key]
        
        # Check if saved index exists
        if index_path.exists() and not force:
            manifest = self._load_manifest(index_path)
            
            # Check if update needed
            if self.config.auto_update:
                needs_update, new_files, deleted = self._needs_update(repo_path, manifest)
                if not needs_update:
                    # Load existing index
                    index = Embeddings.from_saved(str(index_path), self.embedder)
                    self._indexes[cache_key] = index
                    return index
                # TODO: Implement incremental update
                # For now, rebuild entire index
        
        # Build new index
        snippets = self._extract_snippets(repo_path)
        if not snippets:
            raise ValueError(f"No code found in {repo_path}")
        
        corpus = [text for _, text, _ in snippets]
        metadata = {id_: meta for id_, _, meta in snippets}
        
        # Create index
        index = Embeddings(
            corpus=corpus,
            embedder=self.embedder,
            k=10,
            brute_force_threshold=self.config.faiss_threshold if self.config.use_faiss else float('inf'),
        )
        
        # Save index
        index_path.mkdir(parents=True, exist_ok=True)
        index.save(str(index_path))
        
        # Save manifest
        manifest = {
            "files": {str(f): f.stat().st_mtime for f in repo_path.rglob("*.py")},
            "created": time.time(),
            "updated": time.time(),
            "snippet_count": len(snippets),
        }
        self._save_manifest(index_path, manifest)
        
        # Save metadata separately
        (index_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        
        self._indexes[cache_key] = index
        return index
    
    def search(self, repo_path: Path, query: str, k: int = 5) -> list[dict]:
        """Search for relevant code snippets."""
        index = self.index(repo_path)
        result = index(query)
        
        # Load metadata
        index_path = self._get_index_path(repo_path)
        metadata = json.loads((index_path / "metadata.json").read_text())
        
        results = []
        for passage, idx in zip(result.passages, result.indices):
            # Find metadata for this passage
            meta = list(metadata.values())[idx] if idx < len(metadata) else {}
            results.append({
                "text": passage,
                "file": meta.get("file"),
                "line": meta.get("line"),
                "type": meta.get("type"),
                "name": meta.get("name"),
            })
        
        return results[:k]


# Global index manager
_index_manager: CodeIndex | None = None

def get_index_manager() -> CodeIndex:
    global _index_manager
    if _index_manager is None:
        _index_manager = CodeIndex()
    return _index_manager
```

### 2.3 Semantic Search Tool

```python
# Add to src/rlm_dspy/tools.py

def semantic_search(query: str, path: str = ".", k: int = 5) -> str:
    """Search code semantically using embeddings.
    
    This finds code that is conceptually similar to your query,
    even if it doesn't contain the exact words.
    
    Args:
        query: Natural language description of what you're looking for
        path: Directory to search in
        k: Number of results to return
        
    Returns:
        Relevant code snippets with file locations
    """
    from .core.vector_index import get_index_manager
    
    try:
        manager = get_index_manager()
        results = manager.search(Path(path), query, k=k)
        
        output = []
        for r in results:
            output.append(f"=== {r['file']}:{r['line']} ({r['type']} {r['name']}) ===")
            output.append(r['text'][:500])
            output.append("")
        
        return "\n".join(output) if output else "No relevant code found."
    except Exception as e:
        return f"Semantic search error: {e}"
```

### 2.4 Citations Support

```python
# src/rlm_dspy/signatures.py - Add new signature

from dspy.adapters.types import Document, Citations

@register_signature("cited", aliases=["with-citations", "sourced"])
class CitedAnalysis(dspy.Signature):
    """Analyze code and provide citations to specific locations.
    
    You MUST cite the exact source for every claim using the citations field.
    """
    documents: list[Document] = dspy.InputField(desc="Source code documents")
    query: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Detailed answer with inline references")
    citations: Citations = dspy.OutputField(desc="List of citations to source documents")


def citations_to_locations(citations: Citations, documents: list[Document]) -> list[dict]:
    """Convert citations to file:line locations."""
    locations = []
    for citation in citations.citations:
        doc = documents[citation.document_index]
        # Calculate line number from char offset
        text_before = doc.data[:citation.start_char_index]
        line_number = text_before.count('\n') + 1
        locations.append({
            "file": doc.title,
            "line": line_number,
            "text": citation.cited_text,
        })
    return locations
```

### 2.5 Updated RLMConfig

```python
# Update src/rlm_dspy/core/rlm.py

@dataclass
class RLMConfig:
    # ... existing fields ...
    
    # Embedding settings
    embedding_model: str = field(
        default_factory=lambda: _env(
            "RLM_EMBEDDING_MODEL", 
            _get_user_config_default("embedding_model", "openai/text-embedding-3-small")
        )
    )
    local_embedding_model: str = field(
        default_factory=lambda: _get_user_config_default(
            "local_embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_batch_size: int = field(
        default_factory=lambda: _env_get(
            "RLM_EMBEDDING_BATCH_SIZE",
            _get_user_config_default("embedding_batch_size", 100)
        )
    )
    
    # Vector index settings
    index_dir: Path = field(
        default_factory=lambda: Path(_env(
            "RLM_INDEX_DIR",
            _get_user_config_default("index_dir", "~/.rlm/indexes")
        )).expanduser()
    )
    use_faiss: bool = field(
        default_factory=lambda: _get_user_config_default("use_faiss", True)
    )
    auto_update_index: bool = field(
        default_factory=lambda: _get_user_config_default("auto_update_index", True)
    )
```

---

## 3. CLI Updates

```bash
# New CLI options

# Semantic search
rlm-dspy ask "Find authentication logic" src/ --semantic

# With citations
rlm-dspy ask "Explain the caching" src/ --signature cited --format json

# Index management
rlm-dspy index build src/          # Build index
rlm-dspy index update src/         # Update index
rlm-dspy index status src/         # Show index status
rlm-dspy index clear               # Clear all indexes

# Embedding model override
rlm-dspy ask "query" src/ --embedding-model openai/text-embedding-3-large
```

---

## 4. Implementation Phases

### Phase 1: Configuration & Embeddings ✅ COMPLETE
- [x] Update user_config.py with new settings
- [x] Update config template with all options
- [x] Create embeddings.py with EmbeddingConfig
- [x] Create get_embedder() function
- [x] Add tests for embedding config (15 tests)

### Phase 2: Vector Index ✅ COMPLETE
- [x] Create vector_index.py with CodeIndex class
- [x] Implement incremental updates
- [x] Add manifest tracking for file changes
- [x] Add semantic_search tool
- [x] Add index CLI commands (build, status, clear, search)
- [x] Add tests for indexing (13 tests)

### Phase 3: Citations ✅ COMPLETE
- [x] Add CitedAnalysis signature
- [x] Implement citations_to_locations()
- [x] Add Document packaging for code files
- [x] Add --signature cited support
- [x] Add tests for citations (28 tests)

### Phase 4: Integration ✅ COMPLETE
- [x] Integrate semantic search into default tools
- [x] Add auto-indexing on first use
- [x] Update documentation (README.md)
- [x] End-to-end testing

---

## 5. Dependencies

```toml
# Add to pyproject.toml

[project.optional-dependencies]
embeddings = [
    "sentence-transformers>=2.0.0",  # For local embeddings
    "faiss-cpu>=1.7.0",              # For fast vector search
]
```

---

## 5. Future Work

### Phase 5: Index Daemon Service ✅ COMPLETE

**Goal:** Background service that automatically keeps indexes up-to-date.

**Implementation:**
- [x] DaemonConfig dataclass with watch/ignore patterns
- [x] IndexEventHandler for file system events
- [x] IndexWorker with debounced processing
- [x] IndexDaemon class with start/stop/watch/unwatch
- [x] CLI commands: start, stop, status, watch, unwatch, list
- [x] Auto-watch projects with auto_watch=True flag
- [x] Daemonize support (Unix fork to background)
- [x] 12 tests for daemon functionality

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RLM Index Daemon                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐    │
│  │   Watcher    │────▶│    Queue     │────▶│  Index Worker    │    │
│  │  (inotify/   │     │  (debounced  │     │  (background     │    │
│  │   fsevents)  │     │   changes)   │     │   embedding)     │    │
│  └──────────────┘     └──────────────┘     └──────────────────┘    │
│         │                                           │               │
│         ▼                                           ▼               │
│  ┌──────────────┐                          ┌──────────────────┐    │
│  │  Project     │                          │  ~/.rlm/indexes/ │    │
│  │  Registry    │                          │  (persistent)    │    │
│  │  (watched    │                          └──────────────────┘    │
│  │   paths)     │                                                   │
│  └──────────────┘                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### CLI Commands

```bash
# Start daemon
rlm-dspy daemon start
rlm-dspy daemon start --foreground  # Run in foreground

# Stop daemon
rlm-dspy daemon stop

# Check status
rlm-dspy daemon status

# Register project for auto-indexing
rlm-dspy daemon watch ~/projects/my-app
rlm-dspy daemon watch .  # Current directory

# Unregister project
rlm-dspy daemon unwatch ~/projects/my-app

# List watched projects
rlm-dspy daemon list
```

#### Configuration

```yaml
# ~/.rlm/config.yaml

# Daemon settings
daemon:
  enabled: true
  pid_file: ~/.rlm/daemon.pid
  log_file: ~/.rlm/daemon.log
  
  # File watching
  watch_debounce: 5          # Seconds to wait after file change
  watch_ignore:              # Patterns to ignore
    - "*.pyc"
    - "__pycache__"
    - ".git"
    - "node_modules"
    - ".venv"
  
  # Resource limits
  max_concurrent_indexes: 2  # Max parallel index builds
  idle_timeout: 3600         # Stop after 1 hour of no activity (0 = never)
  
  # Startup
  auto_start: false          # Start daemon on first `rlm-dspy` command
```

#### Implementation Notes

1. **File Watcher**: Use `watchdog` library (cross-platform)
2. **Debouncing**: Wait 5s after last change before re-indexing
3. **Incremental**: Only re-embed changed files
4. **Resource-aware**: Limit concurrent index builds
5. **Persistence**: Store watched projects in `~/.rlm/projects.json`

---

### Phase 6: Multi-Project Index Management

**Problem:** How to handle multiple projects efficiently?

#### Current Behavior

```
~/.rlm/indexes/
├── a1b2c3d4e5f6/     # Hash of /home/user/project-a
│   ├── index.npy
│   ├── metadata.json
│   ├── corpus_idx_map.json
│   └── manifest.json
├── 7890abcdef12/     # Hash of /home/user/project-b
│   └── ...
└── ...
```

**Issues:**
1. Hard to know which hash belongs to which project
2. No cross-project search
3. Orphaned indexes when projects are deleted/moved
4. No project naming/aliasing

#### Solution: Project Registry

```
~/.rlm/
├── config.yaml
├── projects.json          # NEW: Project registry
└── indexes/
    ├── project-a/         # Named instead of hash
    │   └── ...
    └── project-b/
        └── ...
```

**projects.json:**
```json
{
  "projects": {
    "project-a": {
      "path": "/home/user/projects/project-a",
      "alias": "my-app",
      "indexed_at": "2025-01-28T12:00:00Z",
      "snippet_count": 450,
      "auto_watch": true,
      "tags": ["python", "web"]
    },
    "project-b": {
      "path": "/home/user/projects/project-b",
      "alias": null,
      "indexed_at": "2025-01-27T10:00:00Z",
      "snippet_count": 120,
      "auto_watch": false,
      "tags": ["rust", "cli"]
    }
  },
  "default_project": "project-a"
}
```

#### CLI Commands

```bash
# Register a project with a name
rlm-dspy project add my-app ~/projects/my-app
rlm-dspy project add . --name current-project

# List all projects
rlm-dspy project list
# Output:
# NAME          PATH                         SNIPPETS  UPDATED
# my-app        ~/projects/my-app            450       2 hours ago
# backend       ~/projects/backend           120       1 day ago
# * current     .                            89        just now

# Set default project (for search without -p)
rlm-dspy project default my-app

# Remove project (keeps files, removes from registry)
rlm-dspy project remove my-app

# Remove project and delete index
rlm-dspy project remove my-app --delete-index

# Search across multiple projects
rlm-dspy index search "auth" --projects my-app,backend
rlm-dspy index search "auth" --all-projects

# Tag projects for grouping
rlm-dspy project tag my-app python web
rlm-dspy index search "database" --tags python

# Cleanup orphaned indexes
rlm-dspy project cleanup
# Found 3 orphaned indexes (paths no longer exist)
# Delete? [y/N]
```

#### Cross-Project Search

```python
from rlm_dspy.core import CodeIndex, get_project_registry

# Search single project
index = CodeIndex()
results = index.search("my-app", "authentication", k=5)

# Search multiple projects
registry = get_project_registry()
results = registry.search_all(
    query="authentication",
    projects=["my-app", "backend"],  # or tags=["python"]
    k=10,
)

# Results include project info
for r in results:
    print(f"[{r.project}] {r.snippet.file}:{r.snippet.line}")
```

#### Migration Path

1. **Auto-migrate**: On first run, scan existing hash-based indexes
2. **Infer names**: Use directory name from manifest's `repo_path`
3. **Keep hashes**: Store hash in project metadata for deduplication

```python
# Migration logic
def migrate_legacy_indexes():
    for hash_dir in index_dir.iterdir():
        manifest = load_manifest(hash_dir)
        repo_path = manifest.get("repo_path")
        project_name = Path(repo_path).name  # e.g., "my-project"
        
        # Rename directory
        hash_dir.rename(index_dir / project_name)
        
        # Register in projects.json
        registry.add(project_name, repo_path)
```

#### Configuration

```yaml
# ~/.rlm/config.yaml

# Project settings
projects:
  default: null                    # Default project for search
  auto_register: true              # Auto-register on first index
  name_from_path: true             # Use directory name as project name
  max_projects: 50                 # Warn if more than this
  cleanup_orphaned_days: 30        # Auto-cleanup after 30 days
```

---

### Phase 7: LSP Integration ✅ COMPLETE

**Goal:** Add IDE-quality code intelligence via Language Server Protocol.

**Implementation:**
- [x] LSPManager class wrapping solidlsp
- [x] 60+ language support via language servers
- [x] 4 new tools: find_references, go_to_definition, get_type_info, get_symbol_hierarchy
- [x] Lazy language server startup
- [x] Graceful degradation if solidlsp not installed

---

### Phase 8: Language Expansion ✅ COMPLETE

**Goal:** Support more programming languages in Tree-sitter AST parsing.

**Implementation:**
- [x] Added 6 new languages: Kotlin, Scala, PHP, Lua, Bash, Haskell
- [x] Updated LANGUAGE_MAP with new extensions
- [x] Language-specific extraction rules (e.g., Haskell `function` nodes)
- [x] Now supports 15 languages total

---

### Phase 9: Bug Fixes & Reliability ✅ COMPLETE

**Goal:** Fix embedding API failures and path resolution issues.

**Implementation:**
- [x] Added snippet chunking for large code blocks (max 18K chars)
- [x] Skip minified files (.min.js, etc.)
- [x] Fixed path resolution in tools (relative to project, not CWD)
- [x] Added `_resolve_project_path()` helper

---

### Phase 10: Code Quality & Testing ✅ COMPLETE

**Goal:** Address gaps identified in code review.

#### High Priority ✅ COMPLETE
- [x] Add tests for `ast_index.py` (Tree-sitter parsing) - 37 tests
- [x] Add tests for `lsp.py` (LSP integration) - 22 tests

#### Medium Priority ✅ COMPLETE
- [x] Add tests for `fileutils.py` (File operations) - 57 tests
- [x] Add tests for `validation.py` (Input validation) - 26 tests
- [x] Add tests for `secrets.py` (Secret handling) - 25 tests
- [x] Sanitize tool outputs in trajectory before logging (already implemented via `_sanitize_trajectory`)

#### Low Priority ✅ COMPLETE
- [x] Add tests for `retry.py` (Retry logic) - 37 tests
- [x] Add tests for `debug.py` (Debug utilities) - 36 tests
- [x] Add tests for `user_config.py` (Config management) - 26 tests
- [x] Document `auto_update` behavior for semantic search (added to README)
- [x] Document shell allowlist security concerns (added security note to README)

#### Test Coverage Status

| Module | Has Tests | Priority |
|--------|-----------|----------|
| `ast_index.py` | ✅ 37 tests | High - DONE |
| `lsp.py` | ✅ 22 tests | High - DONE |
| `fileutils.py` | ✅ 57 tests | Medium - DONE |
| `validation.py` | ✅ 26 tests | Medium - DONE |
| `secrets.py` | ✅ 25 tests | Medium - DONE |
| `retry.py` | ✅ 37 tests | Low - DONE |
| `debug.py` | ✅ 36 tests | Low - DONE |
| `user_config.py` | ✅ 26 tests | Low - DONE |
| `rlm.py` | ✅ | - |
| `vector_index.py` | ✅ | - |
| `daemon.py` | ✅ | - |
| `citations.py` | ✅ | - |
| `embeddings.py` | ✅ | - |
| `project_registry.py` | ✅ | - |
| `token_stats.py` | ✅ | - |

**Total: 478 tests passing (up from 212)**

**All core modules now have dedicated test coverage!**

---

### Phase 11: Hallucination Prevention ✅ COMPLETE

**Goal:** Eliminate hallucinations in code analysis outputs.

**Implementation:**
- [x] Validation enabled by default (`--validate` on all queries)
- [x] Minimum 20 iterations enforced (prevents LLM from guessing)
- [x] Added verification rules to tool instructions
- [x] Cited signatures for file:line references
- [x] LLM-as-judge groundedness checking

**Results:**
| Mode | Grounded % |
|------|------------|
| Before (low iterations) | ~20-33% |
| With min 20 iterations + validation | **100%** |

---

### Phase 12: DSPy Optimizer Integration ✅ COMPLETE

**Goal:** Adopt DSPy's optimization patterns to auto-improve accuracy.

#### 12.1 Trace Bootstrapping (from BootstrapFewShot)

Save successful REPL traces and use as few-shot examples.

```python
# src/rlm_dspy/core/trace_collector.py

@dataclass
class REPLTrace:
    """A successful REPL execution trace."""
    query: str
    reasoning_steps: list[str]
    code_blocks: list[str]
    outputs: list[str]
    final_answer: str
    grounded_score: float
    timestamp: datetime


class TraceCollector:
    """Collects and stores successful REPL traces for bootstrapping."""
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path.home() / ".rlm" / "traces"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.traces: list[REPLTrace] = []
        self._load_traces()
    
    def on_success(self, query: str, trace: dict, answer: str, grounded_score: float):
        """Called when a query succeeds with high grounding score."""
        if grounded_score < 0.8:  # Only save high-quality traces
            return
        
        repl_trace = REPLTrace(
            query=query,
            reasoning_steps=trace.get("reasoning", []),
            code_blocks=trace.get("code", []),
            outputs=trace.get("outputs", []),
            final_answer=answer,
            grounded_score=grounded_score,
            timestamp=datetime.now(UTC),
        )
        self.traces.append(repl_trace)
        self._save_trace(repl_trace)
    
    def get_similar_traces(self, query: str, k: int = 3) -> list[REPLTrace]:
        """Find traces similar to the given query for few-shot prompting."""
        # Use embedding similarity to find relevant traces
        from .embeddings import get_embedder
        embedder = get_embedder()
        
        query_emb = embedder([query])[0]
        trace_embs = embedder([t.query for t in self.traces])
        
        # Cosine similarity
        similarities = np.dot(trace_embs, query_emb)
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.traces[i] for i in top_indices]
    
    def format_as_demos(self, traces: list[REPLTrace]) -> str:
        """Format traces as few-shot demonstrations."""
        demos = []
        for i, trace in enumerate(traces, 1):
            demo = f"=== Example {i} ===\n"
            demo += f"Query: {trace.query}\n\n"
            for j, (reasoning, code, output) in enumerate(
                zip(trace.reasoning_steps, trace.code_blocks, trace.outputs)
            ):
                demo += f"Step {j+1} Reasoning: {reasoning}\n"
                demo += f"```python\n{code}\n```\n"
                demo += f"Output: {output}\n\n"
            demo += f"Final Answer: {trace.final_answer}\n"
            demos.append(demo)
        return "\n\n".join(demos)
```

#### 12.2 Instruction Optimization (from COPRO)

Auto-tune tool instructions based on success/failure history.

```python
# src/rlm_dspy/core/instruction_optimizer.py

class InstructionOptimizer:
    """Optimizes tool instructions using COPRO-style iterative refinement."""
    
    def __init__(self, prompt_model: str = None):
        self.prompt_model = prompt_model
        self.history: list[dict] = []  # {instruction, score, failures}
        self.current_instructions = self._load_default_instructions()
    
    def record_outcome(self, instruction: str, success: bool, failure_reason: str = None):
        """Record the outcome of using an instruction."""
        self.history.append({
            "instruction": instruction,
            "success": success,
            "failure_reason": failure_reason,
            "timestamp": datetime.now(UTC).isoformat(),
        })
    
    def propose_improvement(self, tool_name: str) -> str:
        """Propose improved instruction based on failure history."""
        failures = [h for h in self.history if not h["success"]]
        
        if not failures:
            return self.current_instructions.get(tool_name)
        
        # Use LLM to propose improvement
        proposal_sig = dspy.Signature(
            "current_instruction, failure_examples -> improved_instruction"
        )
        
        proposer = dspy.Predict(proposal_sig)
        result = proposer(
            current_instruction=self.current_instructions.get(tool_name),
            failure_examples=json.dumps(failures[-10:], indent=2),
        )
        
        return result.improved_instruction
    
    def optimize(self, trainset: list[dict], metric: Callable, depth: int = 3):
        """Run COPRO-style optimization over tool instructions."""
        best_score = 0
        best_instructions = self.current_instructions.copy()
        
        for iteration in range(depth):
            # Generate candidates
            candidates = self._generate_candidates()
            
            # Evaluate each candidate
            for candidate in candidates:
                score = self._evaluate(candidate, trainset, metric)
                
                if score > best_score:
                    best_score = score
                    best_instructions = candidate
            
            # Update for next iteration
            self.current_instructions = best_instructions
        
        return best_instructions, best_score
```

#### 12.3 Grounded Proposal (from MIPROv2)

Generate data-aware prompt improvements.

```python
# src/rlm_dspy/core/grounded_proposer.py

class GroundedProposer:
    """Proposes prompt improvements grounded in actual data and failures."""
    
    def __init__(self, prompt_model: str = None):
        self.prompt_model = prompt_model
    
    def propose_tips(self, failures: list[dict], successes: list[dict]) -> list[str]:
        """Generate tips based on patterns in successes vs failures."""
        tip_sig = dspy.Signature("""
            failure_patterns, success_patterns -> tips
            
            Analyze what went wrong in failures vs what worked in successes.
            Generate concrete tips to add to the system prompt.
            
            Example tips:
            - "Always use read_file() to verify line numbers before claiming bugs"
            - "Check for existing error handling before reporting missing guards"
        """)
        
        proposer = dspy.ChainOfThought(tip_sig)
        result = proposer(
            failure_patterns=self._extract_patterns(failures),
            success_patterns=self._extract_patterns(successes),
        )
        
        return result.tips
    
    def augment_prompt(self, base_prompt: str, tips: list[str]) -> str:
        """Add tips to the base prompt."""
        tips_section = "\n".join(f"- {tip}" for tip in tips)
        return f"{base_prompt}\n\nIMPORTANT TIPS:\n{tips_section}"
```

#### 12.4 Configuration

```yaml
# ~/.rlm/config.yaml

# Optimization settings
optimization:
  # Trace collection
  collect_traces: true
  trace_storage: ~/.rlm/traces
  min_grounded_score: 0.8  # Only save high-quality traces
  max_traces: 1000
  
  # Few-shot bootstrapping
  use_bootstrapped_demos: true
  num_demos: 3  # Number of similar traces to include
  
  # Instruction optimization
  auto_optimize_instructions: false  # Manual trigger only
  optimization_depth: 3
  optimization_breadth: 5
  
  # Grounded proposal
  use_grounded_tips: true
  tip_refresh_interval: 100  # Re-generate tips every N queries
```

#### 12.5 CLI Commands

```bash
# View collected traces
rlm-dspy traces list
rlm-dspy traces show <trace-id>
rlm-dspy traces export traces.json

# Run instruction optimization
rlm-dspy optimize instructions --trainset examples.json --depth 3

# Generate grounded tips from history
rlm-dspy optimize tips --output tips.txt

# Apply optimized instructions
rlm-dspy config set optimization.use_bootstrapped_demos true
```

---

### Phase 13: Additional Enhancements ✅ COMPLETE

#### Completed
- [x] **json_repair Integration** - Robust JSON parsing with repair and extraction
- [x] **Index Compression** - Float16 quantization + gzip for ~4x compression
- [x] **Callback Middleware** - Extensibility via `@with_callbacks` pattern

#### Backlog (Lower Priority)
- [x] **SIMBA** - Self-improving optimization (integrated via CLI)

All planned enhancements are now complete!

---

### Optional: MCP Tool Integration

**Status:** Optional / Low Priority

**Goal:** External service support via Model Context Protocol.

MCP allows connecting to external tools (filesystem, GitHub, databases, etc.) via a standardized protocol. This is useful for extending rlm-dspy with external capabilities but is **not required** for core functionality.

**When to implement:**
- When users need to query external services (GitHub issues, Jira, etc.)
- When integrating with IDE extensions that use MCP
- When building multi-agent workflows

**Configuration (if implemented):**

```yaml
# ~/.rlm/config.yaml

# MCP Settings (optional)
mcp_servers:
  - name: filesystem
    command: npx @modelcontextprotocol/server-filesystem /path/to/allowed/dir
  - name: github
    command: npx @modelcontextprotocol/server-github
    env:
      GITHUB_TOKEN: ${GITHUB_TOKEN}
```

**Note:** rlm-dspy already has powerful built-in tools (ripgrep, tree-sitter, LSP, semantic search) that cover most code analysis needs without MCP.

---

## 6. Summary

| Feature | Config Key | Env Var | Default |
|---------|------------|---------|---------|
| Embedding Model | `embedding_model` | `RLM_EMBEDDING_MODEL` | `openai/text-embedding-3-small` |
| Local Model | `local_embedding_model` | - | `sentence-transformers/all-MiniLM-L6-v2` |
| Batch Size | `embedding_batch_size` | `RLM_EMBEDDING_BATCH_SIZE` | `100` |
| Index Dir | `index_dir` | `RLM_INDEX_DIR` | `~/.rlm/indexes` |
| Use FAISS | `use_faiss` | - | `true` |
| Auto Update | `auto_update_index` | - | `true` |
| Cache TTL | `index_cache_ttl` | - | `3600` |

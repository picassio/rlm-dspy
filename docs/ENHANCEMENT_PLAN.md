# RLM-DSPy Enhancement Plan

## Overview

This document outlines the plan to add advanced features to rlm-dspy:
1. Citations Support - Source-backed answers
2. Semantic Code Search - Embedding-based retrieval
3. MCP Tool Integration - External tool support
4. KNNFewShot - Dynamic example selection
5. SIMBA - Self-improving optimization

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
#     command: npx @anthropic/mcp-server-filesystem /path/to/allowed/dir
#   - name: github
#     command: npx @anthropic/mcp-server-github

# ============================================================================
# Optimization Settings
# ============================================================================
# Enable KNN few-shot (requires examples in ~/.rlm/examples/)
use_knn_fewshot: false
knn_k: 3

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
ANTHROPIC_API_KEY=sk-ant-...

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

### Phase 1: Configuration & Embeddings (2-3 days)
- [ ] Update user_config.py with new settings
- [ ] Update config template with all options
- [ ] Create embeddings.py with EmbeddingConfig
- [ ] Create get_embedder() function
- [ ] Add tests for embedding config

### Phase 2: Vector Index (3-4 days)
- [ ] Create vector_index.py with CodeIndex class
- [ ] Implement incremental updates
- [ ] Add manifest tracking for file changes
- [ ] Add semantic_search tool
- [ ] Add index CLI commands
- [ ] Add tests for indexing

### Phase 3: Citations (2-3 days)
- [ ] Add CitedAnalysis signature
- [ ] Implement citations_to_locations()
- [ ] Add Document packaging for code files
- [ ] Add --signature cited support
- [ ] Add tests for citations

### Phase 4: Integration (2 days)
- [ ] Integrate semantic search into default tools
- [ ] Add auto-indexing on first use
- [ ] Update documentation
- [ ] End-to-end testing

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

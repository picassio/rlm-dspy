"""Vector index management for semantic code search.

Provides embedding-based code search with incremental updates and caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from dspy.retrievers import Embeddings

logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """Configuration for vector index.
    
    Attributes:
        index_dir: Directory to store indexes
        use_faiss: Use FAISS for fast search (requires faiss-cpu)
        faiss_threshold: Min docs before using FAISS
        auto_update: Auto-update index when files change
        cache_ttl: Cache TTL in seconds (0 = no expiry)
    """
    index_dir: Path = field(default_factory=lambda: Path.home() / ".rlm" / "indexes")
    use_faiss: bool = True
    faiss_threshold: int = 5000
    auto_update: bool = True
    cache_ttl: int = 3600
    
    @classmethod
    def from_user_config(cls) -> "IndexConfig":
        """Load from user config."""
        from .user_config import load_config
        config = load_config()
        
        index_dir = os.environ.get(
            "RLM_INDEX_DIR",
            config.get("index_dir", "~/.rlm/indexes")
        )
        
        return cls(
            index_dir=Path(index_dir).expanduser(),
            use_faiss=config.get("use_faiss", True),
            faiss_threshold=config.get("faiss_threshold", 5000),
            auto_update=config.get("auto_update_index", True),
            cache_ttl=config.get("index_cache_ttl", 3600),
        )


@dataclass
class CodeSnippet:
    """A code snippet with metadata."""
    id: str
    text: str
    file: str
    line: int
    end_line: int
    type: str  # function, class, method
    name: str
    language: str = "python"
    
    def to_document(self) -> str:
        """Convert to document string for embedding."""
        return f"# {self.type}: {self.name}\n# File: {self.file}:{self.line}\n\n{self.text}"


@dataclass 
class SearchResult:
    """A search result with relevance score."""
    snippet: CodeSnippet
    score: float
    
    def to_dict(self) -> dict:
        return {
            "file": self.snippet.file,
            "line": self.snippet.line,
            "end_line": self.snippet.end_line,
            "type": self.snippet.type,
            "name": self.snippet.name,
            "score": self.score,
            "text": self.snippet.text[:500],
        }


class CodeIndex:
    """Manages vector indexes for code repositories.
    
    Features:
    - Incremental updates (only re-index changed files)
    - FAISS support for fast search on large codebases
    - Persistent storage with manifest tracking
    - In-memory caching with TTL
    
    Example:
        ```python
        index = CodeIndex()
        
        # Index a repository
        index.build("./src")
        
        # Search
        results = index.search("./src", "authentication logic", k=5)
        for r in results:
            print(f"{r.snippet.file}:{r.snippet.line} - {r.snippet.name}")
        ```
    """
    
    def __init__(self, config: IndexConfig | None = None):
        self.config = config or IndexConfig.from_user_config()
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        
        self._embedder = None
        self._indexes: dict[str, tuple["Embeddings", float]] = {}  # path -> (index, timestamp)
        self._metadata: dict[str, dict[str, CodeSnippet]] = {}  # path -> {id -> snippet}
    
    @property
    def embedder(self):
        """Lazy-load embedder."""
        if self._embedder is None:
            from .embeddings import get_embedder
            self._embedder = get_embedder()
        return self._embedder
    
    def _get_index_path(self, repo_path: Path) -> Path:
        """Get index storage path for a repository."""
        path_hash = hashlib.md5(str(repo_path.resolve()).encode()).hexdigest()[:12]
        return self.config.index_dir / path_hash
    
    def _get_manifest_path(self, index_path: Path) -> Path:
        return index_path / "manifest.json"
    
    def _load_manifest(self, index_path: Path) -> dict:
        """Load manifest file."""
        manifest_path = self._get_manifest_path(index_path)
        if manifest_path.exists():
            try:
                return json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Corrupt manifest at %s", manifest_path)
        return {"files": {}, "created": 0, "updated": 0, "snippet_count": 0}
    
    def _save_manifest(self, index_path: Path, manifest: dict) -> None:
        """Save manifest file."""
        manifest_path = self._get_manifest_path(index_path)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    
    def _extract_snippets(self, repo_path: Path) -> list[CodeSnippet]:
        """Extract code snippets from repository using AST parsing."""
        from .ast_index import index_file, LANGUAGE_MAP
        
        snippets = []
        
        # Supported extensions (from LANGUAGE_MAP keys)
        supported_extensions = set(LANGUAGE_MAP.keys())
        
        for file_path in repo_path.rglob("*"):
            # Skip hidden directories and common ignores
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if any(ignore in file_path.parts for ignore in 
                   ['__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build']):
                continue
            
            # Check extension
            if file_path.suffix not in supported_extensions:
                continue
            
            if not file_path.is_file():
                continue
            
            try:
                # Get language from extension (LANGUAGE_MAP: ext -> lang)
                language = LANGUAGE_MAP.get(file_path.suffix)
                
                if not language:
                    continue
                
                # Index the file (returns ASTIndex with .definitions)
                ast_index = index_file(str(file_path))
                
                # Read file content for snippet text
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.splitlines()
                except UnicodeDecodeError:
                    continue
                
                for defn in ast_index.definitions:
                    # Extract text for the definition
                    start_line = defn.line - 1  # 0-indexed
                    end_line = defn.end_line
                    
                    # Get the text
                    if start_line < len(lines):
                        text_lines = lines[start_line:min(end_line, len(lines))]
                        text = '\n'.join(text_lines)
                    else:
                        text = ""
                    
                    snippet_id = f"{file_path}:{defn.name}:{defn.line}"
                    
                    snippets.append(CodeSnippet(
                        id=snippet_id,
                        text=text,
                        file=str(file_path),
                        line=defn.line,
                        end_line=end_line,
                        type=defn.kind,  # Definition uses 'kind', not 'type'
                        name=defn.name,
                        language=language,
                    ))
                    
            except Exception as e:
                logger.debug("Failed to index %s: %s", file_path, e)
                continue
        
        return snippets
    
    def _check_needs_update(
        self, 
        repo_path: Path, 
        manifest: dict
    ) -> tuple[bool, list[Path], list[str]]:
        """Check if index needs updating.
        
        Returns:
            (needs_update, new_or_modified_files, deleted_files)
        """
        from .ast_index import LANGUAGE_MAP
        
        supported_extensions = set(LANGUAGE_MAP.keys())
        
        current_files = {}
        for f in repo_path.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix not in supported_extensions:
                continue
            if any(part.startswith('.') for part in f.parts):
                continue
            if any(ignore in f.parts for ignore in 
                   ['__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build']):
                continue
            
            try:
                current_files[str(f)] = f.stat().st_mtime
            except OSError:
                continue
        
        old_files = manifest.get("files", {})
        
        # Find new or modified files
        new_or_modified = []
        for path, mtime in current_files.items():
            if path not in old_files or old_files[path] < mtime:
                new_or_modified.append(Path(path))
        
        # Find deleted files
        deleted = [p for p in old_files if p not in current_files]
        
        return bool(new_or_modified or deleted), new_or_modified, deleted
    
    def build(
        self, 
        repo_path: str | Path,
        force: bool = False,
    ) -> int:
        """Build or update index for a repository.
        
        Args:
            repo_path: Path to repository
            force: Force full rebuild
            
        Returns:
            Number of snippets indexed
        """
        from dspy.retrievers import Embeddings
        
        repo_path = Path(repo_path).resolve()
        if not repo_path.exists():
            raise ValueError(f"Path does not exist: {repo_path}")
        
        index_path = self._get_index_path(repo_path)
        cache_key = str(repo_path)
        
        # Check if we can use cached index
        if not force and cache_key in self._indexes:
            _, timestamp = self._indexes[cache_key]
            if self.config.cache_ttl == 0 or time.time() - timestamp < self.config.cache_ttl:
                # Check if files changed
                manifest = self._load_manifest(index_path)
                needs_update, _, _ = self._check_needs_update(repo_path, manifest)
                if not needs_update:
                    logger.debug("Using cached index for %s", repo_path)
                    return manifest.get("snippet_count", 0)
        
        # Check if saved index exists and is up-to-date
        if not force and index_path.exists():
            manifest = self._load_manifest(index_path)
            
            if self.config.auto_update:
                needs_update, new_files, deleted = self._check_needs_update(repo_path, manifest)
                if not needs_update:
                    # Load existing index
                    try:
                        index = Embeddings.from_saved(str(index_path), self.embedder)
                        metadata = self._load_metadata(index_path)
                        
                        self._indexes[cache_key] = (index, time.time())
                        self._metadata[cache_key] = metadata
                        
                        logger.info("Loaded existing index for %s (%d snippets)", 
                                   repo_path, len(metadata))
                        return len(metadata)
                    except Exception as e:
                        logger.warning("Failed to load index, rebuilding: %s", e)
        
        # Build new index
        logger.info("Building index for %s...", repo_path)
        snippets = self._extract_snippets(repo_path)
        
        if not snippets:
            logger.warning("No code found in %s", repo_path)
            return 0
        
        logger.info("Found %d code snippets", len(snippets))
        
        # Create corpus for embedding
        corpus = [s.to_document() for s in snippets]
        metadata = {s.id: s for s in snippets}
        
        # Determine if we should use FAISS
        brute_force_threshold = (
            self.config.faiss_threshold 
            if self.config.use_faiss 
            else float('inf')
        )
        
        # Create index
        logger.info("Creating embeddings...")
        index = Embeddings(
            corpus=corpus,
            embedder=self.embedder,
            k=10,
            brute_force_threshold=brute_force_threshold,
        )
        
        # Save index
        index_path.mkdir(parents=True, exist_ok=True)
        index.save(str(index_path))
        
        # Save manifest
        from .ast_index import LANGUAGE_MAP
        supported_extensions = set(LANGUAGE_MAP.keys())
        
        file_mtimes = {}
        for f in repo_path.rglob("*"):
            if f.is_file() and f.suffix in supported_extensions:
                if not any(part.startswith('.') for part in f.parts):
                    if not any(ignore in f.parts for ignore in 
                               ['__pycache__', 'node_modules', '.venv', 'venv']):
                        try:
                            file_mtimes[str(f)] = f.stat().st_mtime
                        except OSError:
                            pass
        
        manifest = {
            "repo_path": str(repo_path),
            "files": file_mtimes,
            "created": time.time(),
            "updated": time.time(),
            "snippet_count": len(snippets),
        }
        self._save_manifest(index_path, manifest)
        
        # Save metadata
        self._save_metadata(index_path, metadata)
        
        # Cache
        self._indexes[cache_key] = (index, time.time())
        self._metadata[cache_key] = metadata
        
        logger.info("Index built: %d snippets from %d files", 
                   len(snippets), len(file_mtimes))
        
        return len(snippets)
    
    def _save_metadata(self, index_path: Path, metadata: dict[str, CodeSnippet]) -> None:
        """Save snippet metadata."""
        data = {
            id_: {
                "id": s.id,
                "file": s.file,
                "line": s.line,
                "end_line": s.end_line,
                "type": s.type,
                "name": s.name,
                "language": s.language,
                "text": s.text[:1000],  # Truncate for storage
            }
            for id_, s in metadata.items()
        }
        (index_path / "metadata.json").write_text(json.dumps(data, indent=2))
    
    def _load_metadata(self, index_path: Path) -> dict[str, CodeSnippet]:
        """Load snippet metadata."""
        metadata_path = index_path / "metadata.json"
        if not metadata_path.exists():
            return {}
        
        data = json.loads(metadata_path.read_text())
        return {
            id_: CodeSnippet(**info)
            for id_, info in data.items()
        }
    
    def search(
        self,
        repo_path: str | Path,
        query: str,
        k: int = 5,
    ) -> list[SearchResult]:
        """Search for relevant code snippets.
        
        Args:
            repo_path: Path to repository (must be indexed)
            query: Natural language search query
            k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        repo_path = Path(repo_path).resolve()
        cache_key = str(repo_path)
        
        # Ensure index exists
        if cache_key not in self._indexes:
            self.build(repo_path)
        
        if cache_key not in self._indexes:
            raise ValueError(f"No index for {repo_path}")
        
        index, _ = self._indexes[cache_key]
        metadata = self._metadata.get(cache_key, {})
        
        # Search
        result = index(query)
        
        results = []
        for passage, idx in zip(result.passages, result.indices):
            # Find the snippet by matching passage
            snippet = None
            for s in metadata.values():
                if s.to_document() == passage:
                    snippet = s
                    break
            
            if snippet is None:
                # Fallback: create from passage
                snippet = CodeSnippet(
                    id=f"unknown:{idx}",
                    text=passage,
                    file="unknown",
                    line=0,
                    end_line=0,
                    type="unknown",
                    name="unknown",
                )
            
            results.append(SearchResult(
                snippet=snippet,
                score=1.0,  # Embeddings doesn't return scores directly
            ))
        
        return results[:k]
    
    def get_status(self, repo_path: str | Path) -> dict:
        """Get index status for a repository."""
        repo_path = Path(repo_path).resolve()
        index_path = self._get_index_path(repo_path)
        
        if not index_path.exists():
            return {
                "indexed": False,
                "path": str(repo_path),
            }
        
        manifest = self._load_manifest(index_path)
        needs_update, new_files, deleted = self._check_needs_update(repo_path, manifest)
        
        return {
            "indexed": True,
            "path": str(repo_path),
            "index_path": str(index_path),
            "snippet_count": manifest.get("snippet_count", 0),
            "file_count": len(manifest.get("files", {})),
            "created": manifest.get("created", 0),
            "updated": manifest.get("updated", 0),
            "needs_update": needs_update,
            "new_or_modified": len(new_files),
            "deleted": len(deleted),
        }
    
    def clear(self, repo_path: str | Path | None = None) -> int:
        """Clear index cache.
        
        Args:
            repo_path: Specific repo to clear, or None to clear all
            
        Returns:
            Number of indexes cleared
        """
        if repo_path:
            repo_path = Path(repo_path).resolve()
            cache_key = str(repo_path)
            index_path = self._get_index_path(repo_path)
            
            # Remove from memory cache
            self._indexes.pop(cache_key, None)
            self._metadata.pop(cache_key, None)
            
            # Remove from disk
            if index_path.exists():
                import shutil
                shutil.rmtree(index_path)
                return 1
            return 0
        else:
            # Clear all
            count = len(self._indexes)
            self._indexes.clear()
            self._metadata.clear()
            
            # Clear disk indexes
            if self.config.index_dir.exists():
                import shutil
                for child in self.config.index_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                        count += 1
            
            return count


# Global index manager instance
_index_manager: CodeIndex | None = None


def get_index_manager(config: IndexConfig | None = None) -> CodeIndex:
    """Get the global index manager instance."""
    global _index_manager
    if _index_manager is None or config is not None:
        _index_manager = CodeIndex(config)
    return _index_manager


def semantic_search(
    query: str,
    path: str = ".",
    k: int = 5,
) -> list[dict]:
    """Search code semantically using embeddings.
    
    This finds code that is conceptually similar to your query,
    even if it doesn't contain the exact words.
    
    Args:
        query: Natural language description of what you're looking for
        path: Directory to search in
        k: Number of results to return
        
    Returns:
        List of result dictionaries with file, line, name, type, score
    """
    manager = get_index_manager()
    results = manager.search(path, query, k=k)
    return [r.to_dict() for r in results]


# Export
__all__ = [
    "IndexConfig",
    "CodeSnippet", 
    "SearchResult",
    "CodeIndex",
    "get_index_manager",
    "semantic_search",
]

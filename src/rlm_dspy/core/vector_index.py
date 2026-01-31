"""Vector index management for semantic code search."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspy.retrievers import Embeddings

from .vector_types import IndexConfig, CodeSnippet, SearchResult

logger = logging.getLogger(__name__)


class CodeIndex:
    """Manages vector indexes for code repositories."""

    # Maximum number of indexes to keep in memory (LRU eviction)
    MAX_CACHED_INDEXES = 50
    
    def __init__(self, config: IndexConfig | None = None):
        self.config = config or IndexConfig.from_user_config()
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = None
        # Use OrderedDict for LRU behavior
        from collections import OrderedDict
        self._indexes: OrderedDict[str, tuple["Embeddings", float]] = OrderedDict()
        self._metadata: OrderedDict[str, dict[str, CodeSnippet]] = OrderedDict()
        self._corpus_idx_map: OrderedDict[str, dict[int, str]] = OrderedDict()
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds MAX_CACHED_INDEXES."""
        while len(self._indexes) > self.MAX_CACHED_INDEXES:
            oldest_key = next(iter(self._indexes))
            self._indexes.pop(oldest_key, None)
            self._metadata.pop(oldest_key, None)
            self._corpus_idx_map.pop(oldest_key, None)
            logger.debug("Evicted cached index for: %s", oldest_key)

    @property
    def embedder(self):
        """Lazy-load embedder."""
        if self._embedder is None:
            from .embeddings import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _get_index_path(self, repo_path: Path) -> Path:
        """Get index storage path for a repository."""
        from .project_registry import get_project_registry

        repo_path = repo_path.resolve()
        registry = get_project_registry()

        # Use O(1) path lookup instead of O(n) list iteration
        project = registry.get_by_path(repo_path)
        if project:
            return self.config.index_dir / project.name

        best_match = registry.find_best_match(repo_path)
        if best_match and best_match.path != str(repo_path):
            return self.config.index_dir / best_match.name

        project = registry.auto_register(repo_path)
        if project:
            return self.config.index_dir / project.name

        path_hash = hashlib.md5(str(repo_path).encode()).hexdigest()[:12]
        return self.config.index_dir / path_hash

    def _get_manifest_path(self, index_path: Path) -> Path:
        return index_path / "manifest.json"

    def _ensure_decompressed(self, index_path: Path) -> None:
        """Ensure index files are decompressed for DSPy compatibility."""
        from .index_compression import is_compressed, decompress_index
        if is_compressed(index_path):
            logger.debug("Decompressing index for loading: %s", index_path)
            decompress_index(index_path)

    # Maximum size for index metadata files (50MB)
    MAX_METADATA_SIZE = 50 * 1024 * 1024
    
    def _load_manifest(self, index_path: Path) -> dict:
        """Load manifest file with size limit."""
        manifest_path = self._get_manifest_path(index_path)
        if manifest_path.exists():
            try:
                # Check file size before loading
                if manifest_path.stat().st_size > self.MAX_METADATA_SIZE:
                    logger.warning("Manifest too large at %s, skipping", manifest_path)
                    return {"files": {}, "created": 0, "updated": 0, "snippet_count": 0}
                return json.loads(manifest_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                logger.warning("Corrupt manifest at %s", manifest_path)
        return {"files": {}, "created": 0, "updated": 0, "snippet_count": 0}

    def _save_manifest(self, index_path: Path, manifest: dict) -> None:
        """Save manifest file atomically."""
        manifest_path = self._get_manifest_path(index_path)
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=index_path, suffix='.tmp',
                                              delete=False, encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
                temp_path = Path(f.name)
            temp_path.replace(manifest_path)
        except OSError as e:
            logger.error("Failed to save manifest: %s", e)
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()

    def _save_metadata(self, index_path: Path, metadata: dict[str, CodeSnippet],
                       corpus_idx_to_id: dict[int, str] | None = None) -> None:
        """Save snippet metadata and corpus index mapping."""
        data = {id_: {"id": s.id, "file": s.file, "line": s.line, "end_line": s.end_line,
                      "type": s.type, "name": s.name, "language": s.language, "text": s.text[:1000]}
                for id_, s in metadata.items()}
        (index_path / "metadata.json").write_text(json.dumps(data, indent=2), encoding='utf-8')

        if corpus_idx_to_id is not None:
            idx_map = {str(k): v for k, v in corpus_idx_to_id.items()}
            (index_path / "corpus_idx_map.json").write_text(json.dumps(idx_map), encoding='utf-8')

        cache_key = str(index_path)
        self._metadata[cache_key] = metadata
        if corpus_idx_to_id is not None:
            self._corpus_idx_map[cache_key] = corpus_idx_to_id

    def _load_metadata(self, index_path: Path) -> tuple[dict[str, CodeSnippet], dict[int, str]]:
        """Load snippet metadata and corpus index mapping with size limits."""
        cache_key = str(index_path)
        if cache_key in self._metadata and cache_key in self._corpus_idx_map:
            return self._metadata[cache_key], self._corpus_idx_map[cache_key]

        metadata_path = index_path / "metadata.json"
        compressed_metadata = index_path / "metadata.json.gz"

        data = {}
        if compressed_metadata.exists():
            # Compressed files checked during decompression
            from .index_compression import load_json
            data = load_json(compressed_metadata)
        elif metadata_path.exists():
            # Check file size before loading
            if metadata_path.stat().st_size > self.MAX_METADATA_SIZE:
                logger.warning("Metadata file too large at %s, skipping", metadata_path)
            else:
                data = json.loads(metadata_path.read_text(encoding='utf-8'))

        metadata = {id_: CodeSnippet(**info) for id_, info in data.items()}

        idx_map_path = index_path / "corpus_idx_map.json"
        compressed_idx_map = index_path / "corpus_idx_map.json.gz"
        corpus_idx_to_id = {}

        if compressed_idx_map.exists():
            from .index_compression import load_json
            idx_data = load_json(compressed_idx_map)
            corpus_idx_to_id = {int(k): v for k, v in idx_data.items()}
        elif idx_map_path.exists():
            # Check file size before loading
            if idx_map_path.stat().st_size > self.MAX_METADATA_SIZE:
                logger.warning("Index map file too large at %s, skipping", idx_map_path)
            else:
                idx_data = json.loads(idx_map_path.read_text(encoding='utf-8'))
                corpus_idx_to_id = {int(k): v for k, v in idx_data.items()}

        self._metadata[cache_key] = metadata
        self._corpus_idx_map[cache_key] = corpus_idx_to_id
        return metadata, corpus_idx_to_id

    def build(self, repo_path: str | Path, force: bool = False) -> int:
        """Build or update index for a repository."""

        repo_path = Path(repo_path).resolve()
        if not repo_path.exists():
            raise ValueError(f"Path does not exist: {repo_path}")

        index_path = self._get_index_path(repo_path)
        lock_file = index_path.parent / f".{index_path.name}.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        lock_fd = None
        try:
            lock_fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            return self._build_locked(repo_path, index_path, force)
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)

    def _build_locked(self, repo_path: Path, index_path: Path, force: bool) -> int:
        """Build index (caller must hold lock)."""
        from dspy.retrievers import Embeddings
        from .vector_build import check_needs_update, build_full_index, incremental_update

        cache_key = str(repo_path)

        # Check cache
        if not force and cache_key in self._indexes:
            _, timestamp = self._indexes[cache_key]
            if self.config.cache_ttl == 0 or time.time() - timestamp < self.config.cache_ttl:
                manifest = self._load_manifest(index_path)
                needs_update, _, _ = check_needs_update(repo_path, manifest)
                if not needs_update:
                    return manifest.get("snippet_count", 0)

        # Check saved index
        if not force and index_path.exists():
            manifest = self._load_manifest(index_path)
            if self.config.auto_update:
                needs_update, new_files, deleted = check_needs_update(repo_path, manifest)
                if not needs_update:
                    try:
                        self._ensure_decompressed(index_path)
                        index = Embeddings.from_saved(str(index_path), self.embedder)
                        metadata, corpus_idx_to_id = self._load_metadata(index_path)
                        self._indexes[cache_key] = (index, time.time())
                        self._metadata[cache_key] = metadata
                        self._corpus_idx_map[cache_key] = corpus_idx_to_id
                        logger.info("Loaded existing index for %s (%d snippets)", repo_path, len(metadata))
                        return len(metadata)
                    except Exception as e:
                        logger.warning("Failed to load index, rebuilding: %s", e)
                elif new_files or deleted:
                    try:
                        old_metadata, old_corpus_idx_map = self._load_metadata(index_path)
                        result = incremental_update(
                            repo_path, index_path, self.embedder, self.config,
                            new_files, deleted, manifest, old_metadata, old_corpus_idx_map
                        )
                        if result:
                            count, new_manifest = result
                            self._save_manifest(index_path, new_manifest)
                            snippets = {s.id: s for s in self._extract_snippets_for_cache(index_path)}
                            corpus_idx_to_id = {i: id_ for i, id_ in enumerate(snippets.keys())}
                            self._save_metadata(index_path, snippets, corpus_idx_to_id)
                            index = Embeddings.from_saved(str(index_path), self.embedder)
                            self._indexes[cache_key] = (index, time.time())
                            self._update_registry_stats(repo_path, count, len(new_manifest.get("files", {})))
                            return count
                    except Exception as e:
                        logger.warning("Incremental update failed: %s", e)

        # Full rebuild
        count, manifest = build_full_index(repo_path, index_path, self.embedder, self.config)
        if count == 0:
            return 0

        self._save_manifest(index_path, manifest)

        # Reload and cache
        index = Embeddings.from_saved(str(index_path), self.embedder)
        metadata, corpus_idx_to_id = self._load_metadata(index_path)

        if not metadata:
            from .vector_build import extract_snippets
            snippets = extract_snippets(repo_path, self.config.max_snippet_chars)
            metadata = {s.id: s for s in snippets}
            corpus_idx_to_id = {i: s.id for i, s in enumerate(snippets)}
            self._save_metadata(index_path, metadata, corpus_idx_to_id)

        self._indexes[cache_key] = (index, time.time())
        self._metadata[cache_key] = metadata
        self._corpus_idx_map[cache_key] = corpus_idx_to_id
        self._evict_if_needed()
        self._update_registry_stats(repo_path, count, len(manifest.get("files", {})))

        return count

    def _extract_snippets_for_cache(self, index_path: Path) -> list[CodeSnippet]:
        """Load snippets from metadata for caching."""
        metadata, _ = self._load_metadata(index_path)
        return list(metadata.values())

    def _update_registry_stats(self, repo_path: Path, snippet_count: int, file_count: int) -> None:
        """Update project registry stats."""
        from .project_registry import get_project_registry
        registry = get_project_registry()
        # Use O(1) path lookup
        project = registry.get_by_path(repo_path)
        if project:
            registry.update_stats(project.name, snippet_count, file_count)

    def search(self, repo_path: str | Path, query: str, k: int = 5) -> list[SearchResult]:
        """Search for relevant code snippets."""
        import re

        repo_path = Path(repo_path).resolve()
        cache_key = str(repo_path)

        if cache_key not in self._indexes:
            self.build(repo_path)

        if cache_key not in self._indexes:
            raise ValueError(f"No index for {repo_path}")

        index, _ = self._indexes[cache_key]
        metadata = self._metadata.get(cache_key, {})
        corpus_idx_map = self._corpus_idx_map.get(cache_key, {})

        result = index(query)
        results = []

        for passage, idx in zip(result.passages, result.indices):
            snippet = None
            if idx in corpus_idx_map:
                snippet = metadata.get(corpus_idx_map[idx])

            if snippet is None:
                header_match = re.match(r'# (\w+): (.+)\n# File: (.+):(\d+)', passage)
                if header_match:
                    type_, name, file_, line = header_match.groups()
                    text_start = passage.find('\n\n')
                    text = passage[text_start+2:] if text_start > 0 else passage
                    snippet = CodeSnippet(
                        id=f"{file_}:{name}:{line}", text=text, file=file_,
                        line=int(line), end_line=int(line) + text.count('\n'),
                        type=type_, name=name,
                    )

            if snippet is None:
                snippet = CodeSnippet(id=f"unknown:{idx}", text=passage, file="unknown",
                                      line=0, end_line=0, type="unknown", name="unknown")

            results.append(SearchResult(snippet=snippet, score=1.0))

        return results[:k]

    def get_status(self, repo_path: str | Path) -> dict:
        """Get index status for a repository."""
        from .vector_build import check_needs_update

        repo_path = Path(repo_path).resolve()
        index_path = self._get_index_path(repo_path)

        if not index_path.exists():
            return {"indexed": False, "path": str(repo_path)}

        manifest = self._load_manifest(index_path)
        needs_update, new_files, deleted = check_needs_update(repo_path, manifest)

        return {
            "indexed": True, "path": str(repo_path), "index_path": str(index_path),
            "snippet_count": manifest.get("snippet_count", 0),
            "file_count": len(manifest.get("files", {})),
            "created": manifest.get("created", 0), "updated": manifest.get("updated", 0),
            "needs_update": needs_update, "new_or_modified": len(new_files), "deleted": len(deleted),
        }

    def clear(self, repo_path: str | Path | None = None) -> int:
        """Clear index cache."""
        import shutil

        if repo_path:
            repo_path = Path(repo_path).resolve()
            cache_key = str(repo_path)
            index_path = self._get_index_path(repo_path)
            self._indexes.pop(cache_key, None)
            self._metadata.pop(cache_key, None)
            self._corpus_idx_map.pop(cache_key, None)
            if index_path.exists():
                shutil.rmtree(index_path)
                return 1
            return 0

        count = len(self._indexes)
        self._indexes.clear()
        self._metadata.clear()
        self._corpus_idx_map.clear()
        if self.config.index_dir.exists():
            for child in self.config.index_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                    count += 1
        return count


# Global singleton
_index_manager: CodeIndex | None = None
_index_manager_lock = threading.Lock()


def get_index_manager(config: IndexConfig | None = None) -> CodeIndex:
    """Get the global index manager instance."""
    global _index_manager
    if _index_manager is not None and config is None:
        return _index_manager
    with _index_manager_lock:
        if _index_manager is None or config is not None:
            _index_manager = CodeIndex(config)
        return _index_manager


def semantic_search(query: str, path: str = ".", k: int = 5) -> list[dict]:
    """Search code semantically using embeddings."""
    manager = get_index_manager()
    results = manager.search(path, query, k=k)
    return [r.to_dict() for r in results]


__all__ = [
    "IndexConfig", "CodeSnippet", "SearchResult", "CodeIndex",
    "get_index_manager", "semantic_search",
]

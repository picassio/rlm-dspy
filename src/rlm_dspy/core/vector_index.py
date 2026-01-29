"""Vector index management for semantic code search.

Provides embedding-based code search with incremental updates and caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    # Lower threshold for faster search - FAISS is much faster than brute-force even for small sets
    faiss_threshold: int = 100
    auto_update: bool = True
    cache_ttl: int = 3600
    # Maximum snippet size in characters.
    # text-embedding-3-small has 8191 token limit.
    # Normal code: ~4 chars/token, minified code: ~2.5 chars/token
    # Using 18K chars = ~7200 tokens at 2.5 ratio, safely under limit
    max_snippet_chars: int = 18000

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
            faiss_threshold=config.get("faiss_threshold", 100),
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
    chunk_index: int = 0  # For chunked large snippets
    total_chunks: int = 1  # Total chunks if split

    def to_document(self) -> str:
        """Convert to document string for embedding."""
        header = f"# {self.type}: {self.name}\n# File: {self.file}:{self.line}"
        if self.total_chunks > 1:
            header += f" (chunk {self.chunk_index + 1}/{self.total_chunks})"
        return f"{header}\n\n{self.text}"


def _chunk_snippet(snippet: CodeSnippet, max_chars: int) -> list[CodeSnippet]:
    """Split a large snippet into smaller chunks.

    Tries to split on logical boundaries (newlines) first.
    Falls back to character-level splitting for minified code with long lines.
    Each chunk preserves context with the snippet name and file info.

    Args:
        snippet: The snippet to chunk
        max_chars: Maximum characters per chunk (for text only, not including header)

    Returns:
        List of snippets (original if small enough, or chunked versions)
    """
    text = snippet.text

    # Reserve space for header in to_document():
    # "# {type}: {name}\n# File: {file}:{line} (chunk N/M)\n\n"
    # Worst case: ~200 chars for long names/paths
    header_reserve = 200
    effective_max = max_chars - header_reserve

    # If small enough, return as-is
    if len(text) <= effective_max:
        return [snippet]

    chunks = []
    lines = text.split('\n')
    current_chunk_lines: list[str] = []
    current_size = 0
    chunk_start_line = snippet.line

    for i, line in enumerate(lines):
        line_size = len(line) + 1  # +1 for newline

        # Handle very long lines (e.g., minified JS)
        if line_size > effective_max:
            # First, save any accumulated lines
            if current_chunk_lines:
                chunk_text = '\n'.join(current_chunk_lines)
                chunks.append((chunk_text, chunk_start_line, snippet.line + i - 1))
                current_chunk_lines = []
                current_size = 0

            # Split the long line by characters
            line_chunks = [line[j:j+effective_max] for j in range(0, len(line), effective_max)]
            for lc in line_chunks:
                chunks.append((lc, snippet.line + i, snippet.line + i))
            chunk_start_line = snippet.line + i + 1
            continue

        # Check if adding this line would exceed limit
        if current_size + line_size > effective_max and current_chunk_lines:
            # Save current chunk
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append((chunk_text, chunk_start_line, snippet.line + i - 1))
            current_chunk_lines = []
            current_size = 0
            chunk_start_line = snippet.line + i

        current_chunk_lines.append(line)
        current_size += line_size

    # Don't forget the last chunk
    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines)
        chunks.append((chunk_text, chunk_start_line, snippet.end_line))

    # Convert to CodeSnippet objects
    total_chunks = len(chunks)
    result = []
    for idx, (chunk_text, start, end) in enumerate(chunks):
        result.append(CodeSnippet(
            id=f"{snippet.id}:chunk{idx}",
            text=chunk_text,
            file=snippet.file,
            line=start,
            end_line=end,
            type=snippet.type,
            name=snippet.name,
            language=snippet.language,
            chunk_index=idx,
            total_chunks=total_chunks,
        ))

    return result


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
        self._corpus_idx_map: dict[str, dict[int, str]] = {}  # path -> {corpus_idx -> snippet_id}

    @property
    def embedder(self):
        """Lazy-load embedder."""
        if self._embedder is None:
            from .embeddings import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _get_index_path(self, repo_path: Path) -> Path:
        """Get index storage path for a repository.

        Uses project registry with best-match semantics for overlapping paths.
        Falls back to hash-based path if no match found.
        """
        from .project_registry import get_project_registry

        repo_path = repo_path.resolve()
        registry = get_project_registry()

        # First check for exact match
        for project in registry.list():
            if project.path == str(repo_path):
                return self.config.index_dir / project.name

        # For overlapping paths, find the most specific (deepest) match
        best_match = registry.find_best_match(repo_path)
        if best_match and best_match.path != str(repo_path):
            # Path is under an existing project - use that project's index
            logger.debug(
                "Path %s is under project '%s' (%s) - using existing index",
                repo_path, best_match.name, best_match.path
            )
            return self.config.index_dir / best_match.name

        # Auto-register if enabled
        project = registry.auto_register(repo_path)
        if project:
            return self.config.index_dir / project.name

        # Fallback to hash-based path
        path_hash = hashlib.md5(str(repo_path).encode()).hexdigest()[:12]
        return self.config.index_dir / path_hash

    def _get_manifest_path(self, index_path: Path) -> Path:
        return index_path / "manifest.json"

    def _load_manifest(self, index_path: Path) -> dict:
        """Load manifest file."""
        manifest_path = self._get_manifest_path(index_path)
        if manifest_path.exists():
            try:
                return json.loads(manifest_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                logger.warning("Corrupt manifest at %s", manifest_path)
        return {"files": {}, "created": 0, "updated": 0, "snippet_count": 0}

    def _save_manifest(self, index_path: Path, manifest: dict) -> None:
        """Save manifest file atomically."""
        import tempfile

        manifest_path = self._get_manifest_path(index_path)

        # Atomic write: temp file + rename
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=index_path,
                suffix='.tmp',
                delete=False,
                encoding='utf-8'
            ) as f:
                json.dump(manifest, f, indent=2)
                temp_path = Path(f.name)

            temp_path.replace(manifest_path)
        except OSError as e:
            logger.error("Failed to save manifest: %s", e)
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()

    def _extract_snippets(self, repo_path: Path) -> list[CodeSnippet]:
        """Extract code snippets from repository using AST parsing."""
        import os
        from .ast_index import index_file, LANGUAGE_MAP

        snippets = []

        # Directories to skip - pruned at walk level for efficiency
        SKIP_DIRS = {'__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build',
                     '.git', '.hg', '.svn', 'eggs', '.eggs', '.tox', '.nox'}

        # Supported extensions
        supported_extensions = set(LANGUAGE_MAP.keys())

        # Single directory walk - more efficient than multiple globs
        for root, dirs, files in os.walk(repo_path):
            # Prune directories in-place (prevents descent into them)
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

            for filename in files:
                # Check extension early
                ext = os.path.splitext(filename)[1]
                if ext not in supported_extensions:
                    continue

                # Skip minified files (not useful for semantic search)
                # Common patterns: .min.js, .min.css, -min.js, _min.js
                if '.min.' in filename or '-min.' in filename or '_min.' in filename:
                    continue

                file_path = Path(root) / filename

                if not file_path.is_file():
                    continue

                try:
                    # Language from extension
                    language = LANGUAGE_MAP.get(ext)

                    if not language:
                        continue

                    # Skip files larger than 1MB to prevent OOM
                    try:
                        if file_path.stat().st_size > 1_000_000:
                            logger.debug("Skipping large file: %s", file_path)
                            continue
                    except OSError:
                        continue

                    # Index the file (returns ASTIndex with .definitions)
                    ast_index = index_file(str(file_path))

                    # Read file content for snippet text
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        lines = content.splitlines()
                    except UnicodeDecodeError:
                        continue

                    # Use relative path for consistent storage
                    # Handle symlinks pointing outside repo gracefully
                    try:
                        rel_path = str(file_path.relative_to(repo_path))
                    except ValueError:
                        # Symlink points outside repo - use absolute path
                        logger.debug("File outside repo (symlink?): %s", file_path)
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

                        snippet_id = f"{rel_path}:{defn.name}:{defn.line}"

                        snippet = CodeSnippet(
                            id=snippet_id,
                            text=text,
                            file=rel_path,
                            line=defn.line,
                            end_line=end_line,
                            type=defn.kind,  # Definition uses 'kind', not 'type'
                            name=defn.name,
                            language=language,
                        )

                        # Chunk large snippets to fit embedding model limits
                        # text-embedding-3-small has 8191 token limit (~24K chars)
                        chunked = _chunk_snippet(snippet, self.config.max_snippet_chars)
                        snippets.extend(chunked)

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
        import os
        from .ast_index import LANGUAGE_MAP

        supported_extensions = set(LANGUAGE_MAP.keys())
        SKIP_DIRS = {'__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build',
                     '.git', '.hg', '.svn', 'eggs', '.eggs', '.tox', '.nox'}

        current_files = {}
        # Use os.walk with directory pruning - much faster than rglob
        for root, dirs, files in os.walk(repo_path):
            # Prune directories in-place
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

            for filename in files:
                ext = os.path.splitext(filename)[1]
                if ext not in supported_extensions:
                    continue

                f = Path(root) / filename
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
        import fcntl

        repo_path = Path(repo_path).resolve()
        if not repo_path.exists():
            raise ValueError(f"Path does not exist: {repo_path}")

        index_path = self._get_index_path(repo_path)

        # Cross-process lock to prevent concurrent builds on same project
        lock_file = index_path.parent / f".{index_path.name}.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        lock_fd = None
        try:
            lock_fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)  # Blocking lock

            return self._build_locked(repo_path, index_path, force)
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)

    def _build_locked(
        self,
        repo_path: Path,
        index_path: Path,
        force: bool,
    ) -> int:
        """Build index (caller must hold lock)."""
        from dspy.retrievers import Embeddings

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
                        metadata, corpus_idx_to_id = self._load_metadata(index_path)

                        self._indexes[cache_key] = (index, time.time())
                        self._metadata[cache_key] = metadata
                        self._corpus_idx_map[cache_key] = corpus_idx_to_id

                        logger.info("Loaded existing index for %s (%d snippets)",
                                   repo_path, len(metadata))
                        return len(metadata)
                    except Exception as e:
                        logger.warning("Failed to load index, rebuilding: %s", e)

                # Incremental update: only re-index changed files
                elif new_files or deleted:
                    try:
                        count = self._incremental_update(
                            repo_path, index_path, cache_key,
                            new_files, deleted, manifest
                        )
                        if count > 0:
                            return count
                        # Fall through to full rebuild if incremental failed
                    except Exception as e:
                        logger.warning("Incremental update failed, doing full rebuild: %s", e)

        # Build new index (full rebuild)
        logger.info("Building full index for %s...", repo_path)
        snippets = self._extract_snippets(repo_path)

        if not snippets:
            logger.warning("No code found in %s", repo_path)
            return 0

        logger.info("Found %d code snippets", len(snippets))

        # Create corpus for embedding
        # Keep order so we can map idx -> snippet
        corpus = [s.to_document() for s in snippets]
        metadata = {s.id: s for s in snippets}
        # Map corpus index to snippet ID for fast lookup during search
        corpus_idx_to_id = {i: s.id for i, s in enumerate(snippets)}

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

        # Note: DSPy Embeddings.save() already saves embeddings as corpus_embeddings.npy

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

        # Save metadata and corpus index mapping
        self._save_metadata(index_path, metadata, corpus_idx_to_id)

        # Cache
        self._indexes[cache_key] = (index, time.time())
        self._metadata[cache_key] = metadata
        self._corpus_idx_map[cache_key] = corpus_idx_to_id

        # Update project registry stats
        from .project_registry import get_project_registry
        registry = get_project_registry()
        for project in registry.list():
            if project.path == str(repo_path):
                registry.update_stats(project.name, len(snippets), len(file_mtimes))
                break

        logger.info("Index built: %d snippets from %d files",
                   len(snippets), len(file_mtimes))

        return len(snippets)

    def _incremental_update(
        self,
        repo_path: Path,
        index_path: Path,
        cache_key: str,
        new_or_modified: list[Path],
        deleted: list[str],
        manifest: dict,
    ) -> int:
        """Incrementally update index with only changed files.

        This is much faster than full rebuild as it only embeds new/modified files.

        Args:
            repo_path: Path to repository
            index_path: Path to index directory
            cache_key: Cache key for this index
            new_or_modified: List of new or modified files
            deleted: List of deleted file paths
            manifest: Current manifest

        Returns:
            Number of snippets in updated index, or 0 if incremental failed
        """
        from dspy.retrievers import Embeddings
        import numpy as np

        logger.info(
            "Incremental update: %d new/modified, %d deleted",
            len(new_or_modified), len(deleted)
        )

        # Load existing metadata and embeddings
        old_metadata, old_corpus_idx_map = self._load_metadata(index_path)

        # Load embeddings saved by DSPy (corpus_embeddings.npy)
        embeddings_path = index_path / "corpus_embeddings.npy"
        if not embeddings_path.exists():
            logger.debug("No cached embeddings, falling back to full rebuild")
            return 0

        old_embeddings = np.load(embeddings_path)

        # Build set of files that were deleted or modified
        # Note: new_or_modified contains absolute paths, deleted contains absolute paths from manifest
        # But snippets store RELATIVE paths, so we need to convert
        changed_files_abs = set(str(f) for f in new_or_modified) | set(deleted)
        changed_files_rel = set()
        for f in new_or_modified:
            try:
                rel = str(f.relative_to(repo_path))
                changed_files_rel.add(rel)
            except ValueError:
                pass
        for d in deleted:
            try:
                rel = str(Path(d).relative_to(repo_path))
                changed_files_rel.add(rel)
            except ValueError:
                pass

        # Keep snippets from unchanged files
        kept_snippets = []
        kept_embeddings = []
        kept_indices = []

        for corpus_idx, snippet_id in old_corpus_idx_map.items():
            snippet = old_metadata.get(snippet_id)
            if snippet:
                # Check both relative and absolute paths
                if snippet.file not in changed_files_abs and snippet.file not in changed_files_rel:
                    kept_snippets.append(snippet)
                    kept_indices.append(corpus_idx)

        if kept_indices:
            kept_embeddings = old_embeddings[kept_indices]

        logger.debug("Kept %d snippets from unchanged files", len(kept_snippets))

        # Extract snippets from new/modified files only
        new_snippets = []
        for file_path in new_or_modified:
            if file_path.exists():
                file_snippets = self._extract_snippets_from_file(file_path, repo_path)
                new_snippets.extend(file_snippets)

        logger.debug("Extracted %d snippets from changed files", len(new_snippets))

        if not new_snippets and not kept_snippets:
            logger.warning("No snippets after incremental update")
            return 0

        # Embed only new snippets
        new_embeddings = None
        if new_snippets:
            new_corpus = [s.to_document() for s in new_snippets]
            logger.info("Embedding %d new snippets...", len(new_snippets))
            new_embeddings = self.embedder(new_corpus)
            new_embeddings = np.array(new_embeddings)

        # Combine old and new
        all_snippets = kept_snippets + new_snippets

        if len(kept_embeddings) > 0 and new_embeddings is not None:
            all_embeddings = np.vstack([kept_embeddings, new_embeddings])
        elif new_embeddings is not None:
            all_embeddings = new_embeddings
        else:
            all_embeddings = np.array(kept_embeddings) if len(kept_embeddings) > 0 else np.array([])

        if len(all_snippets) == 0:
            return 0

        # Build new metadata and corpus mapping
        metadata = {s.id: s for s in all_snippets}
        corpus_idx_to_id = {i: s.id for i, s in enumerate(all_snippets)}
        corpus = [s.to_document() for s in all_snippets]

        # Save combined embeddings first
        np.save(embeddings_path, all_embeddings)

        # Save corpus to config.json (DSPy format)
        import json
        config_path = index_path / "config.json"
        config = {
            "k": 10,
            "normalize": True,
            "corpus": corpus,
            "has_faiss_index": len(corpus) >= self.config.faiss_threshold if self.config.use_faiss else False,
        }
        config_path.write_text(json.dumps(config), encoding='utf-8')

        # Load index from saved files (uses pre-computed embeddings)
        index = Embeddings.from_saved(str(index_path), self.embedder)

        # Update manifest
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
            "created": manifest.get("created", time.time()),
            "updated": time.time(),
            "snippet_count": len(all_snippets),
        }
        self._save_manifest(index_path, manifest)
        self._save_metadata(index_path, metadata, corpus_idx_to_id)

        # Cache
        self._indexes[cache_key] = (index, time.time())
        self._metadata[cache_key] = metadata
        self._corpus_idx_map[cache_key] = corpus_idx_to_id

        # Update project registry
        from .project_registry import get_project_registry
        registry = get_project_registry()
        for project in registry.list():
            if project.path == str(repo_path):
                registry.update_stats(project.name, len(all_snippets), len(file_mtimes))
                break

        logger.info(
            "Incremental update complete: %d total snippets (kept %d, added %d)",
            len(all_snippets), len(kept_snippets), len(new_snippets)
        )

        return len(all_snippets)

    def _extract_snippets_from_file(
        self,
        file_path: Path,
        repo_path: Path
    ) -> list[CodeSnippet]:
        """Extract code snippets from a single file.

        Args:
            file_path: Path to source file
            repo_path: Root repository path (for relative paths)

        Returns:
            List of CodeSnippet objects
        """
        from .ast_index import LANGUAGE_MAP, index_file

        snippets = []

        if file_path.suffix not in LANGUAGE_MAP:
            return snippets

        language = LANGUAGE_MAP.get(file_path.suffix)
        if not language:
            return snippets

        try:
            rel_path = str(file_path.relative_to(repo_path))

            # Read file content
            content = file_path.read_text(encoding='utf-8')
            lines = content.splitlines()

            # Index the file using AST
            ast_index = index_file(str(file_path))

            for defn in ast_index.definitions:
                # Extract text for the definition
                start_line = defn.line - 1  # 0-indexed
                end_line = defn.end_line

                if start_line < len(lines):
                    text_lines = lines[start_line:min(end_line, len(lines))]
                    text = '\n'.join(text_lines)

                    # Limit text size
                    if len(text) > 2000:
                        text = text[:2000] + "\n... (truncated)"

                    snippet = CodeSnippet(
                        id=f"{rel_path}:{defn.line}:{defn.name}",
                        file=rel_path,
                        line=defn.line,
                        end_line=defn.end_line,
                        type=defn.kind,
                        name=defn.name,
                        text=text,
                        language=language,
                    )
                    snippets.append(snippet)

        except UnicodeDecodeError:
            logger.debug("Skipping non-text file: %s", file_path)
        except Exception as e:
            logger.warning("Failed to extract from %s: %s", file_path, e)

        return snippets

    def _save_metadata(
        self,
        index_path: Path,
        metadata: dict[str, CodeSnippet],
        corpus_idx_to_id: dict[int, str] | None = None,
    ) -> None:
        """Save snippet metadata and corpus index mapping."""
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
        (index_path / "metadata.json").write_text(json.dumps(data, indent=2), encoding='utf-8')

        # Save corpus index mapping for fast search lookup
        if corpus_idx_to_id is not None:
            # Convert int keys to strings for JSON
            idx_map = {str(k): v for k, v in corpus_idx_to_id.items()}
            (index_path / "corpus_idx_map.json").write_text(json.dumps(idx_map), encoding='utf-8')

        # Update in-memory cache
        cache_key = str(index_path)
        self._metadata[cache_key] = metadata
        if corpus_idx_to_id is not None:
            self._corpus_idx_map[cache_key] = corpus_idx_to_id

    def _load_metadata(self, index_path: Path) -> tuple[dict[str, CodeSnippet], dict[int, str]]:
        """Load snippet metadata and corpus index mapping.

        Uses in-memory cache to avoid redundant disk reads.

        Returns:
            (metadata dict, corpus_idx_to_id dict)
        """
        cache_key = str(index_path)

        # Check in-memory cache first
        if cache_key in self._metadata and cache_key in self._corpus_idx_map:
            return self._metadata[cache_key], self._corpus_idx_map[cache_key]

        metadata_path = index_path / "metadata.json"
        if not metadata_path.exists():
            return {}, {}

        data = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata = {
            id_: CodeSnippet(**info)
            for id_, info in data.items()
        }

        # Load corpus index mapping
        idx_map_path = index_path / "corpus_idx_map.json"
        corpus_idx_to_id = {}
        if idx_map_path.exists():
            idx_data = json.loads(idx_map_path.read_text(encoding='utf-8'))
            # Convert string keys back to int
            corpus_idx_to_id = {int(k): v for k, v in idx_data.items()}

        # Cache for future calls
        self._metadata[cache_key] = metadata
        self._corpus_idx_map[cache_key] = corpus_idx_to_id

        return metadata, corpus_idx_to_id

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
        corpus_idx_map = self._corpus_idx_map.get(cache_key, {})

        # Search
        result = index(query)

        results = []
        for passage, idx in zip(result.passages, result.indices):
            snippet = None

            # Fast lookup using corpus index mapping
            if idx in corpus_idx_map:
                snippet_id = corpus_idx_map[idx]
                snippet = metadata.get(snippet_id)

            # Fallback: try to parse from passage header
            if snippet is None:
                # Parse the document format: "# type: name\n# File: path:line\n\ntext"
                import re
                header_match = re.match(
                    r'# (\w+): (.+)\n# File: (.+):(\d+)',
                    passage
                )
                if header_match:
                    type_, name, file_, line = header_match.groups()
                    # Extract text after headers
                    text_start = passage.find('\n\n')
                    text = passage[text_start+2:] if text_start > 0 else passage

                    snippet = CodeSnippet(
                        id=f"{file_}:{name}:{line}",
                        text=text,
                        file=file_,
                        line=int(line),
                        end_line=int(line) + text.count('\n'),
                        type=type_,
                        name=name,
                    )

            if snippet is None:
                # Last resort fallback
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
            self._corpus_idx_map.pop(cache_key, None)

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
            self._corpus_idx_map.clear()

            # Clear disk indexes
            if self.config.index_dir.exists():
                import shutil
                for child in self.config.index_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                        count += 1

            return count


# Global index manager instance (thread-safe singleton)
_index_manager: CodeIndex | None = None
_index_manager_lock = threading.Lock()


def get_index_manager(config: IndexConfig | None = None) -> CodeIndex:
    """Get the global index manager instance (thread-safe)."""
    global _index_manager

    # Fast path: already initialized
    if _index_manager is not None and config is None:
        return _index_manager

    # Slow path: need to initialize (with lock)
    with _index_manager_lock:
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

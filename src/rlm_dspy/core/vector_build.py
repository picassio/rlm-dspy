"""Vector index build and incremental update logic."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .vector_types import CodeSnippet

from .vector_types import CodeSnippet, chunk_snippet

logger = logging.getLogger(__name__)


def extract_snippets(repo_path: Path, max_snippet_chars: int = 18000) -> list[CodeSnippet]:
    """Extract code snippets from repository using AST parsing."""
    from .ast_index import index_file, LANGUAGE_MAP

    snippets = []
    SKIP_DIRS = {'__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build',
                 '.git', '.hg', '.svn', 'eggs', '.eggs', '.tox', '.nox'}
    supported_extensions = set(LANGUAGE_MAP.keys())

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext not in supported_extensions:
                continue
            if '.min.' in filename or '-min.' in filename or '_min.' in filename:
                continue

            file_path = Path(root) / filename
            if not file_path.is_file():
                continue

            try:
                language = LANGUAGE_MAP.get(ext)
                if not language:
                    continue

                if file_path.stat().st_size > 1_000_000:
                    logger.debug("Skipping large file: %s", file_path)
                    continue

                ast_index = index_file(str(file_path))
                content = file_path.read_text(encoding='utf-8')
                lines = content.splitlines()

                try:
                    rel_path = str(file_path.relative_to(repo_path))
                except ValueError:
                    continue

                for defn in ast_index.definitions:
                    start_line = defn.line - 1
                    end_line = defn.end_line
                    text_lines = lines[start_line:min(end_line, len(lines))] if start_line < len(lines) else []
                    text = '\n'.join(text_lines)

                    snippet = CodeSnippet(
                        id=f"{rel_path}:{defn.name}:{defn.line}",
                        text=text,
                        file=rel_path,
                        line=defn.line,
                        end_line=end_line,
                        type=defn.kind,
                        name=defn.name,
                        language=language,
                    )
                    snippets.extend(chunk_snippet(snippet, max_snippet_chars))
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.debug("Failed to index %s: %s", file_path, e)

    return snippets


def extract_snippets_from_file(file_path: Path, repo_path: Path) -> list[CodeSnippet]:
    """Extract code snippets from a single file."""
    from .ast_index import LANGUAGE_MAP, index_file

    snippets = []
    if file_path.suffix not in LANGUAGE_MAP:
        return snippets

    language = LANGUAGE_MAP.get(file_path.suffix)
    if not language:
        return snippets

    try:
        rel_path = str(file_path.relative_to(repo_path))
        content = file_path.read_text(encoding='utf-8')
        lines = content.splitlines()
        ast_index = index_file(str(file_path))

        for defn in ast_index.definitions:
            start_line = defn.line - 1
            end_line = defn.end_line
            if start_line < len(lines):
                text = '\n'.join(lines[start_line:min(end_line, len(lines))])
                if len(text) > 2000:
                    text = text[:2000] + "\n... (truncated)"
                snippets.append(CodeSnippet(
                    id=f"{rel_path}:{defn.line}:{defn.name}",
                    file=rel_path,
                    line=defn.line,
                    end_line=defn.end_line,
                    type=defn.kind,
                    name=defn.name,
                    text=text,
                    language=language,
                ))
    except UnicodeDecodeError:
        logger.debug("Skipping non-text file: %s", file_path)
    except Exception as e:
        logger.warning("Failed to extract from %s: %s", file_path, e)

    return snippets


def check_needs_update(repo_path: Path, manifest: dict) -> tuple[bool, list[Path], list[str]]:
    """Check if index needs updating. Returns (needs_update, new_or_modified, deleted)."""
    from .ast_index import LANGUAGE_MAP

    supported_extensions = set(LANGUAGE_MAP.keys())
    SKIP_DIRS = {'__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build',
                 '.git', '.hg', '.svn', 'eggs', '.eggs', '.tox', '.nox'}

    current_files = {}
    for root, dirs, files in os.walk(repo_path):
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
    new_or_modified = [Path(p) for p, mtime in current_files.items()
                       if p not in old_files or old_files[p] < mtime]
    deleted = [p for p in old_files if p not in current_files]

    return bool(new_or_modified or deleted), new_or_modified, deleted


def build_full_index(repo_path: Path, index_path: Path, embedder, config) -> tuple[int, dict]:
    """Build a full index from scratch. Returns (snippet_count, manifest)."""
    from dspy.retrievers import Embeddings

    logger.info("Building full index for %s...", repo_path)
    snippets = extract_snippets(repo_path, config.max_snippet_chars)

    if not snippets:
        logger.warning("No code found in %s", repo_path)
        return 0, {}

    logger.info("Found %d code snippets", len(snippets))

    corpus = [s.to_document() for s in snippets]
    metadata = {s.id: s for s in snippets}
    corpus_idx_to_id = {i: s.id for i, s in enumerate(snippets)}

    brute_force_threshold = config.faiss_threshold if config.use_faiss else float('inf')

    logger.info("Creating embeddings...")
    index = Embeddings(corpus=corpus, embedder=embedder, k=10, brute_force_threshold=brute_force_threshold)

    index_path.mkdir(parents=True, exist_ok=True)
    index.save(str(index_path))

    # Build manifest
    from .ast_index import LANGUAGE_MAP
    supported_extensions = set(LANGUAGE_MAP.keys())
    file_mtimes = {}
    for f in repo_path.rglob("*"):
        if f.is_file() and f.suffix in supported_extensions:
            if not any(part.startswith('.') for part in f.parts):
                if not any(ignore in f.parts for ignore in ['__pycache__', 'node_modules', '.venv', 'venv']):
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

    logger.info("Index built: %d snippets from %d files", len(snippets), len(file_mtimes))
    return len(snippets), manifest


def incremental_update(
    repo_path: Path, index_path: Path, embedder, config,
    new_or_modified: list[Path], deleted: list[str], manifest: dict,
    old_metadata: dict, old_corpus_idx_map: dict,
) -> tuple[int, dict] | None:
    """Incrementally update index. Returns (snippet_count, manifest) or None if failed."""
    from dspy.retrievers import Embeddings

    logger.info("Incremental update: %d new/modified, %d deleted", len(new_or_modified), len(deleted))

    # Load embeddings
    embeddings_path = index_path / "corpus_embeddings.npy"
    compressed_path = index_path / "corpus_embeddings.npz"

    if compressed_path.exists():
        from .index_compression import load_numpy_array
        old_embeddings = load_numpy_array(compressed_path)
    elif embeddings_path.exists():
        old_embeddings = np.load(embeddings_path)
    else:
        logger.debug("No cached embeddings, falling back to full rebuild")
        return None

    # Build changed file sets
    changed_files_rel = set()
    for f in new_or_modified:
        try:
            changed_files_rel.add(str(f.relative_to(repo_path)))
        except ValueError:
            pass
    for d in deleted:
        try:
            changed_files_rel.add(str(Path(d).relative_to(repo_path)))
        except ValueError:
            pass

    # Keep unchanged snippets
    kept_snippets = []
    kept_indices = []
    for corpus_idx, snippet_id in old_corpus_idx_map.items():
        snippet = old_metadata.get(snippet_id)
        if snippet and snippet.file not in changed_files_rel:
            kept_snippets.append(snippet)
            kept_indices.append(corpus_idx)

    kept_embeddings = old_embeddings[kept_indices] if kept_indices else []
    logger.debug("Kept %d snippets from unchanged files", len(kept_snippets))

    # Extract new snippets
    new_snippets = []
    for file_path in new_or_modified:
        if file_path.exists():
            new_snippets.extend(extract_snippets_from_file(file_path, repo_path))
    logger.debug("Extracted %d snippets from changed files", len(new_snippets))

    if not new_snippets and not kept_snippets:
        return None

    # Embed new snippets
    new_embeddings = None
    if new_snippets:
        new_corpus = [s.to_document() for s in new_snippets]
        logger.info("Embedding %d new snippets...", len(new_snippets))
        new_embeddings = np.array(embedder(new_corpus))

    all_snippets = kept_snippets + new_snippets
    if len(kept_embeddings) > 0 and new_embeddings is not None:
        all_embeddings = np.vstack([kept_embeddings, new_embeddings])
    elif new_embeddings is not None:
        all_embeddings = new_embeddings
    else:
        all_embeddings = np.array(kept_embeddings) if len(kept_embeddings) > 0 else np.array([])

    if not all_snippets:
        return None

    # Save combined data
    np.save(embeddings_path, all_embeddings)

    corpus = [s.to_document() for s in all_snippets]
    config_data = {
        "k": 10,
        "normalize": True,
        "corpus": corpus,
        "has_faiss_index": len(corpus) >= config.faiss_threshold if config.use_faiss else False,
    }
    (index_path / "config.json").write_text(json.dumps(config_data), encoding='utf-8')

    # Build manifest
    from .ast_index import LANGUAGE_MAP
    supported_extensions = set(LANGUAGE_MAP.keys())
    file_mtimes = {}
    for f in repo_path.rglob("*"):
        if f.is_file() and f.suffix in supported_extensions:
            if not any(part.startswith('.') for part in f.parts):
                if not any(ignore in f.parts for ignore in ['__pycache__', 'node_modules', '.venv', 'venv']):
                    try:
                        file_mtimes[str(f)] = f.stat().st_mtime
                    except OSError:
                        pass

    new_manifest = {
        "repo_path": str(repo_path),
        "files": file_mtimes,
        "created": manifest.get("created", time.time()),
        "updated": time.time(),
        "snippet_count": len(all_snippets),
    }

    logger.info("Incremental update complete: %d total snippets", len(all_snippets))
    return len(all_snippets), new_manifest


__all__ = [
    "extract_snippets",
    "extract_snippets_from_file",
    "check_needs_update",
    "build_full_index",
    "incremental_update",
]

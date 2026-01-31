"""Vector index types - config, snippets, and search results."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IndexConfig:
    """Configuration for vector index."""
    index_dir: Path = field(default_factory=lambda: Path.home() / ".rlm" / "indexes")
    use_faiss: bool = True
    faiss_threshold: int = 100
    auto_update: bool = True
    cache_ttl: int = 3600
    max_snippet_chars: int = 18000

    @classmethod
    def from_user_config(cls) -> "IndexConfig":
        """Load from user config."""
        from .user_config import load_config
        config = load_config()
        index_dir = os.environ.get("RLM_INDEX_DIR", config.get("index_dir", "~/.rlm/indexes"))
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
    chunk_index: int = 0
    total_chunks: int = 1

    def to_document(self) -> str:
        """Convert to document string for embedding."""
        header = f"# {self.type}: {self.name}\n# File: {self.file}:{self.line}"
        if self.total_chunks > 1:
            header += f" (chunk {self.chunk_index + 1}/{self.total_chunks})"
        return f"{header}\n\n{self.text}"


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


def chunk_snippet(snippet: CodeSnippet, max_chars: int) -> list[CodeSnippet]:
    """Split a large snippet into smaller chunks."""
    text = snippet.text
    header_reserve = 200
    effective_max = max_chars - header_reserve

    if len(text) <= effective_max:
        return [snippet]

    chunks = []
    lines = text.split('\n')
    current_chunk_lines: list[str] = []
    current_size = 0
    chunk_start_line = snippet.line

    for i, line in enumerate(lines):
        line_size = len(line) + 1

        if line_size > effective_max:
            if current_chunk_lines:
                chunk_text = '\n'.join(current_chunk_lines)
                chunks.append((chunk_text, chunk_start_line, snippet.line + i - 1))
                current_chunk_lines = []
                current_size = 0
            line_chunks = [line[j:j+effective_max] for j in range(0, len(line), effective_max)]
            for lc in line_chunks:
                chunks.append((lc, snippet.line + i, snippet.line + i))
            chunk_start_line = snippet.line + i + 1
            continue

        if current_size + line_size > effective_max and current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append((chunk_text, chunk_start_line, snippet.line + i - 1))
            current_chunk_lines = []
            current_size = 0
            chunk_start_line = snippet.line + i

        current_chunk_lines.append(line)
        current_size += line_size

    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines)
        chunks.append((chunk_text, chunk_start_line, snippet.end_line))

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


__all__ = ["IndexConfig", "CodeSnippet", "SearchResult", "chunk_snippet"]

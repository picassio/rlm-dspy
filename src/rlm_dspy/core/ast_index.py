"""AST-based code indexing using tree-sitter.

Provides 100% accurate structural queries (classes, functions, line numbers)
without LLM hallucination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .treesitter import EXTENSION_MAP, get_parser_simple

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
LANGUAGE_MAP = EXTENSION_MAP


@dataclass
class Definition:
    """A code definition (class, function, method)."""
    name: str
    kind: Literal["class", "function", "method"]
    line: int
    end_line: int
    file: str = ""
    parent: str | None = None  # For methods, the class name


@dataclass
class ASTIndex:
    """Index of code definitions extracted via tree-sitter."""
    definitions: list[Definition] = field(default_factory=list)

    def find(self, name: str | None = None, kind: str | None = None) -> list[Definition]:
        """Find definitions matching criteria."""
        results = self.definitions
        if name:
            results = [d for d in results if name.lower() in d.name.lower()]
        if kind:
            results = [d for d in results if d.kind == kind]
        return results

    def classes(self) -> list[Definition]:
        return self.find(kind="class")

    def functions(self) -> list[Definition]:
        return self.find(kind="function")

    def methods(self) -> list[Definition]:
        return self.find(kind="method")

    def get_line(self, name: str) -> int | None:
        """Get exact line number for a definition."""
        for d in self.definitions:
            if d.name == name:
                return d.line
        return None





def _extract_definitions(node: Any, language: str, results: list[Definition], file: str, current_class: str | None = None) -> None:
    """Recursively extract definitions from AST."""

    # Class definitions
    if node.type in ("class_definition", "class_declaration"):
        name_node = node.child_by_field_name("name")
        if name_node:
            name = name_node.text.decode()
            results.append(Definition(
                name=name,
                kind="class",
                line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                file=file,
            ))
            # Recurse with class context
            for child in node.children:
                _extract_definitions(child, language, results, file, current_class=name)
            return

    # Function/method definitions
    if node.type in ("function_definition", "function_declaration", "method_definition"):
        name_node = node.child_by_field_name("name")
        if name_node:
            name = name_node.text.decode()
            results.append(Definition(
                name=name,
                kind="method" if current_class else "function",
                line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                file=file,
                parent=current_class,
            ))

    # Recurse
    for child in node.children:
        _extract_definitions(child, language, results, file, current_class)


def index_file(path: Path | str) -> ASTIndex:
    """Index a single file using tree-sitter."""
    path = Path(path)
    suffix = path.suffix.lower()
    language = LANGUAGE_MAP.get(suffix)

    if not language:
        return ASTIndex()

    parser = get_parser_simple(language)
    if not parser:
        return ASTIndex()

    try:
        code = path.read_text(encoding="utf-8")
        tree = parser.parse(bytes(code, "utf8"))

        definitions: list[Definition] = []
        _extract_definitions(tree.root_node, language, definitions, str(path))

        return ASTIndex(definitions=definitions)
    except UnicodeDecodeError:
        logger.debug("Skipping binary/non-UTF8 file: %s", path)
        return ASTIndex()
    except Exception as e:
        logger.warning("Failed to index %s: %s", path, e)
        return ASTIndex()


def index_files(paths: list[Path | str]) -> ASTIndex:
    """Index multiple files."""
    all_defs: list[Definition] = []
    for path in paths:
        idx = index_file(path)
        all_defs.extend(idx.definitions)
    return ASTIndex(definitions=all_defs)

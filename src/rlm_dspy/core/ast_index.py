"""AST-based code indexing using tree-sitter.

Provides 100% accurate structural queries (classes, functions, line numbers)
without LLM hallucination.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Language mappings (extension -> tree-sitter language name)
LANGUAGE_MAP = {
    # Core languages
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".rb": "ruby",
    ".cs": "c_sharp",
    # Additional languages
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    ".php": "php",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".hs": "haskell",
    ".lhs": "haskell",
}


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
    
    # Lazy-built indexes for O(1) lookup
    _by_name: dict[str, list[Definition]] = field(default_factory=dict, repr=False)
    _by_kind: dict[str, list[Definition]] = field(default_factory=dict, repr=False)
    _indexes_built: bool = field(default=False, repr=False)
    
    def _build_indexes(self) -> None:
        """Build hash indexes for fast lookup."""
        if self._indexes_built:
            return
        
        self._by_name.clear()
        self._by_kind.clear()
        
        for d in self.definitions:
            # Index by lowercase name for case-insensitive lookup
            name_lower = d.name.lower()
            if name_lower not in self._by_name:
                self._by_name[name_lower] = []
            self._by_name[name_lower].append(d)
            
            # Index by kind
            if d.kind not in self._by_kind:
                self._by_kind[d.kind] = []
            self._by_kind[d.kind].append(d)
        
        self._indexes_built = True

    def find(self, name: str | None = None, kind: str | None = None) -> list[Definition]:
        """Find definitions matching criteria. Uses hash indexes for O(1) lookup."""
        self._build_indexes()
        
        if name and kind:
            # Both filters: intersect results using id() for comparison
            name_lower = name.lower()
            name_matches: list[Definition] = []
            for key in self._by_name:
                if name_lower in key:
                    name_matches.extend(self._by_name[key])
            
            # Get kind matches
            kind_matches = self._by_kind.get(kind, [])
            kind_ids = {id(d) for d in kind_matches}
            
            # Return intersection
            return [d for d in name_matches if id(d) in kind_ids]
        elif name:
            # Name filter only (substring match)
            name_lower = name.lower()
            results = []
            for key in self._by_name:
                if name_lower in key:
                    results.extend(self._by_name[key])
            return results
        elif kind:
            # Kind filter only - O(1) lookup
            return list(self._by_kind.get(kind, []))
        else:
            return list(self.definitions)

    def classes(self) -> list[Definition]:
        self._build_indexes()
        return list(self._by_kind.get("class", []))

    def functions(self) -> list[Definition]:
        self._build_indexes()
        return list(self._by_kind.get("function", []))

    def methods(self) -> list[Definition]:
        self._build_indexes()
        return list(self._by_kind.get("method", []))

    def get_line(self, name: str) -> int | None:
        """Get exact line number for a definition."""
        self._build_indexes()
        # Exact match lookup
        matches = self._by_name.get(name.lower(), [])
        for d in matches:
            if d.name == name:  # Case-sensitive final check
                return d.line
        return None


def _get_parser(language: str):
    """Get tree-sitter parser for a language."""
    try:
        from tree_sitter import Language, Parser

        # Core languages
        if language == "python":
            import tree_sitter_python as ts_lang
        elif language == "javascript":
            import tree_sitter_javascript as ts_lang
        elif language == "typescript":
            import tree_sitter_typescript as ts_mod
            ts_lang = ts_mod.language_typescript()
            lang = Language(ts_lang)
            parser = Parser(lang)
            return parser
        elif language == "go":
            import tree_sitter_go as ts_lang
        elif language == "rust":
            import tree_sitter_rust as ts_lang
        elif language == "java":
            import tree_sitter_java as ts_lang
        elif language == "c":
            import tree_sitter_c as ts_lang
        elif language == "cpp":
            import tree_sitter_cpp as ts_lang
        elif language == "ruby":
            import tree_sitter_ruby as ts_lang
        elif language == "c_sharp":
            import tree_sitter_c_sharp as ts_lang
        # Additional languages
        elif language == "kotlin":
            import tree_sitter_kotlin as ts_lang
        elif language == "scala":
            import tree_sitter_scala as ts_lang
        elif language == "php":
            import tree_sitter_php as ts_mod
            # PHP has separate language for PHP code
            ts_lang = ts_mod.language_php()
            lang = Language(ts_lang)
            parser = Parser(lang)
            return parser
        elif language == "lua":
            import tree_sitter_lua as ts_lang
        elif language == "bash":
            import tree_sitter_bash as ts_lang
        elif language == "haskell":
            import tree_sitter_haskell as ts_lang
        else:
            return None

        lang = Language(ts_lang.language())
        parser = Parser(lang)
        return parser
    except ImportError:
        return None


def _extract_definitions(node, language: str, results: list[Definition], file: str, current_class: str | None = None):
    """Recursively extract definitions from AST."""

    # Class definitions (various languages)
    if node.type in ("class_definition", "class_declaration", "class_specifier",
                     "object_declaration",  # Kotlin/Scala companion objects
                     "interface_declaration"):  # Java/Kotlin interfaces
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

    # Function/method definitions (various languages)
    if node.type in ("function_definition", "function_declaration", "method_definition",
                     "function_item",  # Rust
                     "method_declaration",  # Java/Kotlin
                     "function_declaration_statement",  # PHP
                     ):
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
            return  # Don't recurse into function body

    # Haskell: function definitions use "function" with "variable" child for name
    if language == "haskell" and node.type == "function":
        # Get the first variable child (the function name)
        for child in node.children:
            if child.type == "variable":
                name = child.text.decode()
                results.append(Definition(
                    name=name,
                    kind="function",
                    line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file=file,
                ))
                break
        return  # Don't recurse into function body
    
    # Lua: function_declaration with name child
    if language == "lua" and node.type == "function_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = name_node.text.decode()
            results.append(Definition(
                name=name,
                kind="function",
                line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                file=file,
            ))
        return

    # Recurse
    for child in node.children:
        _extract_definitions(child, language, results, file, current_class)


# Cache for parsed AST indexes: {(path, mtime): ASTIndex}
# Using OrderedDict for efficient LRU eviction
_index_cache: OrderedDict[tuple[str, float], ASTIndex] = OrderedDict()
_MAX_CACHE_SIZE = 500  # Max files to cache


def _get_cache_key(path: Path) -> tuple[str, float] | None:
    """Get cache key for a file (path + mtime)."""
    try:
        return (str(path.resolve()), path.stat().st_mtime)
    except OSError:
        return None


def index_file(path: Path | str, use_cache: bool = True) -> ASTIndex:
    """Index a single file using tree-sitter.
    
    Args:
        path: Path to the file to index
        use_cache: Whether to use cached results (default: True)
        
    Returns:
        ASTIndex with definitions found in the file
    """
    path = Path(path)
    suffix = path.suffix.lower()
    language = LANGUAGE_MAP.get(suffix)

    if not language:
        return ASTIndex()

    # Check cache
    if use_cache:
        cache_key = _get_cache_key(path)
        if cache_key and cache_key in _index_cache:
            # Move to end for LRU (most recently used)
            _index_cache.move_to_end(cache_key)
            return _index_cache[cache_key]

    parser = _get_parser(language)
    if not parser:
        return ASTIndex()

    try:
        # Skip files larger than 1MB to prevent OOM
        if path.stat().st_size > 1_000_000:
            logger.debug("Skipping large file: %s", path)
            return ASTIndex()
        
        code = path.read_text(encoding='utf-8', errors='replace')
        tree = parser.parse(bytes(code, "utf8"))

        definitions: list[Definition] = []
        _extract_definitions(tree.root_node, language, definitions, str(path))

        result = ASTIndex(definitions=definitions)
        
        # Store in cache
        if use_cache:
            cache_key = _get_cache_key(path)
            if cache_key:
                # Evict oldest entries if cache is full (LRU)
                while len(_index_cache) >= _MAX_CACHE_SIZE:
                    _index_cache.popitem(last=False)  # Remove oldest (FIFO order)
                _index_cache[cache_key] = result
        
        return result
    except Exception as e:
        logger.warning(f"Failed to index {path}: {e}")
        return ASTIndex()


def index_files(paths: list[Path | str], use_cache: bool = True) -> ASTIndex:
    """Index multiple files.
    
    Args:
        paths: List of file paths to index
        use_cache: Whether to use cached results (default: True)
        
    Returns:
        Combined ASTIndex with all definitions
    """
    all_defs: list[Definition] = []
    for path in paths:
        idx = index_file(path, use_cache=use_cache)
        all_defs.extend(idx.definitions)
    return ASTIndex(definitions=all_defs)


def clear_index_cache() -> int:
    """Clear the AST index cache.
    
    Returns:
        Number of entries cleared
    """
    global _index_cache
    count = len(_index_cache)
    _index_cache = OrderedDict()  # Must stay OrderedDict for LRU eviction
    return count


def get_cache_stats() -> dict[str, int]:
    """Get cache statistics.
    
    Returns:
        Dict with cache size and max size
    """
    return {
        "size": len(_index_cache),
        "max_size": _MAX_CACHE_SIZE,
    }

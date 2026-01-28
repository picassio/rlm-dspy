"""Syntax-aware chunking using tree-sitter.

This module provides intelligent code chunking that respects syntax boundaries,
preventing false positives from truncated code in LLM analysis.

Key features:
- Chunks at function/class boundaries (never mid-definition)
- Falls back to character-based chunking for non-code
- Supports Python, TypeScript, JavaScript, Go, Rust, and more
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Tree-sitter is REQUIRED for syntax-aware chunking
_PARSERS: dict[str, "Parser"] = {}
_PARSERS_LOCK = threading.Lock()
_PARSE_LOCKS: dict[str, threading.Lock] = {}  # Per-language locks for thread-safe parsing

try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.error(
        "tree-sitter is required but not installed. "
        "Install with: pip install tree-sitter tree-sitter-python"
    )
    # Don't raise here - allow graceful degradation for edge cases
    # But log prominently so users know something is wrong

# Language module mappings
LANGUAGE_MODULES = {
    "python": "tree_sitter_python",
    "typescript": "tree_sitter_typescript",
    "javascript": "tree_sitter_javascript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "ruby": "tree_sitter_ruby",
    "php": "tree_sitter_php",
    "csharp": "tree_sitter_c_sharp",
    "kotlin": "tree_sitter_kotlin",
    "lua": "tree_sitter_lua",
}

# Top-level node types that define boundaries (by language)
BOUNDARY_NODES = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "typescript": {"function_declaration", "class_declaration", "interface_declaration",
                   "type_alias_declaration", "export_statement", "lexical_declaration"},
    "javascript": {"function_declaration", "class_declaration", "export_statement",
                   "lexical_declaration", "variable_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item", "mod_item"},
    "java": {"class_declaration", "interface_declaration", "method_declaration"},
    "c": {"function_definition", "struct_specifier", "declaration"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier", "namespace_definition"},
    "ruby": {"method", "class", "module"},
    "php": {"function_definition", "class_declaration", "method_declaration"},
    "csharp": {"class_declaration", "method_declaration", "namespace_declaration"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration"},
    "lua": {"function_definition_statement", "local_function_definition_statement"},
}

# Import/include node types (by language)
IMPORT_NODES = {
    "python": {"import_statement", "import_from_statement"},
    "typescript": {"import_statement", "import_clause"},
    "javascript": {"import_statement"},
    "go": {"import_declaration", "import_spec"},
    "rust": {"use_declaration", "extern_crate_declaration"},
    "java": {"import_declaration", "package_declaration"},
    "c": {"preproc_include"},
    "cpp": {"preproc_include", "using_declaration"},
    "ruby": {"require_statement", "require_relative_statement"},
    "php": {"namespace_use_declaration", "require_expression", "include_expression"},
    "csharp": {"using_directive"},
    "kotlin": {"import_header", "import_list"},
    "lua": {"require_statement"},
}


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""
    content: str
    start_line: int
    end_line: int
    node_type: str | None = None  # e.g., "function_definition", "class_definition"
    name: str | None = None  # e.g., function/class name


def _get_parser(language: str) -> tuple["Parser | None", "threading.Lock | None"]:
    """Get or create a tree-sitter parser for the language (thread-safe).

    Returns:
        Tuple of (parser, parse_lock) - both None if language not supported.
        The parse_lock MUST be held during parser.parse() calls.
    """
    if not TREE_SITTER_AVAILABLE:
        return None, None

    # Fast path: check without lock
    if language in _PARSERS:
        return _PARSERS[language], _PARSE_LOCKS[language]

    module_name = LANGUAGE_MODULES.get(language)
    if not module_name:
        return None, None

    # Slow path: acquire lock and create parser
    with _PARSERS_LOCK:
        # Double-check after acquiring lock
        if language in _PARSERS:
            return _PARSERS[language], _PARSE_LOCKS[language]

        try:
            import importlib
            lang_module = importlib.import_module(module_name)

            # Handle typescript special case (has separate ts/tsx)
            if language == "typescript":
                lang = Language(lang_module.language_typescript())
            elif hasattr(lang_module, f"language_{language}"):
                lang = Language(getattr(lang_module, f"language_{language}")())
            elif hasattr(lang_module, "language"):
                lang = Language(lang_module.language())
            else:
                return None, None

            parser = Parser(lang)
            parse_lock = threading.Lock()
            _PARSERS[language] = parser
            _PARSE_LOCKS[language] = parse_lock
            return parser, parse_lock
        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"Failed to load tree-sitter for {language}: {e}")
            return None, None


def _detect_language(content: str, filename: str | None = None) -> str | None:
    """Detect programming language from content or filename."""
    if filename:
        ext_map = {
            ".py": "python",
            ".ts": "typescript", ".tsx": "typescript",
            ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c", ".h": "c",
            ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".kt": "kotlin", ".kts": "kotlin",
            ".lua": "lua",
        }
        ext = Path(filename).suffix.lower()
        if ext in ext_map:
            return ext_map[ext]

    # Heuristic detection from content
    if "def " in content and ("import " in content or "from " in content):
        return "python"
    if "func " in content and "package " in content:
        return "go"
    if "fn " in content and ("let " in content or "mut " in content):
        return "rust"
    if "function " in content or "const " in content or "=>" in content:
        if "interface " in content or ": " in content:
            return "typescript"
        return "javascript"

    return None


def _extract_name(node, source_bytes: bytes) -> str | None:
    """Extract the name from a definition node."""
    # Look for name/identifier child
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        # For decorated definitions, recurse
        if child.type in ("function_definition", "class_definition"):
            return _extract_name(child, source_bytes)
    return None


@dataclass
class ParseResult:
    """Result of parsing code for boundaries and imports."""
    boundaries: list[tuple[int, int, str, str | None]]  # (start_byte, end_byte, node_type, name)
    imports: list[str]


def _parse_code(content: str, language: str) -> ParseResult:
    """Parse code once to extract both boundaries and imports (single tree traversal)."""
    parser, parse_lock = _get_parser(language)
    if not parser or not parse_lock:
        return ParseResult(boundaries=[], imports=[])

    source_bytes = content.encode("utf-8")

    # Parser.parse() is not thread-safe, must hold lock
    with parse_lock:
        tree = parser.parse(source_bytes)

    boundaries: list[tuple[int, int, str, str | None]] = []
    imports: list[str] = []

    boundary_types = BOUNDARY_NODES.get(language, set())
    import_types = IMPORT_NODES.get(language, set())

    def visit(node):
        # Check for boundaries (functions, classes, etc.)
        if node.type in boundary_types:
            name = _extract_name(node, source_bytes)
            boundaries.append((node.start_byte, node.end_byte, node.type, name))
        # Check for imports
        elif node.type in import_types:
            import_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
            if import_text and import_text not in imports:
                imports.append(import_text)
        # Recurse into children (but not into boundary nodes - they're self-contained)
        if node.type not in boundary_types:
            for child in node.children:
                visit(child)

    visit(tree.root_node)
    return ParseResult(
        boundaries=sorted(boundaries, key=lambda x: x[0]),
        imports=imports,
    )


def _find_boundaries(content: str, language: str) -> list[tuple[int, int, str, str | None]]:
    """Find syntax boundaries (start_byte, end_byte, node_type, name)."""
    return _parse_code(content, language).boundaries


def extract_imports(content: str, language: str | None = None) -> str:
    """Extract import/include statements from code.

    This is used to create a "preamble" for each chunk so the LLM
    knows what's available in the namespace.

    Args:
        content: Source code
        language: Programming language (auto-detected if None)

    Returns:
        Import statements as a string (empty if none found)
    """
    if language is None:
        language = _detect_language(content)
    if not language:
        return ""

    return "\n".join(_parse_code(content, language).imports)


def chunk_code_syntax_aware(
    content: str,
    chunk_size: int = 100_000,
    overlap: int = 500,
    language: str | None = None,
    filename: str | None = None,
    include_imports: bool = True,
) -> list[CodeChunk]:
    """Chunk code respecting syntax boundaries.

    Args:
        content: Source code to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks (for context)
        language: Programming language (auto-detected if None)
        filename: Filename for language detection
        include_imports: If True, prepend imports to each chunk as preamble

    Returns:
        List of CodeChunk objects
    """
    if not content.strip():
        return []

    # Detect language if not provided
    if language is None:
        language = _detect_language(content, filename)

    # If no language detected or no tree-sitter, fall back to character chunking
    if language is None or not TREE_SITTER_AVAILABLE:
        if not TREE_SITTER_AVAILABLE:
            logger.warning(
                "tree-sitter not available, using character-based chunking. "
                "Install with: pip install tree-sitter tree-sitter-python"
            )
        return _chunk_by_characters(content, chunk_size, overlap)

    # Parse once to get both boundaries and imports (single tree traversal)
    parse_result = _parse_code(content, language)
    boundaries = parse_result.boundaries

    # Build imports preamble
    imports_preamble = ""
    if include_imports and parse_result.imports:
        imports_preamble = f"# File imports:\n{chr(10).join(parse_result.imports)}\n\n# Code:\n"

    if not boundaries:
        # No boundaries found, fall back to character chunking
        char_chunks = _chunk_by_characters(content, chunk_size, overlap)
        # Add preamble to all chunks
        if imports_preamble and char_chunks:
            char_chunks = [
                CodeChunk(
                    content=imports_preamble + c.content,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    node_type=c.node_type,
                    name=c.name,
                )
                for c in char_chunks
            ]
        return char_chunks

    # Build chunks respecting boundaries
    chunks = []
    current_start = 0
    current_end = 0
    current_nodes: list[tuple[str, str | None]] = []  # (type, name)

    # Encode content to bytes for correct slicing (tree-sitter uses byte offsets)
    content_bytes = content.encode("utf-8")
    lines = content.split("\n")

    # Precompute line byte offsets for O(1) lookup via binary search
    line_offsets = [0]
    offset = 0
    for line in lines:
        offset += len(line.encode("utf-8")) + 1  # +1 for newline
        line_offsets.append(offset)

    def byte_to_line(byte_offset: int) -> int:
        """Convert byte offset to line number using binary search."""
        import bisect
        idx = bisect.bisect_right(line_offsets, byte_offset)
        return max(1, idx)

    def slice_content(start: int, end: int) -> str:
        """Slice content using byte offsets, decode to string."""
        return content_bytes[start:end].decode("utf-8", errors="replace")

    for start_byte, end_byte, node_type, name in boundaries:
        # If this is the first boundary, set start
        if current_start == 0 and not current_nodes:
            current_start = start_byte

        node_size = end_byte - start_byte
        chunk_size_so_far = current_end - current_start if current_end > current_start else 0

        # If adding this node would exceed chunk_size, flush current chunk
        if chunk_size_so_far + node_size > chunk_size and current_nodes:
            chunk_content = slice_content(current_start, current_end)
            if chunk_content.strip():
                chunks.append(CodeChunk(
                    content=chunk_content,
                    start_line=byte_to_line(current_start),
                    end_line=byte_to_line(current_end),
                    node_type=current_nodes[0][0] if len(current_nodes) == 1 else "multiple",
                    name=current_nodes[0][1] if len(current_nodes) == 1 else None,
                ))
            # Start new chunk at this boundary (no overlap for syntax-aware)
            current_start = start_byte
            current_nodes = []

        # Add this node to current chunk
        current_end = end_byte
        current_nodes.append((node_type, name))

        # If this single node exceeds chunk_size, it goes in its own chunk
        if node_size > chunk_size:
            chunk_content = slice_content(start_byte, end_byte)
            if chunk_content.strip():
                chunks.append(CodeChunk(
                    content=chunk_content,
                    start_line=byte_to_line(start_byte),
                    end_line=byte_to_line(end_byte),
                    node_type=node_type,
                    name=name,
                ))
            current_start = end_byte
            current_end = end_byte
            current_nodes = []

    # Flush remaining content
    if current_start < len(content_bytes):
        remaining = slice_content(current_start, len(content_bytes))
        if remaining.strip():
            chunks.append(CodeChunk(
                content=remaining,
                start_line=byte_to_line(current_start),
                end_line=len(lines),
                node_type=current_nodes[0][0] if len(current_nodes) == 1 else "multiple" if current_nodes else None,
                name=current_nodes[0][1] if len(current_nodes) == 1 else None,
            ))

    result_chunks = chunks if chunks else _chunk_by_characters(content, chunk_size, overlap)

    # Add imports preamble to each chunk (helps LLM understand available symbols)
    if imports_preamble and result_chunks:
        result_chunks = [
            CodeChunk(
                content=imports_preamble + chunk.content,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                node_type=chunk.node_type,
                name=chunk.name,
            )
            for chunk in result_chunks
        ]

    return result_chunks


def _chunk_by_characters(content: str, chunk_size: int, overlap: int) -> list[CodeChunk]:
    """Fallback: chunk by character count with line-aware boundaries."""
    chunks = []
    lines = content.split("\n")

    current_chunk_lines = []
    current_size = 0
    start_line = 1

    for i, line in enumerate(lines, 1):
        line_size = len(line) + 1  # +1 for newline

        if current_size + line_size > chunk_size and current_chunk_lines:
            # Flush current chunk
            chunk_content = "\n".join(current_chunk_lines)
            if chunk_content.strip():
                chunks.append(CodeChunk(
                    content=chunk_content,
                    start_line=start_line,
                    end_line=i - 1,
                ))

            # Calculate overlap (keep last N characters worth of lines)
            overlap_lines: list[str] = []
            overlap_size = 0
            for prev_line in reversed(current_chunk_lines):
                line_len = len(prev_line) + 1
                # If this line alone exceeds overlap but we have nothing yet,
                # still include it to ensure at least some context
                if overlap_size + line_len > overlap and overlap_lines:
                    break
                overlap_lines.append(prev_line)
                overlap_size += line_len
                # Stop if we've reached the overlap target
                if overlap_size >= overlap:
                    break
            overlap_lines.reverse()  # O(n) instead of O(n²) from insert(0,...)

            current_chunk_lines = overlap_lines
            current_size = overlap_size
            start_line = i - len(overlap_lines)

        current_chunk_lines.append(line)
        current_size += line_size

    # Flush remaining
    if current_chunk_lines:
        chunk_content = "\n".join(current_chunk_lines)
        if chunk_content.strip():
            chunks.append(CodeChunk(
                content=chunk_content,
                start_line=start_line,
                end_line=len(lines),
            ))

    return chunks


def chunk_mixed_content(
    content: str,
    chunk_size: int = 100_000,
    overlap: int = 500,
) -> list[CodeChunk]:
    """Chunk content that may contain multiple code blocks.

    Identifies code blocks (```language...```) and chunks them syntax-aware,
    while chunking prose sections by character.
    """
    import re

    # Find code blocks
    code_block_pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

    chunks = []
    last_end = 0

    for match in code_block_pattern.finditer(content):
        # Process prose before this code block
        prose = content[last_end:match.start()]
        if prose.strip():
            chunks.extend(_chunk_by_characters(prose, chunk_size, overlap))

        # Process code block
        language = match.group(1)
        code = match.group(2)
        if code.strip():
            code_chunks = chunk_code_syntax_aware(code, chunk_size, overlap, language)
            chunks.extend(code_chunks)

        last_end = match.end()

    # Process remaining prose
    if last_end < len(content):
        prose = content[last_end:]
        if prose.strip():
            chunks.extend(_chunk_by_characters(prose, chunk_size, overlap))

    return chunks


# Export main functions
__all__ = [
    "CodeChunk",
    "chunk_code_syntax_aware",
    "chunk_mixed_content",
    "TREE_SITTER_AVAILABLE",
]

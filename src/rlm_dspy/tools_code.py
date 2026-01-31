"""Code analysis tools for RLM - AST indexing, file reading, stats."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Import shared utilities
from .tools import _is_safe_path, _resolve_project_path


def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """Read a file or portion of a file.

    Args:
        path: Path to the file
        start_line: First line to read (1-indexed)
        end_line: Last line to read (None for end of file)

    Returns:
        File contents with line numbers
    """
    try:
        path = _resolve_project_path(path)
        is_safe, error = _is_safe_path(path)
        if not is_safe:
            return f"(security: {error})"

        p = Path(path)
        if not p.exists():
            return f"(file not found: {path})"
        if not p.is_file():
            return f"(not a file: {path})"

        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line else len(lines)

        selected = lines[start_idx:end_idx]
        numbered = [f"{start_line + i:4d} | {line}" for i, line in enumerate(selected)]
        return '\n'.join(numbered) if numbered else "(empty file)"
    except Exception as e:
        return f"(read error: {e})"


def file_stats(path: str) -> str:
    """Get statistics about a file or directory.

    Args:
        path: Path to analyze

    Returns:
        JSON with file count, line count, size info
    """
    try:
        path = _resolve_project_path(path)
        is_safe, error = _is_safe_path(path)
        if not is_safe:
            return f"(security: {error})"

        p = Path(path)
        if not p.exists():
            return f"(path not found: {path})"

        if p.is_file():
            content = p.read_text(encoding="utf-8", errors="replace")
            return json.dumps({
                "type": "file",
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "line_count": len(content.splitlines()),
                "char_count": len(content),
            })

        files = list(p.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        dir_count = sum(1 for f in files if f.is_dir())

        code_extensions = {'.py', '.js', '.ts', '.go', '.rs', '.java', '.c', '.cpp', '.rb'}
        code_files = [f for f in files if f.is_file() and f.suffix in code_extensions]
        total_lines = 0
        for f in code_files[:100]:
            try:
                total_lines += len(f.read_text(encoding="utf-8", errors="replace").splitlines())
            except Exception:
                pass

        return json.dumps({
            "type": "directory",
            "path": str(p),
            "file_count": file_count,
            "dir_count": dir_count,
            "code_files": len(code_files),
            "total_code_lines": total_lines,
        })
    except Exception as e:
        return f"(stats error: {e})"


def index_code(path: str, kind: str | None = None, name: str | None = None) -> str:
    """Index code to find classes, functions, and methods with EXACT line numbers.

    Uses tree-sitter for 100% accurate structural analysis.

    Args:
        path: File or directory to index
        kind: Filter by type: "class", "function", "method", or None for all
        name: Filter by name (case-insensitive substring match)

    Returns:
        List of definitions with file:line and type
    """
    try:
        path = _resolve_project_path(path)
        is_safe, error = _is_safe_path(path)
        if not is_safe:
            return f"(security: {error})"

        from .core.ast_index import index_file, index_files, LANGUAGE_MAP

        p = Path(path)
        if p.is_file():
            idx = index_file(p)
        else:
            files = []
            for ext in LANGUAGE_MAP.keys():
                files.extend(p.rglob(f"*{ext}"))
            files = files[:200]
            idx = index_files(files)

        # Filter definitions manually (ASTIndex doesn't have filter method)
        definitions = idx.definitions
        if kind:
            definitions = [d for d in definitions if d.kind == kind]
        if name:
            definitions = [d for d in definitions if name.lower() in d.name.lower()]

        if not definitions:
            return "(no definitions found)"

        output = []
        for d in definitions[:100]:
            parent_info = f" ({d.parent})" if d.parent else ""
            output.append(f"{d.file}:{d.line} | {d.kind}: {d.name}{parent_info}")

        if len(definitions) > 100:
            output.append(f"... ({len(definitions) - 100} more)")

        return '\n'.join(output)
    except Exception as e:
        return f"(index error: {e})"


def find_definitions(path: str, name: str | None = None) -> str:
    """Find function and class definitions."""
    return index_code(path, name=name)


def find_classes(path: str, name: str | None = None) -> str:
    """Find all class definitions."""
    return index_code(path, kind="class", name=name)


def find_functions(path: str, name: str | None = None) -> str:
    """Find all function definitions (not methods)."""
    return index_code(path, kind="function", name=name)


def find_methods(path: str, name: str | None = None) -> str:
    """Find all method definitions."""
    return index_code(path, kind="method", name=name)


def find_imports(path: str) -> str:
    """Find all imports in source files."""
    from .tools_search import ripgrep
    return ripgrep(r"^(import |from .+ import |require\(|use |using )", path, "-n")


def find_calls(path: str, function_name: str) -> str:
    """Find all calls to a specific function or method."""
    import re
    from .tools_search import ripgrep
    pattern = rf"\b{re.escape(function_name)}\s*\("
    return ripgrep(pattern, path, "-n")


def find_usages(file_path: str, symbol_name: str | None = None) -> str:
    """Find all usages of symbols defined in a file.

    Args:
        file_path: Path to the file containing symbol definitions
        symbol_name: Optional specific symbol to find usages for

    Returns:
        Usage report showing where each symbol is used
    """
    from .core.ast_index import index_file
    from .tools_search import ripgrep

    file_path = _resolve_project_path(file_path)
    if not Path(file_path).exists():
        return f"(file not found: {file_path})"

    ast_index = index_file(str(file_path))
    if not ast_index or not ast_index.definitions:
        return f"(no symbols found in {file_path})"

    symbols = [
        {"name": d.name, "kind": d.kind, "line": d.line, "parent": d.parent}
        for d in ast_index.definitions
    ]

    if symbol_name:
        filtered = [s for s in symbols if s["name"] == symbol_name]
        if not filtered:
            available = ", ".join(s["name"] for s in symbols[:10])
            return f"(symbol '{symbol_name}' not found. Available: {available})"
        symbols = filtered

    search_root = _resolve_project_path(".")
    output = []
    
    for symbol in symbols[:20]:
        name = symbol["name"]
        if len(name) < 2 or name.startswith("_"):
            continue

        result = ripgrep(rf"\b{name}\b", search_root, "-l")
        if "(no matches)" in result or "(error)" in result.lower():
            continue

        files = [f for f in result.strip().split('\n') if f and "(no matches)" not in f]
        internal = [f for f in files if f.startswith(str(Path(file_path).parent))]
        external = [f for f in files if f not in internal]

        if internal or external:
            output.append(f"\n=== {symbol['kind']}: {name} (defined at line {symbol['line']}) ===")
            if internal:
                output.append(f"  Internal ({len(internal)}): {', '.join(Path(f).name for f in internal[:5])}")
            if external:
                output.append(f"  External ({len(external)}): {', '.join(Path(f).name for f in external[:5])}")

    return '\n'.join(output) if output else "(no usages found)"

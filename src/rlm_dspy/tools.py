"""Built-in tools for RLM code analysis.

These tools run on the host (not in the sandbox) and are available
to the LLM via the `tools` parameter of dspy.RLM.

Usage:
    from rlm_dspy import RLM
    from rlm_dspy.tools import BUILTIN_TOOLS
    
    rlm = RLM(tools=BUILTIN_TOOLS)
    result = rlm.query("Find all TODO comments", context)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ripgrep(pattern: str, path: str = ".", flags: str = "") -> str:
    """
    Search for pattern using ripgrep (rg).
    
    Args:
        pattern: Regex pattern to search for
        path: Path to search in (file or directory)
        flags: Additional rg flags like "-i" for case-insensitive, "-l" for files-only
        
    Returns:
        Search results as text, or error message
        
    Example:
        ripgrep("TODO|FIXME", "src/", "-i")  # Case-insensitive search for TODOs
        ripgrep("def\\s+\\w+", ".", "-l")    # List files with function definitions
    """
    try:
        cmd = ["rg", "--color=never", "--line-number"]
        if flags:
            cmd.extend(flags.split())
        cmd.extend([pattern, path])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            return result.stdout[:50000]  # Limit output size
        elif result.returncode == 1:
            return "(no matches found)"
        else:
            return f"(ripgrep error: {result.stderr})"
    except FileNotFoundError:
        return "(ripgrep not installed - install with: cargo install ripgrep)"
    except subprocess.TimeoutExpired:
        return "(ripgrep timed out after 30s)"
    except Exception as e:
        return f"(ripgrep error: {e})"


def grep_context(pattern: str, path: str = ".", context_lines: int = 3) -> str:
    """
    Search with context lines around matches.
    
    Args:
        pattern: Regex pattern to search for
        path: Path to search in
        context_lines: Number of lines before/after each match
        
    Returns:
        Matches with surrounding context
    """
    return ripgrep(pattern, path, f"-C {context_lines}")


def find_files(pattern: str, path: str = ".", file_type: str = "") -> str:
    """
    Find files matching a glob pattern.
    
    Args:
        pattern: Glob pattern (e.g., "*.py", "test_*.py")
        path: Directory to search
        file_type: Filter by extension (e.g., "py", "js")
        
    Returns:
        List of matching file paths
    """
    try:
        cmd = ["rg", "--files", "--color=never"]
        if file_type:
            cmd.extend(["-t", file_type])
        if pattern and pattern != "*":
            cmd.extend(["-g", pattern])
        cmd.append(path)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            files = result.stdout.strip().split("\n")
            return "\n".join(files[:500])  # Limit to 500 files
        else:
            return f"(find_files error: {result.stderr})"
    except FileNotFoundError:
        return "(ripgrep not installed)"
    except Exception as e:
        return f"(find_files error: {e})"


def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """
    Read a file or portion of a file.
    
    Args:
        path: Path to the file
        start_line: First line to read (1-indexed)
        end_line: Last line to read (None for end of file)
        
    Returns:
        File contents with line numbers
    """
    try:
        p = Path(path)
        if not p.exists():
            return f"(file not found: {path})"
        if not p.is_file():
            return f"(not a file: {path})"
        
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        
        # Adjust indices (1-indexed to 0-indexed)
        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line else len(lines)
        
        selected = lines[start_idx:end_idx]
        
        # Format with line numbers
        numbered = []
        for i, line in enumerate(selected, start=start_idx + 1):
            numbered.append(f"{i:4d} | {line}")
        
        return "\n".join(numbered[:2000])  # Limit to 2000 lines
    except Exception as e:
        return f"(read_file error: {e})"


def file_stats(path: str) -> str:
    """
    Get statistics about a file or directory.
    
    Args:
        path: Path to analyze
        
    Returns:
        JSON with file count, line count, size info
    """
    try:
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
        
        # Directory stats
        files = list(p.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        
        # Count by extension
        extensions: dict[str, int] = {}
        total_lines = 0
        for f in files:
            if f.is_file():
                ext = f.suffix or "(no ext)"
                extensions[ext] = extensions.get(ext, 0) + 1
                try:
                    total_lines += len(f.read_text(errors="replace").splitlines())
                except:
                    pass
        
        return json.dumps({
            "type": "directory",
            "path": str(p),
            "file_count": file_count,
            "total_lines": total_lines,
            "by_extension": dict(sorted(extensions.items(), key=lambda x: -x[1])[:20]),
        })
    except Exception as e:
        return f"(file_stats error: {e})"


def _find_nodes(node: Any, types: set[str]) -> list[Any]:
    """Recursively find nodes of given types in AST."""
    results = []
    if node.type in types:
        results.append(node)
    for child in node.children:
        results.extend(_find_nodes(child, types))
    return results


def _get_child_by_type(node: Any, child_type: str) -> Any | None:
    """Get first child of a given type."""
    for child in node.children:
        if child.type == child_type:
            return child
    return None


def ast_query(code: str, node_types: str, language: str = "python") -> str:
    """
    Find AST nodes of specific types in code.
    
    Args:
        code: Source code to analyze
        node_types: Comma-separated node types to find (e.g., "function_definition,class_definition")
        language: Programming language ("python" only currently)
        
    Returns:
        Matched AST nodes with locations
        
    Common node types:
        - function_definition: Function definitions
        - class_definition: Class definitions  
        - call: Function calls
        - import_statement, import_from_statement: Imports
        - assignment: Variable assignments
        - if_statement, for_statement, while_statement: Control flow
    """
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        
        if language != "python":
            return f"(ast_query: only 'python' language supported, got '{language}')"
        
        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)
        
        tree = parser.parse(bytes(code, "utf8"))
        
        types = {t.strip() for t in node_types.split(",")}
        nodes = _find_nodes(tree.root_node, types)
        
        results = []
        for node in nodes[:200]:
            start = node.start_point
            text = code[node.start_byte:node.end_byte]
            # Truncate long text
            if len(text) > 100:
                text = text[:97] + "..."
            text = text.replace("\n", "\\n")
            results.append(f"[{start[0]+1}:{start[1]}] {node.type}: {text}")
        
        if not results:
            return "(no matches)"
        
        return "\n".join(results)
    except ImportError:
        return "(tree-sitter not installed - pip install tree-sitter tree-sitter-python)"
    except Exception as e:
        return f"(ast_query error: {e})"


def find_definitions(path: str, name: str | None = None) -> str:
    """
    Find function and class definitions in Python files.
    
    Args:
        path: File or directory to search
        name: Optional name filter (regex)
        
    Returns:
        List of definitions with locations
    """
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        
        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)
        
        p = Path(path)
        files = [p] if p.is_file() else list(p.rglob("*.py"))
        
        results = []
        name_re = re.compile(name) if name else None
        
        for f in files[:100]:  # Limit files
            try:
                code = f.read_text(encoding="utf-8", errors="replace")
                tree = parser.parse(bytes(code, "utf8"))
                
                # Find function and class definitions
                nodes = _find_nodes(tree.root_node, {"function_definition", "class_definition"})
                
                for node in nodes:
                    # Get name from identifier child
                    name_node = _get_child_by_type(node, "identifier")
                    if not name_node:
                        continue
                    
                    def_name = code[name_node.start_byte:name_node.end_byte]
                    if name_re and not name_re.search(def_name):
                        continue
                    
                    line = node.start_point[0] + 1
                    kind = "function" if node.type == "function_definition" else "class"
                    results.append(f"{f}:{line}: {kind} {def_name}")
            except:
                continue
        
        if not results:
            return "(no definitions found)"
        
        return "\n".join(results[:500])
    except ImportError:
        return "(tree-sitter not installed)"
    except Exception as e:
        return f"(find_definitions error: {e})"


def find_imports(path: str) -> str:
    """
    Find all imports in Python files.
    
    Args:
        path: File or directory to analyze
        
    Returns:
        List of imports with locations
    """
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        
        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)
        
        p = Path(path)
        files = [p] if p.is_file() else list(p.rglob("*.py"))
        
        results = []
        
        for f in files[:100]:
            try:
                code = f.read_text(encoding="utf-8", errors="replace")
                tree = parser.parse(bytes(code, "utf8"))
                
                nodes = _find_nodes(tree.root_node, {"import_statement", "import_from_statement"})
                
                for node in nodes:
                    line = node.start_point[0] + 1
                    text = code[node.start_byte:node.end_byte].split("\n")[0]
                    results.append(f"{f}:{line}: {text}")
            except:
                continue
        
        if not results:
            return "(no imports found)"
        
        return "\n".join(results[:500])
    except ImportError:
        return "(tree-sitter not installed)"
    except Exception as e:
        return f"(find_imports error: {e})"


def find_calls(path: str, function_name: str) -> str:
    """
    Find all calls to a specific function or method.
    
    Args:
        path: File or directory to search
        function_name: Name of function/method to find calls to
        
    Returns:
        List of call sites with locations
        
    Examples:
        find_calls("src/", "print")     # Find print() calls
        find_calls("src/", "append")    # Find .append() method calls
        find_calls("src/", "query")     # Find query() or .query() calls
    """
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        
        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)
        
        p = Path(path)
        files = [p] if p.is_file() else list(p.rglob("*.py"))
        
        results = []
        
        for f in files[:100]:
            try:
                code = f.read_text(encoding="utf-8", errors="replace")
                tree = parser.parse(bytes(code, "utf8"))
                
                # Find all call nodes
                call_nodes = _find_nodes(tree.root_node, {"call"})
                
                for call_node in call_nodes:
                    # Get the function being called - could be:
                    # 1. Direct call: func() -> identifier child
                    # 2. Method call: obj.method() -> attribute child with identifier
                    func_node = _get_child_by_type(call_node, "identifier")
                    
                    if not func_node:
                        # Check for attribute access (obj.method)
                        attr_node = _get_child_by_type(call_node, "attribute")
                        if attr_node:
                            # Get the last identifier (method name)
                            identifiers = [c for c in attr_node.children if c.type == "identifier"]
                            if identifiers:
                                func_node = identifiers[-1]  # Last one is the method name
                    
                    if not func_node:
                        continue
                    
                    name = code[func_node.start_byte:func_node.end_byte]
                    if name != function_name:
                        continue
                    
                    line = call_node.start_point[0] + 1
                    call_text = code[call_node.start_byte:call_node.end_byte]
                    # Truncate long calls
                    if len(call_text) > 80:
                        call_text = call_text[:77] + "..."
                    call_text = call_text.replace("\n", " ")
                    results.append(f"{f}:{line}: {call_text}")
            except:
                continue
        
        if not results:
            return f"(no calls to '{function_name}' found)"
        
        return "\n".join(results[:500])
    except ImportError:
        return "(tree-sitter not installed)"
    except Exception as e:
        return f"(find_calls error: {e})"


def shell(command: str, timeout: int = 30) -> str:
    """
    Run a shell command (use with caution).
    
    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Command output (stdout + stderr)
        
    Note:
        This tool is disabled by default for security.
        Enable by setting RLM_ALLOW_SHELL=1.
    """
    import os
    if not os.environ.get("RLM_ALLOW_SHELL"):
        return "(shell disabled - set RLM_ALLOW_SHELL=1 to enable)"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"(command timed out after {timeout}s)"
    except Exception as e:
        return f"(shell error: {e})"


# Collection of all built-in tools
BUILTIN_TOOLS: dict[str, Any] = {
    "ripgrep": ripgrep,
    "grep_context": grep_context,
    "find_files": find_files,
    "read_file": read_file,
    "file_stats": file_stats,
    "ast_query": ast_query,
    "find_definitions": find_definitions,
    "find_imports": find_imports,
    "find_calls": find_calls,
    "shell": shell,
}

# Safer subset without shell
SAFE_TOOLS: dict[str, Any] = {
    k: v for k, v in BUILTIN_TOOLS.items() if k != "shell"
}


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all tools for the LLM prompt."""
    lines = ["Available tools:"]
    for name, func in BUILTIN_TOOLS.items():
        doc = func.__doc__ or "No description"
        # Get first line of docstring
        first_line = doc.strip().split("\n")[0]
        lines.append(f"  - {name}: {first_line}")
    return "\n".join(lines)

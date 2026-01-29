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

# Restricted paths that tools should not access
_RESTRICTED_PATHS = {
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    "~/.ssh", "~/.gnupg", "~/.aws/credentials",
}


def _is_safe_path(path: str, base_dir: str | None = None) -> tuple[bool, str]:
    """Check if a path is safe to access.
    
    Args:
        path: Path to check
        base_dir: Optional base directory to restrict access to.
                  If provided, path must resolve to within base_dir.
    
    Returns:
        (is_safe, error_message)
    """
    try:
        # Resolve to absolute path (resolves symlinks and ..)
        resolved = Path(path).expanduser().resolve()
        resolved_str = str(resolved)
        
        # Check for restricted system paths
        for restricted in _RESTRICTED_PATHS:
            restricted_resolved = str(Path(restricted).expanduser().resolve())
            if resolved_str.startswith(restricted_resolved):
                return False, f"Access to {restricted} is restricted"
        
        # Additional system path restrictions
        system_paths = ['/etc', '/var', '/usr', '/bin', '/sbin', '/boot', '/root', '/proc', '/sys']
        for sys_path in system_paths:
            if resolved_str.startswith(sys_path):
                return False, f"Access to system path {sys_path} is restricted"
        
        # If base_dir provided, ensure path is within it
        if base_dir:
            base_resolved = Path(base_dir).expanduser().resolve()
            try:
                resolved.relative_to(base_resolved)
            except ValueError:
                return False, f"Path must be within {base_dir}"
        
        return True, ""
        
    except (OSError, ValueError) as e:
        return False, f"Invalid path: {e}"


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
        # Safety check
        is_safe, error = _is_safe_path(path)
        if not is_safe:
            return f"(security: {error})"
        
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
        # Safety check
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
        
        # Directory stats
        files = list(p.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        
        # Count by extension
        extensions: dict[str, int] = {}
        total_lines = 0
        skipped_files = 0
        for f in files:
            if f.is_file():
                ext = f.suffix or "(no ext)"
                extensions[ext] = extensions.get(ext, 0) + 1
                try:
                    total_lines += len(f.read_text(errors="replace").splitlines())
                except (OSError, UnicodeDecodeError):
                    # Skip binary files and unreadable files
                    skipped_files += 1
        
        return json.dumps({
            "type": "directory",
            "path": str(p),
            "file_count": file_count,
            "total_lines": total_lines,
            "by_extension": dict(sorted(extensions.items(), key=lambda x: -x[1])[:20]),
        })
    except Exception as e:
        return f"(file_stats error: {e})"


def index_code(path: str, kind: str | None = None, name: str | None = None) -> str:
    """
    Index code to find classes, functions, and methods with EXACT line numbers.
    
    Uses tree-sitter for 100% accurate structural analysis - NO hallucination.
    Supports: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, C#
    
    Args:
        path: File or directory to index
        kind: Filter by type: "class", "function", "method", or None for all
        name: Filter by name (case-insensitive substring match)
        
    Returns:
        List of definitions with file:line and type
        
    Examples:
        index_code("src/")                    # All definitions
        index_code("src/", kind="class")      # Only classes
        index_code("src/", name="query")      # Definitions containing 'query'
        index_code("file.py", kind="method")  # Methods in a single file
    """
    try:
        # Safety check
        is_safe, error = _is_safe_path(path)
        if not is_safe:
            return f"(security: {error})"
        
        from .core.ast_index import index_file, index_files, LANGUAGE_MAP
        
        p = Path(path)
        if p.is_file():
            idx = index_file(p)
        else:
            # Find all supported files
            files = []
            for ext in LANGUAGE_MAP.keys():
                files.extend(p.rglob(f"*{ext}"))
            files = files[:200]  # Limit
            idx = index_files(files)
        
        # Apply filters
        defs = idx.find(name=name, kind=kind)
        
        if not defs:
            return "(no definitions found)"
        
        results = []
        for d in defs[:500]:
            parent_info = f" (in {d.parent})" if d.parent else ""
            results.append(f"{d.file}:{d.line}-{d.end_line}: {d.kind} {d.name}{parent_info}")
        
        return "\n".join(results)
    except ImportError as e:
        return f"(tree-sitter not installed: {e})"
    except Exception as e:
        return f"(index_code error: {e})"


def find_definitions(path: str, name: str | None = None) -> str:
    """
    Find function and class definitions (wrapper for index_code).
    
    Supports: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, C#
    
    Args:
        path: File or directory to search
        name: Optional name filter (case-insensitive)
        
    Returns:
        List of definitions with file:line
    """
    return index_code(path, name=name)


def find_classes(path: str, name: str | None = None) -> str:
    """
    Find all class definitions.
    
    Args:
        path: File or directory to search  
        name: Optional name filter
        
    Returns:
        List of classes with file:line
    """
    return index_code(path, kind="class", name=name)


def find_functions(path: str, name: str | None = None) -> str:
    """
    Find all function definitions (not methods).
    
    Args:
        path: File or directory to search
        name: Optional name filter
        
    Returns:
        List of functions with file:line
    """
    return index_code(path, kind="function", name=name)


def find_methods(path: str, name: str | None = None) -> str:
    """
    Find all method definitions (functions inside classes).
    
    Args:
        path: File or directory to search
        name: Optional name filter
        
    Returns:
        List of methods with file:line and parent class
    """
    return index_code(path, kind="method", name=name)


def find_imports(path: str) -> str:
    """
    Find all imports in source files using ripgrep.
    
    Args:
        path: File or directory to analyze
        
    Returns:
        List of imports with locations
    """
    # Use ripgrep for imports - faster and works for all languages
    return ripgrep(r"^(import |from .+ import |require\(|use |using )", path, "-n")


def find_calls(path: str, function_name: str) -> str:
    """
    Find all calls to a specific function or method using ripgrep.
    
    Args:
        path: File or directory to search
        function_name: Name of function/method to find calls to
        
    Returns:
        List of call sites with locations
        
    Examples:
        find_calls("src/", "print")     # Find print() calls
        find_calls("src/", "query")     # Find query() calls
    """
    # Use ripgrep - matches function_name followed by (
    pattern = rf"\b{re.escape(function_name)}\s*\("
    return ripgrep(pattern, path, "-n")


def shell(command: str, timeout: int = 30) -> str:
    """
    Run a shell command (use with caution).
    
    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Command output (stdout + stderr)
        
    Security:
        - Disabled by default (requires RLM_ALLOW_SHELL=1)
        - Allowlist of safe commands enforced
        - Dangerous patterns blocked
        - Commands are logged for audit purposes
    """
    import os
    import shlex
    
    if not os.environ.get("RLM_ALLOW_SHELL"):
        return "(shell disabled - set RLM_ALLOW_SHELL=1 to enable)"
    
    # Allowlist of safe base commands
    ALLOWED_COMMANDS = {
        'ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc', 'sort', 'uniq',
        'echo', 'pwd', 'date', 'whoami', 'env', 'which', 'file', 'stat',
        'diff', 'tree', 'du', 'df', 'uname', 'hostname', 'ps', 'top',
        'git', 'python', 'python3', 'pip', 'node', 'npm', 'cargo', 'go',
    }
    
    # Parse command to get base command
    try:
        parts = shlex.split(command)
        if not parts:
            return "(empty command)"
        base_cmd = Path(parts[0]).name  # Get basename (e.g., /usr/bin/ls -> ls)
    except ValueError as e:
        return f"(invalid command syntax: {e})"
    
    # Check if command is allowed
    if base_cmd not in ALLOWED_COMMANDS:
        return f"(command '{base_cmd}' not in allowlist. Allowed: {', '.join(sorted(ALLOWED_COMMANDS)[:10])}...)"
    
    # Block dangerous patterns
    dangerous_patterns = [
        "rm -rf /", "rm -rf ~", "rm -rf .", "mkfs", "> /dev/", "dd if=",
        "chmod 777", "curl | sh", "wget | sh", "; rm", "&& rm",
        "eval ", "exec ", "$(",  "`", ">/dev/sd",
    ]
    cmd_lower = command.lower()
    for pattern in dangerous_patterns:
        if pattern in cmd_lower:
            return f"(blocked dangerous pattern: {pattern})"
    
    # Log command for audit
    logger.warning("Shell command executed: %s", command[:200])
    
    try:
        # Use shell=False with parsed arguments for safety
        result = subprocess.run(
            parts,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),  # Run in current directory
        )
        output = result.stdout + result.stderr
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"(command timed out after {timeout}s)"
    except Exception as e:
        return f"(shell error: {e})"


def semantic_search(query: str, path: str = ".", k: int = 5) -> str:
    """Search code semantically using embeddings (finds conceptually similar code).
    
    Unlike ripgrep which matches text patterns, semantic search finds code
    that is conceptually related to your query even without exact word matches.
    
    Best for queries like:
    - "authentication logic"
    - "error handling patterns"
    - "database connection code"
    
    Args:
        query: Natural language description of what you're looking for
        path: Directory to search in (default: current directory)
        k: Number of results to return (default: 5)
        
    Returns:
        Formatted results with file locations and code snippets
    """
    try:
        from .core.vector_index import get_index_manager
        
        manager = get_index_manager()
        results = manager.search(path, query, k=k)
        
        if not results:
            return f"No results found for: {query}"
        
        output = [f"Found {len(results)} semantically similar code snippets:\n"]
        for i, r in enumerate(results, 1):
            output.append(f"--- Result {i}: {r.snippet.file}:{r.snippet.line} ---")
            output.append(f"Type: {r.snippet.type} | Name: {r.snippet.name}")
            output.append(r.snippet.text[:500])
            if len(r.snippet.text) > 500:
                output.append("... (truncated)")
            output.append("")
        
        return "\n".join(output)
    except Exception as e:
        return f"Semantic search error: {e}\nTip: Run 'rlm-dspy index build {path}' first."


# Collection of all built-in tools
BUILTIN_TOOLS: dict[str, Any] = {
    "ripgrep": ripgrep,
    "grep_context": grep_context,
    "find_files": find_files,
    "read_file": read_file,
    "file_stats": file_stats,
    "index_code": index_code,
    "find_definitions": find_definitions,
    "find_classes": find_classes,
    "find_functions": find_functions,
    "find_methods": find_methods,
    "find_imports": find_imports,
    "find_calls": find_calls,
    "semantic_search": semantic_search,
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

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
        path: Path to search in (file or directory, relative to current project)
        flags: Additional rg flags like "-i" for case-insensitive, "-l" for files-only

    Returns:
        Search results as text, or error message

    Example:
        ripgrep("TODO|FIXME", "src/", "-i")  # Case-insensitive search for TODOs
        ripgrep("def\\s+\\w+", ".", "-l")    # List files with function definitions
    """
    try:
        # Resolve path relative to current project
        path = _resolve_project_path(path)

        # Safety check on path
        is_safe, error = _is_safe_path(path)
        if not is_safe:
            return f"(security: {error})"

        # Validate flags - only allow safe ripgrep flags (strict matching)
        # Short flags that take optional numeric args
        SHORT_FLAGS_WITH_NUM = {'-A', '-B', '-C', '-m'}
        # Short flags without args
        SHORT_FLAGS = {'-i', '-l', '-c', '-n', '-w', '-v', '-F', '-s', '-S'}
        # Long flags (exact match or with =value)
        LONG_FLAGS = {'--glob', '--type', '-t', '-g'}

        cmd = ["rg", "--color=never", "--line-number"]
        validated_flags = []
        if flags:
            for flag in flags.split():
                if not flag.startswith('-'):
                    # Non-flag arguments not allowed - could be injection
                    return f"(security: unexpected argument '{flag}' - only flags allowed)"

                # Check if it's a known short flag with numeric arg (e.g., -C5, -A3)
                if len(flag) >= 2 and flag[:2] in SHORT_FLAGS_WITH_NUM:
                    # Verify rest is numeric
                    if flag[2:] and not flag[2:].isdigit():
                        return f"(security: invalid flag '{flag}')"
                    validated_flags.append(flag)
                    continue

                # Check exact short flag match
                if flag in SHORT_FLAGS or flag in SHORT_FLAGS_WITH_NUM:
                    validated_flags.append(flag)
                    continue

                # Check long flags (exact or with =)
                flag_name = flag.split('=')[0]
                if flag_name in LONG_FLAGS:
                    validated_flags.append(flag)
                    continue

                # Not in allowlist
                return f"(security: flag '{flag}' not allowed)"

            cmd.extend(validated_flags)
        # Use -- to separate flags from pattern (prevents pattern injection)
        cmd.extend(["--", pattern, path])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return result.stdout[:50000]  # Limit output size
        elif result.returncode == 1:
            # No matches - check if pattern looks like an identifier and suggest -i
            if "-i" not in flags and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', pattern):
                return f"(no matches found for '{pattern}' - try with '-i' flag for case-insensitive search)"
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
    return ripgrep(pattern, path, f"-C{context_lines}")


def find_files(pattern: str, path: str = ".", file_type: str = "") -> str:
    """
    Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "*.py", "test_*.py")
        path: Directory to search (relative to current project)
        file_type: Filter by extension (e.g., "py", "js")

    Returns:
        List of matching file paths
    """
    try:
        # Resolve path relative to current project
        path = _resolve_project_path(path)

        # Safety check on path
        is_safe, error = _is_safe_path(path)
        if not is_safe:
            return f"(security: {error})"

        cmd = ["rg", "--files", "--color=never"]
        if file_type:
            # Sanitize file_type to prevent injection
            file_type_clean = re.sub(r'[^a-zA-Z0-9]', '', file_type)
            cmd.extend(["-t", file_type_clean])
        if pattern and pattern != "*":
            # Pattern is passed to -g, which is a glob, relatively safe
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
        path: Path to the file (relative to current project)
        start_line: First line to read (1-indexed)
        end_line: Last line to read (None for end of file)

    Returns:
        File contents with line numbers
    """
    try:
        # Resolve path relative to current project
        path = _resolve_project_path(path)

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
        path: Path to analyze (relative to current project)

    Returns:
        JSON with file count, line count, size info
    """
    try:
        # Resolve path relative to current project
        path = _resolve_project_path(path)

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
        path: File or directory to index (relative to current project)
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
        # Resolve path relative to current project
        path = _resolve_project_path(path)

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
    
    Note: If no matches found, automatically tries case-insensitive search.
    """
    # Use ripgrep - matches function_name followed by (
    pattern = rf"\b{re.escape(function_name)}\s*\("
    result = ripgrep(pattern, path, "-n")
    
    # Smart fallback: if no matches, try case-insensitive
    if "(no matches found)" in result:
        result_ci = ripgrep(pattern, path, "-n -i")  # -i = case insensitive
        if "(no matches found)" not in result_ci:
            # Found with different case - show results with warning
            return f"⚠ No exact match for '{function_name}', but found with different case:\n{result_ci}"
    
    return result


def find_usages(file_path: str, symbol_name: str | None = None) -> str:
    """
    Find all references to symbols defined in a file across the codebase.
    
    KEY PRINCIPLE: This tool extracts EXACT symbol names from AST first,
    then searches. No case-sensitivity issues, no guessing.
    
    Use cases:
    - Dead code detection: Which symbols have no external usages?
    - Refactoring: Where is this class/function used?
    - Impact analysis: What will break if I change this?
    - Understanding: How is this module connected to the rest?
    
    Args:
        file_path: File containing the symbol definitions
        symbol_name: Optional - specific symbol to check. If None, checks ALL symbols.
    
    Returns:
        For each symbol: definition location + usage locations across codebase
    
    Examples:
        find_usages("src/rlm_dspy/core/simba_optimizer.py")  # All symbols in file
        find_usages("src/rlm_dspy/core/rlm.py", "RLM")  # Specific symbol
    
    Note: If you get the case wrong, it suggests the correct spelling.
    """
    try:
        from .core.ast_index import index_file
        
        file_path = _resolve_project_path(file_path)
        
        if not Path(file_path).exists():
            return f"(file not found: {file_path})"
        
        # Step 1: Get exact symbol names from the file using AST
        ast_index = index_file(str(file_path))
        
        if not ast_index or not ast_index.definitions:
            return f"(no symbols found in {file_path})"
        
        # Convert to list of dicts for easier processing
        symbols = [
            {"name": d.name, "kind": d.kind, "line": d.line, "parent": d.parent}
            for d in ast_index.definitions
        ]
        
        # Filter to specific symbol if requested
        if symbol_name:
            filtered = [s for s in symbols if s["name"] == symbol_name]
            if not filtered:
                # Try case-insensitive match and suggest correct name
                all_names = [s["name"] for s in symbols]
                matches = [n for n in all_names if n.lower() == symbol_name.lower()]
                if matches:
                    return f"(symbol '{symbol_name}' not found. Did you mean: {', '.join(set(matches))}?)"
                return f"(symbol '{symbol_name}' not found in {file_path})"
            symbols = filtered
        
        # Step 2: For each top-level symbol (not methods), search for usages
        results = []
        base_path = Path(file_path).parent.parent  # Go up to find project root
        search_path = str(base_path) if base_path.exists() else "."
        
        # Only check top-level symbols (classes, functions) - not methods
        top_level = [s for s in symbols if s["parent"] is None and s["kind"] in ("class", "function")]
        
        for sym in top_level:
            name = sym["name"]
            kind = sym["kind"]
            line = sym["line"]
            
            # Search for usages using exact name (with word boundaries)
            pattern = rf"\b{re.escape(name)}\b"
            
            # Get INTERNAL usages (within the same file)
            internal_result = ripgrep(pattern, str(file_path), "-c")  # -c = count only
            if "(no matches" in internal_result or not internal_result.strip():
                internal_count = 0
            else:
                # Parse count from output like "file.py:5"
                try:
                    internal_count = int(internal_result.strip().split(":")[-1])
                except (ValueError, IndexError):
                    internal_count = 0
            # Subtract 1 for the definition itself
            internal_usages = max(0, internal_count - 1)
            
            # Get EXTERNAL usages (other files)
            usage_result = ripgrep(pattern, search_path, "-l")  # -l = files only
            if "(no matches found)" in usage_result or not usage_result.strip():
                external_files = []
            else:
                external_files = [f for f in usage_result.strip().split("\n") 
                                 if f and not f.endswith(Path(file_path).name)]
            
            # Format result with BOTH internal and external counts
            if external_files:
                status = f"USED: {internal_usages} internal, {len(external_files)} external files"
                results.append(f"{kind} {name} (line {line}): {status}")
                short_files = [Path(f).name for f in external_files[:5]]
                results.append(f"  → {', '.join(short_files)}")
                if len(external_files) > 5:
                    results.append(f"  ... and {len(external_files) - 5} more files")
            elif internal_usages > 0:
                status = f"INTERNAL ONLY: {internal_usages} usages within same file"
                results.append(f"{kind} {name} (line {line}): {status}")
                results.append(f"  ℹ Used internally but not imported elsewhere")
            else:
                results.append(f"{kind} {name} (line {line}): ⚠ DEAD CODE (0 usages)")
                results.append(f"  ⚠ Never used - candidate for removal")
        
        if not results:
            return f"(no top-level classes/functions found in {file_path})"
            
        return "\n".join(results)
        
    except Exception as e:
        return f"(find_usages error: {e})"


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


def semantic_search(query: str, path: str | None = None, k: int = 5) -> str:
    """Search code semantically using embeddings (finds conceptually similar code).

    Unlike ripgrep which matches text patterns, semantic search finds code
    that is conceptually related to your query even without exact word matches.

    Best for queries like:
    - "authentication logic"
    - "error handling patterns"
    - "database connection code"

    Args:
        query: Natural language description of what you're looking for
        path: Directory to search in (default: current project, relative paths resolved to project)
        k: Number of results to return (default: 5)

    Returns:
        Formatted results with file locations and code snippets
    """
    try:
        from .core.vector_index import get_index_manager

        # Resolve path relative to current project
        search_path = _resolve_project_path(path)

        manager = get_index_manager()
        results = manager.search(search_path, query, k=k)

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


# Module-level project context (set by CLI when loading paths)
_current_project_path: str | None = None


def set_current_project(path: str | None) -> None:
    """Set the current project path for tools (called by CLI)."""
    global _current_project_path
    _current_project_path = path


def get_current_project() -> str | None:
    """Get the current project path."""
    return _current_project_path


def _resolve_project_path(path: str | None) -> str:
    """Resolve a path relative to the current project.

    When the LLM specifies a path like "." or "./src", it should resolve
    relative to the project being analyzed, not the CWD.

    Args:
        path: Path from tool argument (may be None, ".", or relative)

    Returns:
        Resolved absolute path
    """
    # Use current project if no path or "." specified
    if not path or path == ".":
        return _current_project_path or "."

    # If already absolute, still need to validate it
    # (don't just return - let _is_safe_path check it later)
    if Path(path).is_absolute():
        return str(Path(path).resolve())

    # Try multiple resolution strategies for relative paths
    if _current_project_path:
        project_path = Path(_current_project_path)

        # Strategy 1: Resolve relative to project path
        resolved = (project_path / path).resolve()
        if resolved.exists():
            return str(resolved)

        # Strategy 2: Maybe the path is relative to CWD (common when LLM
        # sees paths like "src/rlm_dspy/cli.py" in the context)
        cwd_resolved = Path(path).resolve()
        if cwd_resolved.exists():
            return str(cwd_resolved)

        # Strategy 3: Try stripping common prefixes if path duplicates project structure
        # e.g., project=/a/b/src/pkg, path=src/pkg/file.py -> /a/b/src/pkg/file.py
        path_parts = Path(path).parts
        proj_parts = project_path.parts
        for i in range(len(path_parts)):
            # Check if path_parts[i:] starts with a suffix of proj_parts
            for j in range(len(proj_parts)):
                if path_parts[i:i+1] == proj_parts[j:j+1]:
                    # Found overlap - try resolving from project parent
                    overlap_start = j
                    candidate = Path(*proj_parts[:overlap_start], *path_parts[i:])
                    if candidate.exists():
                        return str(candidate)

        # Fallback: return project-relative (may not exist)
        return str(resolved)

    # No project path set - resolve relative to CWD
    return str(Path(path).resolve())


def list_projects(include_empty: bool = False) -> str:
    """List all indexed projects available for semantic search.

    Use this to discover what codebases are indexed and their paths.
    Then use semantic_search with the project path to search that specific project.

    Args:
        include_empty: Include projects with 0 snippets (default: False)

    Returns:
        Formatted list of projects with names, paths, and snippet counts
    """
    try:
        from .core.project_registry import get_project_registry
        from .core.vector_index import get_index_manager

        registry = get_project_registry()
        projects = registry.list()

        if not projects:
            return "No projects indexed. Use 'rlm-dspy index build <path>' to index a project."

        manager = get_index_manager()
        default_project = registry.get_default()
        default_name = default_project.name if default_project else None

        # Collect projects with their snippet counts
        project_data = []
        for p in projects:
            # Use snippet_count from Project if available, else check manifest
            snippet_count = p.snippet_count
            if not snippet_count:
                index_path = manager.config.index_dir / p.name
                manifest_path = index_path / "manifest.json"
                if manifest_path.exists():
                    try:
                        import json
                        manifest = json.loads(manifest_path.read_text())
                        snippet_count = manifest.get("snippet_count", 0)
                    except (json.JSONDecodeError, OSError) as e:
                        logger.debug("Failed to read manifest %s: %s", manifest_path, e)

            # Filter empty projects unless requested
            if snippet_count > 0 or include_empty:
                project_data.append((p, snippet_count))

        if not project_data:
            return "No indexed projects with code found. Use 'rlm-dspy index build <path>' to index."

        output = [f"Found {len(project_data)} indexed projects:\n"]

        for p, snippet_count in project_data:
            output.append(f"  • {p.name}")
            output.append(f"    Path: {p.path}")
            output.append(f"    Snippets: {snippet_count}")
            if p.name == default_name:
                output.append("    [DEFAULT]")
            output.append("")

        return "\n".join(output)
    except Exception as e:
        return f"Error listing projects: {e}"


def search_all_projects(query: str, k: int = 3) -> str:
    """Search semantically across ALL indexed projects (without duplicates).

    Unlike semantic_search which searches one project, this searches
    all registered projects and returns the best matches from each.

    Automatically skips overlapping projects to avoid duplicate results
    (e.g., if /project and /project/src are both indexed, only searches /project).

    Args:
        query: Natural language description of what you're looking for
        k: Number of results per project (default: 3)

    Returns:
        Aggregated results from all projects, sorted by relevance
    """
    try:
        from .core.project_registry import get_project_registry
        from .core.vector_index import get_index_manager
        from pathlib import Path

        registry = get_project_registry()
        projects = registry.list()

        if not projects:
            return "No projects indexed. Use 'rlm-dspy index build <path>' first."

        # Filter out child projects that overlap with parent projects
        # Keep only the most specific (deepest) non-overlapping projects
        overlaps = registry.find_overlaps()
        projects_to_skip = set()

        for project_name, overlap_list in overlaps.items():
            project_path = Path(registry.get(project_name).path)
            for overlap in overlap_list:
                overlap_path = Path(overlap.path)
                # If this project is a parent of the overlap, skip the overlap (child)
                try:
                    overlap_path.relative_to(project_path)
                    # overlap is a child of project - skip the child to avoid duplicates
                    projects_to_skip.add(overlap.name)
                except ValueError:
                    pass

        manager = get_index_manager()
        all_results = []
        searched_projects = 0

        for p in projects:
            # Skip overlapping child projects
            if p.name in projects_to_skip:
                logger.debug(f"Skipping '{p.name}' - overlaps with parent project")
                continue

            # Skip projects with no snippets
            if p.snippet_count == 0:
                index_path = manager.config.index_dir / p.name
                manifest_path = index_path / "manifest.json"
                if manifest_path.exists():
                    try:
                        import json
                        manifest = json.loads(manifest_path.read_text())
                        if manifest.get("snippet_count", 0) == 0:
                            continue
                    except Exception:
                        continue
                else:
                    continue

            try:
                results = manager.search(p.path, query, k=k)
                searched_projects += 1
                for r in results:
                    all_results.append((p.name, r))
            except Exception as e:
                logger.debug(f"Search failed for {p.name}: {e}")
                continue

        if not all_results:
            return f"No results found for: {query}"

        # Sort by score descending
        all_results.sort(key=lambda x: x[1].score, reverse=True)

        # Take top results
        top_results = all_results[:k * 2]  # Return more since it's cross-project

        output = [f"Found {len(top_results)} results across {searched_projects} projects:\n"]
        for i, (project_name, r) in enumerate(top_results, 1):
            output.append(f"--- Result {i}: [{project_name}] {r.snippet.file}:{r.snippet.line} ---")
            output.append(f"Type: {r.snippet.type} | Name: {r.snippet.name} | Score: {r.score:.3f}")
            output.append(r.snippet.text[:400])
            if len(r.snippet.text) > 400:
                output.append("... (truncated)")
            output.append("")

        return "\n".join(output)
    except Exception as e:
        return f"Cross-project search error: {e}"


# Collection of all built-in tools
BUILTIN_TOOLS: dict[str, Any] = {
    # Structural search
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
    "find_usages": find_usages,  # Dead code detection - extracts exact names first
    # Semantic search (uses current project by default)
    "semantic_search": semantic_search,
    # Shell (unsafe)
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

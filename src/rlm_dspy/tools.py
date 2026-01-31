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

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Restricted paths that tools should not access
_RESTRICTED_PATHS = {
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    "~/.ssh", "~/.gnupg", "~/.aws/credentials",
}

# Current project path (set by CLI)
_current_project_path: str | None = None


def _is_safe_path(path: str, base_dir: str | None = None) -> tuple[bool, str]:
    """Check if a path is safe to access."""
    try:
        resolved = Path(path).expanduser().resolve()
        resolved_str = str(resolved)

        for restricted in _RESTRICTED_PATHS:
            restricted_resolved = str(Path(restricted).expanduser().resolve())
            if resolved_str.startswith(restricted_resolved):
                return False, f"Access to {restricted} is restricted"

        system_paths = ['/etc', '/var', '/usr', '/bin', '/sbin', '/boot', '/root', '/proc', '/sys']
        for sys_path in system_paths:
            if resolved_str.startswith(sys_path):
                return False, f"Access to system path {sys_path} is restricted"

        if base_dir:
            base_resolved = Path(base_dir).expanduser().resolve()
            try:
                resolved.relative_to(base_resolved)
            except ValueError:
                return False, f"Path must be within {base_dir}"

        return True, ""
    except Exception as e:
        return False, f"Path validation error: {e}"


def set_current_project(path: str | None) -> None:
    """Set the current project path for tools (called by CLI)."""
    global _current_project_path
    _current_project_path = path


def get_current_project() -> str | None:
    """Get the current project path."""
    return _current_project_path


def _resolve_project_path(path: str | None) -> str:
    """Resolve a path relative to the current project."""
    if not path or path == ".":
        return _current_project_path or "."

    if Path(path).is_absolute():
        return str(Path(path).resolve())

    if _current_project_path:
        project_path = Path(_current_project_path)

        resolved = (project_path / path).resolve()
        if resolved.exists():
            return str(resolved)

        cwd_resolved = Path(path).resolve()
        if cwd_resolved.exists():
            return str(cwd_resolved)

        path_parts = Path(path).parts
        proj_parts = project_path.parts
        for i in range(len(path_parts)):
            for j in range(len(proj_parts)):
                if path_parts[i:i+1] == proj_parts[j:j+1]:
                    overlap_start = j
                    candidate = Path(*proj_parts[:overlap_start], *path_parts[i:])
                    if candidate.exists():
                        return str(candidate)

        return str((project_path / path).resolve())

    return str(Path(path).resolve())


def shell(command: str, timeout: int = 30) -> str:
    """Run a shell command (use with caution).

    Security: Disabled by default (requires RLM_ALLOW_SHELL=1)
    """
    import shlex

    if not os.environ.get("RLM_ALLOW_SHELL"):
        return "(shell disabled - set RLM_ALLOW_SHELL=1 to enable)"

    ALLOWED_COMMANDS = {
        'ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc', 'sort', 'uniq',
        'echo', 'pwd', 'date', 'whoami', 'env', 'which', 'file', 'stat',
        'diff', 'tree', 'du', 'df', 'uname', 'hostname', 'ps', 'top',
        'git', 'python', 'python3', 'pip', 'node', 'npm', 'cargo', 'go',
    }

    DANGEROUS = ['rm ', 'mv ', 'cp ', 'chmod ', 'chown ', 'sudo ', 'su ', 'dd ',
                 '>', '>>', '|', ';', '&&', '||', '`', '$(',
                 'curl ', 'wget ', 'nc ', 'netcat ']

    for pattern in DANGEROUS:
        if pattern in command:
            return f"(blocked: dangerous pattern '{pattern}')"

    try:
        parts = shlex.split(command)
        if not parts:
            return "(empty command)"
        base_cmd = parts[0]
        if base_cmd not in ALLOWED_COMMANDS:
            return f"(blocked: '{base_cmd}' not in allowed commands)"
    except ValueError as e:
        return f"(invalid command: {e})"

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=_current_project_path
        )
        output = result.stdout + result.stderr
        if len(output) > 10000:
            output = output[:10000] + "\n... (truncated)"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"
    except Exception as e:
        return f"(error: {e})"


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all tools for the LLM prompt."""
    lines = ["Available tools:"]
    for name, func in BUILTIN_TOOLS.items():
        doc = func.__doc__ or "No description"
        first_line = doc.strip().split("\n")[0]
        lines.append(f"  - {name}: {first_line}")
    return "\n".join(lines)


# Import all tools from submodules
from .tools_search import (
    ripgrep,
    grep_context,
    find_files,
    semantic_search,
    list_projects,
    search_all_projects,
)
from .tools_code import (
    read_file,
    file_stats,
    index_code,
    find_definitions,
    find_classes,
    find_functions,
    find_methods,
    find_imports,
    find_calls,
    find_usages,
)

# Tool registry
BUILTIN_TOOLS: dict[str, callable] = {
    # Search tools
    "ripgrep": ripgrep,
    "grep_context": grep_context,
    "find_files": find_files,
    "semantic_search": semantic_search,
    "list_projects": list_projects,
    "search_all_projects": search_all_projects,
    # Code analysis tools
    "read_file": read_file,
    "file_stats": file_stats,
    "index_code": index_code,
    "find_definitions": find_definitions,
    "find_classes": find_classes,
    "find_functions": find_functions,
    "find_methods": find_methods,
    "find_imports": find_imports,
    "find_calls": find_calls,
    "find_usages": find_usages,
    # Shell (disabled by default)
    "shell": shell,
}

# Re-export for convenience
__all__ = [
    "BUILTIN_TOOLS",
    "get_tool_descriptions",
    "set_current_project",
    "get_current_project",
    # Search
    "ripgrep",
    "grep_context", 
    "find_files",
    "semantic_search",
    "list_projects",
    "search_all_projects",
    # Code
    "read_file",
    "file_stats",
    "index_code",
    "find_definitions",
    "find_classes",
    "find_functions",
    "find_methods",
    "find_imports",
    "find_calls",
    "find_usages",
    "shell",
    # Utilities (for submodules)
    "_is_safe_path",
    "_resolve_project_path",
]

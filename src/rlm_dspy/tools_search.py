"""Search tools for RLM - ripgrep, find_files, semantic search."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Import shared utilities
from .tools import _is_safe_path, _resolve_project_path


def ripgrep(pattern: str, path: str = ".", flags: str = "") -> str:
    """Search for pattern using ripgrep (rg).

    Args:
        pattern: Regex pattern to search for
        path: Path to search in (file or directory)
        flags: Additional rg flags like "-i" for case-insensitive

    Returns:
        Search results as text, or error message
    """
    try:
        path = _resolve_project_path(path)
        is_safe, error = _is_safe_path(path)
        if not is_safe:
            return f"(security: {error})"

        SHORT_FLAGS_WITH_NUM = {'-A', '-B', '-C', '-m'}
        SHORT_FLAGS = {'-i', '-l', '-c', '-n', '-w', '-v', '-F', '-s', '-S'}
        LONG_FLAGS = {'--glob', '--type', '-t', '-g'}

        cmd = ["rg", "--color=never", "--line-number"]
        validated_flags = []
        if flags:
            for flag in flags.split():
                if not flag.startswith('-'):
                    continue
                if flag in SHORT_FLAGS:
                    validated_flags.append(flag)
                elif any(flag.startswith(f) for f in SHORT_FLAGS_WITH_NUM):
                    base = flag[:2]
                    rest = flag[2:]
                    if base in SHORT_FLAGS_WITH_NUM and (not rest or rest.isdigit()):
                        validated_flags.append(flag)
                elif any(flag.startswith(f) for f in LONG_FLAGS):
                    validated_flags.append(flag)

        cmd.extend(validated_flags)
        cmd.extend([pattern, path])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 200:
                return '\n'.join(lines[:200]) + f'\n... ({len(lines) - 200} more matches)'
            return result.stdout.strip() or "(no matches)"
        elif result.returncode == 1:
            return "(no matches)"
        else:
            return f"(ripgrep error: {result.stderr[:200]})"
    except subprocess.TimeoutExpired:
        return "(search timed out after 30s)"
    except FileNotFoundError:
        return "(ripgrep not installed - use `apt install ripgrep`)"
    except Exception as e:
        return f"(ripgrep error: {e})"


def grep_context(pattern: str, path: str = ".", context_lines: int = 3) -> str:
    """Search with context lines around matches."""
    return ripgrep(pattern, path, f"-C{context_lines}")


def find_files(pattern: str, path: str = ".", file_type: str = "") -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "*.py", "test_*.py")
        path: Directory to search
        file_type: Filter by extension (e.g., "py", "js")

    Returns:
        List of matching file paths
    """
    try:
        path = _resolve_project_path(path)
        is_safe, error = _is_safe_path(path)
        if not is_safe:
            return f"(security: {error})"

        cmd = ["rg", "--files", "--color=never"]
        if file_type:
            file_type_clean = re.sub(r'[^a-zA-Z0-9]', '', file_type)
            cmd.extend(["-t", file_type_clean])
        if pattern and pattern != "*":
            cmd.extend(["-g", pattern])
        cmd.append(path)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            files = result.stdout.strip().split('\n')
            if len(files) > 200:
                return '\n'.join(files[:200]) + f'\n... ({len(files) - 200} more files)'
            return result.stdout.strip() or "(no files found)"
        else:
            return f"(find error: {result.stderr[:200]})"
    except Exception as e:
        return f"(find error: {e})"


def semantic_search(query: str, path: str | None = None, k: int = 5) -> str:
    """Search code semantically using embeddings.

    Args:
        query: Natural language description of what you're looking for
        path: Directory to search in
        k: Number of results to return

    Returns:
        Formatted results with file locations and code snippets
    """
    try:
        from .core.vector_index import get_index_manager

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


def list_projects(include_empty: bool = False) -> str:
    """List all indexed projects available for semantic search."""
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

        project_data = []
        for project in projects:
            count = project.snippet_count
            if not include_empty and count == 0:
                continue
            project_data.append({
                "name": project.name,
                "path": project.path,
                "snippets": count,
                "is_default": project.name == default_name,
            })

        if not project_data:
            return "No indexed projects with code found. Use 'rlm-dspy index build <path>' to index."

        output = [f"Found {len(project_data)} indexed projects:\n"]
        for p in project_data:
            default_marker = " (default)" if p["is_default"] else ""
            output.append(f"  {p['name']}{default_marker}: {p['path']} ({p['snippets']} snippets)")

        return "\n".join(output)
    except Exception as e:
        return f"Error listing projects: {e}"


def search_all_projects(query: str, k: int = 3) -> str:
    """Search semantically across ALL indexed projects."""
    try:
        from .core.project_registry import get_project_registry
        from .core.vector_index import get_index_manager

        registry = get_project_registry()
        projects = registry.list()

        if not projects:
            return "No projects indexed. Use 'rlm-dspy index build <path>' first."

        overlaps = registry.find_overlaps()
        projects_to_skip = set()
        for overlapping in overlaps.values():
            for proj in overlapping:
                projects_to_skip.add(proj.name)

        manager = get_index_manager()
        all_results = []

        for project in projects:
            if project.name in projects_to_skip:
                continue
            if project.snippet_count == 0:
                continue
            try:
                results = manager.search(project.path, query, k=k)
                for r in results:
                    all_results.append((project.name, r))
            except Exception:
                continue

        if not all_results:
            return f"No results found for: {query}"

        output = [f"Found {len(all_results)} results across projects:\n"]
        for project_name, r in all_results[:k * 3]:
            output.append(f"--- [{project_name}] {r.snippet.file}:{r.snippet.line} ---")
            output.append(f"Type: {r.snippet.type} | Name: {r.snippet.name}")
            output.append(r.snippet.text[:400])
            output.append("")

        return "\n".join(output)
    except Exception as e:
        return f"Search error: {e}"

"""LSP (Language Server Protocol) integration for IDE-quality code intelligence.

Provides precise code navigation, type information, and refactoring capabilities
by leveraging actual language servers (pyright, rust-analyzer, gopls, etc.).

Uses solidlsp from the Serena project for unified LSP management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Singleton LSP manager
_lsp_manager: "LSPManager | None" = None


@dataclass
class LSPConfig:
    """Configuration for LSP integration."""
    enabled: bool = True
    timeout: int = 30  # Request timeout in seconds
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".rlm" / "lsp_cache")

    @classmethod
    def from_user_config(cls) -> "LSPConfig":
        """Load from user config."""
        try:
            from .user_config import load_config
            config = load_config()

            return cls(
                enabled=config.get("lsp_enabled", True),
                timeout=config.get("lsp_timeout", 30),
                cache_dir=Path(config.get("lsp_cache_dir", "~/.rlm/lsp_cache")).expanduser(),
            )
        except ImportError:
            return cls()


class LSPManager:
    """Manages language server connections for different projects/languages.

    Provides a unified interface for LSP operations across all supported languages.
    Lazily starts language servers as needed.

    Example:
        ```python
        manager = get_lsp_manager()

        # Find all references to a symbol
        refs = manager.find_references("/path/to/file.py", "MyClass", line=10)

        # Get type info for a symbol
        info = manager.get_hover_info("/path/to/file.py", line=10, column=5)
        ```
    """

    def __init__(self, config: LSPConfig | None = None):
        self.config = config or LSPConfig.from_user_config()
        self._servers: dict[str, Any] = {}  # project_root -> SolidLanguageServer
        self._solidlsp_available = self._check_solidlsp()

    def _check_solidlsp(self) -> bool:
        """Check if solidlsp is available."""
        try:
            from solidlsp import SolidLanguageServer  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "solidlsp not installed. LSP features unavailable. "
                "Install with: pip install -e path/to/serena"
            )
            return False

    def _get_server(self, file_path: str) -> Any | None:
        """Get or create a language server for the given file."""
        if not self._solidlsp_available or not self.config.enabled:
            return None

        from solidlsp import SolidLanguageServer

        path = Path(file_path).resolve()
        if not path.exists():
            return None

        # Find project root (look for common markers)
        project_root = self._find_project_root(path)
        project_key = str(project_root)

        if project_key in self._servers:
            server = self._servers[project_key]
            if server.is_running():
                return server
            else:
                # Server stopped, remove from cache
                del self._servers[project_key]

        # Determine language from file extension
        language = self._get_language(path)
        if not language:
            return None

        # Create and start language server
        try:
            from solidlsp.ls_config import LanguageServerConfig

            ls_config = LanguageServerConfig(code_language=language)
            server = SolidLanguageServer.create(
                config=ls_config,
                repository_root_path=str(project_root),
                timeout=float(self.config.timeout),
            )
            server.start()
            self._servers[project_key] = server
            logger.info("Started language server for %s (%s)", project_root, language)
            return server
        except Exception as e:
            logger.warning("Failed to start language server: %s", e)
            return None

    def _find_project_root(self, path: Path) -> Path:
        """Find project root by looking for common markers."""
        markers = [
            ".git", "pyproject.toml", "package.json", "Cargo.toml",
            "go.mod", "pom.xml", "build.gradle", "build.sbt",
            ".project", "Makefile", "CMakeLists.txt"
        ]

        current = path if path.is_dir() else path.parent
        while current != current.parent:
            for marker in markers:
                if (current / marker).exists():
                    return current
            current = current.parent

        # Fallback to file's directory
        return path if path.is_dir() else path.parent

    def _get_language(self, path: Path) -> Any | None:
        """Get Language enum for a file extension."""
        try:
            from solidlsp.ls_config import Language

            ext_map = {
                # Python
                ".py": Language.PYTHON,
                ".pyi": Language.PYTHON,
                # JavaScript/TypeScript (use TypeScript server for both)
                ".js": Language.TYPESCRIPT,
                ".jsx": Language.TYPESCRIPT,
                ".mjs": Language.TYPESCRIPT,
                ".ts": Language.TYPESCRIPT,
                ".tsx": Language.TYPESCRIPT,
                # Other languages
                ".go": Language.GO,
                ".rs": Language.RUST,
                ".java": Language.JAVA,
                ".kt": Language.KOTLIN,
                ".kts": Language.KOTLIN,
                ".scala": Language.SCALA,
                ".sc": Language.SCALA,
                ".cpp": Language.CPP,
                ".cc": Language.CPP,
                ".cxx": Language.CPP,
                ".hpp": Language.CPP,
                ".hxx": Language.CPP,
                ".c": Language.CPP,  # Use CPP server for C too
                ".h": Language.CPP,
                ".rb": Language.RUBY,
                ".cs": Language.CSHARP,
                ".lua": Language.LUA,
                ".hs": Language.HASKELL,
                ".lhs": Language.HASKELL,
                ".php": Language.PHP,
                ".swift": Language.SWIFT,
                ".ex": Language.ELIXIR,
                ".exs": Language.ELIXIR,
                ".erl": Language.ERLANG,
                ".hrl": Language.ERLANG,
                ".sh": Language.BASH,
                ".bash": Language.BASH,
                ".zsh": Language.BASH,
                ".dart": Language.DART,
                ".clj": Language.CLOJURE,
                ".cljs": Language.CLOJURE,
                ".nix": Language.NIX,
                ".zig": Language.ZIG,
                ".jl": Language.JULIA,
                ".r": Language.R,
                ".R": Language.R,
                ".yaml": Language.YAML,
                ".yml": Language.YAML,
                ".toml": Language.TOML,
                ".md": Language.MARKDOWN,
                ".ps1": Language.POWERSHELL,
                ".vue": Language.VUE,
                ".elm": Language.ELM,
                ".fs": Language.FSHARP,
                ".fsx": Language.FSHARP,
                ".pl": Language.PERL,
                ".pm": Language.PERL,
                ".f90": Language.FORTRAN,
                ".f95": Language.FORTRAN,
                ".tf": Language.TERRAFORM,
                ".pas": Language.PASCAL,
                ".groovy": Language.GROOVY,
                ".m": Language.MATLAB,
            }
            return ext_map.get(path.suffix.lower())
        except ImportError:
            return None

    def find_references(
        self,
        file_path: str,
        line: int,
        column: int = 0,
    ) -> list[dict[str, Any]]:
        """Find all references to the symbol at the given position.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            List of references with file, line, and column info
        """
        server = self._get_server(file_path)
        if not server:
            return []

        try:
            # solidlsp uses relative path from project root
            abs_path = Path(file_path).resolve()
            project_root = self._find_project_root(abs_path)
            try:
                rel_path = str(abs_path.relative_to(project_root))
            except ValueError:
                rel_path = str(abs_path)

            refs = server.request_references(
                rel_path,
                line - 1,  # LSP uses 0-indexed lines
                column,
            )

            if not refs:
                return []

            # solidlsp returns Location objects (TypedDict-like)
            result = []
            for ref in refs:
                if hasattr(ref, 'get'):
                    # Dict-like
                    result.append({
                        "file": ref.get("uri", "").replace("file://", ""),
                        "line": ref.get("range", {}).get("start", {}).get("line", 0) + 1,
                        "column": ref.get("range", {}).get("start", {}).get("character", 0),
                    })
                elif hasattr(ref, 'uri'):
                    # Object with attributes
                    result.append({
                        "file": getattr(ref, 'uri', '').replace("file://", ""),
                        "line": getattr(ref, 'range', {}).get("start", {}).get("line", 0) + 1,
                        "column": getattr(ref, 'range', {}).get("start", {}).get("character", 0),
                    })
            return result
        except Exception as e:
            logger.warning("Failed to find references: %s", e)
            return []

    def go_to_definition(
        self,
        file_path: str,
        line: int,
        column: int = 0
    ) -> dict[str, Any] | None:
        """Go to the definition of the symbol at the given position.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            Definition location with file, line, and column
        """
        server = self._get_server(file_path)
        if not server:
            return None

        try:
            result = server.request_definition(file_path, line - 1, column)
            if not result:
                return None

            # Handle both single location and list of locations
            loc = result[0] if isinstance(result, list) else result

            return {
                "file": loc.get("uri", "").replace("file://", ""),
                "line": loc.get("range", {}).get("start", {}).get("line", 0) + 1,
                "column": loc.get("range", {}).get("start", {}).get("character", 0),
            }
        except Exception as e:
            logger.warning("Failed to go to definition: %s", e)
            return None

    def get_hover_info(
        self,
        file_path: str,
        line: int,
        column: int = 0
    ) -> str | None:
        """Get hover information (type signature, docs) for symbol at position.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            Hover information as string (may contain markdown)
        """
        server = self._get_server(file_path)
        if not server:
            return None

        try:
            result = server.request_hover(file_path, line - 1, column)
            if not result:
                return None

            # Extract contents from hover response
            contents = result.get("contents", "")
            if isinstance(contents, dict):
                return contents.get("value", str(contents))
            elif isinstance(contents, list):
                return "\n".join(
                    c.get("value", str(c)) if isinstance(c, dict) else str(c)
                    for c in contents
                )
            return str(contents)
        except Exception as e:
            logger.warning("Failed to get hover info: %s", e)
            return None

    def get_document_symbols(self, file_path: str) -> list[dict[str, Any]]:
        """Get all symbols in a document.

        Args:
            file_path: Path to the file

        Returns:
            List of symbols with name, kind, line, and children
        """
        server = self._get_server(file_path)
        if not server:
            return []

        try:
            result = server.request_document_symbols(file_path)
            if result is None:
                return []

            # solidlsp returns DocumentSymbols object with root_symbols attribute
            if hasattr(result, 'root_symbols'):
                symbols = result.root_symbols
            elif isinstance(result, list):
                symbols = result
            else:
                symbols = []

            return self._flatten_symbols(symbols)
        except Exception as e:
            logger.warning("Failed to get document symbols: %s", e)
            return []

    def _flatten_symbols(
        self,
        symbols: list[dict],
        parent: str | None = None
    ) -> list[dict[str, Any]]:
        """Flatten nested document symbols into a list."""
        result = []
        for sym in symbols:
            name = sym.get("name", "")
            kind = sym.get("kind", 0)
            range_info = sym.get("range", sym.get("location", {}).get("range", {}))

            result.append({
                "name": name,
                "kind": self._symbol_kind_name(kind),
                "line": range_info.get("start", {}).get("line", 0) + 1,
                "end_line": range_info.get("end", {}).get("line", 0) + 1,
                "parent": parent,
            })

            # Recurse into children
            children = sym.get("children", [])
            if children:
                result.extend(self._flatten_symbols(children, parent=name))

        return result

    def _symbol_kind_name(self, kind: int) -> str:
        """Convert LSP SymbolKind number to name."""
        kinds = {
            1: "file", 2: "module", 3: "namespace", 4: "package",
            5: "class", 6: "method", 7: "property", 8: "field",
            9: "constructor", 10: "enum", 11: "interface", 12: "function",
            13: "variable", 14: "constant", 15: "string", 16: "number",
            17: "boolean", 18: "array", 19: "object", 20: "key",
            21: "null", 22: "enum_member", 23: "struct", 24: "event",
            25: "operator", 26: "type_parameter",
        }
        return kinds.get(kind, f"kind_{kind}")

    def rename_symbol(
        self,
        file_path: str,
        line: int,
        column: int,
        new_name: str
    ) -> dict[str, list[dict]] | None:
        """Get edits needed to rename a symbol.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed)
            column: Column number (0-indexed)
            new_name: New name for the symbol

        Returns:
            Dict mapping file paths to list of edits
        """
        server = self._get_server(file_path)
        if not server:
            return None

        try:
            result = server.request_rename_symbol_edit(
                file_path, line - 1, column, new_name
            )
            if not result:
                return None

            # Convert workspace edit to our format
            edits: dict[str, list[dict]] = {}
            for uri, changes in result.get("changes", {}).items():
                file = uri.replace("file://", "")
                edits[file] = [
                    {
                        "start_line": c["range"]["start"]["line"] + 1,
                        "start_col": c["range"]["start"]["character"],
                        "end_line": c["range"]["end"]["line"] + 1,
                        "end_col": c["range"]["end"]["character"],
                        "new_text": c["newText"],
                    }
                    for c in changes
                ]
            return edits
        except Exception as e:
            logger.warning("Failed to rename symbol: %s", e)
            return None

    def shutdown(self) -> None:
        """Shutdown all running language servers."""
        for project, server in list(self._servers.items()):
            try:
                server.stop()
                logger.info("Stopped language server for %s", project)
            except Exception as e:
                logger.warning("Error stopping server for %s: %s", project, e)
        self._servers.clear()

    def __del__(self) -> None:
        """Cleanup on garbage collection - prevent orphaned subprocesses."""
        try:
            self.shutdown()
        except Exception:
            pass  # Best effort cleanup during GC


def get_lsp_manager(config: LSPConfig | None = None) -> LSPManager:
    """Get the singleton LSP manager instance."""
    global _lsp_manager
    if _lsp_manager is None:
        _lsp_manager = LSPManager(config)
    return _lsp_manager


def clear_lsp_manager() -> None:
    """Shutdown and clear the LSP manager."""
    global _lsp_manager
    if _lsp_manager:
        _lsp_manager.shutdown()
        _lsp_manager = None

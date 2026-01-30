"""LSP (Language Server Protocol) integration for IDE-quality code intelligence.

Provides precise code navigation, type information, and refactoring capabilities
by leveraging actual language servers (pyright, rust-analyzer, gopls, etc.).

Uses vendored solidlsp for unified LSP management with auto-installation support.
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
    """Whether LSP features are enabled."""
    
    auto_install: bool = True
    """Whether to auto-install LSP servers when needed."""
    
    timeout: int = 30
    """Request timeout in seconds."""
    
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".rlm" / "lsp_cache")
    """Directory for LSP cache files."""

    @classmethod
    def from_user_config(cls) -> "LSPConfig":
        """Load from user config."""
        try:
            from .user_config import load_config
            config = load_config()

            return cls(
                enabled=config.get("lsp_enabled", True),
                auto_install=config.get("lsp_auto_install", True),
                timeout=config.get("lsp_timeout", 30),
                cache_dir=Path(config.get("lsp_cache_dir", "~/.rlm/lsp_cache")).expanduser(),
            )
        except ImportError:
            return cls()


# Extension to language mapping
EXT_TO_LANGUAGE = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # JavaScript/TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    # Other languages
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".rb": "ruby",
    ".cs": "csharp",
    ".lua": "lua",
    ".hs": "haskell",
    ".lhs": "haskell",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
}


class LSPManager:
    """Manages language server connections for different projects/languages.

    Provides a unified interface for LSP operations across all supported languages.
    Lazily starts language servers as needed and auto-installs them if configured.

    Example:
        ```python
        manager = get_lsp_manager()

        # Find all references to a symbol
        refs = manager.find_references("/path/to/file.py", line=10, column=5)

        # Get type info for a symbol
        info = manager.get_hover_info("/path/to/file.py", line=10, column=5)
        ```
    """

    def __init__(self, config: LSPConfig | None = None):
        self.config = config or LSPConfig.from_user_config()
        self._servers: dict[str, Any] = {}  # project_root -> SolidLanguageServer
        self._solidlsp_available = True  # We vendored it, so it's always available

    def _get_server(self, file_path: str) -> Any | None:
        """Get or create a language server for the given file."""
        if not self.config.enabled:
            return None

        path = Path(file_path).resolve()
        if not path.exists():
            return None

        # Find project root (look for common markers)
        project_root = self._find_project_root(path)
        
        # Get language for this file
        language = self._get_language(path)
        if not language:
            logger.debug(f"No language detected for {path}")
            return None
        
        # Project key includes language since we might have multiple servers per project
        project_key = f"{project_root}:{language}"

        if project_key in self._servers:
            return self._servers[project_key]

        # Check if server is installed, auto-install if configured
        if self.config.auto_install:
            if not self._ensure_server_installed(language):
                return None
        
        # Create and start language server
        try:
            server = self._create_server(language, project_root)
            if server:
                self._servers[project_key] = server
                logger.info(f"Started language server for {project_root} ({language})")
            return server
        except Exception as e:
            logger.warning(f"Failed to start language server for {language}: {e}")
            return None

    def _ensure_server_installed(self, language: str) -> bool:
        """Ensure LSP server is installed for the language."""
        try:
            from .lsp_installer import (
                get_server_for_language,
                check_server_installed,
                install_for_language,
                LANGUAGE_TO_SERVER,
            )
            
            server_id = LANGUAGE_TO_SERVER.get(language)
            if not server_id:
                logger.debug(f"No LSP server configured for {language}")
                return False
            
            if check_server_installed(server_id):
                return True
            
            logger.info(f"Auto-installing LSP server for {language}...")
            return install_for_language(language)
            
        except Exception as e:
            logger.warning(f"Failed to check/install LSP server: {e}")
            return False

    def _create_server(self, language: str, project_root: Path) -> Any | None:
        """Create a language server instance."""
        try:
            from rlm_dspy.vendor.solidlsp import SolidLanguageServer
            from rlm_dspy.vendor.solidlsp.ls_config import Language, LanguageServerConfig
            
            # Map our language string to solidlsp Language enum
            lang_enum = self._get_language_enum(language)
            if not lang_enum:
                return None
            
            config = LanguageServerConfig(code_language=lang_enum)
            server = SolidLanguageServer.create(
                config=config,
                repository_root_path=str(project_root),
                timeout=float(self.config.timeout),
            )
            server.start()
            return server
            
        except Exception as e:
            logger.error(f"Failed to create server for {language}: {e}")
            return None

    def _get_language_enum(self, language: str) -> Any | None:
        """Convert language string to solidlsp Language enum."""
        try:
            from rlm_dspy.vendor.solidlsp.ls_config import Language
            
            mapping = {
                "python": Language.PYTHON,
                "javascript": Language.TYPESCRIPT,  # TypeScript server handles both
                "typescript": Language.TYPESCRIPT,
                "go": Language.GO,
                "rust": Language.RUST,
                "java": Language.JAVA,
                "kotlin": Language.KOTLIN,
                "scala": Language.SCALA,
                "cpp": Language.CPP,
                "c": Language.CPP,  # clangd handles both
                "ruby": Language.RUBY,
                "csharp": Language.CSHARP,
                "lua": Language.LUA,
                "haskell": Language.HASKELL,
                "php": Language.PHP,
                "bash": Language.BASH,
            }
            return mapping.get(language)
        except ImportError:
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

    def _get_language(self, path: Path) -> str | None:
        """Get language for a file extension."""
        return EXT_TO_LANGUAGE.get(path.suffix.lower())

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

            result = []
            for ref in refs:
                if hasattr(ref, 'get'):
                    result.append({
                        "file": ref.get("uri", "").replace("file://", ""),
                        "line": ref.get("range", {}).get("start", {}).get("line", 0) + 1,
                        "column": ref.get("range", {}).get("start", {}).get("character", 0),
                    })
                elif hasattr(ref, 'uri'):
                    result.append({
                        "file": getattr(ref, 'uri', '').replace("file://", ""),
                        "line": getattr(ref, 'range', {}).get("start", {}).get("line", 0) + 1,
                        "column": getattr(ref, 'range', {}).get("start", {}).get("character", 0),
                    })
            return result
        except Exception as e:
            logger.warning(f"Failed to find references: {e}")
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
            abs_path = Path(file_path).resolve()
            project_root = self._find_project_root(abs_path)
            try:
                rel_path = str(abs_path.relative_to(project_root))
            except ValueError:
                rel_path = str(abs_path)

            result = server.request_definition(rel_path, line - 1, column)
            if not result:
                return None

            loc = result[0] if isinstance(result, list) else result

            return {
                "file": loc.get("uri", "").replace("file://", ""),
                "line": loc.get("range", {}).get("start", {}).get("line", 0) + 1,
                "column": loc.get("range", {}).get("start", {}).get("character", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to go to definition: {e}")
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
            abs_path = Path(file_path).resolve()
            project_root = self._find_project_root(abs_path)
            try:
                rel_path = str(abs_path.relative_to(project_root))
            except ValueError:
                rel_path = str(abs_path)

            result = server.request_hover(rel_path, line - 1, column)
            if not result:
                return None

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
            logger.warning(f"Failed to get hover info: {e}")
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
            abs_path = Path(file_path).resolve()
            project_root = self._find_project_root(abs_path)
            try:
                rel_path = str(abs_path.relative_to(project_root))
            except ValueError:
                rel_path = str(abs_path)

            result = server.request_document_symbols(rel_path)
            if result is None:
                return []

            if hasattr(result, 'root_symbols'):
                symbols = result.root_symbols
            elif isinstance(result, list):
                symbols = result
            else:
                symbols = []

            return self._flatten_symbols(symbols)
        except Exception as e:
            logger.warning(f"Failed to get document symbols: {e}")
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

    def shutdown(self) -> None:
        """Shutdown all running language servers."""
        for project, server in list(self._servers.items()):
            try:
                server.stop()
                logger.info(f"Stopped language server for {project}")
            except Exception as e:
                logger.warning(f"Error stopping server for {project}: {e}")
        self._servers.clear()

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            self.shutdown()
        except Exception:
            pass


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

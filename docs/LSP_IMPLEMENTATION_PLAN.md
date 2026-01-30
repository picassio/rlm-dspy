# solidlsp Vendor Implementation Plan

## Overview

**Goal**: Vendor solidlsp into rlm-dspy for full LSP support across all tree-sitter languages with auto-install capability.

**Update Strategy**: Sync with upstream solidlsp once per quarter.

## Language Coverage

| Language | Tree-sitter | LSP Server | Auto-Install |
|----------|-------------|------------|--------------|
| Python | ✅ | pyright | npm |
| JavaScript | ✅ | typescript-language-server | npm |
| TypeScript | ✅ | typescript-language-server | npm |
| Go | ✅ | gopls | go install |
| Rust | ✅ | rust-analyzer | rustup / download |
| Java | ✅ | eclipse_jdtls | download |
| C | ✅ | clangd | apt/brew / download |
| C++ | ✅ | clangd | apt/brew / download |
| C# | ✅ | omnisharp | download |
| Ruby | ✅ | ruby_lsp | gem |
| PHP | ✅ | intelephense | npm |
| Kotlin | ✅ | kotlin-language-server | download |
| Scala | ✅ | metals | coursier |
| Lua | ✅ | lua_ls | download |
| Haskell | ✅ | haskell-language-server | ghcup |
| Bash | ✅ | bash-language-server | npm |

## Directory Structure

```
src/rlm_dspy/
├── core/
│   ├── lsp.py                    # Our high-level wrapper (UPDATE)
│   └── lsp_installer.py          # NEW: Auto-install logic
└── vendor/
    └── solidlsp/                 # Vendored from serena
        ├── __init__.py
        ├── _compat.py            # NEW: Replace sensai/serena deps
        ├── ls.py
        ├── ls_config.py
        ├── ls_handler.py
        ├── ls_request.py
        ├── ls_types.py
        ├── ls_utils.py
        ├── ls_exceptions.py
        ├── settings.py
        ├── util/
        │   └── cache.py
        ├── lsp_protocol_handler/
        │   ├── __init__.py
        │   ├── server.py
        │   ├── lsp_types.py
        │   ├── lsp_requests.py
        │   └── lsp_constants.py
        └── language_servers/
            ├── __init__.py
            ├── common.py
            ├── pyright_server.py
            ├── typescript_language_server.py
            ├── gopls.py
            ├── rust_analyzer.py
            ├── eclipse_jdtls.py
            ├── clangd_language_server.py
            ├── omnisharp.py
            ├── ruby_lsp.py
            ├── intelephense.py
            ├── kotlin_language_server.py
            ├── scala_language_server.py
            ├── lua_ls.py
            ├── haskell_language_server.py
            └── bash_language_server.py
```

## Implementation Phases

### Phase 1: Create Compatibility Layer (~2 hours)

Create `src/rlm_dspy/vendor/solidlsp/_compat.py`:

```python
"""
Compatibility layer replacing sensai/serena dependencies.
"""
import pickle
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Self
import pathspec

# === Replace sensai.util.pickle ===

def getstate(cls, obj, transient_properties=None):
    """Get picklable state, excluding transient properties."""
    state = obj.__dict__.copy()
    for prop in (transient_properties or []):
        state.pop(prop, None)
    return state

def load_pickle(path):
    """Load pickled object from file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def dump_pickle(obj, path):
    """Dump object to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

# === Replace sensai.util.string ===

class ToStringMixin:
    """Mixin for pretty __str__ representation."""
    def __str__(self):
        attrs = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items() 
                         if not k.startswith('_'))
        return f'{self.__class__.__name__}({attrs})'

# === Replace sensai.util.logging ===

class LogTime:
    """Context manager for timing operations."""
    def __init__(self, description: str, logger=None):
        self.description = description
        self.logger = logger or logging.getLogger(__name__)
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        self.logger.info(f"{self.description}: {elapsed:.2f}s")

# === Replace serena.text_utils ===

class LineType(Enum):
    BEFORE_MATCH = "before"
    MATCH = "match"
    AFTER_MATCH = "after"

@dataclass
class TextLine:
    """A single line of text with metadata."""
    line_number: int
    content: str
    match_type: LineType = LineType.MATCH
    
    def format_line(self, include_line_numbers: bool = True) -> str:
        if include_line_numbers:
            return f"{self.line_number:4d} | {self.content}"
        return self.content

@dataclass
class MatchedConsecutiveLines:
    """Collection of consecutive lines from a file."""
    lines: list[TextLine]
    source_file_path: str | None = None
    lines_before_matched: list[TextLine] = field(default_factory=list)
    matched_lines: list[TextLine] = field(default_factory=list)
    lines_after_matched: list[TextLine] = field(default_factory=list)
    
    def __post_init__(self):
        for line in self.lines:
            if line.match_type == LineType.BEFORE_MATCH:
                self.lines_before_matched.append(line)
            elif line.match_type == LineType.MATCH:
                self.matched_lines.append(line)
            elif line.match_type == LineType.AFTER_MATCH:
                self.lines_after_matched.append(line)
    
    @property
    def start_line(self) -> int:
        return self.lines[0].line_number
    
    @property
    def end_line(self) -> int:
        return self.lines[-1].line_number
    
    def to_display_string(self, include_line_numbers: bool = True) -> str:
        return "\n".join(line.format_line(include_line_numbers) for line in self.lines)
    
    @classmethod
    def from_file_contents(
        cls, 
        file_contents: str, 
        line: int, 
        context_lines_before: int = 0, 
        context_lines_after: int = 0,
        source_file_path: str | None = None
    ) -> Self:
        all_lines = file_contents.split("\n")
        start = max(0, line - context_lines_before)
        end = min(len(all_lines) - 1, line + context_lines_after)
        
        text_lines = []
        for i in range(start, end + 1):
            if i < line:
                match_type = LineType.BEFORE_MATCH
            elif i == line:
                match_type = LineType.MATCH
            else:
                match_type = LineType.AFTER_MATCH
            text_lines.append(TextLine(i + 1, all_lines[i], match_type))
        
        return cls(lines=text_lines, source_file_path=source_file_path)

# === Replace serena.util.file_system ===

def match_path(relative_path: str, path_spec: pathspec.PathSpec, root_path: str = "") -> bool:
    """Match path against pathspec."""
    import os
    normalized = str(relative_path).replace(os.path.sep, "/")
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return path_spec.match_file(normalized)
```

### Phase 2: Vendor solidlsp Core (~1 hour)

1. Copy files from `~/projects/serena/src/solidlsp/` to `src/rlm_dspy/vendor/solidlsp/`
2. Update imports in all files:
   ```python
   # OLD
   from sensai.util.pickle import getstate, load_pickle
   from serena.text_utils import MatchedConsecutiveLines
   
   # NEW
   from rlm_dspy.vendor.solidlsp._compat import getstate, load_pickle
   from rlm_dspy.vendor.solidlsp._compat import MatchedConsecutiveLines
   ```

3. Create `__init__.py` files for proper package structure

### Phase 3: Vendor Language Servers (~1 hour)

Copy these language server configs:
- `common.py` (required base)
- `pyright_server.py`
- `typescript_language_server.py`
- `gopls.py`
- `rust_analyzer.py`
- `eclipse_jdtls.py`
- `clangd_language_server.py`
- `omnisharp.py`
- `ruby_lsp.py`
- `intelephense.py`
- `kotlin_language_server.py`
- `scala_language_server.py`
- `lua_ls.py`
- `haskell_language_server.py`
- `bash_language_server.py`

### Phase 4: Create Auto-Installer (~3 hours)

Create `src/rlm_dspy/core/lsp_installer.py`:

```python
"""
Auto-installer for LSP servers.
Downloads and configures language servers automatically.
"""
import shutil
import subprocess
import platform
from pathlib import Path
from typing import Callable
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)

LSP_INSTALL_DIR = Path.home() / ".rlm" / "lsp_servers"

@dataclass
class ServerInfo:
    name: str
    languages: list[str]
    check_cmd: str | Callable[[], bool]  # Command to check if installed
    install_methods: dict[str, str]  # platform -> install command
    download_url: str | None = None  # For direct download

SERVERS = {
    "pyright": ServerInfo(
        name="pyright",
        languages=["python"],
        check_cmd="pyright --version",
        install_methods={
            "any": "npm install -g pyright",
        }
    ),
    "typescript-language-server": ServerInfo(
        name="typescript-language-server", 
        languages=["javascript", "typescript"],
        check_cmd="typescript-language-server --version",
        install_methods={
            "any": "npm install -g typescript-language-server typescript",
        }
    ),
    "gopls": ServerInfo(
        name="gopls",
        languages=["go"],
        check_cmd="gopls version",
        install_methods={
            "any": "go install golang.org/x/tools/gopls@latest",
        }
    ),
    "rust-analyzer": ServerInfo(
        name="rust-analyzer",
        languages=["rust"],
        check_cmd="rust-analyzer --version",
        install_methods={
            "any": "rustup component add rust-analyzer",
        },
        download_url="https://github.com/rust-lang/rust-analyzer/releases/latest"
    ),
    # ... more servers
}

def check_server_installed(server_id: str) -> bool:
    """Check if LSP server is installed and working."""
    info = SERVERS.get(server_id)
    if not info:
        return False
    
    if callable(info.check_cmd):
        return info.check_cmd()
    
    try:
        result = subprocess.run(
            info.check_cmd.split(),
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False

def install_server(server_id: str, force: bool = False) -> bool:
    """Install LSP server."""
    if not force and check_server_installed(server_id):
        log.info(f"{server_id} already installed")
        return True
    
    info = SERVERS.get(server_id)
    if not info:
        log.error(f"Unknown server: {server_id}")
        return False
    
    # Get install command for current platform
    system = platform.system().lower()
    cmd = info.install_methods.get(system) or info.install_methods.get("any")
    
    if not cmd:
        log.error(f"No install method for {server_id} on {system}")
        return False
    
    log.info(f"Installing {server_id}: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            log.info(f"Successfully installed {server_id}")
            return True
        else:
            log.error(f"Failed to install {server_id}: {result.stderr}")
            return False
    except Exception as e:
        log.error(f"Error installing {server_id}: {e}")
        return False

def install_for_language(language: str) -> bool:
    """Install LSP server for given language."""
    for server_id, info in SERVERS.items():
        if language in info.languages:
            return install_server(server_id)
    log.warning(f"No LSP server configured for {language}")
    return False

def get_server_status() -> dict[str, dict]:
    """Get status of all LSP servers."""
    status = {}
    for server_id, info in SERVERS.items():
        status[server_id] = {
            "installed": check_server_installed(server_id),
            "languages": info.languages,
            "install_cmd": info.install_methods.get("any", "N/A")
        }
    return status
```

### Phase 5: Update lsp.py Wrapper (~2 hours)

Update `src/rlm_dspy/core/lsp.py` to:
1. Use vendored solidlsp
2. Auto-install servers when needed
3. Graceful fallback to tree-sitter

```python
"""
LSP client wrapper with auto-install support.
"""
from pathlib import Path
from typing import Any
import logging

from rlm_dspy.vendor.solidlsp import SolidLanguageServer
from rlm_dspy.vendor.solidlsp.ls_config import Language, LanguageServerConfig
from rlm_dspy.core.lsp_installer import (
    check_server_installed, 
    install_for_language,
    get_server_status
)

log = logging.getLogger(__name__)

# Map file extensions to languages
EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".hs": "haskell",
    ".sh": "bash",
}

class LSPClient:
    """High-level LSP client with auto-install."""
    
    def __init__(self, project_root: str, auto_install: bool = True):
        self.project_root = Path(project_root)
        self.auto_install = auto_install
        self._servers: dict[str, SolidLanguageServer] = {}
    
    def _get_server(self, language: str) -> SolidLanguageServer | None:
        """Get or create LSP server for language."""
        if language in self._servers:
            return self._servers[language]
        
        # Check if server is installed
        if not check_server_installed(self._get_server_id(language)):
            if self.auto_install:
                log.info(f"Auto-installing LSP server for {language}")
                if not install_for_language(language):
                    return None
            else:
                return None
        
        # Create server
        try:
            config = self._get_config(language)
            server = SolidLanguageServer(config, str(self.project_root))
            self._servers[language] = server
            return server
        except Exception as e:
            log.error(f"Failed to create LSP server for {language}: {e}")
            return None
    
    def find_references(self, file_path: str, line: int, column: int) -> list[dict]:
        """Find all references to symbol at position."""
        lang = self._detect_language(file_path)
        server = self._get_server(lang)
        if not server:
            return []
        
        refs = server.find_references_to_item(file_path, line, column)
        return [self._format_location(r) for r in refs]
    
    def go_to_definition(self, file_path: str, line: int, column: int) -> list[dict]:
        """Go to definition of symbol at position."""
        lang = self._detect_language(file_path)
        server = self._get_server(lang)
        if not server:
            return []
        
        defs = server.find_definition(file_path, line, column)
        return [self._format_location(d) for d in defs]
    
    def get_hover(self, file_path: str, line: int, column: int) -> str:
        """Get hover information for symbol at position."""
        lang = self._detect_language(file_path)
        server = self._get_server(lang)
        if not server:
            return ""
        
        return server.get_hover(file_path, line, column)
    
    def get_document_symbols(self, file_path: str) -> list[dict]:
        """Get all symbols in document."""
        lang = self._detect_language(file_path)
        server = self._get_server(lang)
        if not server:
            return []
        
        symbols = server.get_document_symbols(file_path)
        return [self._format_symbol(s) for s in symbols]
    
    def _detect_language(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        return EXT_TO_LANG.get(ext, "unknown")
    
    def _get_server_id(self, language: str) -> str:
        # Map language to server ID
        mapping = {
            "python": "pyright",
            "javascript": "typescript-language-server",
            "typescript": "typescript-language-server",
            "go": "gopls",
            "rust": "rust-analyzer",
            # ...
        }
        return mapping.get(language, language)
    
    def shutdown(self):
        """Shutdown all LSP servers."""
        for server in self._servers.values():
            try:
                server.shutdown()
            except Exception:
                pass
        self._servers.clear()
```

### Phase 6: Add CLI Commands (~1 hour)

Add to `cli.py`:

```python
# LSP command group
lsp_app = typer.Typer(help="LSP server management")
app.add_typer(lsp_app, name="lsp")

@lsp_app.command("status")
def lsp_status():
    """Show LSP server status."""
    from rlm_dspy.core.lsp_installer import get_server_status
    
    status = get_server_status()
    for server_id, info in status.items():
        icon = "✅" if info["installed"] else "❌"
        langs = ", ".join(info["languages"])
        console.print(f"{icon} {server_id} ({langs})")
        if not info["installed"]:
            console.print(f"   Install: {info['install_cmd']}", style="dim")

@lsp_app.command("install")
def lsp_install(
    language: str = typer.Argument(..., help="Language to install server for"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall")
):
    """Install LSP server for a language."""
    from rlm_dspy.core.lsp_installer import install_for_language, install_server
    
    if install_for_language(language):
        console.print(f"[green]✓ Installed LSP server for {language}[/green]")
    else:
        console.print(f"[red]✗ Failed to install LSP server for {language}[/red]")
        raise typer.Exit(1)

@lsp_app.command("install-all")
def lsp_install_all():
    """Install all LSP servers."""
    from rlm_dspy.core.lsp_installer import SERVERS, install_server
    
    for server_id in SERVERS:
        console.print(f"Installing {server_id}...", end=" ")
        if install_server(server_id):
            console.print("[green]✓[/green]")
        else:
            console.print("[red]✗[/red]")
```

### Phase 7: Update pyproject.toml (~30 min)

```toml
dependencies = [
    # ... existing ...
    
    # LSP support
    "psutil>=5.9.0",           # Process management for LSP servers
    "charset-normalizer>=3.0", # Encoding detection
]
```

### Phase 8: Write Tests (~2 hours)

Create `tests/test_lsp_integration.py`:

```python
"""Integration tests for LSP functionality."""
import pytest
from pathlib import Path

from rlm_dspy.core.lsp import LSPClient
from rlm_dspy.core.lsp_installer import check_server_installed, SERVERS

# Test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"

class TestLSPInstaller:
    def test_get_server_status(self):
        from rlm_dspy.core.lsp_installer import get_server_status
        status = get_server_status()
        assert "pyright" in status
        assert "languages" in status["pyright"]
    
    def test_check_server_installed(self):
        # This should work even if server not installed
        result = check_server_installed("nonexistent-server")
        assert result is False

class TestLSPClient:
    @pytest.fixture
    def client(self, tmp_path):
        # Create a simple Python project
        (tmp_path / "main.py").write_text("""
def hello():
    return "world"

result = hello()
""")
        return LSPClient(str(tmp_path), auto_install=False)
    
    @pytest.mark.skipif(
        not check_server_installed("pyright"),
        reason="pyright not installed"
    )
    def test_find_references_python(self, client, tmp_path):
        refs = client.find_references(str(tmp_path / "main.py"), 5, 9)
        # Should find both definition and usage of hello()
        assert len(refs) >= 2
    
    @pytest.mark.skipif(
        not check_server_installed("pyright"),
        reason="pyright not installed"
    )
    def test_go_to_definition_python(self, client, tmp_path):
        defs = client.go_to_definition(str(tmp_path / "main.py"), 5, 9)
        assert len(defs) == 1
        assert defs[0]["line"] == 2  # def hello() is on line 2

# Parametrized tests for each language
@pytest.mark.parametrize("lang,server", [
    ("python", "pyright"),
    ("javascript", "typescript-language-server"),
    ("typescript", "typescript-language-server"),
    ("go", "gopls"),
    ("rust", "rust-analyzer"),
])
def test_server_configured(lang, server):
    """Verify each language has a configured server."""
    assert server in SERVERS
    assert lang in SERVERS[server].languages
```

### Phase 9: Update Documentation (~1 hour)

Update README.md with:
- LSP feature description
- `rlm-dspy lsp status` command
- `rlm-dspy lsp install` commands
- Auto-install behavior explanation
- Manual install instructions as fallback

## Timeline Summary

| Phase | Task | Time |
|-------|------|------|
| 1 | Create _compat.py | 2 hours |
| 2 | Vendor solidlsp core | 1 hour |
| 3 | Vendor language servers | 1 hour |
| 4 | Create auto-installer | 3 hours |
| 5 | Update lsp.py wrapper | 2 hours |
| 6 | Add CLI commands | 1 hour |
| 7 | Update pyproject.toml | 30 min |
| 8 | Write tests | 2 hours |
| 9 | Update documentation | 1 hour |
| **Total** | | **~14 hours** |

## Testing Checklist

- [x] Python LSP (pyright) works - Found 24 references ✓
- [x] TypeScript/JS LSP works - Found 2 symbols ✓
- [x] Go LSP works - gopls installed ✓
- [x] Rust LSP works - rust-analyzer installed ✓
- [x] Auto-install works for each server ✓
- [x] Graceful fallback when server unavailable ✓
- [x] `rlm-dspy lsp status` shows correct info - 7/16 servers ✓
- [x] `rlm-dspy lsp install <lang>` works ✓
- [x] Tools (`find_references`, `go_to_definition`) work in RLM queries ✓
- [x] All existing tests still pass - 646 passed ✓

## Current Installation Status (as of 2025-01-30)

| Server | Status | Languages |
|--------|--------|-----------|
| Pyright | ✅ Installed | Python |
| TypeScript LS | ✅ Installed | JavaScript, TypeScript |
| gopls | ✅ Installed | Go |
| rust-analyzer | ✅ Installed | Rust |
| clangd | ✅ Installed | C, C++ |
| Intelephense | ✅ Installed | PHP |
| Bash LS | ✅ Installed | Bash |
| Jedi | ○ Not installed | Python (alternative) |
| Eclipse JDTLS | ○ Missing java | Java |
| OmniSharp | ○ Auto-install | C# |
| Ruby LSP | ○ Missing gem | Ruby |
| Kotlin LS | ○ Missing java | Kotlin |
| Metals | ○ Not installed | Scala |
| Lua LS | ○ Auto-install | Lua |
| Haskell LS | ○ Not installed | Haskell |

## Rollback Plan

If issues arise:
1. LSP tools return "(LSP not available)" message
2. Fall back to tree-sitter for structural queries
3. Semantic search still works for conceptual queries

## Future Improvements

1. **Caching**: Cache LSP server connections across queries
2. **Parallel startup**: Start multiple servers concurrently
3. **Health monitoring**: Auto-restart crashed servers
4. **IDE integration**: VS Code extension support

"""
Auto-installer for LSP servers.

Downloads and configures language servers automatically based on the
programming languages detected in the project.
"""

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

__all__ = [
    "ServerInfo",
    "SERVERS",
    "LSP_INSTALL_DIR",
    "check_server_installed",
    "install_server",
    "install_for_language",
    "get_server_status",
    "get_server_for_language",
]

log = logging.getLogger(__name__)

# Directory where we install LSP servers
LSP_INSTALL_DIR = Path.home() / ".rlm" / "lsp_servers"


@dataclass
class ServerInfo:
    """Information about an LSP server."""
    
    name: str
    """Human-readable name of the server."""
    
    languages: list[str]
    """List of languages this server supports."""
    
    check_cmd: str | list[str] | Callable[[], bool]
    """Command to check if server is installed, or a callable that returns bool."""
    
    install_methods: dict[str, str | list[str]]
    """Platform -> install command mapping. Use 'any' for cross-platform."""
    
    server_class: str
    """Full path to the SolidLanguageServer subclass."""
    
    requires: list[str] = field(default_factory=list)
    """Required system tools (e.g., 'npm', 'go', 'rustup')."""
    
    download_url: str | None = None
    """URL for direct download (for servers without package managers)."""
    
    notes: str = ""
    """Additional notes for the user."""


def _check_npm() -> bool:
    """Check if npm is available."""
    return shutil.which("npm") is not None


def _check_go() -> bool:
    """Check if go is available."""
    return shutil.which("go") is not None


def _check_rustup() -> bool:
    """Check if rustup is available."""
    return shutil.which("rustup") is not None


def _check_gem() -> bool:
    """Check if gem (Ruby) is available."""
    return shutil.which("gem") is not None


def _check_java() -> bool:
    """Check if java is available."""
    return shutil.which("java") is not None


# =============================================================================
# Server Registry
# =============================================================================

SERVERS: dict[str, ServerInfo] = {
    # Python
    "pyright": ServerInfo(
        name="Pyright",
        languages=["python"],
        check_cmd=["pyright", "--version"],
        install_methods={
            "any": "npm install -g pyright",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.pyright_server.PyrightServer",
        requires=["npm"],
        notes="Fast Python type checker and language server from Microsoft.",
    ),
    "jedi": ServerInfo(
        name="Jedi Language Server",
        languages=["python"],
        check_cmd=["jedi-language-server", "--version"],
        install_methods={
            "any": "pip install jedi-language-server",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.jedi_server.JediServer",
        requires=[],
        notes="Pure Python language server using Jedi.",
    ),
    
    # JavaScript/TypeScript
    "typescript-language-server": ServerInfo(
        name="TypeScript Language Server",
        languages=["javascript", "typescript"],
        check_cmd=["typescript-language-server", "--version"],
        install_methods={
            "any": "npm install -g typescript-language-server typescript",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.typescript_language_server.TypeScriptLanguageServer",
        requires=["npm"],
        notes="Official TypeScript/JavaScript language server.",
    ),
    
    # Go
    "gopls": ServerInfo(
        name="gopls",
        languages=["go"],
        check_cmd=["gopls", "version"],
        install_methods={
            "any": "go install golang.org/x/tools/gopls@latest",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.gopls.Gopls",
        requires=["go"],
        notes="Official Go language server.",
    ),
    
    # Rust
    "rust-analyzer": ServerInfo(
        name="rust-analyzer",
        languages=["rust"],
        check_cmd=["rust-analyzer", "--version"],
        install_methods={
            "any": "rustup component add rust-analyzer",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.rust_analyzer.RustAnalyzer",
        requires=["rustup"],
        download_url="https://github.com/rust-lang/rust-analyzer/releases",
        notes="Official Rust language server.",
    ),
    
    # Java
    "jdtls": ServerInfo(
        name="Eclipse JDT Language Server",
        languages=["java"],
        check_cmd=lambda: (LSP_INSTALL_DIR / "jdtls").exists(),
        install_methods={
            # JDTLS requires manual download - the server class handles this
            "any": "auto",  # Handled by EclipseJDTLS class
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.eclipse_jdtls.EclipseJDTLS",
        requires=["java"],
        download_url="https://download.eclipse.org/jdtls/milestones/",
        notes="Eclipse Java Development Tools Language Server.",
    ),
    
    # C/C++
    "clangd": ServerInfo(
        name="clangd",
        languages=["c", "cpp"],
        check_cmd=["clangd", "--version"],
        install_methods={
            "linux": "sudo apt install clangd || sudo dnf install clangd",
            "darwin": "brew install llvm",
            "windows": "choco install llvm",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.clangd_language_server.ClangdLanguageServer",
        requires=[],
        notes="LLVM-based C/C++ language server.",
    ),
    
    # C#
    "omnisharp": ServerInfo(
        name="OmniSharp",
        languages=["csharp"],
        check_cmd=lambda: (LSP_INSTALL_DIR / "omnisharp").exists() or shutil.which("omnisharp") is not None,
        install_methods={
            "any": "auto",  # Handled by OmniSharp class
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.omnisharp.OmniSharp",
        requires=[],
        download_url="https://github.com/OmniSharp/omnisharp-roslyn/releases",
        notes="Cross-platform .NET development server.",
    ),
    
    # Ruby
    "ruby-lsp": ServerInfo(
        name="Ruby LSP",
        languages=["ruby"],
        check_cmd=["ruby-lsp", "--version"],
        install_methods={
            "any": "gem install ruby-lsp",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.ruby_lsp.RubyLsp",
        requires=["gem"],
        notes="Official Ruby language server from Shopify.",
    ),
    "solargraph": ServerInfo(
        name="Solargraph",
        languages=["ruby"],
        check_cmd=["solargraph", "--version"],
        install_methods={
            "any": "gem install solargraph",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.solargraph.Solargraph",
        requires=["gem"],
        notes="Ruby language server with documentation support.",
    ),
    
    # PHP
    "intelephense": ServerInfo(
        name="Intelephense",
        languages=["php"],
        check_cmd=lambda: shutil.which("intelephense") is not None,
        install_methods={
            "any": "npm install -g intelephense",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.intelephense.Intelephense",
        requires=["npm"],
        notes="High-performance PHP language server.",
    ),
    
    # Kotlin
    "kotlin-language-server": ServerInfo(
        name="Kotlin Language Server",
        languages=["kotlin"],
        check_cmd=lambda: (LSP_INSTALL_DIR / "kotlin-language-server").exists(),
        install_methods={
            "any": "auto",  # Handled by KotlinLanguageServer class
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.kotlin_language_server.KotlinLanguageServer",
        requires=["java"],
        download_url="https://github.com/fwcd/kotlin-language-server/releases",
        notes="Community Kotlin language server.",
    ),
    
    # Scala
    "metals": ServerInfo(
        name="Metals",
        languages=["scala"],
        check_cmd=["metals", "--version"],
        install_methods={
            "any": "coursier install metals",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.scala_language_server.ScalaLanguageServer",
        requires=[],  # Coursier is downloaded automatically
        notes="Scala language server.",
    ),
    
    # Lua
    "lua-language-server": ServerInfo(
        name="Lua Language Server",
        languages=["lua"],
        check_cmd=["lua-language-server", "--version"],
        install_methods={
            "linux": "auto",  # Downloaded by LuaLanguageServer class
            "darwin": "brew install lua-language-server",
            "windows": "auto",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.lua_ls.LuaLanguageServer",
        requires=[],
        download_url="https://github.com/LuaLS/lua-language-server/releases",
        notes="Lua language server by sumneko.",
    ),
    
    # Haskell
    "haskell-language-server": ServerInfo(
        name="Haskell Language Server",
        languages=["haskell"],
        check_cmd=["haskell-language-server-wrapper", "--version"],
        install_methods={
            "any": "ghcup install hls",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.haskell_language_server.HaskellLanguageServer",
        requires=[],  # ghcup handles installation
        notes="Official Haskell language server.",
    ),
    
    # Bash
    "bash-language-server": ServerInfo(
        name="Bash Language Server",
        languages=["bash"],
        check_cmd=["bash-language-server", "--version"],
        install_methods={
            "any": "npm install -g bash-language-server",
        },
        server_class="rlm_dspy.vendor.solidlsp.language_servers.bash_language_server.BashLanguageServer",
        requires=["npm"],
        notes="Bash/shell script language server.",
    ),
}

# Language to preferred server mapping (when multiple servers support a language)
LANGUAGE_TO_SERVER: dict[str, str] = {
    "python": "pyright",  # Prefer pyright over jedi for speed
    "javascript": "typescript-language-server",
    "typescript": "typescript-language-server",
    "go": "gopls",
    "rust": "rust-analyzer",
    "java": "jdtls",
    "c": "clangd",
    "cpp": "clangd",
    "csharp": "omnisharp",
    "ruby": "ruby-lsp",  # Prefer ruby-lsp over solargraph
    "php": "intelephense",
    "kotlin": "kotlin-language-server",
    "scala": "metals",
    "lua": "lua-language-server",
    "haskell": "haskell-language-server",
    "bash": "bash-language-server",
}


# =============================================================================
# Public API
# =============================================================================


def check_server_installed(server_id: str) -> bool:
    """
    Check if an LSP server is installed and working.
    
    Args:
        server_id: The server identifier (e.g., 'pyright', 'gopls')
        
    Returns:
        True if the server is installed and responds to version check
    """
    info = SERVERS.get(server_id)
    if not info:
        return False
    
    check = info.check_cmd
    
    # Callable check
    if callable(check):
        try:
            return check()
        except Exception:
            return False
    
    # Command check
    try:
        cmd = check if isinstance(check, list) else check.split()
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def check_requirements(server_id: str) -> tuple[bool, list[str]]:
    """
    Check if requirements for a server are met.
    
    Args:
        server_id: The server identifier
        
    Returns:
        Tuple of (all_met, list_of_missing_requirements)
    """
    info = SERVERS.get(server_id)
    if not info:
        return False, [f"Unknown server: {server_id}"]
    
    missing = []
    checkers = {
        "npm": _check_npm,
        "go": _check_go,
        "rustup": _check_rustup,
        "gem": _check_gem,
        "java": _check_java,
    }
    
    for req in info.requires:
        checker = checkers.get(req)
        if checker and not checker():
            missing.append(req)
    
    return len(missing) == 0, missing


def install_server(server_id: str, force: bool = False) -> bool:
    """
    Install an LSP server.
    
    Args:
        server_id: The server identifier (e.g., 'pyright', 'gopls')
        force: Force reinstall even if already installed
        
    Returns:
        True if installation succeeded
    """
    if not force and check_server_installed(server_id):
        log.info(f"{server_id} already installed")
        return True
    
    info = SERVERS.get(server_id)
    if not info:
        log.error(f"Unknown server: {server_id}")
        return False
    
    # Check requirements
    reqs_met, missing = check_requirements(server_id)
    if not reqs_met:
        log.error(f"Missing requirements for {server_id}: {', '.join(missing)}")
        return False
    
    # Get install command for current platform
    system = platform.system().lower()
    cmd = info.install_methods.get(system) or info.install_methods.get("any")
    
    if not cmd:
        log.error(f"No install method for {server_id} on {system}")
        return False
    
    # Handle auto-install servers (they manage their own installation)
    if cmd == "auto":
        log.info(f"{server_id} uses auto-install - will be set up on first use")
        return True
    
    log.info(f"Installing {server_id}: {cmd}")
    try:
        # Ensure LSP install directory exists
        LSP_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Run install command
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode == 0:
            log.info(f"Successfully installed {server_id}")
            return True
        else:
            log.error(f"Failed to install {server_id}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        log.error(f"Installation of {server_id} timed out")
        return False
    except Exception as e:
        log.error(f"Error installing {server_id}: {e}")
        return False


def install_for_language(language: str) -> bool:
    """
    Install the preferred LSP server for a given language.
    
    Args:
        language: The programming language (e.g., 'python', 'go')
        
    Returns:
        True if installation succeeded
    """
    server_id = LANGUAGE_TO_SERVER.get(language.lower())
    if not server_id:
        log.warning(f"No LSP server configured for {language}")
        return False
    
    return install_server(server_id)


def get_server_for_language(language: str) -> ServerInfo | None:
    """
    Get the preferred server info for a language.
    
    Args:
        language: The programming language
        
    Returns:
        ServerInfo or None if no server configured
    """
    server_id = LANGUAGE_TO_SERVER.get(language.lower())
    if server_id:
        return SERVERS.get(server_id)
    return None


def get_server_status() -> dict[str, dict]:
    """
    Get status of all LSP servers.
    
    Returns:
        Dictionary mapping server_id to status dict with keys:
        - installed: bool
        - languages: list[str]
        - install_cmd: str
        - requirements: list[str]
        - requirements_met: bool
    """
    status = {}
    for server_id, info in SERVERS.items():
        reqs_met, missing = check_requirements(server_id)
        
        # Get install command
        system = platform.system().lower()
        cmd = info.install_methods.get(system) or info.install_methods.get("any", "N/A")
        
        status[server_id] = {
            "name": info.name,
            "installed": check_server_installed(server_id),
            "languages": info.languages,
            "install_cmd": cmd if cmd != "auto" else "(automatic)",
            "requirements": info.requires,
            "requirements_met": reqs_met,
            "missing_requirements": missing,
            "notes": info.notes,
        }
    return status


def install_all_servers(skip_missing_requirements: bool = True) -> dict[str, bool]:
    """
    Install all LSP servers.
    
    Args:
        skip_missing_requirements: Skip servers with missing requirements
        
    Returns:
        Dictionary mapping server_id to installation success
    """
    results = {}
    for server_id in SERVERS:
        if skip_missing_requirements:
            reqs_met, _ = check_requirements(server_id)
            if not reqs_met:
                log.info(f"Skipping {server_id} - missing requirements")
                results[server_id] = False
                continue
        
        results[server_id] = install_server(server_id)
    
    return results

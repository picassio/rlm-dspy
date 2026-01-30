"""Integration tests for LSP functionality.

These tests verify that the vendored solidlsp and auto-installer work correctly.
"""

import pytest
from pathlib import Path

from rlm_dspy.core.lsp import LSPManager, LSPConfig, get_lsp_manager, clear_lsp_manager
from rlm_dspy.core.lsp_installer import (
    SERVERS,
    LANGUAGE_TO_SERVER,
    check_server_installed,
    check_requirements,
    get_server_status,
)


class TestLSPInstaller:
    """Tests for the LSP installer module."""
    
    def test_servers_configured(self):
        """All expected servers are configured."""
        expected_servers = [
            "pyright", "jedi", "typescript-language-server", "gopls",
            "rust-analyzer", "jdtls", "clangd", "omnisharp",
            "ruby-lsp", "intelephense", "kotlin-language-server",
            "metals", "lua-language-server", "haskell-language-server",
            "bash-language-server",
        ]
        for server in expected_servers:
            assert server in SERVERS, f"Server {server} not configured"
    
    def test_language_mapping_complete(self):
        """All tree-sitter languages have LSP mappings."""
        expected_languages = [
            "python", "javascript", "typescript", "go", "rust",
            "java", "c", "cpp", "csharp", "ruby", "php",
            "kotlin", "scala", "lua", "haskell", "bash",
        ]
        for lang in expected_languages:
            assert lang in LANGUAGE_TO_SERVER, f"Language {lang} has no server mapping"
    
    def test_server_info_complete(self):
        """All server info objects have required fields."""
        for server_id, info in SERVERS.items():
            assert info.name, f"{server_id} missing name"
            assert info.languages, f"{server_id} missing languages"
            assert info.check_cmd, f"{server_id} missing check_cmd"
            assert info.install_methods, f"{server_id} missing install_methods"
            assert info.server_class, f"{server_id} missing server_class"
    
    def test_get_server_status(self):
        """get_server_status returns valid status dict."""
        status = get_server_status()
        
        assert len(status) == len(SERVERS)
        for server_id, info in status.items():
            assert "name" in info
            assert "installed" in info
            assert "languages" in info
            assert "install_cmd" in info
            assert "requirements_met" in info


class TestLSPManagerWithPyright:
    """Tests for LSP Manager with pyright (if installed)."""
    
    @pytest.fixture
    def manager(self):
        """Create manager with auto-install disabled."""
        clear_lsp_manager()
        return LSPManager(LSPConfig(enabled=True, auto_install=False))
    
    @pytest.fixture
    def test_file(self, tmp_path):
        """Create a test Python file."""
        code = '''
class MyClass:
    """A test class."""
    
    def __init__(self, value: int):
        self.value = value
    
    def get_value(self) -> int:
        return self.value


def main():
    obj = MyClass(42)
    print(obj.get_value())


if __name__ == "__main__":
    main()
'''
        test_file = tmp_path / "test_code.py"
        test_file.write_text(code)
        return test_file
    
    @pytest.mark.skipif(
        not check_server_installed("pyright"),
        reason="pyright not installed"
    )
    def test_find_references(self, manager, test_file):
        """Find references to MyClass."""
        # Line 2 is the class definition
        refs = manager.find_references(str(test_file), line=2, column=6)
        
        # Should find at least one reference
        assert len(refs) >= 1
    
    @pytest.mark.skipif(
        not check_server_installed("pyright"),
        reason="pyright not installed"
    )
    def test_go_to_definition(self, manager, test_file):
        """Go to definition of MyClass from usage."""
        # Line 13 has `obj = MyClass(42)`
        defn = manager.go_to_definition(str(test_file), line=13, column=10)
        
        assert defn is not None
        assert defn["line"] == 2  # Class definition is on line 2
    
    @pytest.mark.skipif(
        not check_server_installed("pyright"),
        reason="pyright not installed"
    )
    def test_get_hover_info(self, manager, test_file):
        """Get hover info for MyClass."""
        hover = manager.get_hover_info(str(test_file), line=2, column=6)
        
        assert hover is not None
        assert "MyClass" in hover
    
    @pytest.mark.skipif(
        not check_server_installed("pyright"),
        reason="pyright not installed"
    )
    def test_get_document_symbols(self, manager, test_file):
        """Get all symbols in document."""
        symbols = manager.get_document_symbols(str(test_file))
        
        assert len(symbols) > 0
        
        # Find class and function
        names = [s["name"] for s in symbols]
        assert "MyClass" in names
        assert "main" in names


class TestLSPManagerWithGopls:
    """Tests for LSP Manager with gopls (if installed)."""
    
    @pytest.fixture
    def manager(self):
        """Create manager with auto-install disabled."""
        clear_lsp_manager()
        return LSPManager(LSPConfig(enabled=True, auto_install=False))
    
    @pytest.fixture
    def test_file(self, tmp_path):
        """Create a test Go file."""
        # Create go.mod for proper project detection
        (tmp_path / "go.mod").write_text("module test\ngo 1.21\n")
        
        code = '''package main

import "fmt"

type MyStruct struct {
    Value int
}

func (m *MyStruct) GetValue() int {
    return m.Value
}

func main() {
    obj := MyStruct{Value: 42}
    fmt.Println(obj.GetValue())
}
'''
        test_file = tmp_path / "main.go"
        test_file.write_text(code)
        return test_file
    
    @pytest.mark.skipif(
        not check_server_installed("gopls"),
        reason="gopls not installed"
    )
    def test_find_references_go(self, manager, test_file):
        """Find references to MyStruct."""
        # Line 5 is the struct definition
        refs = manager.find_references(str(test_file), line=5, column=5)
        
        # Should find at least the definition and usage
        assert len(refs) >= 2
    
    @pytest.mark.skipif(
        not check_server_installed("gopls"),
        reason="gopls not installed"
    )
    def test_get_document_symbols_go(self, manager, test_file):
        """Get all symbols in Go document."""
        symbols = manager.get_document_symbols(str(test_file))
        
        assert len(symbols) > 0
        
        names = [s["name"] for s in symbols]
        assert "MyStruct" in names or "main" in names

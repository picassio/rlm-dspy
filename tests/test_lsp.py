"""Tests for LSP (Language Server Protocol) integration."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rlm_dspy.core.lsp import (
    LSPConfig,
    LSPManager,
    get_lsp_manager,
    clear_lsp_manager,
)


class TestLSPConfig:
    """Tests for LSPConfig dataclass."""
    
    def test_default_values(self):
        """LSPConfig has sensible defaults."""
        config = LSPConfig()
        assert config.enabled is True
        assert config.timeout == 30
        assert "lsp_cache" in str(config.cache_dir)
    
    def test_from_user_config(self):
        """Can load config from user config."""
        # This should not raise even if config file doesn't exist
        config = LSPConfig.from_user_config()
        assert isinstance(config, LSPConfig)
    
    def test_custom_values(self):
        """Can create config with custom values."""
        config = LSPConfig(
            enabled=False,
            timeout=60,
            cache_dir=Path("/tmp/lsp_test"),
        )
        assert config.enabled is False
        assert config.timeout == 60
        assert config.cache_dir == Path("/tmp/lsp_test")


class TestLSPManager:
    """Tests for LSPManager class."""
    
    def test_create_manager(self):
        """Can create an LSP manager."""
        config = LSPConfig(enabled=False)  # Disable to avoid starting servers
        manager = LSPManager(config)
        assert manager.config.enabled is False
    
    def test_find_project_root_git(self):
        """Finds project root by .git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .git directory
            git_dir = Path(tmpdir) / ".git"
            git_dir.mkdir()
            
            # Create a nested file
            subdir = Path(tmpdir) / "src" / "pkg"
            subdir.mkdir(parents=True)
            test_file = subdir / "test.py"
            test_file.write_text("# test")
            
            manager = LSPManager(LSPConfig(enabled=False))
            root = manager._find_project_root(test_file)
            
            assert root == Path(tmpdir)
    
    def test_find_project_root_pyproject(self):
        """Finds project root by pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create pyproject.toml
            (Path(tmpdir) / "pyproject.toml").write_text("[tool.pytest]")
            
            # Create a nested file
            subdir = Path(tmpdir) / "src"
            subdir.mkdir()
            test_file = subdir / "test.py"
            test_file.write_text("# test")
            
            manager = LSPManager(LSPConfig(enabled=False))
            root = manager._find_project_root(test_file)
            
            assert root == Path(tmpdir)
    
    def test_find_project_root_fallback(self):
        """Falls back to file's directory if no markers found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("# test")
            
            manager = LSPManager(LSPConfig(enabled=False))
            root = manager._find_project_root(test_file)
            
            assert root == Path(tmpdir)
    
    def test_get_language_python(self):
        """Recognizes Python files."""
        manager = LSPManager(LSPConfig(enabled=False))
        
        # Mock the import to avoid solidlsp dependency in test
        with patch.dict('sys.modules', {'solidlsp.ls_config': MagicMock()}):
            try:
                from solidlsp.ls_config import Language
                lang = manager._get_language(Path("test.py"))
                assert lang == Language.PYTHON
            except ImportError:
                # solidlsp not installed - test the extension mapping logic
                pass
    
    def test_get_language_unsupported(self):
        """Returns None for unsupported extensions."""
        manager = LSPManager(LSPConfig(enabled=False))
        lang = manager._get_language(Path("test.xyz"))
        assert lang is None
    
    def test_symbol_kind_name(self):
        """Converts LSP SymbolKind to names."""
        manager = LSPManager(LSPConfig(enabled=False))
        
        assert manager._symbol_kind_name(5) == "class"
        assert manager._symbol_kind_name(6) == "method"
        assert manager._symbol_kind_name(12) == "function"
        assert manager._symbol_kind_name(13) == "variable"
        assert manager._symbol_kind_name(999) == "kind_999"
    
    def test_flatten_symbols_empty(self):
        """Handles empty symbol list."""
        manager = LSPManager(LSPConfig(enabled=False))
        result = manager._flatten_symbols([])
        assert result == []
    
    def test_flatten_symbols_nested(self):
        """Flattens nested symbols."""
        manager = LSPManager(LSPConfig(enabled=False))
        
        symbols = [
            {
                "name": "MyClass",
                "kind": 5,  # class
                "range": {"start": {"line": 0}, "end": {"line": 20}},
                "children": [
                    {
                        "name": "method1",
                        "kind": 6,  # method
                        "range": {"start": {"line": 5}, "end": {"line": 10}},
                        "children": [],
                    },
                ],
            },
        ]
        
        result = manager._flatten_symbols(symbols)
        
        assert len(result) == 2
        assert result[0]["name"] == "MyClass"
        assert result[0]["kind"] == "class"
        assert result[1]["name"] == "method1"
        assert result[1]["parent"] == "MyClass"
    
    def test_disabled_manager_returns_empty(self):
        """Disabled manager returns empty results."""
        manager = LSPManager(LSPConfig(enabled=False))
        
        # All methods should return empty/None without trying to start servers
        refs = manager.find_references("/fake/path.py", 10, 5)
        assert refs == []
        
        defn = manager.go_to_definition("/fake/path.py", 10, 5)
        assert defn is None
        
        hover = manager.get_hover_info("/fake/path.py", 10, 5)
        assert hover is None
        
        symbols = manager.get_document_symbols("/fake/path.py")
        assert symbols == []
    
    def test_shutdown(self):
        """Can shutdown manager without error."""
        manager = LSPManager(LSPConfig(enabled=False))
        manager.shutdown()  # Should not raise


class TestGetLSPManager:
    """Tests for singleton get_lsp_manager function."""
    
    def setup_method(self):
        """Clear singleton before each test."""
        clear_lsp_manager()
    
    def test_returns_singleton(self):
        """Returns the same instance on multiple calls."""
        manager1 = get_lsp_manager()
        manager2 = get_lsp_manager()
        assert manager1 is manager2
    
    def test_clear_manager(self):
        """clear_lsp_manager() resets the singleton."""
        manager1 = get_lsp_manager()
        clear_lsp_manager()
        manager2 = get_lsp_manager()
        assert manager1 is not manager2


def _solidlsp_available() -> bool:
    """Check if solidlsp is installed."""
    try:
        from solidlsp import SolidLanguageServer
        return True
    except ImportError:
        return False


class TestLSPTools:
    """Tests for LSP-based tools in tools.py."""
    
    def test_find_references_without_lsp(self):
        """find_references returns message when LSP unavailable."""
        from rlm_dspy.tools import find_references
        
        # With disabled LSP, should return "no references" message
        result = find_references("/fake/path.py", 10, 5)
        assert "No references found" in result or "LSP not available" in result
    
    def test_go_to_definition_without_lsp(self):
        """go_to_definition returns message when LSP unavailable."""
        from rlm_dspy.tools import go_to_definition
        
        result = go_to_definition("/fake/path.py", 10, 5)
        assert "No definition found" in result or "LSP not available" in result
    
    def test_get_type_info_without_lsp(self):
        """get_type_info returns message when LSP unavailable."""
        from rlm_dspy.tools import get_type_info
        
        result = get_type_info("/fake/path.py", 10, 5)
        assert "No type info found" in result or "LSP not available" in result
    
    def test_get_symbol_hierarchy_without_lsp(self):
        """get_symbol_hierarchy returns message when LSP unavailable."""
        from rlm_dspy.tools import get_symbol_hierarchy
        
        result = get_symbol_hierarchy("/fake/path.py")
        assert "No symbols found" in result or "LSP not available" in result


@pytest.mark.skipif(
    not _solidlsp_available(),
    reason="solidlsp not installed"
)
class TestLSPIntegration:
    """Integration tests that require solidlsp to be installed."""
    
    def test_python_symbol_hierarchy(self):
        """Can get symbol hierarchy for a Python file."""
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write('''
class MyClass:
    def method1(self):
        pass
    
    def method2(self):
        pass

def standalone():
    pass
''')
            f.flush()
            
            from rlm_dspy.tools import get_symbol_hierarchy
            result = get_symbol_hierarchy(f.name)
            
            assert "MyClass" in result
            assert "method1" in result
            assert "standalone" in result
    
    def test_find_references_real(self):
        """Can find references in a real Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple project
            (Path(tmpdir) / "pyproject.toml").write_text("[tool.pytest]")
            
            main_file = Path(tmpdir) / "main.py"
            main_file.write_text('''
def greet(name):
    return f"Hello, {name}"

def main():
    print(greet("World"))
    print(greet("Python"))
''')
            
            from rlm_dspy.tools import find_references
            result = find_references(str(main_file), 2, 4)  # "greet" function
            
            # Should find at least the definition and usages
            assert "Found" in result or "references" in result.lower()

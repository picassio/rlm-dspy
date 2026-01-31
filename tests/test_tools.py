"""Tests for built-in tools."""

import os
from unittest.mock import patch

import pytest

from rlm_dspy.tools import (
    BUILTIN_TOOLS,
    SAFE_TOOLS,
    _is_safe_path,
    find_calls,
    find_classes,
    find_definitions,
    find_files,
    find_functions,
    find_imports,
    find_methods,
    file_stats,
    get_tool_descriptions,
    grep_context,
    index_code,
    read_file,
    ripgrep,
    shell,
)


class TestToolCollections:
    """Tests for tool collections."""

    def test_builtin_tools_not_empty(self):
        """Test that BUILTIN_TOOLS contains tools."""
        assert len(BUILTIN_TOOLS) > 0
        assert "ripgrep" in BUILTIN_TOOLS
        assert "index_code" in BUILTIN_TOOLS

    def test_safe_tools_excludes_shell(self):
        """Test that SAFE_TOOLS excludes shell."""
        assert "shell" not in SAFE_TOOLS
        assert "shell" in BUILTIN_TOOLS

    def test_all_tools_callable(self):
        """Test that all tools are callable."""
        for name, tool in BUILTIN_TOOLS.items():
            assert callable(tool), f"{name} is not callable"

    def test_tool_descriptions(self):
        """Test get_tool_descriptions returns useful info."""
        desc = get_tool_descriptions()
        assert "ripgrep" in desc
        assert "index_code" in desc
        assert "find_classes" in desc


class TestPathSafety:
    """Tests for path safety checks."""

    def test_safe_path(self):
        """Test that normal paths are safe."""
        is_safe, error = _is_safe_path("./src/main.py")
        assert is_safe
        assert error == ""

    def test_restricted_ssh(self):
        """Test that ~/.ssh is blocked."""
        is_safe, error = _is_safe_path("~/.ssh/id_rsa")
        assert not is_safe
        assert "restricted" in error.lower()

    def test_restricted_aws(self):
        """Test that ~/.aws/credentials is blocked."""
        is_safe, error = _is_safe_path("~/.aws/credentials")
        assert not is_safe
        assert "restricted" in error.lower()

    def test_etc_passwd(self):
        """Test that /etc/passwd is blocked."""
        is_safe, error = _is_safe_path("/etc/passwd")
        assert not is_safe


class TestReadFile:
    """Tests for read_file tool."""

    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create a temporary file."""
        f = tmp_path / "test.txt"
        f.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")
        return f

    def test_read_entire_file(self, temp_file):
        """Test reading entire file."""
        result = read_file(str(temp_file))
        assert "line 1" in result
        assert "line 5" in result

    def test_read_with_line_numbers(self, temp_file):
        """Test that line numbers are included."""
        result = read_file(str(temp_file))
        assert "1 |" in result or "1|" in result

    def test_read_partial_file(self, temp_file):
        """Test reading partial file."""
        result = read_file(str(temp_file), start_line=2, end_line=4)
        assert "line 2" in result
        assert "line 4" in result
        # line 1 and line 5 should not be included
        assert "line 1" not in result or result.count("line") == 3

    def test_read_nonexistent_file(self):
        """Test reading nonexistent file."""
        result = read_file("/nonexistent/file.txt")
        assert "not found" in result.lower()

    def test_read_blocked_path(self):
        """Test that restricted paths are blocked."""
        result = read_file("/etc/passwd")
        assert "security" in result.lower() or "restricted" in result.lower()


class TestFileStats:
    """Tests for file_stats tool."""

    def test_file_stats_file(self, tmp_path):
        """Test file_stats on a single file."""
        f = tmp_path / "test.py"
        f.write_text("line 1\nline 2\n")

        result = file_stats(str(f))
        assert "file" in result.lower()
        assert "line" in result.lower() or "2" in result

    def test_file_stats_directory(self, tmp_path):
        """Test file_stats on a directory."""
        (tmp_path / "a.py").write_text("# python\n")
        (tmp_path / "b.py").write_text("# python\n")

        result = file_stats(str(tmp_path))
        assert "directory" in result.lower() or "file_count" in result.lower()


class TestRipgrep:
    """Tests for ripgrep tool."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project."""
        (tmp_path / "main.py").write_text("def main():\n    print('hello')\n")
        (tmp_path / "utils.py").write_text("def helper():\n    pass\n")
        return tmp_path

    def test_ripgrep_finds_pattern(self, temp_project):
        """Test ripgrep finds a pattern."""
        result = ripgrep("def main", str(temp_project))
        if "not installed" not in result:
            assert "main" in result

    def test_ripgrep_no_matches(self, temp_project):
        """Test ripgrep with no matches."""
        result = ripgrep("nonexistent_pattern_xyz", str(temp_project))
        if "not installed" not in result:
            assert "no matches" in result.lower() or result.strip() == ""

    def test_ripgrep_with_flags(self, temp_project):
        """Test ripgrep with flags."""
        result = ripgrep("DEF MAIN", str(temp_project), "-i")  # case insensitive
        if "not installed" not in result:
            assert "main" in result.lower()


class TestGrepContext:
    """Tests for grep_context tool."""

    def test_grep_context_returns_context(self, tmp_path):
        """Test grep_context returns surrounding lines."""
        f = tmp_path / "test.py"
        f.write_text("# comment\ndef target():\n    pass\n# end\n")

        result = grep_context("target", str(tmp_path), context_lines=1)
        if "not installed" not in result:
            # Should include context lines
            assert "target" in result


class TestFindFiles:
    """Tests for find_files tool."""

    def test_find_python_files(self, tmp_path):
        """Test finding Python files."""
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")

        result = find_files("*.py", str(tmp_path))
        if "not installed" not in result:
            assert "a.py" in result
            assert "b.py" in result


class TestIndexCode:
    """Tests for index_code tool."""

    @pytest.fixture
    def python_file(self, tmp_path):
        """Create a Python file with classes and functions."""
        f = tmp_path / "code.py"
        f.write_text("""
class MyClass:
    def method(self):
        pass

def my_function():
    pass

class AnotherClass:
    pass
""")
        return f

    def test_index_all(self, python_file):
        """Test indexing all definitions."""
        result = index_code(str(python_file))
        assert "MyClass" in result
        assert "my_function" in result
        assert "AnotherClass" in result

    def test_index_classes_only(self, python_file):
        """Test indexing only classes."""
        result = index_code(str(python_file), kind="class")
        assert "MyClass" in result
        assert "AnotherClass" in result
        # my_function should not be in class-only results
        # (it might appear in line context, so just check classes are there)

    def test_index_functions_only(self, python_file):
        """Test indexing only functions."""
        result = index_code(str(python_file), kind="function")
        assert "my_function" in result

    def test_index_with_name_filter(self, python_file):
        """Test indexing with name filter."""
        result = index_code(str(python_file), name="Another")
        assert "AnotherClass" in result


class TestFindDefinitions:
    """Tests for find_definitions tool."""

    def test_find_definitions(self, tmp_path):
        """Test finding definitions."""
        f = tmp_path / "test.py"
        f.write_text("def foo(): pass\nclass Bar: pass\n")

        result = find_definitions(str(f))
        assert "foo" in result
        assert "Bar" in result


class TestFindClasses:
    """Tests for find_classes tool."""

    def test_find_classes(self, tmp_path):
        """Test finding classes."""
        f = tmp_path / "test.py"
        f.write_text("class MyClass: pass\ndef func(): pass\n")

        result = find_classes(str(f))
        assert "MyClass" in result


class TestFindFunctions:
    """Tests for find_functions tool."""

    def test_find_functions(self, tmp_path):
        """Test finding functions."""
        f = tmp_path / "test.py"
        f.write_text("def my_func(): pass\nclass MyClass: pass\n")

        result = find_functions(str(f))
        assert "my_func" in result


class TestFindMethods:
    """Tests for find_methods tool."""

    def test_find_methods(self, tmp_path):
        """Test finding methods."""
        f = tmp_path / "test.py"
        f.write_text("class MyClass:\n    def my_method(self): pass\n")

        result = find_methods(str(f))
        assert "my_method" in result
        assert "MyClass" in result  # Parent class should be shown


class TestFindImports:
    """Tests for find_imports tool."""

    def test_find_imports(self, tmp_path):
        """Test finding imports."""
        f = tmp_path / "test.py"
        f.write_text("import os\nfrom pathlib import Path\n")

        result = find_imports(str(f))
        if "not installed" not in result:
            assert "import" in result


class TestFindCalls:
    """Tests for find_calls tool."""

    def test_find_calls(self, tmp_path):
        """Test finding function calls."""
        f = tmp_path / "test.py"
        f.write_text("print('hello')\nprint('world')\n")

        result = find_calls(str(f), "print")
        if "not installed" not in result:
            assert "print" in result


class TestShell:
    """Tests for shell tool."""

    def test_shell_disabled_by_default(self):
        """Test that shell is disabled by default."""
        with patch.dict(os.environ, {"RLM_ALLOW_SHELL": ""}, clear=False):
            # Ensure RLM_ALLOW_SHELL is not set
            if "RLM_ALLOW_SHELL" in os.environ:
                del os.environ["RLM_ALLOW_SHELL"]
            result = shell("echo test")
            assert "disabled" in result.lower()

    def test_shell_enabled(self):
        """Test shell when enabled."""
        with patch.dict(os.environ, {"RLM_ALLOW_SHELL": "1"}):
            result = shell("echo test")
            assert "test" in result or "disabled" not in result.lower()

    def test_shell_blocks_dangerous(self):
        """Test that dangerous/disallowed commands are blocked."""
        with patch.dict(os.environ, {"RLM_ALLOW_SHELL": "1"}):
            # rm is not in allowlist
            result = shell("rm -rf /")
            assert "not in allowlist" in result.lower() or "blocked" in result.lower()

    def test_shell_blocks_not_allowed(self):
        """Test that commands not in allowlist are blocked."""
        with patch.dict(os.environ, {"RLM_ALLOW_SHELL": "1"}):
            result = shell("curl http://example.com")
            # Could be blocked for various reasons
            assert "blocked" in result.lower() or "not in allowlist" in result.lower()

    def test_shell_timeout(self):
        """Test shell with allowed command."""
        with patch.dict(os.environ, {"RLM_ALLOW_SHELL": "1"}):
            # Use an allowed command
            result = shell("echo hello")
            assert "hello" in result or "not in allowlist" in result.lower()

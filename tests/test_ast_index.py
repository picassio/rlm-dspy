"""Tests for AST-based code indexing."""

import tempfile
from pathlib import Path

from rlm_dspy.core.ast_index import (
    LANGUAGE_MAP,
    ASTIndex,
    Definition,
    index_file,
    index_files,
)
from rlm_dspy.core.treesitter import get_parser_simple


class TestLanguageSupport:
    """Test language detection and parser availability."""

    def test_language_map_has_common_extensions(self):
        """Common file extensions should be mapped."""
        assert ".py" in LANGUAGE_MAP
        assert ".js" in LANGUAGE_MAP
        assert ".ts" in LANGUAGE_MAP
        assert ".go" in LANGUAGE_MAP
        assert ".rs" in LANGUAGE_MAP
        assert ".java" in LANGUAGE_MAP
        assert ".c" in LANGUAGE_MAP
        assert ".cpp" in LANGUAGE_MAP

    def test_python_parser_available(self):
        """Python parser should always be available."""
        parser = get_parser_simple("python")
        assert parser is not None

    def test_unknown_language_returns_none(self):
        """Unknown language should return None."""
        parser = get_parser_simple("unknown_language_xyz")
        assert parser is None


class TestDefinition:
    """Test Definition dataclass."""

    def test_definition_creation(self):
        """Test creating a Definition."""
        d = Definition(
            name="MyClass",
            kind="class",
            line=10,
            end_line=50,
            file="test.py",
            parent=None,
        )
        assert d.name == "MyClass"
        assert d.kind == "class"
        assert d.line == 10
        assert d.end_line == 50
        assert d.file == "test.py"
        assert d.parent is None

    def test_definition_with_parent(self):
        """Test Definition with parent (method)."""
        d = Definition(
            name="my_method",
            kind="method",
            line=20,
            end_line=30,
            file="test.py",
            parent="MyClass",
        )
        assert d.parent == "MyClass"
        assert d.kind == "method"


class TestASTIndex:
    """Test ASTIndex class."""

    def test_empty_index(self):
        """Empty index should have no definitions."""
        idx = ASTIndex()
        assert len(idx.definitions) == 0
        assert idx.classes() == []
        assert idx.functions() == []
        assert idx.methods() == []

    def test_find_by_name(self):
        """Test finding definitions by name."""
        idx = ASTIndex(definitions=[
            Definition("MyClass", "class", 1, 10, "test.py"),
            Definition("my_func", "function", 15, 20, "test.py"),
            Definition("MyOtherClass", "class", 25, 35, "test.py"),
        ])

        results = idx.find(name="Class")
        assert len(results) == 2
        assert all("Class" in d.name for d in results)

    def test_find_by_kind(self):
        """Test finding definitions by kind."""
        idx = ASTIndex(definitions=[
            Definition("MyClass", "class", 1, 10, "test.py"),
            Definition("my_func", "function", 15, 20, "test.py"),
            Definition("my_method", "method", 5, 8, "test.py", parent="MyClass"),
        ])

        assert len(idx.classes()) == 1
        assert len(idx.functions()) == 1
        assert len(idx.methods()) == 1

    def test_find_by_name_and_kind(self):
        """Test finding with both name and kind filter."""
        idx = ASTIndex(definitions=[
            Definition("MyClass", "class", 1, 10, "test.py"),
            Definition("MyFunc", "function", 15, 20, "test.py"),
        ])

        results = idx.find(name="My", kind="class")
        assert len(results) == 1
        assert results[0].name == "MyClass"

    def test_get_line(self):
        """Test getting exact line number."""
        idx = ASTIndex(definitions=[
            Definition("MyClass", "class", 42, 100, "test.py"),
        ])

        assert idx.get_line("MyClass") == 42
        assert idx.get_line("NonExistent") is None


class TestIndexFile:
    """Test file indexing."""

    def test_index_python_file(self):
        """Test indexing a Python file."""
        code = '''
class MyClass:
    def my_method(self):
        pass

def my_function():
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            idx = index_file(f.name)

            assert len(idx.classes()) == 1
            assert len(idx.functions()) == 1
            assert len(idx.methods()) == 1

            assert idx.classes()[0].name == "MyClass"
            assert idx.functions()[0].name == "my_function"
            assert idx.methods()[0].name == "my_method"
            assert idx.methods()[0].parent == "MyClass"

    def test_index_nonexistent_file(self):
        """Non-existent file should return empty index."""
        idx = index_file("/nonexistent/path/file.py")
        assert len(idx.definitions) == 0

    def test_index_unsupported_extension(self):
        """Unsupported file extension should return empty index."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("some content")
            f.flush()

            idx = index_file(f.name)
            assert len(idx.definitions) == 0

    def test_index_file_line_numbers(self):
        """Test that line numbers are accurate."""
        code = '''# Line 1
# Line 2
class MyClass:  # Line 3
    pass
# Line 5
def my_func():  # Line 6
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            idx = index_file(f.name)

            assert idx.get_line("MyClass") == 3
            assert idx.get_line("my_func") == 6


class TestIndexFiles:
    """Test indexing multiple files."""

    def test_index_multiple_files(self):
        """Test indexing multiple files."""
        code1 = "class ClassA: pass"
        code2 = "class ClassB: pass"

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"

            file1.write_text(code1)
            file2.write_text(code2)

            idx = index_files([file1, file2])

            assert len(idx.classes()) == 2
            names = {d.name for d in idx.classes()}
            assert names == {"ClassA", "ClassB"}

    def test_index_empty_list(self):
        """Empty file list should return empty index."""
        idx = index_files([])
        assert len(idx.definitions) == 0

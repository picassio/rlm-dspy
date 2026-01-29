"""Tests for AST-based code indexing using tree-sitter."""

import tempfile
from pathlib import Path


from rlm_dspy.core.ast_index import (
    ASTIndex,
    Definition,
    LANGUAGE_MAP,
    index_file,
    index_files,
    clear_index_cache,
    clear_parser_cache,
    get_cache_stats,
)


class TestLanguageMap:
    """Tests for language extension mapping."""

    def test_python_extensions(self):
        """Python files are recognized."""
        assert LANGUAGE_MAP[".py"] == "python"
        assert LANGUAGE_MAP[".pyi"] == "python"

    def test_javascript_extensions(self):
        """JavaScript files are recognized."""
        assert LANGUAGE_MAP[".js"] == "javascript"
        assert LANGUAGE_MAP[".jsx"] == "javascript"
        assert LANGUAGE_MAP[".mjs"] == "javascript"

    def test_typescript_extensions(self):
        """TypeScript files are recognized."""
        assert LANGUAGE_MAP[".ts"] == "typescript"
        assert LANGUAGE_MAP[".tsx"] == "typescript"

    def test_go_extension(self):
        """Go files are recognized."""
        assert LANGUAGE_MAP[".go"] == "go"

    def test_rust_extension(self):
        """Rust files are recognized."""
        assert LANGUAGE_MAP[".rs"] == "rust"

    def test_java_extension(self):
        """Java files are recognized."""
        assert LANGUAGE_MAP[".java"] == "java"

    def test_cpp_extensions(self):
        """C++ files are recognized."""
        assert LANGUAGE_MAP[".cpp"] == "cpp"
        assert LANGUAGE_MAP[".cc"] == "cpp"
        assert LANGUAGE_MAP[".hpp"] == "cpp"

    def test_c_extensions(self):
        """C files are recognized."""
        assert LANGUAGE_MAP[".c"] == "c"
        assert LANGUAGE_MAP[".h"] == "c"

    def test_ruby_extension(self):
        """Ruby files are recognized."""
        assert LANGUAGE_MAP[".rb"] == "ruby"

    def test_csharp_extension(self):
        """C# files are recognized."""
        assert LANGUAGE_MAP[".cs"] == "c_sharp"

    def test_kotlin_extensions(self):
        """Kotlin files are recognized."""
        assert LANGUAGE_MAP[".kt"] == "kotlin"
        assert LANGUAGE_MAP[".kts"] == "kotlin"

    def test_scala_extensions(self):
        """Scala files are recognized."""
        assert LANGUAGE_MAP[".scala"] == "scala"
        assert LANGUAGE_MAP[".sc"] == "scala"

    def test_php_extension(self):
        """PHP files are recognized."""
        assert LANGUAGE_MAP[".php"] == "php"

    def test_lua_extension(self):
        """Lua files are recognized."""
        assert LANGUAGE_MAP[".lua"] == "lua"

    def test_bash_extensions(self):
        """Bash files are recognized."""
        assert LANGUAGE_MAP[".sh"] == "bash"
        assert LANGUAGE_MAP[".bash"] == "bash"

    def test_haskell_extensions(self):
        """Haskell files are recognized."""
        assert LANGUAGE_MAP[".hs"] == "haskell"
        assert LANGUAGE_MAP[".lhs"] == "haskell"


class TestDefinition:
    """Tests for Definition dataclass."""

    def test_create_function(self):
        """Can create a function definition."""
        defn = Definition(
            name="my_func",
            kind="function",
            line=10,
            end_line=20,
            file="test.py",
        )
        assert defn.name == "my_func"
        assert defn.kind == "function"
        assert defn.line == 10
        assert defn.end_line == 20
        assert defn.parent is None

    def test_create_method_with_parent(self):
        """Can create a method with parent class."""
        defn = Definition(
            name="do_something",
            kind="method",
            line=15,
            end_line=25,
            file="test.py",
            parent="MyClass",
        )
        assert defn.name == "do_something"
        assert defn.kind == "method"
        assert defn.parent == "MyClass"


class TestASTIndex:
    """Tests for ASTIndex class."""

    def test_empty_index(self):
        """Empty index has no definitions."""
        idx = ASTIndex()
        assert len(idx.definitions) == 0
        assert idx.classes() == []
        assert idx.functions() == []
        assert idx.methods() == []

    def test_find_by_name(self):
        """Can find definitions by name."""
        idx = ASTIndex(definitions=[
            Definition(name="foo", kind="function", line=1, end_line=5, file="a.py"),
            Definition(name="bar", kind="function", line=10, end_line=15, file="a.py"),
            Definition(name="foobar", kind="class", line=20, end_line=30, file="a.py"),
        ])

        # Substring match
        results = idx.find(name="foo")
        assert len(results) == 2  # foo and foobar

        # Exact match
        results = idx.find(name="bar")
        assert len(results) == 2  # bar and foobar

    def test_find_by_kind(self):
        """Can find definitions by kind."""
        idx = ASTIndex(definitions=[
            Definition(name="MyClass", kind="class", line=1, end_line=20, file="a.py"),
            Definition(name="func1", kind="function", line=25, end_line=30, file="a.py"),
            Definition(name="method1", kind="method", line=5, end_line=10, file="a.py", parent="MyClass"),
        ])

        assert len(idx.find(kind="class")) == 1
        assert len(idx.find(kind="function")) == 1
        assert len(idx.find(kind="method")) == 1

    def test_find_by_name_and_kind(self):
        """Can find definitions by name AND kind."""
        idx = ASTIndex(definitions=[
            Definition(name="process", kind="function", line=1, end_line=10, file="a.py"),
            Definition(name="process", kind="method", line=15, end_line=25, file="a.py", parent="Handler"),
            Definition(name="Processor", kind="class", line=30, end_line=50, file="a.py"),
        ])

        # Find functions named "process"
        results = idx.find(name="process", kind="function")
        assert len(results) == 1
        assert results[0].kind == "function"

        # Find methods named "process"
        results = idx.find(name="process", kind="method")
        assert len(results) == 1
        assert results[0].kind == "method"

    def test_classes_filter(self):
        """classes() returns only classes."""
        idx = ASTIndex(definitions=[
            Definition(name="MyClass", kind="class", line=1, end_line=20, file="a.py"),
            Definition(name="func", kind="function", line=25, end_line=30, file="a.py"),
        ])

        classes = idx.classes()
        assert len(classes) == 1
        assert classes[0].name == "MyClass"

    def test_functions_filter(self):
        """functions() returns only functions."""
        idx = ASTIndex(definitions=[
            Definition(name="MyClass", kind="class", line=1, end_line=20, file="a.py"),
            Definition(name="func", kind="function", line=25, end_line=30, file="a.py"),
        ])

        funcs = idx.functions()
        assert len(funcs) == 1
        assert funcs[0].name == "func"

    def test_methods_filter(self):
        """methods() returns only methods."""
        idx = ASTIndex(definitions=[
            Definition(name="method1", kind="method", line=5, end_line=10, file="a.py", parent="MyClass"),
            Definition(name="func", kind="function", line=25, end_line=30, file="a.py"),
        ])

        methods = idx.methods()
        assert len(methods) == 1
        assert methods[0].name == "method1"

    def test_get_line(self):
        """get_line() returns line number for exact name match."""
        idx = ASTIndex(definitions=[
            Definition(name="MyClass", kind="class", line=10, end_line=50, file="a.py"),
            Definition(name="myclass", kind="function", line=100, end_line=110, file="a.py"),
        ])

        # Case-sensitive exact match
        assert idx.get_line("MyClass") == 10
        assert idx.get_line("myclass") == 100
        assert idx.get_line("MYCLASS") is None  # No match


class TestIndexFile:
    """Tests for index_file function."""

    def test_index_python_file(self):
        """Can index a Python file."""
        code = '''
class MyClass:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1

def standalone_func(x, y):
    return x + y
'''
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(code)
            f.flush()

            idx = index_file(f.name, use_cache=False)

            # Should find class, 2 methods, and 1 function
            assert len(idx.definitions) == 4

            classes = idx.classes()
            assert len(classes) == 1
            assert classes[0].name == "MyClass"

            methods = idx.methods()
            assert len(methods) == 2
            method_names = {m.name for m in methods}
            assert "__init__" in method_names
            assert "increment" in method_names

            funcs = idx.functions()
            assert len(funcs) == 1
            assert funcs[0].name == "standalone_func"

    def test_index_javascript_file(self):
        """Can index a JavaScript file."""
        code = '''
class Calculator {
    add(a, b) {
        return a + b;
    }
}

function multiply(a, b) {
    return a * b;
}
'''
        with tempfile.NamedTemporaryFile(suffix='.js', mode='w', delete=False) as f:
            f.write(code)
            f.flush()

            idx = index_file(f.name, use_cache=False)

            classes = idx.classes()
            assert len(classes) == 1
            assert classes[0].name == "Calculator"

    def test_index_go_file(self):
        """Can index a Go file."""
        code = '''
package main

type Server struct {
    port int
}

func (s *Server) Start() {
    // start server
}

func main() {
    s := &Server{port: 8080}
    s.Start()
}
'''
        with tempfile.NamedTemporaryFile(suffix='.go', mode='w', delete=False) as f:
            f.write(code)
            f.flush()

            idx = index_file(f.name, use_cache=False)

            # Should find function definitions
            assert len(idx.definitions) > 0

    def test_index_kotlin_file(self):
        """Can index a Kotlin file."""
        code = '''
class Calculator {
    fun add(a: Int, b: Int): Int {
        return a + b
    }
}

fun main() {
    println("Hello")
}
'''
        with tempfile.NamedTemporaryFile(suffix='.kt', mode='w', delete=False) as f:
            f.write(code)
            f.flush()

            idx = index_file(f.name, use_cache=False)

            classes = idx.classes()
            assert len(classes) == 1
            assert classes[0].name == "Calculator"

    def test_index_unsupported_extension(self):
        """Returns empty index for unsupported extensions."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', mode='w', delete=False) as f:
            f.write("some content")
            f.flush()

            idx = index_file(f.name, use_cache=False)
            assert len(idx.definitions) == 0

    def test_index_nonexistent_file(self):
        """Returns empty index for nonexistent files."""
        idx = index_file("/nonexistent/path/file.py", use_cache=False)
        assert len(idx.definitions) == 0

    def test_caching(self):
        """File indexing is cached."""
        code = "def func(): pass"

        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(code)
            f.flush()

            clear_index_cache()

            # First call
            idx1 = index_file(f.name, use_cache=True)
            # Second call (should be cached)
            idx2 = index_file(f.name, use_cache=True)

            # Same object from cache
            assert idx1 is idx2


class TestIndexFiles:
    """Tests for index_files function."""

    def test_index_multiple_files(self):
        """Can index multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two Python files
            file1 = Path(tmpdir) / "a.py"
            file1.write_text("class A: pass")

            file2 = Path(tmpdir) / "b.py"
            file2.write_text("def func_b(): pass")

            idx = index_files([file1, file2])

            assert len(idx.definitions) == 2
            assert len(idx.classes()) == 1
            assert len(idx.functions()) == 1

    def test_index_empty_list(self):
        """Returns empty index for empty file list."""
        idx = index_files([])
        assert len(idx.definitions) == 0

    def test_index_mixed_languages(self):
        """Can index files from different languages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "code.py"
            py_file.write_text("def py_func(): pass")

            js_file = Path(tmpdir) / "code.js"
            js_file.write_text("function jsFunc() {}")

            idx = index_files([py_file, js_file])

            assert len(idx.definitions) >= 2


class TestCacheManagement:
    """Tests for cache management."""

    def test_clear_index_cache(self):
        """Can clear the index cache."""
        code = "def func(): pass"

        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(code)
            f.flush()

            # Populate cache
            idx1 = index_file(f.name, use_cache=True)

            # Clear cache
            clear_index_cache()

            # Should be a new object
            idx2 = index_file(f.name, use_cache=True)
            assert idx1 is not idx2

    def test_parser_cache_populated(self):
        """Parser cache is populated after indexing."""
        code = "def func(): pass"

        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(code)
            f.flush()

            clear_parser_cache()
            clear_index_cache()

            # Index a Python file
            index_file(f.name, use_cache=False)

            # Parser cache should have python parser
            stats = get_cache_stats()
            assert stats["parser_cache_size"] >= 1

    def test_clear_parser_cache(self):
        """Can clear the parser cache."""
        code = "def func(): pass"

        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(code)
            f.flush()

            # Populate parser cache
            index_file(f.name, use_cache=False)

            # Clear parser cache
            count = clear_parser_cache()
            assert count >= 1

            # Cache should be empty
            stats = get_cache_stats()
            assert stats["parser_cache_size"] == 0

    def test_get_cache_stats(self):
        """get_cache_stats returns valid stats."""
        stats = get_cache_stats()
        assert "index_cache_size" in stats
        assert "parser_cache_size" in stats
        assert "max_index_cache_size" in stats
        assert stats["max_index_cache_size"] > 0

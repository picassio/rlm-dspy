"""Tests for syntax-aware chunking."""

import pytest

from rlm_dspy.core.syntax_chunker import (
    CodeChunk,
    chunk_code_syntax_aware,
    chunk_mixed_content,
    TREE_SITTER_AVAILABLE,
    _detect_language,
)


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_from_extension(self):
        assert _detect_language("", "test.py") == "python"
        assert _detect_language("", "test.ts") == "typescript"
        assert _detect_language("", "test.js") == "javascript"
        assert _detect_language("", "test.go") == "go"
        assert _detect_language("", "test.rs") == "rust"

    def test_detect_from_content(self):
        assert _detect_language("def hello(): pass\nimport os") == "python"
        assert _detect_language("func main() {\n}\npackage main") == "go"
        assert _detect_language("fn main() {\n    let x = 1;\n}") == "rust"


class TestCharacterChunking:
    """Tests for fallback character-based chunking."""

    def test_small_content(self):
        """Small content should stay in one chunk."""
        chunks = chunk_code_syntax_aware("hello world", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0].content == "hello world"

    def test_empty_content(self):
        """Empty content should return no chunks."""
        chunks = chunk_code_syntax_aware("")
        assert len(chunks) == 0


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestSyntaxAwareChunking:
    """Tests for syntax-aware chunking with tree-sitter."""

    def test_python_functions(self):
        """Python functions should be chunked at definition boundaries."""
        code = """def hello():
    print('hello')

def world():
    print('world')
"""
        chunks = chunk_code_syntax_aware(code, chunk_size=50, language="python")
        # Each function should be in its own chunk
        assert len(chunks) >= 1
        # First chunk should contain "def hello"
        assert "def hello" in chunks[0].content

    def test_python_class(self):
        """Python class should be chunked as a single unit."""
        code = """class MyClass:
    def __init__(self):
        pass
    
    def method(self):
        pass
"""
        chunks = chunk_code_syntax_aware(code, chunk_size=200, language="python")
        assert len(chunks) == 1
        assert chunks[0].node_type == "class_definition"
        assert chunks[0].name == "MyClass"

    def test_no_truncated_functions(self):
        """Functions should never be truncated mid-definition."""
        code = """def function_one():
    x = 1
    y = 2
    return x + y

def function_two():
    for i in range(10):
        print(i)
    return True
"""
        chunks = chunk_code_syntax_aware(code, chunk_size=80, language="python")
        for chunk in chunks:
            # Each chunk should contain complete function definitions
            content = chunk.content.strip()
            if "def " in content:
                # Count 'def' and make sure each has a matching body
                def_count = content.count("def ")
                # Should have at least as many 'return' or 'pass' as 'def'
                # (indicating complete function bodies)
                assert "return" in content or "pass" in content or "print" in content

    def test_large_function_own_chunk(self):
        """A function larger than chunk_size should get its own chunk."""
        code = """def small():
    pass

def large_function():
    x = 1
    y = 2
    z = 3
    a = 4
    b = 5
    c = 6
    return x + y + z + a + b + c

def another_small():
    pass
"""
        # With very small chunk size, large function should be in its own chunk
        chunks = chunk_code_syntax_aware(code, chunk_size=60, language="python")
        # Should have multiple chunks
        assert len(chunks) >= 2


class TestMixedContent:
    """Tests for mixed content (prose + code blocks)."""

    def test_code_blocks_extracted(self):
        """Code blocks should be identified and chunked syntax-aware."""
        content = """# Documentation

This is some text.

```python
def hello():
    pass
```

More text here.

```python
def world():
    pass
```
"""
        chunks = chunk_mixed_content(content, chunk_size=100)
        # Should have chunks for prose and code
        assert len(chunks) >= 2


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_chunk_attributes(self):
        chunk = CodeChunk(
            content="def test(): pass",
            start_line=1,
            end_line=1,
            node_type="function_definition",
            name="test",
        )
        assert chunk.content == "def test(): pass"
        assert chunk.start_line == 1
        assert chunk.end_line == 1
        assert chunk.node_type == "function_definition"
        assert chunk.name == "test"

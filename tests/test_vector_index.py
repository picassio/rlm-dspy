"""Tests for vector index and semantic search."""

import os
import pytest
from pathlib import Path


class TestIndexConfig:
    """Test IndexConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from rlm_dspy.core.vector_index import IndexConfig
        
        config = IndexConfig()
        
        assert config.use_faiss is True
        assert config.faiss_threshold == 5000
        assert config.auto_update is True
        assert config.cache_ttl == 3600

    def test_from_user_config(self):
        """Test loading from user config."""
        from rlm_dspy.core.vector_index import IndexConfig
        
        config = IndexConfig.from_user_config()
        
        assert config.index_dir is not None
        assert isinstance(config.index_dir, Path)

    def test_env_var_override(self, monkeypatch, tmp_path):
        """Test environment variable override."""
        from rlm_dspy.core.vector_index import IndexConfig
        
        test_dir = str(tmp_path / "test_indexes")
        monkeypatch.setenv("RLM_INDEX_DIR", test_dir)
        
        config = IndexConfig.from_user_config()
        
        assert str(config.index_dir) == test_dir


class TestCodeSnippet:
    """Test CodeSnippet dataclass."""

    def test_to_document(self):
        """Test conversion to document string."""
        from rlm_dspy.core.vector_index import CodeSnippet
        
        snippet = CodeSnippet(
            id="test.py:my_func:10",
            text="def my_func():\n    pass",
            file="test.py",
            line=10,
            end_line=12,
            type="function",
            name="my_func",
        )
        
        doc = snippet.to_document()
        
        assert "function" in doc
        assert "my_func" in doc
        assert "test.py:10" in doc
        assert "def my_func" in doc


class TestCodeIndex:
    """Test CodeIndex class."""

    def test_create_index(self):
        """Test index creation."""
        from rlm_dspy.core.vector_index import CodeIndex, IndexConfig
        
        config = IndexConfig(use_faiss=False)  # Don't require FAISS
        index = CodeIndex(config)
        
        assert index is not None
        assert index.config == config

    def test_get_index_path(self, tmp_path):
        """Test index path generation."""
        from rlm_dspy.core.vector_index import CodeIndex, IndexConfig
        
        config = IndexConfig(index_dir=tmp_path / "indexes")
        index = CodeIndex(config)
        
        repo_path = tmp_path / "my_repo"
        repo_path.mkdir()
        
        index_path = index._get_index_path(repo_path)
        
        assert index_path.parent == config.index_dir
        # Should be a hash-based name
        assert len(index_path.name) == 12

    def test_extract_snippets(self, tmp_path):
        """Test snippet extraction from Python files."""
        from rlm_dspy.core.vector_index import CodeIndex, IndexConfig
        
        # Create test Python file
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def hello():
    '''Say hello'''
    print("Hello")

class MyClass:
    def method(self):
        pass
""")
        
        config = IndexConfig(index_dir=tmp_path / "indexes", use_faiss=False)
        index = CodeIndex(config)
        
        snippets = index._extract_snippets(tmp_path)
        
        # Should find function and class
        names = {s.name for s in snippets}
        assert "hello" in names or "MyClass" in names

    def test_get_status_not_indexed(self, tmp_path):
        """Test status for non-indexed directory."""
        from rlm_dspy.core.vector_index import CodeIndex, IndexConfig
        
        config = IndexConfig(index_dir=tmp_path / "indexes")
        index = CodeIndex(config)
        
        status = index.get_status(tmp_path)
        
        assert status["indexed"] is False

    def test_clear_specific(self, tmp_path):
        """Test clearing a specific index."""
        from rlm_dspy.core.vector_index import CodeIndex, IndexConfig
        
        config = IndexConfig(index_dir=tmp_path / "indexes")
        index = CodeIndex(config)
        
        # Create a fake index directory
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        index_path = index._get_index_path(repo_path)
        index_path.mkdir(parents=True)
        (index_path / "manifest.json").write_text("{}")
        
        # Clear it
        count = index.clear(repo_path)
        
        assert count == 1
        assert not index_path.exists()


class TestSemanticSearch:
    """Test semantic search tool."""

    def test_semantic_search_no_index(self, tmp_path):
        """Test semantic search on non-indexed directory."""
        from rlm_dspy.tools import semantic_search
        
        # Should handle gracefully
        result = semantic_search("test query", str(tmp_path))
        
        # Either returns no results or error message
        assert "error" in result.lower() or "no results" in result.lower() or "found" in result.lower()


class TestGetIndexManager:
    """Test global index manager."""

    def test_get_index_manager_singleton(self):
        """Test that get_index_manager returns same instance."""
        from rlm_dspy.core.vector_index import get_index_manager
        
        manager1 = get_index_manager()
        manager2 = get_index_manager()
        
        assert manager1 is manager2

    def test_get_index_manager_with_config(self, tmp_path):
        """Test creating manager with custom config."""
        from rlm_dspy.core.vector_index import get_index_manager, IndexConfig
        
        config = IndexConfig(index_dir=tmp_path / "custom_indexes")
        manager = get_index_manager(config)
        
        assert manager.config.index_dir == tmp_path / "custom_indexes"


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from rlm_dspy.core.vector_index import CodeSnippet, SearchResult
        
        snippet = CodeSnippet(
            id="test.py:func:1",
            text="def func(): pass",
            file="test.py",
            line=1,
            end_line=1,
            type="function",
            name="func",
        )
        
        result = SearchResult(snippet=snippet, score=0.95)
        result_dict = result.to_dict()
        
        assert result_dict["file"] == "test.py"
        assert result_dict["line"] == 1
        assert result_dict["type"] == "function"
        assert result_dict["name"] == "func"
        assert result_dict["score"] == 0.95

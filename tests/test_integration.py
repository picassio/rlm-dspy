"""End-to-end integration tests for rlm-dspy.

These tests verify the complete pipeline works together.
"""

import os
import pytest
from pathlib import Path


# Skip tests that require embedding API keys (OpenAI by default)
requires_embedding_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY for embeddings"
)


class TestSemanticSearchIntegration:
    """Test semantic search integration."""

    def test_semantic_search_tool_in_builtin(self):
        """Test that semantic_search is in BUILTIN_TOOLS."""
        from rlm_dspy import BUILTIN_TOOLS
        
        assert "semantic_search" in BUILTIN_TOOLS
        assert callable(BUILTIN_TOOLS["semantic_search"])

    @requires_embedding_api_key
    def test_semantic_search_auto_indexes(self, tmp_path):
        """Test that semantic search auto-indexes on first use."""
        from rlm_dspy.tools import semantic_search
        from rlm_dspy.core.vector_index import get_index_manager
        
        # Create test file
        (tmp_path / "test.py").write_text("""
def hello_world():
    '''Say hello to the world'''
    print("Hello, World!")

def goodbye_world():
    '''Say goodbye to the world'''
    print("Goodbye, World!")
""")
        
        # Clear any existing index
        manager = get_index_manager()
        manager.clear(tmp_path)
        
        # Search should auto-build index
        result = semantic_search("greeting function", path=str(tmp_path))
        
        # Should find hello_world
        assert "hello" in result.lower() or "found" in result.lower()


class TestCitedSignaturesIntegration:
    """Test cited signatures integration."""

    def test_all_cited_signatures_registered(self):
        """Test all cited signatures are in registry."""
        from rlm_dspy.signatures import get_signature, list_signatures
        
        # Check signatures exist
        assert get_signature("cited") is not None
        assert get_signature("cited-security") is not None
        assert get_signature("cited-bugs") is not None
        assert get_signature("cited-review") is not None
        
        # Check they're in the list
        sigs = list_signatures()
        assert "cited" in sigs

    def test_cited_signature_has_locations(self):
        """Test cited signatures include locations field."""
        from rlm_dspy.signatures import (
            CitedAnalysis, CitedSecurityAudit, CitedBugFinder, CitedCodeReview
        )
        
        for sig in [CitedAnalysis, CitedSecurityAudit, CitedBugFinder, CitedCodeReview]:
            assert "locations" in sig.output_fields


class TestCodeToDocumentIntegration:
    """Test code document conversion integration."""

    def test_code_to_document_with_real_file(self, tmp_path):
        """Test converting real file to document."""
        from rlm_dspy import code_to_document
        
        # Create test file
        test_file = tmp_path / "example.py"
        test_file.write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")
        
        doc = code_to_document(test_file)
        
        # Check structure
        assert doc["title"] == str(test_file)
        assert doc["media_type"] == "text/plain"
        
        # Check line numbers
        assert "1 |" in doc["data"]
        assert "def add" in doc["data"]

    def test_files_to_documents_multiple(self, tmp_path):
        """Test converting multiple files."""
        from rlm_dspy import files_to_documents
        
        # Create test files
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        (tmp_path / "c.py").write_text("z = 3")
        
        docs = files_to_documents([
            tmp_path / "a.py",
            tmp_path / "b.py",
            tmp_path / "c.py",
        ])
        
        assert len(docs) == 3
        assert all("data" in d for d in docs)


class TestEmbeddingConfigIntegration:
    """Test embedding config integration."""

    def test_embedder_from_config(self):
        """Test creating embedder from user config."""
        from rlm_dspy.core.embeddings import EmbeddingConfig, get_embedder
        
        config = EmbeddingConfig.from_user_config()
        embedder = get_embedder(config)
        
        assert embedder is not None
        assert callable(embedder)

    def test_embedder_caching_works(self):
        """Test that embedder caching returns same instance."""
        from rlm_dspy.core.embeddings import get_embedder, clear_embedder_cache
        
        clear_embedder_cache()
        
        embedder1 = get_embedder()
        embedder2 = get_embedder()
        
        assert embedder1 is embedder2


class TestVectorIndexIntegration:
    """Test vector index integration."""

    def test_index_manager_singleton(self):
        """Test index manager is singleton."""
        from rlm_dspy.core.vector_index import get_index_manager
        
        manager1 = get_index_manager()
        manager2 = get_index_manager()
        
        assert manager1 is manager2

    @requires_embedding_api_key
    def test_build_and_search_pipeline(self, tmp_path):
        """Test full build and search pipeline."""
        from rlm_dspy.core.vector_index import CodeIndex, IndexConfig
        
        # Create test files
        (tmp_path / "auth.py").write_text("""
def login(username, password):
    '''Authenticate user with credentials'''
    return verify_password(username, password)

def logout(user):
    '''Log out the current user'''
    user.session = None
""")
        
        (tmp_path / "db.py").write_text("""
def connect():
    '''Connect to database'''
    return Database()

def query(sql):
    '''Execute SQL query'''
    return db.execute(sql)
""")
        
        # Build index
        config = IndexConfig(
            index_dir=tmp_path / "indexes",
            use_faiss=False,  # Don't require FAISS
        )
        index = CodeIndex(config)
        count = index.build(tmp_path)
        
        # Should have indexed some snippets
        assert count > 0
        
        # Search should work
        results = index.search(tmp_path, "authentication", k=3)
        
        # Should find auth-related code
        assert len(results) > 0


class TestCitationsIntegration:
    """Test citations utilities integration."""

    def test_extract_references_from_text(self):
        """Test extracting file references from analysis text."""
        from rlm_dspy.core.citations import extract_file_references
        
        text = """
        Found several issues:
        1. SQL injection at db.py:45
        2. Missing validation in api.py:23
        3. Hardcoded secret at config.py:10-15
        """
        
        refs = extract_file_references(text)
        
        # Should find all three references
        assert len(refs) >= 3
        assert any("db.py" in r[0] for r in refs)
        assert any("api.py" in r[0] for r in refs)
        assert any("config.py" in r[0] for r in refs)

    def test_parse_findings_with_citations(self):
        """Test parsing findings and auto-detecting citations."""
        from rlm_dspy.core.citations import (
            parse_findings_from_text, code_to_document
        )
        
        text = """
        - [CRITICAL] SQL injection vulnerability at db.py:45
        - [WARNING] Missing input validation at api.py:23
        - [INFO] Consider adding logging at utils.py:100
        """
        
        docs = [
            {"title": "db.py", "data": ""},
            {"title": "api.py", "data": ""},
            {"title": "utils.py", "data": ""},
        ]
        
        findings = parse_findings_from_text(text, docs)
        
        assert len(findings) >= 3
        
        # Check severities were detected
        severities = {f.severity for f in findings}
        assert "critical" in severities or "error" in severities


class TestRLMConfigIntegration:
    """Test RLM configuration integration."""

    def test_config_loads_defaults(self):
        """Test that RLMConfig loads with defaults."""
        from rlm_dspy import RLMConfig
        
        config = RLMConfig()
        
        assert config.max_iterations > 0
        assert config.max_llm_calls > 0
        assert config.model is not None

    def test_rlm_creates_with_config(self):
        """Test that RLM can be created with config."""
        from rlm_dspy import RLM, RLMConfig
        
        config = RLMConfig(
            max_iterations=5,
            max_llm_calls=10,
        )
        
        rlm = RLM(config=config)
        
        assert rlm.config.max_iterations == 5
        assert rlm.config.max_llm_calls == 10


class TestToolsIntegration:
    """Test tools integration."""

    def test_all_tools_callable(self):
        """Test all built-in tools are callable."""
        from rlm_dspy import BUILTIN_TOOLS
        
        for name, tool in BUILTIN_TOOLS.items():
            assert callable(tool), f"{name} is not callable"

    def test_tool_descriptions_complete(self):
        """Test all tools have descriptions."""
        from rlm_dspy import BUILTIN_TOOLS
        
        for name, tool in BUILTIN_TOOLS.items():
            assert tool.__doc__, f"{name} missing docstring"

    def test_safe_tools_excludes_shell(self):
        """Test SAFE_TOOLS doesn't include shell."""
        from rlm_dspy import SAFE_TOOLS
        
        assert "shell" not in SAFE_TOOLS


class TestExportsIntegration:
    """Test that all exports work correctly."""

    def test_main_exports(self):
        """Test main module exports."""
        from rlm_dspy import (
            # Core
            RLM, RLMConfig, RLMResult, ProgressCallback,
            # Signatures
            SecurityAudit, CodeReview, BugFinder,
            ArchitectureAnalysis, PerformanceAnalysis, DiffReview,
            # Cited signatures
            CitedAnalysis, CitedSecurityAudit, CitedBugFinder, CitedCodeReview,
            # Registry
            SIGNATURES, get_signature, list_signatures,
            # Guards
            validate_groundedness, validate_completeness, semantic_f1,
            # Tools
            BUILTIN_TOOLS, SAFE_TOOLS, semantic_search,
            # Citations
            SourceLocation, CitedFinding, CitedAnalysisResult,
            code_to_document, files_to_documents,
        )
        
        # All should be importable
        assert RLM is not None
        assert CitedAnalysis is not None
        assert semantic_search is not None
        assert SourceLocation is not None

    def test_core_exports(self):
        """Test core module exports."""
        from rlm_dspy.core import (
            # Embeddings
            EmbeddingConfig, get_embedder, embed_texts, get_embedding_dim,
            # Vector Index
            IndexConfig, CodeSnippet, SearchResult, CodeIndex, get_index_manager,
            # Citations
            SourceLocation, CitedFinding, CitedAnalysisResult,
            code_to_document, files_to_documents, citations_to_locations,
        )
        
        assert EmbeddingConfig is not None
        assert CodeIndex is not None
        assert SourceLocation is not None

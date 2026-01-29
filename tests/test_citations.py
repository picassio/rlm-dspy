"""Tests for citations module."""

import pytest
from pathlib import Path


class TestSourceLocation:
    """Test SourceLocation dataclass."""

    def test_str_single_line(self):
        """Test string representation for single line."""
        from rlm_dspy.core.citations import SourceLocation
        
        loc = SourceLocation(file="test.py", line=42)
        assert str(loc) == "test.py:42"

    def test_str_line_range(self):
        """Test string representation for line range."""
        from rlm_dspy.core.citations import SourceLocation
        
        loc = SourceLocation(file="test.py", line=42, end_line=50)
        assert str(loc) == "test.py:42-50"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from rlm_dspy.core.citations import SourceLocation
        
        loc = SourceLocation(
            file="test.py",
            line=42,
            end_line=45,
            snippet="def foo():\n    pass",
        )
        d = loc.to_dict()
        
        assert d["file"] == "test.py"
        assert d["line"] == 42
        assert d["end_line"] == 45
        assert "def foo" in d["snippet"]


class TestCitedFinding:
    """Test CitedFinding dataclass."""

    def test_format_simple(self):
        """Test basic formatting."""
        from rlm_dspy.core.citations import CitedFinding, SourceLocation
        
        finding = CitedFinding(
            text="Found a bug",
            severity="error",
        )
        
        formatted = finding.format()
        assert "[ERROR]" in formatted
        assert "Found a bug" in formatted

    def test_format_with_sources(self):
        """Test formatting with source citations."""
        from rlm_dspy.core.citations import CitedFinding, SourceLocation
        
        finding = CitedFinding(
            text="SQL injection vulnerability",
            sources=[
                SourceLocation(file="db.py", line=45),
                SourceLocation(file="api.py", line=123),
            ],
            severity="critical",
            category="security",
        )
        
        formatted = finding.format()
        assert "[CRITICAL:security]" in formatted
        assert "→ db.py:45" in formatted
        assert "→ api.py:123" in formatted

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from rlm_dspy.core.citations import CitedFinding, SourceLocation
        
        finding = CitedFinding(
            text="Test finding",
            sources=[SourceLocation(file="test.py", line=1)],
            severity="warning",
            category="style",
        )
        
        d = finding.to_dict()
        assert d["text"] == "Test finding"
        assert d["severity"] == "warning"
        assert d["category"] == "style"
        assert len(d["sources"]) == 1


class TestCodeToDocument:
    """Test code_to_document function."""

    def test_adds_line_numbers(self):
        """Test that line numbers are added."""
        from rlm_dspy.core.citations import code_to_document
        
        content = "def foo():\n    pass\n    return 42"
        doc = code_to_document("test.py", content=content)
        
        assert "   1 | def foo():" in doc["data"]
        assert "   2 |     pass" in doc["data"]
        assert "   3 |     return 42" in doc["data"]

    def test_sets_title(self):
        """Test that title is set to file path."""
        from rlm_dspy.core.citations import code_to_document
        
        doc = code_to_document("/path/to/test.py", content="x = 1")
        assert doc["title"] == "/path/to/test.py"

    def test_sets_media_type(self):
        """Test media type is text/plain."""
        from rlm_dspy.core.citations import code_to_document
        
        doc = code_to_document("test.py", content="x = 1")
        assert doc["media_type"] == "text/plain"


class TestFilesToDocuments:
    """Test files_to_documents function."""

    def test_converts_multiple_files(self, tmp_path):
        """Test converting multiple files."""
        from rlm_dspy.core.citations import files_to_documents
        
        # Create test files
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        
        docs = files_to_documents([tmp_path / "a.py", tmp_path / "b.py"])
        
        assert len(docs) == 2

    def test_skips_nonexistent_files(self, tmp_path):
        """Test that nonexistent files are skipped."""
        from rlm_dspy.core.citations import files_to_documents
        
        (tmp_path / "exists.py").write_text("x = 1")
        
        docs = files_to_documents([
            tmp_path / "exists.py",
            tmp_path / "nonexistent.py",
        ])
        
        assert len(docs) == 1

    def test_respects_max_files(self, tmp_path):
        """Test max_files limit."""
        from rlm_dspy.core.citations import files_to_documents
        
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"x = {i}")
        
        docs = files_to_documents(
            [tmp_path / f"file{i}.py" for i in range(5)],
            max_files=2,
        )
        
        assert len(docs) == 2


class TestExtractFileReferences:
    """Test extract_file_references function."""

    def test_extract_colon_format(self):
        """Test file.py:123 format."""
        from rlm_dspy.core.citations import extract_file_references
        
        refs = extract_file_references("Found issue at test.py:42")
        assert ("test.py", 42) in refs

    def test_extract_range_format(self):
        """Test file.py:123-456 format."""
        from rlm_dspy.core.citations import extract_file_references
        
        refs = extract_file_references("See test.py:10-20 for details")
        assert ("test.py", 10) in refs

    def test_extract_line_keyword_format(self):
        """Test 'file.py, line 123' format."""
        from rlm_dspy.core.citations import extract_file_references
        
        refs = extract_file_references("Check test.py, line 55")
        assert ("test.py", 55) in refs

    def test_extract_at_line_format(self):
        """Test 'at line X of file.py' format."""
        from rlm_dspy.core.citations import extract_file_references
        
        refs = extract_file_references("Error at line 99 of api.py")
        assert ("api.py", 99) in refs

    def test_extract_multiple_refs(self):
        """Test extracting multiple references."""
        from rlm_dspy.core.citations import extract_file_references
        
        text = "Issues in test.py:10, also api.py:20 and utils.py:30"
        refs = extract_file_references(text)
        
        assert len(refs) >= 3


class TestCitationsToLocations:
    """Test citations_to_locations function."""

    def test_matches_exact_file(self):
        """Test matching exact file path."""
        from rlm_dspy.core.citations import citations_to_locations
        
        docs = [{"title": "src/test.py", "data": "line1\nline2\nline3"}]
        
        locations = citations_to_locations("Issue at src/test.py:2", docs)
        
        assert len(locations) >= 1
        assert any(loc.file == "src/test.py" for loc in locations)

    def test_matches_filename_only(self):
        """Test matching by filename when full path not in text."""
        from rlm_dspy.core.citations import citations_to_locations
        
        docs = [{"title": "/full/path/to/test.py", "data": "line1\nline2"}]
        
        locations = citations_to_locations("Issue at test.py:1", docs)
        
        assert len(locations) >= 1


class TestParseFindingsFromText:
    """Test parse_findings_from_text function."""

    def test_parse_bullet_points(self):
        """Test parsing bullet-point findings."""
        from rlm_dspy.core.citations import parse_findings_from_text
        
        text = """
        - Found SQL injection at db.py:45
        - Missing input validation in api.py:23
        - Hardcoded credentials in config.py:10
        """
        
        docs = [
            {"title": "db.py", "data": ""},
            {"title": "api.py", "data": ""},
            {"title": "config.py", "data": ""},
        ]
        
        findings = parse_findings_from_text(text, docs)
        
        assert len(findings) >= 3

    def test_parse_severity_markers(self):
        """Test parsing severity from text."""
        from rlm_dspy.core.citations import parse_findings_from_text
        
        text = "[CRITICAL] Security vulnerability at vuln.py:1"
        
        findings = parse_findings_from_text(text, [{"title": "vuln.py", "data": ""}])
        
        # Should detect critical severity
        assert len(findings) >= 1

    def test_detects_security_category(self):
        """Test that security issues are categorized."""
        from rlm_dspy.core.citations import parse_findings_from_text
        
        text = "- SQL injection vulnerability at db.py:10"
        
        findings = parse_findings_from_text(text, [{"title": "db.py", "data": ""}])
        
        assert len(findings) >= 1
        assert any(f.category == "security" for f in findings)


class TestCitedSignatures:
    """Test cited signature classes."""

    def test_cited_analysis_exists(self):
        """Test CitedAnalysis signature."""
        from rlm_dspy.signatures import CitedAnalysis
        
        # Check it's a valid DSPy signature
        assert hasattr(CitedAnalysis, 'input_fields')
        assert hasattr(CitedAnalysis, 'output_fields')
        # Check fields exist
        fields = CitedAnalysis.input_fields
        assert 'context' in fields
        assert 'query' in fields

    def test_cited_security_exists(self):
        """Test CitedSecurityAudit signature."""
        from rlm_dspy.signatures import CitedSecurityAudit
        
        assert hasattr(CitedSecurityAudit, 'output_fields')
        fields = CitedSecurityAudit.output_fields
        assert 'vulnerabilities' in fields
        assert 'locations' in fields

    def test_get_signature_cited(self):
        """Test getting cited signature by name."""
        from rlm_dspy.signatures import get_signature
        
        sig = get_signature("cited")
        assert sig is not None
        assert sig.__name__ == "CitedAnalysis"

    def test_get_signature_aliases(self):
        """Test signature aliases work."""
        from rlm_dspy.signatures import get_signature
        
        # Test alias
        sig = get_signature("cited-audit")
        assert sig is not None
        assert sig.__name__ == "CitedSecurityAudit"


class TestCitedAnalysisResult:
    """Test CitedAnalysisResult dataclass."""

    def test_format_output(self):
        """Test formatting full result."""
        from rlm_dspy.core.citations import (
            CitedAnalysisResult, CitedFinding, SourceLocation
        )
        
        result = CitedAnalysisResult(
            summary="Found 2 issues",
            findings=[
                CitedFinding(
                    text="SQL injection",
                    sources=[SourceLocation(file="db.py", line=10)],
                    severity="critical",
                    category="security",
                ),
                CitedFinding(
                    text="Missing validation",
                    sources=[SourceLocation(file="api.py", line=20)],
                    severity="warning",
                    category="security",
                ),
            ],
            documents_analyzed=5,
        )
        
        formatted = result.format()
        
        assert "ANALYSIS SUMMARY" in formatted
        assert "Found 2 issues" in formatted
        assert "Documents analyzed: 5" in formatted
        assert "SQL injection" in formatted
        assert "db.py:10" in formatted

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from rlm_dspy.core.citations import CitedAnalysisResult, CitedFinding
        
        result = CitedAnalysisResult(
            summary="Test",
            findings=[CitedFinding(text="Issue", severity="info")],
            documents_analyzed=1,
        )
        
        d = result.to_dict()
        assert d["summary"] == "Test"
        assert len(d["findings"]) == 1
        assert d["documents_analyzed"] == 1

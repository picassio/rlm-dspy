"""Citations support for grounded code analysis.

Provides source references in analysis results, linking findings back
to specific files and line numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import dspy

# Cache for code_to_document results: (path, mtime) -> document
_document_cache: dict[tuple[str, float], dict] = {}
_DOCUMENT_CACHE_MAX_SIZE = 200


@dataclass
class SourceLocation:
    """A location in source code."""
    file: str
    line: int
    end_line: int | None = None
    snippet: str = ""

    def __str__(self) -> str:
        if self.end_line and self.end_line != self.line:
            return f"{self.file}:{self.line}-{self.end_line}"
        return f"{self.file}:{self.line}"

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "snippet": self.snippet[:200] if self.snippet else "",
        }


@dataclass
class CitedFinding:
    """A finding with source citations."""
    text: str
    sources: list[SourceLocation] = field(default_factory=list)
    severity: str = "info"  # info, warning, error, critical
    category: str = ""  # security, performance, bug, style, etc.

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "severity": self.severity,
            "category": self.category,
            "sources": [s.to_dict() for s in self.sources],
        }

    def format(self) -> str:
        """Format finding with citations."""
        lines = [f"[{self.severity.upper()}] {self.text}"]
        if self.category:
            lines[0] = f"[{self.severity.upper()}:{self.category}] {self.text}"
        for source in self.sources:
            lines.append(f"  → {source}")
        return "\n".join(lines)


@dataclass
class CitedAnalysisResult:
    """Analysis result with citations."""
    summary: str
    findings: list[CitedFinding] = field(default_factory=list)
    documents_analyzed: int = 0

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "documents_analyzed": self.documents_analyzed,
        }

    def format(self) -> str:
        """Format full result."""
        lines = [
            "=" * 60,
            "ANALYSIS SUMMARY",
            "=" * 60,
            self.summary,
            "",
            f"Documents analyzed: {self.documents_analyzed}",
            f"Findings: {len(self.findings)}",
            "",
        ]

        if self.findings:
            lines.append("-" * 60)
            lines.append("FINDINGS")
            lines.append("-" * 60)
            for i, finding in enumerate(self.findings, 1):
                lines.append(f"\n{i}. {finding.format()}")

        return "\n".join(lines)


def code_to_document(
    file_path: str | Path,
    content: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Convert a code file to a Document-like dict for citation.

    Results are cached by (path, mtime) for performance in recursive workflows.

    Args:
        file_path: Path to the file
        content: Optional content (reads from file if not provided)
        use_cache: Whether to use caching (default True)

    Returns:
        Dict in DSPy Document format
    """
    file_path = Path(file_path)

    # Check cache if content not provided and caching enabled
    if content is None and use_cache and file_path.exists():
        try:
            mtime = file_path.stat().st_mtime
            cache_key = (str(file_path.resolve()), mtime)

            if cache_key in _document_cache:
                return _document_cache[cache_key]
        except OSError:
            pass

    if content is None:
        content = file_path.read_text(encoding='utf-8')

    # Add line numbers for easier reference
    lines = content.splitlines()
    numbered_content = "\n".join(
        f"{i+1:4d} | {line}" for i, line in enumerate(lines)
    )

    result = {
        "data": numbered_content,
        "title": str(file_path),
        "media_type": "text/plain",
        "context": f"Source code file: {file_path.name}",
    }

    # Cache result if caching enabled
    if use_cache and 'cache_key' in locals():
        # Evict oldest entries if cache is full
        if len(_document_cache) >= _DOCUMENT_CACHE_MAX_SIZE:
            # Remove first 50 entries
            keys_to_remove = list(_document_cache.keys())[:50]
            for k in keys_to_remove:
                del _document_cache[k]
        _document_cache[cache_key] = result

    return result


def files_to_documents(
    paths: list[str | Path],
    max_files: int = 50,
) -> list[dict]:
    """Convert multiple files to Document dicts.

    Args:
        paths: List of file paths
        max_files: Maximum number of files to include

    Returns:
        List of Document dicts
    """
    documents = []

    for path in paths[:max_files]:
        path = Path(path)
        if not path.is_file():
            continue

        try:
            documents.append(code_to_document(path))
        except (UnicodeDecodeError, OSError):
            continue

    return documents


def extract_file_references(text: str) -> list[tuple[str, int | None]]:
    """Extract file:line references from text.

    Patterns recognized:
    - file.py:123
    - file.py:123-456
    - file.py, line 123
    - at line 123 of file.py

    Returns:
        List of (file, line) tuples
    """
    references = []

    # Pattern: file.py:123 or file.py:123-456
    pattern1 = r'([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]+):(\d+)(?:-\d+)?'
    for match in re.finditer(pattern1, text):
        references.append((match.group(1), int(match.group(2))))

    # Pattern: file.py, line 123
    pattern2 = r'([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]+),?\s+line\s+(\d+)'
    for match in re.finditer(pattern2, text, re.IGNORECASE):
        references.append((match.group(1), int(match.group(2))))

    # Pattern: at line 123 of file.py
    pattern3 = r'(?:at\s+)?line\s+(\d+)\s+(?:of|in)\s+([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]+)'
    for match in re.finditer(pattern3, text, re.IGNORECASE):
        references.append((match.group(2), int(match.group(1))))

    return list(set(references))


def citations_to_locations(
    analysis_text: str,
    documents: list[dict],
) -> list[SourceLocation]:
    """Extract source locations from analysis text.

    Matches file references in the text to the provided documents.

    Args:
        analysis_text: The analysis result text
        documents: List of Document dicts that were analyzed

    Returns:
        List of SourceLocation objects
    """
    # Build file title -> content map
    file_contents = {}
    for doc in documents:
        title = doc.get("title", "")
        if title:
            file_contents[title] = doc.get("data", "")
            # Also map just the filename
            file_contents[Path(title).name] = doc.get("data", "")

    locations = []
    references = extract_file_references(analysis_text)

    for file_ref, line in references:
        # Try to find matching document
        content = file_contents.get(file_ref, "")
        if not content:
            # Try with just filename
            for title in file_contents:
                if title.endswith(file_ref) or Path(title).name == file_ref:
                    content = file_contents[title]
                    file_ref = title
                    break

        # Extract snippet around the line
        snippet = ""
        if content and line:
            lines = content.splitlines()
            # Account for line number prefix in our format
            if lines and lines[0].strip().startswith("1 |"):
                # Lines are numbered, extract actual content
                actual_lines = []
                for line_text in lines:
                    if " | " in line_text:
                        actual_lines.append(line_text.split(" | ", 1)[1])
                    else:
                        actual_lines.append(line_text)
                lines = actual_lines

            start = max(0, line - 2)  # 1 line before
            end = min(len(lines), line + 3)  # 2 lines after
            snippet = "\n".join(lines[start:end])

        locations.append(SourceLocation(
            file=file_ref,
            line=line or 0,
            snippet=snippet,
        ))

    return locations


def parse_findings_from_text(
    analysis_text: str,
    documents: list[dict],
) -> list[CitedFinding]:
    """Parse findings from analysis text and add citations.

    Looks for common patterns like:
    - Bullet points starting with - or *
    - Numbered lists
    - Lines starting with severity markers

    Args:
        analysis_text: The analysis result text
        documents: List of Document dicts for citation lookup

    Returns:
        List of CitedFinding objects
    """
    findings = []

    # Split into potential findings
    lines = analysis_text.splitlines()
    current_finding = []

    for line in lines:
        stripped = line.strip()

        # Check if this starts a new finding
        is_new_finding = (
            stripped.startswith(('-', '*', '•')) or
            re.match(r'^\d+\.', stripped) or
            re.match(r'^\[(WARNING|ERROR|INFO|CRITICAL)\]', stripped, re.IGNORECASE) or
            re.match(r'^(Bug|Issue|Problem|Warning|Error|Security|Performance):', stripped, re.IGNORECASE)
        )

        if is_new_finding and current_finding:
            # Process previous finding
            finding_text = "\n".join(current_finding)
            finding = _create_finding(finding_text, documents)
            if finding:
                findings.append(finding)
            current_finding = [stripped]
        elif stripped:
            current_finding.append(stripped)

    # Process last finding
    if current_finding:
        finding_text = "\n".join(current_finding)
        finding = _create_finding(finding_text, documents)
        if finding:
            findings.append(finding)

    return findings


def _create_finding(text: str, documents: list[dict]) -> CitedFinding | None:
    """Create a CitedFinding from text."""
    if not text or len(text) < 10:
        return None

    # Detect severity
    severity = "info"
    if re.search(r'\b(critical|severe|dangerous)\b', text, re.IGNORECASE):
        severity = "critical"
    elif re.search(r'\b(error|bug|broken|fail)\b', text, re.IGNORECASE):
        severity = "error"
    elif re.search(r'\b(warning|caution|potential|possible)\b', text, re.IGNORECASE):
        severity = "warning"

    # Detect category
    category = ""
    if re.search(r'\b(security|vulnerab|injection|xss|csrf|auth)\b', text, re.IGNORECASE):
        category = "security"
    elif re.search(r'\b(performance|slow|optimization|memory|leak)\b', text, re.IGNORECASE):
        category = "performance"
    elif re.search(r'\b(bug|error|exception|crash|null)\b', text, re.IGNORECASE):
        category = "bug"
    elif re.search(r'\b(style|naming|convention|format)\b', text, re.IGNORECASE):
        category = "style"

    # Extract citations
    sources = citations_to_locations(text, documents)

    # Clean up text (remove bullet markers, etc.)
    clean_text = re.sub(r'^[-*•]\s*', '', text)
    clean_text = re.sub(r'^\d+\.\s*', '', clean_text)
    clean_text = re.sub(r'^\[(WARNING|ERROR|INFO|CRITICAL)\]\s*', '', clean_text, flags=re.IGNORECASE)

    return CitedFinding(
        text=clean_text.strip(),
        sources=sources,
        severity=severity,
        category=category,
    )


# DSPy Signature for cited analysis
class CitedCodeAnalysis(dspy.Signature):
    """Analyze code and provide findings with source citations.

    For each finding, include the exact file and line number where the issue occurs.
    Format references as: filename.py:line_number
    """
    context: str = dspy.InputField(desc="Source code with line numbers")
    question: str = dspy.InputField(desc="What to analyze or look for")

    analysis: str = dspy.OutputField(
        desc="Detailed analysis with file:line citations for each finding"
    )
    findings_count: int = dspy.OutputField(
        desc="Number of distinct findings"
    )


class CitedSecurityAudit(dspy.Signature):
    """Security audit with precise source citations.

    Identify security vulnerabilities and reference exact locations.
    Format: [SEVERITY] Description - filename.py:line
    """
    context: str = dspy.InputField(desc="Source code with line numbers")

    vulnerabilities: str = dspy.OutputField(
        desc="List of vulnerabilities with [CRITICAL/HIGH/MEDIUM/LOW] severity and file:line citations"
    )
    summary: str = dspy.OutputField(
        desc="Overall security assessment"
    )
    risk_score: int = dspy.OutputField(
        desc="Risk score from 0 (safe) to 100 (critical)"
    )


class CitedBugFinder(dspy.Signature):
    """Find bugs with precise source citations.

    Identify potential bugs and reference exact locations.
    Format: [BUG] Description - filename.py:line
    """
    context: str = dspy.InputField(desc="Source code with line numbers")

    bugs: str = dspy.OutputField(
        desc="List of bugs with descriptions and file:line citations"
    )
    summary: str = dspy.OutputField(
        desc="Overall code quality assessment"
    )


# Export
__all__ = [
    "SourceLocation",
    "CitedFinding",
    "CitedAnalysisResult",
    "code_to_document",
    "files_to_documents",
    "extract_file_references",
    "citations_to_locations",
    "parse_findings_from_text",
    "CitedCodeAnalysis",
    "CitedSecurityAudit",
    "CitedBugFinder",
]

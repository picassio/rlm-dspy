"""
RLM Query Answering Reward Function

Inspired by QMD's multi-dimensional scoring approach.
Scores code exploration answers on multiple dimensions.

Dimensions:
  Completeness (30) - Answer addresses the query, has substance
  Accuracy (30)     - Key terms from expected answer are present
  Specificity (20)  - References specific files, lines, code snippets
  Format (20)       - Well-structured, not too short/long

Returns 0.0-1.0 for optimization rewards, or detailed breakdown dict.
"""

import re
from collections import Counter

# =============================================================================
# Constants
# =============================================================================

STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'so', 'yet',
    'both', 'either', 'neither', 'not', 'only', 'own', 'same', 'than',
    'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'each', 'every', 'any', 'some', 'no', 'none',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
    'they', 'what', 'which', 'who', 'whom', 'whose', 'if', 'because',
    'while', 'although', 'though', 'unless', 'until', 'about', 'can',
})

# Patterns that indicate specific code references
CODE_REFERENCE_PATTERNS = [
    r'\b[\w/]+\.py\b',           # Python files
    r'\b[\w/]+\.ts\b',           # TypeScript files
    r'\b[\w/]+\.js\b',           # JavaScript files
    r'\bline[s]?\s*\d+',         # Line references
    r':\d+',                      # file:line format
    r'`[^`]+`',                   # Inline code
    r'```[\s\S]*?```',           # Code blocks
    r'\bclass\s+\w+',            # Class definitions
    r'\bdef\s+\w+',              # Function definitions
    r'\bfunction\s+\w+',         # JS function definitions
]

# Generic phrases that indicate a non-specific answer
GENERIC_PHRASES = [
    'let me explore',
    'let me search',
    'let me find',
    'i will analyze',
    'i would need to',
    'i can help you',
    'would you like me to',
    'shall i',
    'i\'ll start by',
    'first, i\'ll',
]


# =============================================================================
# Helpers
# =============================================================================

def extract_key_terms(text: str) -> set[str]:
    """Extract meaningful terms from text, removing stopwords."""
    if not text or not isinstance(text, str):
        return set()
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return words - STOPWORDS


def extract_code_references(text: str) -> list[str]:
    """Extract code-specific references from text."""
    refs = []
    for pattern in CODE_REFERENCE_PATTERNS:
        refs.extend(re.findall(pattern, text, re.IGNORECASE))
    return refs


def is_generic_response(text: str) -> bool:
    """Check if response is generic without specific findings."""
    lower = text.lower()
    for phrase in GENERIC_PHRASES:
        if phrase in lower:
            # Check if there's substance after the generic phrase
            idx = lower.find(phrase)
            after = lower[idx + len(phrase):]
            # If nothing substantial follows, it's generic
            if len(after.strip()) < 100:
                return True
    return False


def echoes_query(answer: str, query: str, threshold: float = 0.8) -> bool:
    """Check if answer mostly just repeats the query."""
    answer_terms = extract_key_terms(answer[:200])  # First 200 chars
    query_terms = extract_key_terms(query)
    
    if not query_terms:
        return False
    
    overlap = len(answer_terms & query_terms) / len(query_terms)
    return overlap >= threshold


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


# =============================================================================
# Scoring Functions
# =============================================================================

def score_completeness(answer: str, query: str) -> tuple[int, list[str]]:
    """
    Score how completely the answer addresses the query.
    
    Returns (score 0-30, deductions list)
    """
    deductions = []
    score = 0
    
    # Check minimum length
    wc = word_count(answer)
    if wc < 20:
        deductions.append(f"too short ({wc} words)")
        return 0, deductions
    
    # Base score for having content
    score = 10
    
    # Length-based scoring
    if wc >= 50:
        score += 5
    if wc >= 100:
        score += 5
    if wc >= 200:
        score += 5
    
    # Penalty for being too long without substance
    if wc > 500:
        refs = extract_code_references(answer)
        if len(refs) < 3:
            score -= 5
            deductions.append("very long but few code references")
    
    # Penalty for generic responses
    if is_generic_response(answer):
        score -= 10
        deductions.append("generic response without findings")
    
    # Penalty for echoing query
    if echoes_query(answer, query):
        score -= 5
        deductions.append("echoes query")
    
    return max(0, min(30, score)), deductions


def score_accuracy(answer: str, expected: str) -> tuple[int, list[str]]:
    """
    Score how accurately the answer matches expected content.
    
    Returns (score 0-30, deductions list)
    """
    deductions = []
    
    if not expected:
        # No expected answer - give partial credit for having content
        if len(answer) > 100:
            return 15, ["no expected answer to compare"]
        return 10, ["no expected answer, short response"]
    
    answer_terms = extract_key_terms(answer)
    expected_terms = extract_key_terms(expected)
    
    if not expected_terms:
        return 15, ["no key terms in expected answer"]
    
    overlap = answer_terms & expected_terms
    coverage = len(overlap) / len(expected_terms)
    
    # Scoring based on coverage
    if coverage >= 0.9:
        score = 30
    elif coverage >= 0.7:
        score = 25
    elif coverage >= 0.5:
        score = 20
    elif coverage >= 0.3:
        score = 15
    elif coverage >= 0.1:
        score = 10
    else:
        score = 5
        deductions.append(f"low term overlap ({coverage:.0%})")
    
    # Bonus for exact substring match
    if expected.lower()[:100] in answer.lower():
        score = min(30, score + 5)
    
    missing = expected_terms - answer_terms
    if missing and len(missing) <= 5:
        deductions.append(f"missing terms: {', '.join(list(missing)[:3])}")
    
    return score, deductions


def score_specificity(answer: str) -> tuple[int, list[str]]:
    """
    Score how specific the answer is (code references, line numbers, etc).
    
    Returns (score 0-20, deductions list)
    """
    deductions = []
    refs = extract_code_references(answer)
    
    # Score based on number of code references
    if len(refs) >= 5:
        score = 20
    elif len(refs) >= 3:
        score = 15
    elif len(refs) >= 1:
        score = 10
    else:
        score = 5
        deductions.append("no code references found")
    
    # Check for specific patterns
    has_file_refs = bool(re.search(r'\b[\w/]+\.(py|ts|js|md)\b', answer))
    has_line_refs = bool(re.search(r'line[s]?\s*\d+|:\d+', answer, re.I))
    has_code_blocks = '```' in answer or '`' in answer
    
    if has_file_refs and has_line_refs:
        score = min(20, score + 3)
    
    if has_code_blocks:
        score = min(20, score + 2)
    
    return min(20, score), deductions


def score_format(answer: str) -> tuple[int, list[str]]:
    """
    Score the format/structure of the answer.
    
    Returns (score 0-20, deductions list)
    """
    deductions = []
    score = 10  # Base score
    
    wc = word_count(answer)
    
    # Length checks
    if wc < 20:
        score -= 5
        deductions.append("too short")
    elif wc > 1000:
        score -= 3
        deductions.append("very long")
    
    # Structure checks
    has_paragraphs = '\n\n' in answer or answer.count('\n') >= 3
    has_lists = bool(re.search(r'^\s*[-*•]\s', answer, re.M)) or bool(re.search(r'^\s*\d+\.\s', answer, re.M))
    has_headers = bool(re.search(r'^#+\s', answer, re.M))
    
    if has_paragraphs:
        score += 3
    if has_lists:
        score += 3
    if has_headers:
        score += 2
    
    # Check for reasonable sentence structure
    sentences = re.split(r'[.!?]', answer)
    avg_sentence_len = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    
    if 5 < avg_sentence_len < 30:
        score += 2
    
    return min(20, max(0, score)), deductions


# =============================================================================
# Main Scoring Function
# =============================================================================

def score_answer_detailed(query: str, answer: str, expected: str = "") -> dict:
    """
    Score an answer with full breakdown.
    
    Args:
        query: The original question
        answer: The generated answer
        expected: Expected/reference answer (optional)
    
    Returns:
        Dict with scores, percentage, rating, and deductions
    """
    all_deductions = []
    
    # Handle empty answer
    if not answer or not answer.strip():
        return {
            "completeness": 0,
            "accuracy": 0,
            "specificity": 0,
            "format": 0,
            "total": 0,
            "max_possible": 100,
            "percentage": 0.0,
            "score": 0.0,
            "rating": "Failed",
            "deductions": ["empty answer"],
        }
    
    # Score each dimension
    completeness, comp_ded = score_completeness(answer, query)
    all_deductions.extend(comp_ded)
    
    accuracy, acc_ded = score_accuracy(answer, expected)
    all_deductions.extend(acc_ded)
    
    specificity, spec_ded = score_specificity(answer)
    all_deductions.extend(spec_ded)
    
    format_score, fmt_ded = score_format(answer)
    all_deductions.extend(fmt_ded)
    
    # Calculate total
    total = completeness + accuracy + specificity + format_score
    max_possible = 100
    percentage = total / max_possible * 100
    
    # Rating
    if percentage >= 80:
        rating = "Excellent"
    elif percentage >= 60:
        rating = "Good"
    elif percentage >= 40:
        rating = "Acceptable"
    elif percentage >= 20:
        rating = "Poor"
    else:
        rating = "Failed"
    
    return {
        "completeness": completeness,
        "accuracy": accuracy,
        "specificity": specificity,
        "format": format_score,
        "total": total,
        "max_possible": max_possible,
        "percentage": round(percentage, 1),
        "score": round(total / max_possible, 3),
        "rating": rating,
        "deductions": all_deductions,
    }


def score_answer(query: str, answer: str, expected: str = "") -> float:
    """
    Score an answer as a float in [0.0, 1.0] for optimization.
    
    Args:
        query: The original question
        answer: The generated answer
        expected: Expected/reference answer (optional)
    
    Returns:
        Float score between 0.0 and 1.0
    """
    result = score_answer_detailed(query, answer, expected)
    return result["score"]


# =============================================================================
# CLI: run standalone to test
# =============================================================================

if __name__ == "__main__":
    print("RLM Reward Function Self-Test")
    print("=" * 60)
    
    tests = [
        # (query, answer, expected)
        (
            "What is the main purpose of this project?",
            "This is a code exploration tool that helps developers understand codebases.",
            "A code exploration tool for developers",
        ),
        (
            "Find bugs in the code",
            "Let me explore the codebase...",
            "Found a bug in file.py:123",
        ),
        (
            "How does authentication work?",
            """Authentication is handled in `src/auth/oauth.py`. The main class is `OAuthHandler` 
            which implements OAuth 2.0 flow. Key methods:
            
            - `authenticate()` at line 45: Initiates the OAuth flow
            - `callback()` at line 78: Handles the callback
            
            ```python
            def authenticate(self):
                return redirect(oauth_url)
            ```""",
            "OAuth authentication in oauth.py with authenticate and callback methods",
        ),
        (
            "What files exist?",
            "",
            "List of files",
        ),
    ]
    
    for query, answer, expected in tests:
        result = score_answer_detailed(query, answer, expected)
        print(f"\nQuery: {query[:50]}...")
        print(f"Score: {result['score']:.2f} ({result['rating']})")
        print(f"  Completeness: {result['completeness']}/30")
        print(f"  Accuracy: {result['accuracy']}/30")
        print(f"  Specificity: {result['specificity']}/20")
        print(f"  Format: {result['format']}/20")
        if result['deductions']:
            print(f"  Issues: {', '.join(result['deductions'][:3])}")

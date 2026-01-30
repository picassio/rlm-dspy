"""
KNN Few-Shot example selection.

Dynamically selects relevant examples for few-shot prompting based on
query similarity, following DSPy's KNNFewShot pattern.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KNNFewShotConfig:
    """Configuration for KNN few-shot selection."""
    
    k: int = 3  # Number of examples to select
    min_similarity: float = 0.3  # Minimum cosine similarity threshold
    use_traces: bool = True  # Use collected traces as examples
    use_static_examples: bool = True  # Use manually defined examples
    enabled: bool = True
    
    @classmethod
    def from_user_config(cls) -> "KNNFewShotConfig":
        """Load from user config."""
        try:
            from .user_config import load_config
            config = load_config()
            opt_config = config.get("optimization", {})
            
            return cls(
                k=opt_config.get("knn_k", 3),
                min_similarity=opt_config.get("knn_min_similarity", 0.3),
                use_traces=opt_config.get("use_bootstrapped_demos", True),
                enabled=opt_config.get("use_knn_fewshot", True),
            )
        except Exception:
            return cls()


# Static examples for common query types
STATIC_EXAMPLES = {
    "bugs": [
        {
            "query": "Find bugs in this code",
            "demo": """Query: Find bugs in this code

Reasoning: I'll search for common bug patterns and then verify each finding.
```python
# First, find all functions to understand the codebase structure
funcs = find_functions()
print(funcs[:20])
```
Output: Found 15 functions: parse_config, validate_input, process_data...

Reasoning: Let me search for potential null checks and error handling issues.
```python
# Search for potential None/null issues
results = ripgrep("if.*is None|if.*== None", ".")
print(results)
```
Output: src/parser.py:45: if config is None:
        src/utils.py:23: if data == None:  # Should use 'is None'

Reasoning: Found a style issue. Let me verify by reading the actual code.
```python
# Read the context around line 23
content = read_file("src/utils.py", 20, 30)
print(content)
```
Output: 
20: def process_data(data):
21:     \"\"\"Process input data.\"\"\"
22:     # BUG: Using == instead of 'is' for None comparison
23:     if data == None:
24:         return []
25:     return data.split()

Final Answer: Found 1 bug:
- [STYLE] src/utils.py:23 - Using `== None` instead of `is None` for None comparison""",
        },
    ],
    "security": [
        {
            "query": "Check for security vulnerabilities",
            "demo": """Query: Check for security vulnerabilities

Reasoning: I'll search for common security issues like SQL injection, command injection, and path traversal.
```python
# Search for potential SQL injection
sql_patterns = ripgrep("execute.*%|cursor.*format|f\".*SELECT", ".")
print(sql_patterns)
```
Output: src/db.py:34: cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

Reasoning: Found potential SQL injection. Let me verify by reading the context.
```python
# Read the database code
content = read_file("src/db.py", 30, 45)
print(content)
```
Output:
30: def get_user(user_id):
31:     conn = get_connection()
32:     cursor = conn.cursor()
33:     # VULNERABILITY: SQL injection - user_id is not sanitized
34:     cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
35:     return cursor.fetchone()

Reasoning: Confirmed SQL injection. The user_id is directly interpolated without parameterization.

Final Answer: Found 1 critical vulnerability:
- [SQL_INJECTION] src/db.py:34 - User input directly interpolated into SQL query. Use parameterized queries: `cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))`""",
        },
    ],
    "explanation": [
        {
            "query": "Explain how this function works",
            "demo": """Query: Explain how the parse_config function works

Reasoning: I'll find the function definition and read its implementation.
```python
# Find the function
results = find_functions("parse_config")
print(results)
```
Output: src/config.py:15 - parse_config (function)

Reasoning: Found it. Let me read the implementation.
```python
content = read_file("src/config.py", 15, 40)
print(content)
```
Output:
15: def parse_config(path: str) -> dict:
16:     \"\"\"Parse a YAML configuration file.\"\"\"
17:     with open(path, 'r') as f:
18:         data = yaml.safe_load(f)
19:     
20:     # Apply defaults
21:     defaults = {'timeout': 30, 'retries': 3}
22:     return {**defaults, **data}

Final Answer: The `parse_config` function:
1. Takes a file path as input
2. Opens and reads the YAML file using `yaml.safe_load` (safe parsing)
3. Defines default values for 'timeout' (30) and 'retries' (3)
4. Merges defaults with loaded config, where loaded values override defaults
5. Returns the merged dictionary""",
        },
    ],
}


class KNNFewShot:
    """Selects relevant few-shot examples using KNN similarity."""
    
    def __init__(self, config: KNNFewShotConfig | None = None):
        self.config = config or KNNFewShotConfig.from_user_config()
        self._embedder = None
        self._static_embeddings: dict[str, np.ndarray] | None = None
    
    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from .embeddings import get_embedder
            self._embedder = get_embedder()
        return self._embedder
    
    def _get_static_examples(self, query_type: str | None = None) -> list[dict]:
        """Get static examples, optionally filtered by type."""
        if query_type and query_type in STATIC_EXAMPLES:
            return STATIC_EXAMPLES[query_type]
        
        # Return all examples
        all_examples = []
        for examples in STATIC_EXAMPLES.values():
            all_examples.extend(examples)
        return all_examples
    
    def _embed_static_examples(self) -> tuple[list[dict], np.ndarray]:
        """Embed all static examples (cached)."""
        if self._static_embeddings is not None:
            return self._get_static_examples(), self._static_embeddings
        
        examples = self._get_static_examples()
        if not examples:
            return [], np.array([])
        
        queries = [ex["query"] for ex in examples]
        embeddings = np.array(self.embedder(queries))
        self._static_embeddings = embeddings
        
        return examples, embeddings
    
    def _get_trace_examples(self, query_type: str | None = None) -> list[dict]:
        """Get examples from collected traces."""
        if not self.config.use_traces:
            return []
        
        try:
            from .trace_collector import get_trace_collector
            collector = get_trace_collector()
            
            traces = collector.traces
            if query_type:
                traces = [t for t in traces if t.query_type == query_type]
            
            # Convert traces to example format
            examples = []
            for trace in traces:
                examples.append({
                    "query": trace.query,
                    "demo": trace.format_as_demo(),
                    "score": trace.grounded_score,
                })
            
            return examples
        except Exception as e:
            logger.debug("Failed to get trace examples: %s", e)
            return []
    
    def select_examples(
        self,
        query: str,
        k: int | None = None,
        query_type: str | None = None,
    ) -> list[dict]:
        """
        Select the most relevant examples for a query.
        
        Args:
            query: The query to find examples for
            k: Number of examples (default from config)
            query_type: Optional query type filter
            
        Returns:
            List of selected examples with 'query' and 'demo' keys
        """
        if not self.config.enabled:
            return []
        
        k = k or self.config.k
        
        # Collect candidate examples
        candidates = []
        
        # Add static examples
        if self.config.use_static_examples:
            for ex in self._get_static_examples(query_type):
                candidates.append({
                    "query": ex["query"],
                    "demo": ex["demo"],
                    "source": "static",
                    "score": 1.0,  # Static examples are pre-verified
                })
        
        # Add trace examples
        if self.config.use_traces:
            for ex in self._get_trace_examples(query_type):
                candidates.append({
                    "query": ex["query"],
                    "demo": ex["demo"],
                    "source": "trace",
                    "score": ex.get("score", 0.8),
                })
        
        if not candidates:
            return []
        
        # Compute similarity
        try:
            query_emb = np.array(self.embedder([query])[0])
            candidate_queries = [c["query"] for c in candidates]
            candidate_embs = np.array(self.embedder(candidate_queries))
            
            # Cosine similarity
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
            candidate_norms = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-9)
            similarities = np.dot(candidate_norms, query_norm)
            
            # Filter by minimum similarity
            valid_indices = np.where(similarities >= self.config.min_similarity)[0]
            
            if len(valid_indices) == 0:
                logger.debug("No examples above similarity threshold %.2f", self.config.min_similarity)
                return []
            
            # Sort by similarity and take top-k
            sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
            selected_indices = sorted_indices[:k]
            
            selected = []
            for idx in selected_indices:
                candidate = candidates[idx]
                selected.append({
                    "query": candidate["query"],
                    "demo": candidate["demo"],
                    "similarity": float(similarities[idx]),
                    "source": candidate["source"],
                })
            
            logger.debug(
                "Selected %d examples for query (similarities: %s)",
                len(selected),
                [f"{s['similarity']:.2f}" for s in selected]
            )
            
            return selected
            
        except Exception as e:
            logger.warning("KNN selection failed, returning first %d examples: %s", k, e)
            return candidates[:k]
    
    def format_examples_for_prompt(
        self,
        examples: list[dict],
        max_chars: int = 6000,
    ) -> str:
        """
        Format selected examples as a prompt section.
        
        Args:
            examples: List of examples from select_examples()
            max_chars: Maximum characters for all examples
            
        Returns:
            Formatted string to prepend to prompts
        """
        if not examples:
            return ""
        
        lines = ["Here are some examples of successful analyses:\n"]
        total_chars = len(lines[0])
        
        for i, ex in enumerate(examples, 1):
            header = f"=== Example {i} ===\n"
            demo = ex["demo"]
            
            # Check if we'd exceed limit
            example_chars = len(header) + len(demo) + 2
            if total_chars + example_chars > max_chars:
                break
            
            lines.append(header)
            lines.append(demo)
            lines.append("")
            total_chars += example_chars
        
        lines.append("Now analyze the given context:\n")
        
        return "\n".join(lines)


# Global instance
_knn_fewshot: KNNFewShot | None = None


def get_knn_fewshot() -> KNNFewShot:
    """Get the global KNN few-shot instance."""
    global _knn_fewshot
    if _knn_fewshot is None:
        _knn_fewshot = KNNFewShot()
    return _knn_fewshot


def clear_knn_fewshot() -> None:
    """Clear the global KNN few-shot instance."""
    global _knn_fewshot
    _knn_fewshot = None

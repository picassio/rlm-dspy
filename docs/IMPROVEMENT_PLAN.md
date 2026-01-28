# rlm-dspy Improvement Plan

Based on analysis of **llm-tldr** and **microcode** codebases.

## Executive Summary

| Feature | Source | Priority | Complexity | Status |
|---------|--------|----------|------------|--------|
| Syntax-aware Chunking | llm-tldr | HIGH | Medium | ✅ Done |
| Salsa-style Incremental Computation | llm-tldr | HIGH | Medium | ✅ Done |
| Paste Store (Large Input Handling) | microcode | HIGH | Low | ✅ Done |
| Token Usage Tracking | llm-tldr | MEDIUM | Low | ✅ Done |
| Atomic Config + Secure Writes | microcode | MEDIUM | Low | ✅ Done |
| Content-Hashed Deduplication | llm-tldr | MEDIUM | Low | ✅ Done |
| Dirty File Tracking | llm-tldr | MEDIUM | Low | ✅ Done |
| User-Friendly Error Messages | microcode | LOW | Low | ✅ Done |
| Context Injection | microcode | LOW | Low | ✅ Done |
| Hierarchical Config Resolution | microcode | LOW | Low | ✅ Done |
| Program Slicing | llm-tldr | MEDIUM | High | 📋 Planned |
| Daemon Architecture | llm-tldr | LOW | High | 📋 Planned |

---

## 1. Salsa-style Incremental Computation (HIGH PRIORITY)

### What llm-tldr Does
```python
@salsa_query
def parse_file(db: SalsaDB, path: str) -> dict:
    content = db.query(read_file, db, path)
    return parse(content)

# When file changes, only affected queries recompute
db.set_file("auth.py", new_content)
result = db.query(parse_file, db, "auth.py")  # Auto-invalidates & recomputes
```

### How It Works
- **Dependency Tracking**: Queries record which other queries they call
- **Revision Numbers**: Files have revision counters; cache entries track last-seen revision
- **Cascade Invalidation**: When a file changes, all dependent queries are invalidated
- **Minimal Re-computation**: Only affected queries re-run

### Implementation for rlm-dspy

```python
# src/rlm_dspy/core/salsa.py
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Set, Tuple

QueryKey = Tuple[Callable, Tuple[Any, ...]]

@dataclass
class CacheEntry:
    result: Any
    dependencies: Set[QueryKey] = field(default_factory=set)
    file_revisions: Dict[str, int] = field(default_factory=dict)

class IncrementalCache:
    """Salsa-style query memoization for rlm-dspy."""
    
    def __init__(self):
        self._file_revisions: Dict[str, int] = {}
        self._query_cache: Dict[QueryKey, CacheEntry] = {}
        self._reverse_deps: Dict[QueryKey, Set[QueryKey]] = {}
    
    def invalidate_file(self, path: str) -> None:
        """Invalidate all queries depending on this file."""
        self._file_revisions[path] = self._file_revisions.get(path, 0) + 1
        # Cascade to dependent queries
        ...
```

### Use Cases in rlm-dspy
- **Compiled Prompts**: Cache DSPy-optimized prompts, invalidate when training data changes
- **File Analysis**: Cache AST/chunk analysis, recompute only for changed files
- **Multi-file Queries**: "Find all usages of X" can reuse cached file parses

---

## 2. Paste Store Pattern (HIGH PRIORITY)

### What microcode Does
```python
paste_store = {}
paste_counter = 0

paste_payload = consume_paste_for_input(user_input)
if paste_payload:
    paste_counter += 1
    paste_id = f"paste_{paste_counter}"
    paste_store[paste_id] = paste_payload["text"]
    user_input = user_input.replace(paste_payload["placeholder"], f"[{paste_id}]")
```

### Why It Matters
- Prevents large code blocks from overwhelming the conversation context
- Allows LLM to reference content by ID without seeing it in every turn
- Content is injected at final task construction, not in history

### Implementation for rlm-dspy

```python
# src/rlm_dspy/core/paste_store.py
import re
from dataclasses import dataclass, field

@dataclass
class PasteStore:
    """Manages large content placeholders to prevent context overflow."""
    
    _store: dict[str, str] = field(default_factory=dict)
    _counter: int = 0
    threshold: int = 2000  # Characters
    
    def maybe_store(self, text: str) -> tuple[str, str | None]:
        """If text exceeds threshold, store and return placeholder."""
        if len(text) <= self.threshold:
            return text, None
        
        self._counter += 1
        paste_id = f"paste_{self._counter}"
        self._store[paste_id] = text
        return f"[{paste_id}: {len(text)} chars]", paste_id
    
    def inject_context(self) -> str:
        """Generate context section with all stored pastes."""
        if not self._store:
            return ""
        lines = ["## Stored Content"]
        for paste_id, content in self._store.items():
            lines.append(f"### [{paste_id}]")
            lines.append(content)
        return "\n".join(lines)
```

### Integration with RLM
```python
class RLM:
    def __init__(self, ...):
        self.paste_store = PasteStore()
    
    def __call__(self, query: str, context: str, ...) -> RLMResult:
        # Store large context sections
        processed_context, paste_id = self.paste_store.maybe_store(context)
        
        # Add paste store to final context if needed
        if self.paste_store._store:
            context = self.paste_store.inject_context() + "\n\n" + processed_context
```

---

## 3. Token Usage Tracking (MEDIUM PRIORITY)

### What llm-tldr Does
```python
@dataclass
class SessionStats:
    raw_tokens: int = 0      # What vanilla would use
    tldr_tokens: int = 0     # What was actually used
    requests: int = 0
    
    @property
    def savings_percent(self) -> float:
        if self.raw_tokens == 0:
            return 0.0
        return ((self.raw_tokens - self.tldr_tokens) / self.raw_tokens) * 100
```

### Implementation for rlm-dspy

```python
# src/rlm_dspy/core/token_stats.py
import tiktoken
from dataclasses import dataclass

_encoder = None

def count_tokens(text: str) -> int:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return len(_encoder.encode(text))

@dataclass  
class TokenStats:
    """Track token savings from chunking/compression."""
    
    raw_context_tokens: int = 0      # Original context size
    processed_tokens: int = 0        # After chunking/filtering
    llm_input_tokens: int = 0        # Actual tokens sent to LLM
    llm_output_tokens: int = 0       # Response tokens
    
    @property
    def context_savings(self) -> float:
        """Percentage of context tokens saved by chunking."""
        if self.raw_context_tokens == 0:
            return 0.0
        return ((self.raw_context_tokens - self.processed_tokens) 
                / self.raw_context_tokens) * 100
```

### Add to RLMResult
```python
@dataclass
class RLMResult:
    answer: str
    success: bool
    
    # Existing fields...
    total_tokens: int = 0
    total_cost: float = 0.0
    
    # New: Token savings tracking
    token_stats: TokenStats | None = None
```

---

## 4. Program Slicing (MEDIUM PRIORITY, HIGH COMPLEXITY)

### What llm-tldr Does
Instead of sending entire files, it uses:
- **Forward Slicing**: Lines that are affected by a variable
- **Backward Slicing**: Lines that affect a variable
- **Call Graph Traversal**: Only functions reachable from entry point

### Concept
```python
# Instead of sending entire auth.py (1000 lines)
# Send only the slice relevant to "password validation"

def get_program_slice(file: str, line: int, variable: str) -> str:
    """Extract only lines that affect/are affected by variable at line."""
    dfg = build_data_flow_graph(file)
    backward_slice = dfg.backward_slice(line, variable)
    return extract_lines(file, backward_slice)
```

### Implementation Notes
This requires integrating with llm-tldr's DFG extractor or building similar:
- Use tree-sitter for AST
- Build data flow graph
- Implement slicing algorithm

**Recommendation**: Import from llm-tldr as optional dependency rather than reimplementing.

---

## 5. Daemon Architecture (LOW PRIORITY)

### What llm-tldr Does
- Persistent `TLDRDaemon` holds indexes in memory
- Avoids re-parsing project for every query
- Uses dirty flag tracking for lazy re-indexing

### Why Lower Priority for rlm-dspy
- rlm-dspy is primarily a one-shot CLI tool
- Compiled prompts already provide caching
- Adding daemon complexity may not be worth it for typical use cases

### If Implemented
```python
# rlm-dspy daemon --start
# rlm-dspy ask "query" --daemon  # Uses running daemon
```

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
- [ ] Paste Store for large input handling
- [ ] Token usage tracking in RLMResult

### Phase 2: Incremental Computation (3-5 days)
- [ ] Salsa-style cache for file analysis
- [ ] Integration with syntax-aware chunking

### Phase 3: Advanced Features (1-2 weeks)
- [ ] Program slicing (optional llm-tldr integration)
- [ ] Daemon mode for persistent indexing

---

## Testing Strategy

```python
# tests/test_paste_store.py
def test_large_input_stored():
    store = PasteStore(threshold=100)
    text = "x" * 200
    result, paste_id = store.maybe_store(text)
    assert "[paste_1:" in result
    assert store._store["paste_1"] == text

# tests/test_token_stats.py
def test_savings_calculation():
    stats = TokenStats(raw_context_tokens=1000, processed_tokens=200)
    assert stats.context_savings == 80.0
```

---

## References

- llm-tldr Salsa implementation: `tldr/salsa.py`
- llm-tldr Token stats: `tldr/stats.py`
- microcode Paste store: `microcode/main.py:194-299`
- llm-tldr Program slicing: `tldr/semantic.py`, `tldr/dfg_extractor.py`

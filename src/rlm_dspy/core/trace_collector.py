"""
Trace Collector for REPL execution traces.

Collects successful REPL traces and uses them for few-shot bootstrapping,
following DSPy's BootstrapFewShot pattern.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class REPLTrace:
    """A successful REPL execution trace."""
    
    query: str
    query_type: str  # e.g., "bugs", "security", "review", "general"
    reasoning_steps: list[str] = field(default_factory=list)
    code_blocks: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    final_answer: str = ""
    grounded_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    trace_id: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f"))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "REPLTrace":
        """Create from dictionary."""
        return cls(**data)
    
    def format_as_demo(self) -> str:
        """Format this trace as a few-shot demonstration."""
        lines = [f"Query: {self.query}", ""]
        
        # Interleave reasoning, code, and outputs
        max_steps = max(
            len(self.reasoning_steps),
            len(self.code_blocks),
            len(self.outputs)
        )
        
        for i in range(max_steps):
            if i < len(self.reasoning_steps) and self.reasoning_steps[i]:
                lines.append(f"Reasoning: {self.reasoning_steps[i]}")
            
            if i < len(self.code_blocks) and self.code_blocks[i]:
                lines.append(f"```python\n{self.code_blocks[i]}\n```")
            
            if i < len(self.outputs) and self.outputs[i]:
                # Truncate long outputs
                output = self.outputs[i]
                if len(output) > 500:
                    output = output[:500] + "\n... (truncated)"
                lines.append(f"Output:\n{output}")
            
            lines.append("")
        
        lines.append(f"Final Answer: {self.final_answer}")
        
        return "\n".join(lines)


@dataclass
class TraceCollectorConfig:
    """Configuration for trace collection."""
    
    storage_path: Path = field(default_factory=lambda: Path.home() / ".rlm" / "traces")
    min_grounded_score: float = 0.8  # Only save high-quality traces
    max_traces: int = 1000  # Maximum traces to keep
    max_traces_per_type: int = 200  # Max traces per query type
    enabled: bool = True
    
    @classmethod
    def from_user_config(cls) -> "TraceCollectorConfig":
        """Load from user config."""
        try:
            from .user_config import load_config
            config = load_config()
            opt_config = config.get("optimization", {})
            
            return cls(
                storage_path=Path(opt_config.get("trace_storage", "~/.rlm/traces")).expanduser(),
                min_grounded_score=opt_config.get("min_grounded_score", 0.8),
                max_traces=opt_config.get("max_traces", 1000),
                enabled=opt_config.get("collect_traces", True),
            )
        except Exception:
            return cls()


class TraceCollector:
    """Collects and stores successful REPL traces for bootstrapping."""
    
    def __init__(self, config: TraceCollectorConfig | None = None):
        self.config = config or TraceCollectorConfig.from_user_config()
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        self.traces: list[REPLTrace] = []
        self._embeddings_cache: dict[str, np.ndarray] = {}
        self._load_traces()
    
    def _get_traces_file(self) -> Path:
        """Get the traces storage file path."""
        return self.config.storage_path / "traces.json"
    
    def _load_traces(self) -> None:
        """Load existing traces from storage."""
        traces_file = self._get_traces_file()
        if traces_file.exists():
            try:
                data = json.loads(traces_file.read_text())
                self.traces = [REPLTrace.from_dict(t) for t in data.get("traces", [])]
                logger.debug("Loaded %d traces from %s", len(self.traces), traces_file)
            except Exception as e:
                logger.warning("Failed to load traces: %s", e)
                self.traces = []
    
    def _save_traces(self) -> None:
        """Save traces to storage."""
        traces_file = self._get_traces_file()
        try:
            data = {
                "version": 1,
                "updated": datetime.now(UTC).isoformat(),
                "count": len(self.traces),
                "traces": [t.to_dict() for t in self.traces],
            }
            traces_file.write_text(json.dumps(data, indent=2))
            logger.debug("Saved %d traces to %s", len(self.traces), traces_file)
        except Exception as e:
            logger.warning("Failed to save traces: %s", e)
    
    def _enforce_limits(self) -> None:
        """Enforce trace limits by removing oldest traces."""
        # Enforce per-type limits
        type_counts: dict[str, list[REPLTrace]] = {}
        for trace in self.traces:
            if trace.query_type not in type_counts:
                type_counts[trace.query_type] = []
            type_counts[trace.query_type].append(trace)
        
        kept_traces = []
        for query_type, traces in type_counts.items():
            # Sort by timestamp (newest first) and grounded score
            traces.sort(key=lambda t: (t.grounded_score, t.timestamp), reverse=True)
            kept_traces.extend(traces[:self.config.max_traces_per_type])
        
        # Enforce total limit
        kept_traces.sort(key=lambda t: (t.grounded_score, t.timestamp), reverse=True)
        self.traces = kept_traces[:self.config.max_traces]
    
    def _infer_query_type(self, query: str) -> str:
        """Infer the query type from the query text."""
        query_lower = query.lower()
        
        # Check security FIRST (before bugs, since "vulnerability" shouldn't match "bug")
        if any(word in query_lower for word in ["security", "vulnerabil", "injection", "xss", "csrf", "auth"]):
            return "security"
        elif any(word in query_lower for word in ["bug", "error", "issue", "problem", "fix"]):
            return "bugs"
        elif any(word in query_lower for word in ["review", "quality", "improve", "refactor"]):
            return "review"
        elif any(word in query_lower for word in ["explain", "what", "how", "why", "describe"]):
            return "explanation"
        elif any(word in query_lower for word in ["find", "search", "where", "locate"]):
            return "search"
        else:
            return "general"
    
    def _extract_tools_used(self, code_blocks: list[str]) -> list[str]:
        """Extract tool names used in code blocks."""
        tools = set()
        tool_patterns = [
            "read_file", "ripgrep", "find_files", "find_classes", "find_functions",
            "find_methods", "find_imports", "find_calls", "index_code",
            "semantic_search", "grep_context", "run_shell_command",
            "find_references", "go_to_definition", "get_type_info", "get_symbol_hierarchy",
        ]
        
        for code in code_blocks:
            for tool in tool_patterns:
                if tool in code:
                    tools.add(tool)
        
        return sorted(tools)
    
    def record(
        self,
        query: str,
        reasoning_steps: list[str],
        code_blocks: list[str],
        outputs: list[str],
        final_answer: str,
        grounded_score: float,
        query_type: str | None = None,
    ) -> bool:
        """
        Record a successful REPL trace.
        
        Returns True if trace was saved, False if skipped (low score or disabled).
        """
        if not self.config.enabled:
            return False
        
        if grounded_score < self.config.min_grounded_score:
            logger.debug(
                "Skipping trace with low grounded score: %.2f < %.2f",
                grounded_score, self.config.min_grounded_score
            )
            return False
        
        trace = REPLTrace(
            query=query,
            query_type=query_type or self._infer_query_type(query),
            reasoning_steps=reasoning_steps,
            code_blocks=code_blocks,
            outputs=outputs,
            tools_used=self._extract_tools_used(code_blocks),
            final_answer=final_answer,
            grounded_score=grounded_score,
        )
        
        self.traces.append(trace)
        self._enforce_limits()
        self._save_traces()
        
        # Clear embeddings cache since traces changed
        self._embeddings_cache.clear()
        
        logger.info(
            "Recorded trace %s (type=%s, score=%.2f, tools=%s)",
            trace.trace_id, trace.query_type, grounded_score, trace.tools_used
        )
        
        return True
    
    def get_similar_traces(
        self,
        query: str,
        k: int = 3,
        query_type: str | None = None,
        min_score: float = 0.0,
    ) -> list[REPLTrace]:
        """
        Find traces similar to the given query for few-shot prompting.
        
        Args:
            query: The query to find similar traces for
            k: Number of traces to return
            query_type: Filter by query type (optional)
            min_score: Minimum grounded score threshold
            
        Returns:
            List of similar traces, sorted by relevance
        """
        if not self.traces:
            return []
        
        # Filter by type and score
        candidates = self.traces
        if query_type:
            candidates = [t for t in candidates if t.query_type == query_type]
        if min_score > 0:
            candidates = [t for t in candidates if t.grounded_score >= min_score]
        
        if not candidates:
            return []
        
        # Use embedding similarity
        try:
            from .embeddings import get_embedder
            embedder = get_embedder()
            
            # Get query embedding
            query_emb = np.array(embedder([query])[0])
            
            # Get trace embeddings (cached)
            cache_key = "_".join(t.trace_id for t in candidates)
            if cache_key not in self._embeddings_cache:
                trace_queries = [t.query for t in candidates]
                self._embeddings_cache[cache_key] = np.array(embedder(trace_queries))
            
            trace_embs = self._embeddings_cache[cache_key]
            
            # Cosine similarity
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
            trace_norms = trace_embs / (np.linalg.norm(trace_embs, axis=1, keepdims=True) + 1e-9)
            similarities = np.dot(trace_norms, query_norm)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            return [candidates[i] for i in top_indices]
            
        except Exception as e:
            logger.warning("Embedding similarity failed, using recency: %s", e)
            # Fallback: return most recent traces
            candidates.sort(key=lambda t: t.timestamp, reverse=True)
            return candidates[:k]
    
    def format_demos(self, traces: list[REPLTrace], max_chars: int = 8000) -> str:
        """
        Format traces as few-shot demonstrations.
        
        Args:
            traces: List of traces to format
            max_chars: Maximum total characters for demos
            
        Returns:
            Formatted demonstration string
        """
        if not traces:
            return ""
        
        demos = []
        total_chars = 0
        
        for i, trace in enumerate(traces, 1):
            demo = f"=== Example {i} (score: {trace.grounded_score:.0%}) ===\n"
            demo += trace.format_as_demo()
            demo += "\n"
            
            if total_chars + len(demo) > max_chars:
                break
            
            demos.append(demo)
            total_chars += len(demo)
        
        return "\n".join(demos)
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about collected traces."""
        if not self.traces:
            return {"total": 0, "by_type": {}, "avg_score": 0.0}
        
        by_type: dict[str, int] = {}
        scores = []
        
        for trace in self.traces:
            by_type[trace.query_type] = by_type.get(trace.query_type, 0) + 1
            scores.append(trace.grounded_score)
        
        return {
            "total": len(self.traces),
            "by_type": by_type,
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "storage_path": str(self.config.storage_path),
        }
    
    def clear(self) -> int:
        """Clear all traces. Returns number of traces deleted."""
        count = len(self.traces)
        self.traces = []
        self._embeddings_cache.clear()
        self._save_traces()
        return count
    
    def export(self, path: Path) -> int:
        """Export traces to a file. Returns number of traces exported."""
        data = {
            "version": 1,
            "exported": datetime.now(UTC).isoformat(),
            "count": len(self.traces),
            "traces": [t.to_dict() for t in self.traces],
        }
        path.write_text(json.dumps(data, indent=2))
        return len(self.traces)
    
    def import_traces(self, path: Path) -> int:
        """Import traces from a file. Returns number of traces imported."""
        data = json.loads(path.read_text())
        imported = 0
        
        for trace_data in data.get("traces", []):
            trace = REPLTrace.from_dict(trace_data)
            # Avoid duplicates by trace_id
            if not any(t.trace_id == trace.trace_id for t in self.traces):
                self.traces.append(trace)
                imported += 1
        
        self._enforce_limits()
        self._save_traces()
        
        return imported


# Global trace collector instance
_trace_collector: TraceCollector | None = None


def get_trace_collector() -> TraceCollector:
    """Get the global trace collector instance."""
    global _trace_collector
    if _trace_collector is None:
        _trace_collector = TraceCollector()
    return _trace_collector


def clear_trace_collector() -> None:
    """Clear the global trace collector instance."""
    global _trace_collector
    _trace_collector = None

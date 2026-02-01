"""Grounded Proposer for data-aware prompt improvements.

Generates tips based on failure patterns, following DSPy's MIPROv2 pattern.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """Record of a failed query for pattern analysis."""
    
    query: str
    query_type: str
    failure_reason: str
    grounded_score: float
    ungrounded_claims: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tools_missing: list[str] = field(default_factory=list)  # Tools that should have been used
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureRecord":
        return cls(**data)


@dataclass
class SuccessRecord:
    """Record of a successful query for pattern analysis."""
    
    query: str
    query_type: str
    grounded_score: float
    tools_used: list[str] = field(default_factory=list)
    key_patterns: list[str] = field(default_factory=list)  # What made it successful
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuccessRecord":
        return cls(**data)


@dataclass
class ProposerConfig:
    """Configuration for grounded proposer."""
    
    storage_path: Path = field(default_factory=lambda: Path.home() / ".rlm" / "proposer")
    max_failures: int = 200
    max_successes: int = 200
    tip_refresh_interval: int = 50  # Re-generate tips every N queries
    max_tips: int = 10
    enabled: bool = True
    
    @classmethod
    def from_user_config(cls) -> "ProposerConfig":
        """Load from user config."""
        try:
            from .user_config import load_config
            config = load_config()
            opt_config = config.get("optimization", {})
            
            return cls(
                storage_path=Path(opt_config.get("proposer_storage", "~/.rlm/proposer")).expanduser(),
                tip_refresh_interval=opt_config.get("tip_refresh_interval", 50),
                max_tips=opt_config.get("max_tips", 10),
                enabled=opt_config.get("use_grounded_tips", True),
            )
        except Exception:
            return cls()


# Default tips based on common failure patterns
DEFAULT_TIPS = [
    "Use read_file() to verify line numbers before claiming bugs exist",
    "Use ripgrep/find_functions first, then read_file to verify",
    "Quote actual code when reporting issues",
    "Search tools are CASE-SENSITIVE - use '-i' flag or check exact spelling",
]


class GroundedProposer:
    """Proposes prompt improvements grounded in actual data and failures."""
    
    def __init__(self, config: ProposerConfig | None = None):
        self.config = config or ProposerConfig.from_user_config()
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.failures: list[FailureRecord] = []
        self.successes: list[SuccessRecord] = []
        self.current_tips: list[str] = DEFAULT_TIPS.copy()
        self.queries_since_refresh: int = 0
        
        self._load_state()
    
    def _get_state_file(self) -> Path:
        return self.config.storage_path / "proposer_state.json"
    
    def _load_state(self) -> None:
        """Load proposer state from storage."""
        state_file = self._get_state_file()
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self.failures = [FailureRecord.from_dict(f) for f in data.get("failures", [])]
                self.successes = [SuccessRecord.from_dict(s) for s in data.get("successes", [])]
                self.current_tips = data.get("current_tips", DEFAULT_TIPS.copy())
                self.queries_since_refresh = data.get("queries_since_refresh", 0)
                logger.debug(
                    "Loaded proposer state: %d failures, %d successes, %d tips",
                    len(self.failures), len(self.successes), len(self.current_tips)
                )
            except Exception as e:
                logger.warning("Failed to load proposer state: %s", e)
    
    def _save_state(self) -> None:
        """Save proposer state to storage."""
        state_file = self._get_state_file()
        try:
            data = {
                "version": 1,
                "updated": datetime.now(UTC).isoformat(),
                "failures": [f.to_dict() for f in self.failures[-self.config.max_failures:]],
                "successes": [s.to_dict() for s in self.successes[-self.config.max_successes:]],
                "current_tips": self.current_tips,
                "queries_since_refresh": self.queries_since_refresh,
            }
            state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save proposer state: %s", e)
    
    def record_failure(
        self,
        query: str,
        query_type: str,
        failure_reason: str,
        grounded_score: float,
        ungrounded_claims: list[str] | None = None,
        tools_used: list[str] | None = None,
    ) -> None:
        """Record a failed query for pattern analysis."""
        if not self.config.enabled:
            return
        
        # Infer missing tools based on common patterns
        tools_missing = self._infer_missing_tools(
            failure_reason, tools_used or [], ungrounded_claims or []
        )
        
        record = FailureRecord(
            query=query,
            query_type=query_type,
            failure_reason=failure_reason,
            grounded_score=grounded_score,
            ungrounded_claims=ungrounded_claims or [],
            tools_used=tools_used or [],
            tools_missing=tools_missing,
        )
        
        self.failures.append(record)
        self.queries_since_refresh += 1
        
        # Enforce limit efficiently (remove oldest entries in place)
        if len(self.failures) > self.config.max_failures:
            excess = len(self.failures) - self.config.max_failures
            del self.failures[:excess]
        
        self._save_state()
        
        # Check if tips need refresh
        if self.queries_since_refresh >= self.config.tip_refresh_interval:
            self._maybe_refresh_tips()
    
    def record_success(
        self,
        query: str,
        query_type: str,
        grounded_score: float,
        tools_used: list[str] | None = None,
    ) -> None:
        """Record a successful query for pattern analysis."""
        if not self.config.enabled:
            return
        
        # Extract key patterns from successful execution
        key_patterns = self._extract_success_patterns(tools_used or [])
        
        record = SuccessRecord(
            query=query,
            query_type=query_type,
            grounded_score=grounded_score,
            tools_used=tools_used or [],
            key_patterns=key_patterns,
        )
        
        self.successes.append(record)
        self.queries_since_refresh += 1
        
        # Enforce limit efficiently (remove oldest entries in place)
        if len(self.successes) > self.config.max_successes:
            excess = len(self.successes) - self.config.max_successes
            del self.successes[:excess]
        
        self._save_state()
    
    def _infer_missing_tools(
        self,
        failure_reason: str,
        tools_used: list[str],
        ungrounded_claims: list[str],
    ) -> list[str]:
        """Infer which tools should have been used to prevent the failure."""
        missing = []
        
        # If claims are about specific lines but read_file wasn't used
        if any("line" in claim.lower() or ":" in claim for claim in ungrounded_claims):
            if "read_file" not in tools_used:
                missing.append("read_file")
        
        # If claims are about code existence but no search was done
        if any(word in failure_reason.lower() for word in ["not found", "doesn't exist", "missing"]):
            if not any(t in tools_used for t in ["ripgrep", "find_functions", "find_classes", "semantic_search"]):
                missing.append("ripgrep or find_functions")
        
        # If claims are about vulnerabilities but no verification
        if any(word in failure_reason.lower() for word in ["vulnerability", "security", "injection"]):
            if "read_file" not in tools_used:
                missing.append("read_file (to verify vulnerability)")
        
        return missing
    
    def _extract_success_patterns(self, tools_used: list[str]) -> list[str]:
        """Extract patterns that contributed to success."""
        patterns = []
        
        if "read_file" in tools_used:
            patterns.append("verified with read_file")
        
        if any(t in tools_used for t in ["ripgrep", "find_functions", "find_classes"]):
            patterns.append("used structural search first")
        
        if "semantic_search" in tools_used:
            patterns.append("used semantic search for discovery")
        
        if len(tools_used) >= 3:
            patterns.append("multiple verification steps")
        
        return patterns
    
    def _maybe_refresh_tips(self) -> None:
        """Refresh tips if enough queries have accumulated."""
        if self.queries_since_refresh < self.config.tip_refresh_interval:
            return
        
        if len(self.failures) < 5:  # Need enough data
            return
        
        try:
            new_tips = self.generate_tips()
            if new_tips:
                self.current_tips = new_tips
                self.queries_since_refresh = 0
                self._save_state()
                logger.info("Refreshed tips based on %d failures", len(self.failures))
        except Exception as e:
            logger.warning("Failed to refresh tips: %s", e)
    
    def generate_tips(self) -> list[str]:
        """Generate tips based on patterns in successes vs failures."""
        if not self.failures:
            return DEFAULT_TIPS.copy()
        
        # Analyze failure patterns
        failure_patterns = self._analyze_failure_patterns()
        success_patterns = self._analyze_success_patterns()
        
        try:
            import dspy
            from .user_config import load_config
            
            # Configure LM from user config
            config = load_config()
            model = config.get("model", "openrouter/google/gemini-2.0-flash-001")
            api_key = config.get("api_key")
            api_base = config.get("api_base")
            
            if not api_key:
                # Try environment
                import os
                api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            
            if not api_key:
                logger.debug("No API key for tip generation, using heuristics")
                return self._generate_heuristic_tips(failure_patterns, success_patterns)
            
            lm_kwargs = {"model": model, "api_key": api_key}
            if api_base:
                lm_kwargs["api_base"] = api_base
            
            lm = dspy.LM(**lm_kwargs)
            
            class GenerateTips(dspy.Signature):
                """Generate actionable tips from failure/success patterns."""
                failure_patterns: str = dspy.InputField(desc="Common patterns in failed queries")
                success_patterns: str = dspy.InputField(desc="Common patterns in successful queries")
                tips: list[str] = dspy.OutputField(desc="List of actionable tips (max 10)")
            
            generator = dspy.ChainOfThought(GenerateTips)
            
            with dspy.settings.context(lm=lm):
                result = generator(
                    failure_patterns=json.dumps(failure_patterns, indent=2),
                    success_patterns=json.dumps(success_patterns, indent=2),
                )
            
            # Ensure we have valid tips
            tips = result.tips if isinstance(result.tips, list) else []
            tips = [t for t in tips if isinstance(t, str) and len(t) > 10]
            
            if not tips:
                return DEFAULT_TIPS.copy()
            
            return tips[:self.config.max_tips]
            
        except Exception as e:
            logger.warning("Failed to generate tips with LLM: %s", e)
            return self._generate_heuristic_tips(failure_patterns, success_patterns)
    
    def _analyze_failure_patterns(self) -> dict[str, Any]:
        """Analyze patterns in failures."""
        if not self.failures:
            return {}
        
        # Count failure reasons
        reason_counts: dict[str, int] = {}
        missing_tools: dict[str, int] = {}
        query_types: dict[str, int] = {}
        
        for f in self.failures:
            reason = f.failure_reason or "unknown"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            query_types[f.query_type] = query_types.get(f.query_type, 0) + 1
            
            for tool in f.tools_missing:
                missing_tools[tool] = missing_tools.get(tool, 0) + 1
        
        return {
            "total_failures": len(self.failures),
            "top_reasons": sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "missing_tools": sorted(missing_tools.items(), key=lambda x: x[1], reverse=True)[:5],
            "by_query_type": query_types,
            "avg_grounded_score": sum(f.grounded_score for f in self.failures) / len(self.failures),
        }
    
    def _analyze_success_patterns(self) -> dict[str, Any]:
        """Analyze patterns in successes."""
        if not self.successes:
            return {}
        
        # Count successful patterns
        pattern_counts: dict[str, int] = {}
        tool_counts: dict[str, int] = {}
        query_types: dict[str, int] = {}
        
        for s in self.successes:
            query_types[s.query_type] = query_types.get(s.query_type, 0) + 1
            
            for pattern in s.key_patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            for tool in s.tools_used:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        return {
            "total_successes": len(self.successes),
            "top_patterns": sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "most_used_tools": sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "by_query_type": query_types,
            "avg_grounded_score": sum(s.grounded_score for s in self.successes) / len(self.successes),
        }
    
    def _generate_heuristic_tips(
        self,
        failure_patterns: dict[str, Any],
        success_patterns: dict[str, Any],
    ) -> list[str]:
        """Generate tips using heuristics when LLM is unavailable."""
        tips = []
        
        # Tips based on missing tools
        missing_tools = failure_patterns.get("missing_tools", [])
        for tool, count in missing_tools[:3]:
            if "read_file" in tool:
                tips.append("Always use read_file() to verify claims about specific code locations")
            elif "ripgrep" in tool or "find" in tool:
                tips.append("Use ripgrep or structural search before making claims about code patterns")
        
        # Tips based on successful patterns
        top_patterns = success_patterns.get("top_patterns", [])
        for pattern, count in top_patterns[:3]:
            if "verified" in pattern:
                tips.append("Verify all claims by reading the actual source code")
            elif "structural" in pattern:
                tips.append("Start with structural search (find_functions, find_classes) to get exact locations")
            elif "multiple" in pattern:
                tips.append("Use multiple verification steps before finalizing your answer")
        
        # Add defaults if not enough tips
        while len(tips) < 5:
            for default_tip in DEFAULT_TIPS:
                if default_tip not in tips:
                    tips.append(default_tip)
                    break
            else:
                break
        
        return tips[:self.config.max_tips]
    
    def get_tips(self) -> list[str]:
        """Get current tips for prompt augmentation."""
        return self.current_tips.copy()
    
    def format_tips_for_prompt(self) -> str:
        """Format tips as a prompt section."""
        if not self.current_tips:
            return ""
        
        lines = ["IMPORTANT TIPS (learned from past queries):"]
        for tip in self.current_tips:
            lines.append(f"- {tip}")
        
        return "\n".join(lines)
    
    def augment_prompt(self, base_prompt: str) -> str:
        """Add tips to a base prompt."""
        tips_section = self.format_tips_for_prompt()
        if not tips_section:
            return base_prompt
        
        return f"{base_prompt}\n\n{tips_section}"
    
    def get_stats(self) -> dict[str, Any]:
        """Get proposer statistics."""
        return {
            "total_failures": len(self.failures),
            "total_successes": len(self.successes),
            "current_tips_count": len(self.current_tips),
            "queries_since_refresh": self.queries_since_refresh,
            "refresh_interval": self.config.tip_refresh_interval,
            "failure_patterns": self._analyze_failure_patterns(),
            "success_patterns": self._analyze_success_patterns(),
            "storage_path": str(self.config.storage_path),
        }
    
    def clear(self) -> tuple[int, int]:
        """Clear all records. Returns (failures_deleted, successes_deleted)."""
        failures_count = len(self.failures)
        successes_count = len(self.successes)
        
        self.failures = []
        self.successes = []
        self.current_tips = DEFAULT_TIPS.copy()
        self.queries_since_refresh = 0
        
        self._save_state()
        
        return failures_count, successes_count
    
    def reset_tips(self) -> None:
        """Reset tips to defaults."""
        self.current_tips = DEFAULT_TIPS.copy()
        self._save_state()
    
    def set_optimized_tips(self, tips: list[str]) -> None:
        """Set tips from optimization (merges with existing, prioritizes optimized).
        
        Args:
            tips: List of optimized tips to apply
        """
        if not tips:
            return
        
        # Merge: optimized tips first, then existing non-duplicate tips
        merged = list(tips)  # Start with optimized tips
        
        for existing_tip in self.current_tips:
            # Add existing tips that aren't duplicates
            if existing_tip not in merged and len(merged) < self.config.max_tips:
                merged.append(existing_tip)
        
        self.current_tips = merged[:self.config.max_tips]
        self._save_state()
        
        logger.info("Applied %d optimized tips (total: %d)", len(tips), len(self.current_tips))
    
    def set_tips(self, tips: list[str]) -> None:
        """Replace current tips with the given list.
        
        Args:
            tips: List of tips to set
        """
        self.current_tips = tips[:self.config.max_tips] if tips else DEFAULT_TIPS.copy()
        self._save_state()


# Global proposer instance
_proposer: GroundedProposer | None = None


def get_grounded_proposer() -> GroundedProposer:
    """Get the global grounded proposer instance."""
    global _proposer
    if _proposer is None:
        _proposer = GroundedProposer()
    return _proposer

def clear_grounded_proposer() -> None:
    """Clear the global grounded proposer instance."""
    global _proposer
    _proposer = None

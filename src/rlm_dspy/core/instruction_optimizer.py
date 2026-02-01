"""
Instruction Optimizer using COPRO-style iterative refinement.

Analyzes successful and failed REPL executions to propose improved
tool instructions, following DSPy's COPRO pattern.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class InstructionOutcome:
    """Record of an instruction's outcome."""
    
    instruction_key: str  # e.g., "tool_instructions", "verification_rules"
    instruction_text: str
    success: bool
    grounded_score: float = 0.0
    failure_reason: str | None = None
    query: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstructionOutcome":
        return cls(**data)


@dataclass
class InstructionCandidate:
    """A candidate instruction with its score."""
    
    key: str
    text: str
    score: float = 0.0
    eval_count: int = 0
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstructionCandidate":
        return cls(**data)


@dataclass
class OptimizerConfig:
    """Configuration for instruction optimization."""
    
    storage_path: Path = field(default_factory=lambda: Path.home() / ".rlm" / "optimizer")
    max_history: int = 500  # Maximum outcome records to keep
    optimization_depth: int = 3  # Number of refinement iterations
    optimization_breadth: int = 5  # Candidates per iteration
    min_samples_for_optimization: int = 10  # Minimum outcomes before optimizing
    enabled: bool = True
    
    @classmethod
    def from_user_config(cls) -> "OptimizerConfig":
        """Load from user config."""
        try:
            from .user_config import load_config
            config = load_config()
            opt_config = config.get("optimization", {})
            
            return cls(
                storage_path=Path(opt_config.get("optimizer_storage", "~/.rlm/optimizer")).expanduser(),
                optimization_depth=opt_config.get("optimization_depth", 3),
                optimization_breadth=opt_config.get("optimization_breadth", 5),
                enabled=opt_config.get("auto_optimize_instructions", False),
            )
        except Exception:
            return cls()


# Default tool instructions
DEFAULT_INSTRUCTIONS = {
    "tool_instructions": """CONTEXT: You are exploring a LARGE CODEBASE (potentially thousands of files).
You CANNOT read everything - use tools strategically to find what matters.

EXPLORATION STRATEGY (follow this order):
1. GET THE BIG PICTURE FIRST:
   - `file_stats(".")` - Understand project size and structure
   - `find_files("*.py", ".")` or `find_files("*.ts", ".")` - See all relevant files
   - `semantic_search("your topic")` - Find conceptually related code across the project

2. NARROW DOWN TO RELEVANT AREAS:
   - `index_code("src/", kind="class")` - Find all classes in a directory
   - `find_functions("src/")` - Find all functions in a directory
   - `ripgrep("keyword", ".")` - Search for specific patterns across all files

3. THEN READ SPECIFIC CODE:
   - `read_file(path, start_line, end_line)` - Read only what you need
   - Don't read entire files - use line ranges from index_code results

AVAILABLE TOOLS:
- `file_stats(path)` - Get file/directory size, line counts, structure
- `find_files(pattern, path)` - Find files by name pattern (*.py, *test*, etc.)
- `semantic_search(query)` - Conceptual search across indexed project
- `index_code(path, kind, name)` - Find classes/functions/methods with EXACT line numbers
- `find_classes(path)`, `find_functions(path)`, `find_methods(path)` - Structural queries
- `find_calls(path, function_name)` - Find call sites (CASE-SENSITIVE - know exact name first!)
- `find_usages(file_path)` - Find ALL references to symbols in a file (extracts exact names from AST)
- `ripgrep(pattern, path, flags)` - Fast regex search (use flags="-i" for case-insensitive)
- `read_file(path, start_line, end_line)` - Read specific sections

CRITICAL: EXTRACT EXACT NAMES BEFORE SEARCHING
- DON'T guess symbol names from filenames (simba_optimizer.py ≠ SimbaOptimizer)
- DO use `find_usages(file)` or `index_code(file)` to get EXACT names first
- THEN search with exact names

Use `find_usages(file)` for:
- Dead code detection: Which symbols have no external usages?
- Refactoring: Where is this class/function used?
- Impact analysis: What will break if I change this?
- Understanding: How is this module connected to the rest?

SEARCH TIPS:
- `find_calls` and `ripgrep` are CASE-SENSITIVE by default
- Use `ripgrep("pattern", ".", "-i")` for case-insensitive search
- If search returns nothing, you probably have the wrong case

ANTI-PATTERNS (avoid these):
- DON'T analyze files one by one - search across the project first
- DON'T guess at file locations - use find_files or ripgrep
- DON'T claim issues exist without reading the actual code
- DON'T assume case - use -i flag or verify exact name first""",

    "verification_rules": """CRITICAL VERIFICATION RULES:
1. NEVER claim a bug/issue exists without using read_file() to see the actual code
2. ALWAYS verify line numbers by reading the file - don't guess or assume
3. If you claim "file.py:123 has bug X", you MUST have read lines 120-130 to verify
4. Check for existing protections before claiming vulnerabilities (guards, validation, etc.)
5. When reporting issues, quote the actual problematic code you found
6. After using find_usages(), VERIFY with read_file() if "INTERNAL ONLY" - check the actual usage
7. Don't trust tool summaries blindly - write Python to analyze results when needed""",

    "iteration_guidance": """ITERATION STRATEGY FOR LARGE CODEBASES:
Iteration 1-2: EXPLORE (understand the project structure)
  - file_stats(".") to see project size
  - find_files() to see what files exist
  - semantic_search() or ripgrep() to find relevant areas

Iteration 3-5: LOCATE (find specific code)
  - index_code() on relevant directories
  - find_classes(), find_functions() for structure
  - Narrow down to specific files and line ranges

Iteration 6+: VERIFY (read and confirm)
  - read_file() with specific line ranges
  - Quote actual code in your findings
  - Then call SUBMIT(answer="your answer") to finish

EARLY TERMINATION - Use SUBMIT() to end immediately:
```python
# When you have your answer, call SUBMIT to stop:
SUBMIT(answer="Your final answer here")
```

Call SUBMIT() immediately when:
- Simple calculations: SUBMIT(answer="4") for "2+2"
- Enough evidence gathered - don't repeat findings
- Query fully answered - use SUBMIT() to stop

REMEMBER: You're exploring a codebase that may have 100+ files.
Use search tools to find relevant code, don't try to read everything.""",
}


class InstructionOptimizer:
    """Optimizes tool instructions using COPRO-style iterative refinement."""
    
    def __init__(self, config: OptimizerConfig | None = None):
        self.config = config or OptimizerConfig.from_user_config()
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.history: list[InstructionOutcome] = []
        self.current_instructions: dict[str, str] = DEFAULT_INSTRUCTIONS.copy()
        self.candidates: dict[str, list[InstructionCandidate]] = {}
        
        self._load_state()
    
    def _get_state_file(self) -> Path:
        return self.config.storage_path / "optimizer_state.json"
    
    def _load_state(self) -> None:
        """Load optimizer state from storage."""
        state_file = self._get_state_file()
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self.history = [InstructionOutcome.from_dict(h) for h in data.get("history", [])]
                self.current_instructions = data.get("current_instructions", DEFAULT_INSTRUCTIONS.copy())
                self.candidates = {
                    k: [InstructionCandidate.from_dict(c) for c in v]
                    for k, v in data.get("candidates", {}).items()
                }
                logger.debug("Loaded optimizer state with %d history records", len(self.history))
            except Exception as e:
                logger.warning("Failed to load optimizer state: %s", e)
    
    def _save_state(self) -> None:
        """Save optimizer state to storage."""
        state_file = self._get_state_file()
        try:
            data = {
                "version": 1,
                "updated": datetime.now(UTC).isoformat(),
                "history": [h.to_dict() for h in self.history[-self.config.max_history:]],
                "current_instructions": self.current_instructions,
                "candidates": {
                    k: [c.to_dict() for c in v]
                    for k, v in self.candidates.items()
                },
            }
            state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save optimizer state: %s", e)
    
    def record_outcome(
        self,
        instruction_key: str,
        success: bool,
        grounded_score: float = 0.0,
        failure_reason: str | None = None,
        query: str = "",
    ) -> None:
        """Record the outcome of using an instruction."""
        if not self.config.enabled:
            return
        
        outcome = InstructionOutcome(
            instruction_key=instruction_key,
            instruction_text=self.current_instructions.get(instruction_key, ""),
            success=success,
            grounded_score=grounded_score,
            failure_reason=failure_reason,
            query=query,
        )
        
        self.history.append(outcome)
        
        # Enforce limit
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]
        
        self._save_state()
        
        logger.debug(
            "Recorded outcome for %s: success=%s, score=%.2f",
            instruction_key, success, grounded_score
        )
    
    def get_instruction(self, key: str) -> str:
        """Get current instruction by key."""
        return self.current_instructions.get(key, DEFAULT_INSTRUCTIONS.get(key, ""))
    
    def get_all_instructions(self) -> dict[str, str]:
        """Get all current instructions."""
        return self.current_instructions.copy()
    
    def _get_failure_patterns(self, key: str) -> list[dict[str, Any]]:
        """Extract failure patterns for an instruction key."""
        failures = [
            h for h in self.history
            if h.instruction_key == key and not h.success
        ]
        
        # Group by failure reason
        patterns: dict[str, list[dict]] = {}
        for f in failures:
            reason = f.failure_reason or "unknown"
            if reason not in patterns:
                patterns[reason] = []
            patterns[reason].append({
                "query": f.query,
                "score": f.grounded_score,
            })
        
        return [
            {"reason": reason, "examples": examples[:3], "count": len(examples)}
            for reason, examples in patterns.items()
        ]
    
    def _get_success_patterns(self, key: str) -> list[dict[str, Any]]:
        """Extract success patterns for an instruction key."""
        successes = [
            h for h in self.history
            if h.instruction_key == key and h.success and h.grounded_score >= 0.8
        ]
        
        return [
            {"query": s.query, "score": s.grounded_score}
            for s in successes[-10:]  # Last 10 successes
        ]
    
    def propose_improvement(self, key: str) -> str | None:
        """
        Propose an improved instruction based on failure history.
        
        Uses LLM to analyze failures and generate improved instruction.
        Returns None if not enough data or LLM unavailable.
        """
        # Check if we have enough data
        key_history = [h for h in self.history if h.instruction_key == key]
        if len(key_history) < self.config.min_samples_for_optimization:
            logger.debug(
                "Not enough samples for %s: %d < %d",
                key, len(key_history), self.config.min_samples_for_optimization
            )
            return None
        
        failures = self._get_failure_patterns(key)
        if not failures:
            logger.debug("No failures to learn from for %s", key)
            return None
        
        successes = self._get_success_patterns(key)
        
        try:
            import dspy
            
            # Define proposal signature
            class ProposeImprovement(dspy.Signature):
                """Analyze instruction failures and propose an improvement.
                
                Look at what went wrong and suggest a better instruction that
                would prevent those failures while maintaining what works.
                """
                current_instruction: str = dspy.InputField(desc="The current instruction text")
                failure_patterns: str = dspy.InputField(desc="Common failure patterns with examples")
                success_patterns: str = dspy.InputField(desc="Examples of successful uses")
                improved_instruction: str = dspy.OutputField(desc="Improved instruction that addresses failures")
            
            proposer = dspy.ChainOfThought(ProposeImprovement)
            
            result = proposer(
                current_instruction=self.current_instructions.get(key, ""),
                failure_patterns=json.dumps(failures, indent=2),
                success_patterns=json.dumps(successes, indent=2),
            )
            
            return result.improved_instruction
            
        except Exception as e:
            logger.warning("Failed to propose improvement for %s: %s", key, e)
            return None
    
    def optimize(
        self,
        key: str,
        evaluate_fn: Callable[[str], float],
        depth: int | None = None,
        breadth: int | None = None,
    ) -> tuple[str, float]:
        """
        Run COPRO-style optimization for an instruction.
        
        Args:
            key: Instruction key to optimize
            evaluate_fn: Function that takes instruction text and returns score
            depth: Number of iterations (default from config)
            breadth: Number of candidates per iteration (default from config)
            
        Returns:
            Tuple of (best_instruction, best_score)
        """
        depth = depth or self.config.optimization_depth
        breadth = breadth or self.config.optimization_breadth
        
        current = self.current_instructions.get(key, DEFAULT_INSTRUCTIONS.get(key, ""))
        best_instruction = current
        best_score = evaluate_fn(current)
        
        logger.info("Starting optimization for %s (initial score: %.2f)", key, best_score)
        
        for iteration in range(depth):
            logger.info("Optimization iteration %d/%d", iteration + 1, depth)
            
            # Generate candidates
            candidates = [current]  # Always include current
            
            for _ in range(breadth - 1):
                proposed = self.propose_improvement(key)
                if proposed and proposed not in candidates:
                    candidates.append(proposed)
            
            # Evaluate candidates
            for candidate in candidates:
                score = evaluate_fn(candidate)
                logger.debug("Candidate score: %.2f", score)
                
                if score > best_score:
                    best_score = score
                    best_instruction = candidate
                    logger.info("New best score: %.2f", best_score)
            
            # Update current for next iteration
            current = best_instruction
        
        # Save best instruction
        self.current_instructions[key] = best_instruction
        self._save_state()
        
        logger.info("Optimization complete for %s (final score: %.2f)", key, best_score)
        
        return best_instruction, best_score
    
    def reset_to_defaults(self, key: str | None = None) -> None:
        """Reset instructions to defaults."""
        if key:
            self.current_instructions[key] = DEFAULT_INSTRUCTIONS.get(key, "")
        else:
            self.current_instructions = DEFAULT_INSTRUCTIONS.copy()
        self._save_state()
    
    def get_stats(self) -> dict[str, Any]:
        """Get optimizer statistics."""
        if not self.history:
            return {"total_outcomes": 0, "by_key": {}}
        
        by_key: dict[str, dict] = {}
        for outcome in self.history:
            key = outcome.instruction_key
            if key not in by_key:
                by_key[key] = {"total": 0, "successes": 0, "avg_score": 0.0, "scores": []}
            
            by_key[key]["total"] += 1
            if outcome.success:
                by_key[key]["successes"] += 1
            by_key[key]["scores"].append(outcome.grounded_score)
        
        # Calculate averages
        for key, stats in by_key.items():
            scores = stats.pop("scores")
            stats["avg_score"] = sum(scores) / len(scores) if scores else 0.0
            stats["success_rate"] = stats["successes"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return {
            "total_outcomes": len(self.history),
            "by_key": by_key,
            "storage_path": str(self.config.storage_path),
        }
    
    def clear_history(self) -> int:
        """Clear all history. Returns number of records deleted."""
        count = len(self.history)
        self.history = []
        self._save_state()
        return count
    
    def set_instruction(self, key: str, text: str) -> None:
        """Set instruction text for a key.
        
        Args:
            key: Instruction key (e.g., "tool_instructions")
            text: New instruction text
        """
        self.current_instructions[key] = text
        self._save_state()
        logger.debug("Set instruction for key '%s' (%d chars)", key, len(text))
    
    def add_rules(self, rules: list[str]) -> None:
        """Add SIMBA-generated rules to the tool_instructions.
        
        Rules are appended as a "LEARNED RULES" section.
        
        Args:
            rules: List of rule strings to add
        """
        if not rules:
            return
        
        # Get current tool_instructions
        current = self.current_instructions.get("tool_instructions", DEFAULT_INSTRUCTIONS.get("tool_instructions", ""))
        
        # Remove any existing LEARNED RULES section
        if "\n\nLEARNED RULES:" in current:
            current = current.split("\n\nLEARNED RULES:")[0]
        
        # Add new rules section
        rules_section = "\n\nLEARNED RULES (from optimization):\n" + "\n".join(f"- {rule}" for rule in rules)
        
        self.current_instructions["tool_instructions"] = current + rules_section
        self._save_state()
        
        logger.info("Added %d rules to tool_instructions", len(rules))
    
    def get_all_instructions(self) -> dict[str, str]:
        """Get all current instructions as a dictionary.
        
        Returns:
            Dict mapping instruction key to instruction text
        """
        return self.current_instructions.copy()


# Global optimizer instance
_optimizer: InstructionOptimizer | None = None


def get_instruction_optimizer() -> InstructionOptimizer:
    """Get the global instruction optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = InstructionOptimizer()
    return _optimizer


def clear_instruction_optimizer() -> None:
    """Clear the global instruction optimizer instance."""
    global _optimizer
    _optimizer = None

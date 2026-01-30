"""Tests for instruction optimizer module."""

import pytest
from pathlib import Path

from rlm_dspy.core.instruction_optimizer import (
    InstructionOptimizer,
    InstructionOutcome,
    InstructionCandidate,
    OptimizerConfig,
    DEFAULT_INSTRUCTIONS,
    get_instruction_optimizer,
    clear_instruction_optimizer,
)


class TestInstructionOutcome:
    """Tests for InstructionOutcome dataclass."""

    def test_create_outcome(self):
        """Test creating an outcome."""
        outcome = InstructionOutcome(
            instruction_key="tool_instructions",
            instruction_text="Use tools first",
            success=True,
            grounded_score=0.95,
        )
        assert outcome.instruction_key == "tool_instructions"
        assert outcome.success is True
        assert outcome.grounded_score == 0.95

    def test_outcome_to_dict(self):
        """Test converting outcome to dict."""
        outcome = InstructionOutcome(
            instruction_key="test",
            instruction_text="text",
            success=False,
            failure_reason="Low score",
        )
        data = outcome.to_dict()
        assert data["instruction_key"] == "test"
        assert data["success"] is False
        assert data["failure_reason"] == "Low score"

    def test_outcome_from_dict(self):
        """Test creating outcome from dict."""
        data = {
            "instruction_key": "test",
            "instruction_text": "text",
            "success": True,
            "grounded_score": 0.9,
            "failure_reason": None,
            "query": "test query",
            "timestamp": "2025-01-30T00:00:00Z",
        }
        outcome = InstructionOutcome.from_dict(data)
        assert outcome.success is True
        assert outcome.grounded_score == 0.9


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = OptimizerConfig()
        assert config.max_history == 500
        assert config.optimization_depth == 3
        assert config.optimization_breadth == 5

    def test_custom_config(self, tmp_path):
        """Test custom configuration."""
        config = OptimizerConfig(
            storage_path=tmp_path / "optimizer",
            max_history=100,
            optimization_depth=5,
        )
        assert config.max_history == 100
        assert config.optimization_depth == 5


class TestInstructionOptimizer:
    """Tests for InstructionOptimizer."""

    @pytest.fixture
    def optimizer(self, tmp_path):
        """Create an optimizer with temp storage."""
        config = OptimizerConfig(
            storage_path=tmp_path / "optimizer",
            max_history=100,
            enabled=True,
        )
        return InstructionOptimizer(config)

    def test_record_outcome(self, optimizer):
        """Test recording an outcome."""
        optimizer.record_outcome(
            instruction_key="tool_instructions",
            success=True,
            grounded_score=0.95,
            query="Find bugs",
        )
        assert len(optimizer.history) == 1
        assert optimizer.history[0].success is True

    def test_record_failure(self, optimizer):
        """Test recording a failure."""
        optimizer.record_outcome(
            instruction_key="verification_rules",
            success=False,
            grounded_score=0.3,
            failure_reason="Claims not verified",
            query="Audit code",
        )
        assert len(optimizer.history) == 1
        assert optimizer.history[0].success is False
        assert optimizer.history[0].failure_reason == "Claims not verified"

    def test_get_instruction(self, optimizer):
        """Test getting instructions."""
        instruction = optimizer.get_instruction("tool_instructions")
        assert "IMPORTANT" in instruction
        assert "read_file" in instruction

    def test_get_all_instructions(self, optimizer):
        """Test getting all instructions."""
        instructions = optimizer.get_all_instructions()
        assert "tool_instructions" in instructions
        assert "verification_rules" in instructions

    def test_default_instructions_exist(self, optimizer):
        """Test that default instructions are loaded."""
        assert "tool_instructions" in DEFAULT_INSTRUCTIONS
        assert "verification_rules" in DEFAULT_INSTRUCTIONS
        assert "iteration_guidance" in DEFAULT_INSTRUCTIONS

    def test_persistence(self, tmp_path):
        """Test that history persists across instances."""
        config = OptimizerConfig(
            storage_path=tmp_path / "optimizer",
            enabled=True,
        )

        # Create and record
        opt1 = InstructionOptimizer(config)
        opt1.record_outcome("tool_instructions", True, 0.9)

        # Create new instance
        opt2 = InstructionOptimizer(config)
        assert len(opt2.history) == 1
        assert opt2.history[0].grounded_score == 0.9

    def test_enforce_history_limit(self, tmp_path):
        """Test that history limit is enforced."""
        config = OptimizerConfig(
            storage_path=tmp_path / "optimizer",
            max_history=5,
            enabled=True,
        )
        optimizer = InstructionOptimizer(config)

        # Add more than max
        for i in range(10):
            optimizer.record_outcome("test", True, 0.9, query=f"Query {i}")

        assert len(optimizer.history) <= 5

    def test_get_failure_patterns(self, optimizer):
        """Test extracting failure patterns."""
        # Add some failures
        optimizer.record_outcome("test", False, 0.3, "not verified", "Q1")
        optimizer.record_outcome("test", False, 0.2, "not verified", "Q2")
        optimizer.record_outcome("test", False, 0.4, "wrong tool", "Q3")

        patterns = optimizer._get_failure_patterns("test")
        assert len(patterns) == 2  # Two different failure reasons
        
        # Check "not verified" pattern
        verified_pattern = next((p for p in patterns if p["reason"] == "not verified"), None)
        assert verified_pattern is not None
        assert verified_pattern["count"] == 2

    def test_get_success_patterns(self, optimizer):
        """Test extracting success patterns."""
        # Add some successes
        optimizer.record_outcome("test", True, 0.95, query="Q1")
        optimizer.record_outcome("test", True, 0.9, query="Q2")

        patterns = optimizer._get_success_patterns("test")
        assert len(patterns) == 2
        assert patterns[0]["score"] >= 0.9

    def test_reset_to_defaults(self, optimizer):
        """Test resetting instructions to defaults."""
        # Modify instruction
        optimizer.current_instructions["tool_instructions"] = "Modified"
        
        # Reset
        optimizer.reset_to_defaults("tool_instructions")
        
        assert optimizer.current_instructions["tool_instructions"] == DEFAULT_INSTRUCTIONS["tool_instructions"]

    def test_reset_all_to_defaults(self, optimizer):
        """Test resetting all instructions."""
        # Modify
        optimizer.current_instructions["tool_instructions"] = "Modified 1"
        optimizer.current_instructions["verification_rules"] = "Modified 2"

        # Reset all
        optimizer.reset_to_defaults()

        assert optimizer.current_instructions == DEFAULT_INSTRUCTIONS

    def test_get_stats(self, optimizer):
        """Test getting statistics."""
        optimizer.record_outcome("tool_instructions", True, 0.9)
        optimizer.record_outcome("tool_instructions", True, 0.95)
        optimizer.record_outcome("tool_instructions", False, 0.4)
        optimizer.record_outcome("verification_rules", True, 0.85)

        stats = optimizer.get_stats()
        assert stats["total_outcomes"] == 4
        assert "tool_instructions" in stats["by_key"]
        assert stats["by_key"]["tool_instructions"]["total"] == 3
        assert stats["by_key"]["tool_instructions"]["successes"] == 2

    def test_clear_history(self, optimizer):
        """Test clearing history."""
        optimizer.record_outcome("test", True, 0.9)
        optimizer.record_outcome("test", False, 0.3)
        assert len(optimizer.history) == 2

        deleted = optimizer.clear_history()
        assert deleted == 2
        assert len(optimizer.history) == 0


class TestGlobalOptimizer:
    """Tests for global optimizer functions."""

    def test_get_instruction_optimizer(self):
        """Test getting global optimizer."""
        clear_instruction_optimizer()
        o1 = get_instruction_optimizer()
        o2 = get_instruction_optimizer()
        assert o1 is o2

    def test_clear_instruction_optimizer(self):
        """Test clearing global optimizer."""
        o1 = get_instruction_optimizer()
        clear_instruction_optimizer()
        o2 = get_instruction_optimizer()
        assert o1 is not o2

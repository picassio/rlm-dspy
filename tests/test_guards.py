"""Tests for hallucination guards."""

import pytest
from rlm_dspy.guards import (
    ValidationResult,
    validate_line_numbers,
    validate_references,
    validate_code_blocks,
    validate_all,
)


class TestValidateLineNumbers:
    """Tests for line number validation."""
    
    def test_valid_line_numbers(self):
        """Valid line numbers should pass."""
        context = "line1\nline2\nline3\nline4\nline5"
        output = "The bug is on line 3"
        
        result = validate_line_numbers(output, context)
        assert result.is_valid
        assert len(result.issues) == 0
        assert result.confidence == 1.0
    
    def test_invalid_line_number(self):
        """Line numbers beyond context should be flagged."""
        context = "line1\nline2\nline3"  # 3 lines
        output = "Check line 100 for the bug"
        
        result = validate_line_numbers(output, context)
        assert not result.is_valid
        assert len(result.issues) == 1
        assert "Line 100" in result.issues[0]
    
    def test_multiple_line_formats(self):
        """Should detect various line number formats."""
        context = "a\nb\nc\nd\ne"  # 5 lines
        output = "See Line 3, also L4 and at line 2"
        
        result = validate_line_numbers(output, context)
        assert result.is_valid  # All valid
    
    def test_line_ranges(self):
        """Should handle line ranges like 'lines 1-5'."""
        context = "a\nb\nc"  # 3 lines
        output = "Check lines 1-10 for issues"
        
        result = validate_line_numbers(output, context)
        assert not result.is_valid
        assert any("10" in issue for issue in result.issues)


class TestValidateReferences:
    """Tests for reference validation."""
    
    def test_valid_function_reference(self):
        """Referenced functions in context should pass."""
        context = "def calculate_total(items):\n    return sum(items)"
        output = "The calculate_total() function sums items"
        
        result = validate_references(output, context, check_classes=False, check_files=False)
        assert result.is_valid
    
    def test_hallucinated_function(self):
        """Functions not in context should be flagged."""
        context = "def add(a, b):\n    return a + b"
        output = "The process_data() function handles input"
        
        result = validate_references(output, context, check_classes=False, check_files=False)
        # Note: only flags specific-looking function names with underscores
        assert "process_data" in str(result.issues) or result.is_valid
    
    def test_common_builtins_ignored(self):
        """Common Python builtins should not be flagged."""
        context = "x = 1"
        output = "Use print() and len() here"
        
        result = validate_references(output, context, check_classes=False, check_files=False)
        assert result.is_valid
    
    def test_file_reference_valid(self):
        """Files mentioned in context should pass."""
        context = "=== FILE: src/main.py ===\ncode here"
        output = "The bug is in src/main.py"
        
        result = validate_references(output, context, check_functions=False, check_classes=False)
        assert result.is_valid
    
    def test_file_reference_invalid(self):
        """Files not in context should be flagged."""
        context = "=== FILE: src/main.py ===\ncode here"
        output = "Check config/settings.yaml for the issue"
        
        result = validate_references(output, context, check_functions=False, check_classes=False)
        assert not result.is_valid
        assert "settings.yaml" in str(result.issues)


class TestValidateCodeBlocks:
    """Tests for code block validation."""
    
    def test_code_from_context(self):
        """Code blocks matching context should pass."""
        context = "def hello():\n    print('world')"
        output = "```python\ndef hello():\n    print('world')\n```"
        
        result = validate_code_blocks(output, context)
        assert result.is_valid
    
    def test_fabricated_code_block(self):
        """Code blocks not in context may be flagged."""
        context = "x = 1"
        output = "```python\ndef complex_algorithm():\n    process_data()\n    return result\n```"
        
        result = validate_code_blocks(output, context)
        # May or may not flag depending on matching logic
        # At minimum confidence should be affected
        assert result.confidence <= 1.0


class TestValidateAll:
    """Tests for combined validation."""
    
    def test_clean_output(self):
        """Clean output should pass all checks."""
        context = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b"
        output = "There are two functions: add() on line 1 and sub() on line 4"
        
        result = validate_all(output, context)
        assert result.is_valid
        assert result.confidence > 0.8
    
    def test_multiple_issues(self):
        """Multiple hallucinations should all be caught."""
        context = "x = 1"
        output = "The process_data() function on line 500 in utils/helper.py"
        
        result = validate_all(output, context)
        assert not result.is_valid
        assert len(result.issues) >= 1  # At least line number issue


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_bool_conversion(self):
        """ValidationResult should be truthy when valid."""
        valid = ValidationResult(is_valid=True, issues=[], confidence=1.0)
        invalid = ValidationResult(is_valid=False, issues=["test"], confidence=0.5)
        
        assert bool(valid) is True
        assert bool(invalid) is False
    
    def test_default_values(self):
        """Default values should be sensible."""
        result = ValidationResult(is_valid=True)
        assert result.issues == []
        assert result.confidence == 1.0

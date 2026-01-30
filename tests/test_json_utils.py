"""Tests for JSON utilities module."""

import pytest

from rlm_dspy.core.json_utils import (
    repair_json,
    extract_json,
    parse_json_safe,
    parse_json_strict,
    parse_list_safe,
    parse_dict_safe,
    ensure_json_serializable,
)


class TestRepairJson:
    """Tests for repair_json function."""

    def test_valid_json_unchanged(self):
        """Test that valid JSON is unchanged."""
        text = '{"key": "value"}'
        result = repair_json(text)
        assert '"key"' in result
        assert '"value"' in result

    def test_repair_trailing_comma(self):
        """Test repairing trailing comma."""
        text = '{"key": "value",}'
        result = repair_json(text)
        # Should be parseable after repair
        import json
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_repair_single_quotes(self):
        """Test repairing single quotes."""
        text = "{'key': 'value'}"
        result = repair_json(text)
        import json
        parsed = json.loads(result)
        assert parsed["key"] == "value"


class TestExtractJson:
    """Tests for extract_json function."""

    def test_extract_from_code_fence(self):
        """Test extracting JSON from markdown code fence."""
        text = '''Here's the result:
```json
{"key": "value"}
```
Done!'''
        result = extract_json(text)
        assert result == '{"key": "value"}'

    def test_extract_from_plain_fence(self):
        """Test extracting from plain code fence."""
        text = '''```
{"key": "value"}
```'''
        result = extract_json(text)
        assert '{"key": "value"}' in result

    def test_extract_object(self):
        """Test extracting JSON object from text."""
        text = 'The result is {"key": "value"} as shown.'
        result = extract_json(text)
        assert '{"key": "value"}' in result

    def test_extract_array(self):
        """Test extracting JSON array from text."""
        text = 'Items: ["a", "b", "c"]'
        result = extract_json(text)
        assert '["a", "b", "c"]' in result

    def test_plain_json_unchanged(self):
        """Test that plain JSON is unchanged."""
        text = '{"key": "value"}'
        result = extract_json(text)
        assert result == text


class TestParseJsonSafe:
    """Tests for parse_json_safe function."""

    def test_valid_json(self):
        """Test parsing valid JSON."""
        result = parse_json_safe('{"key": "value"}')
        assert result == {"key": "value"}

    def test_empty_returns_default(self):
        """Test empty input returns default."""
        assert parse_json_safe("") is None
        assert parse_json_safe("", default={}) == {}

    def test_invalid_returns_default(self):
        """Test invalid JSON returns default."""
        result = parse_json_safe("not json at all", default="fallback")
        assert result == "fallback"

    def test_repairs_and_parses(self):
        """Test that repair is applied."""
        result = parse_json_safe("{'key': 'value'}")
        assert result == {"key": "value"}

    def test_extracts_from_markdown(self):
        """Test extraction from markdown."""
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_safe(text)
        assert result == {"key": "value"}


class TestParseJsonStrict:
    """Tests for parse_json_strict function."""

    def test_valid_json(self):
        """Test parsing valid JSON."""
        result = parse_json_strict('{"key": "value"}')
        assert result == {"key": "value"}

    def test_empty_raises(self):
        """Test empty input raises."""
        with pytest.raises(ValueError):
            parse_json_strict("")

    def test_invalid_raises(self):
        """Test invalid JSON raises."""
        with pytest.raises(ValueError):
            parse_json_strict("not json")


class TestParseListSafe:
    """Tests for parse_list_safe function."""

    def test_json_array(self):
        """Test parsing JSON array."""
        result = parse_list_safe('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_newline_separated(self):
        """Test parsing newline-separated items."""
        result = parse_list_safe("item1\nitem2\nitem3")
        assert result == ["item1", "item2", "item3"]

    def test_bullet_points(self):
        """Test parsing bullet points."""
        text = """- First item
- Second item
- Third item"""
        result = parse_list_safe(text)
        assert result == ["First item", "Second item", "Third item"]

    def test_numbered_list(self):
        """Test parsing numbered list."""
        text = """1. First item
2. Second item
3. Third item"""
        result = parse_list_safe(text)
        assert result == ["First item", "Second item", "Third item"]

    def test_empty_returns_empty(self):
        """Test empty input returns empty list."""
        assert parse_list_safe("") == []

    def test_filters_empty_items(self):
        """Test that empty items are filtered."""
        result = parse_list_safe("a\n\nb\n\nc")
        assert result == ["a", "b", "c"]


class TestParseDictSafe:
    """Tests for parse_dict_safe function."""

    def test_json_object(self):
        """Test parsing JSON object."""
        result = parse_dict_safe('{"key": "value"}')
        assert result == {"key": "value"}

    def test_key_value_format(self):
        """Test parsing key: value format."""
        text = """name: John
age: 30
city: NYC"""
        result = parse_dict_safe(text)
        assert result["name"] == "John"
        assert result["age"] == "30"
        assert result["city"] == "NYC"

    def test_empty_returns_empty(self):
        """Test empty input returns empty dict."""
        assert parse_dict_safe("") == {}


class TestEnsureJsonSerializable:
    """Tests for ensure_json_serializable function."""

    def test_primitives_unchanged(self):
        """Test primitives are unchanged."""
        assert ensure_json_serializable(None) is None
        assert ensure_json_serializable(True) is True
        assert ensure_json_serializable(42) == 42
        assert ensure_json_serializable(3.14) == 3.14
        assert ensure_json_serializable("hello") == "hello"

    def test_list_converted(self):
        """Test lists are converted recursively."""
        result = ensure_json_serializable([1, "two", {"three": 3}])
        assert result == [1, "two", {"three": 3}]

    def test_dict_converted(self):
        """Test dicts are converted recursively."""
        result = ensure_json_serializable({"a": 1, "b": [2, 3]})
        assert result == {"a": 1, "b": [2, 3]}

    def test_set_converted_to_list(self):
        """Test sets are converted to lists."""
        result = ensure_json_serializable({1, 2, 3})
        assert sorted(result) == [1, 2, 3]

    def test_tuple_converted_to_list(self):
        """Test tuples are converted to lists."""
        result = ensure_json_serializable((1, 2, 3))
        assert result == [1, 2, 3]

    def test_custom_object_to_string(self):
        """Test custom objects without __dict__ converted to string."""
        class CustomClass:
            __slots__ = ()  # No __dict__
            def __str__(self):
                return "custom"
        
        result = ensure_json_serializable(CustomClass())
        assert result == "custom"

    def test_object_with_to_dict(self):
        """Test objects with to_dict method."""
        class DictableClass:
            def to_dict(self):
                return {"key": "value"}
        
        result = ensure_json_serializable(DictableClass())
        assert result == {"key": "value"}

    def test_nested_conversion(self):
        """Test deeply nested structures."""
        data = {
            "items": [{1, 2}, (3, 4)],
            "nested": {"set": {5, 6}},
        }
        result = ensure_json_serializable(data)
        
        # Should be fully JSON serializable
        import json
        json.dumps(result)  # Should not raise

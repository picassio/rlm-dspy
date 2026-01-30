"""
Robust JSON parsing utilities.

Uses json_repair to handle malformed JSON from LLM outputs,
following DSPy's JSONAdapter pattern.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def repair_json(text: str) -> str:
    """
    Repair malformed JSON string.
    
    Handles common LLM output issues:
    - Missing quotes around keys
    - Trailing commas
    - Single quotes instead of double
    - Unescaped control characters
    - Markdown code fences
    
    Args:
        text: Potentially malformed JSON string
        
    Returns:
        Repaired JSON string
    """
    try:
        import json_repair
        return json_repair.repair_json(text)
    except ImportError:
        logger.warning("json_repair not installed, returning original text")
        return text
    except Exception as e:
        logger.debug("json_repair failed: %s", e)
        return text


def extract_json(text: str) -> str:
    """
    Extract JSON from text that may contain markdown or other content.
    
    Handles:
    - ```json ... ``` code blocks
    - ``` ... ``` code blocks
    - Raw JSON objects/arrays
    - JSON embedded in other text
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Extracted JSON string
    """
    text = text.strip()
    
    # Try to extract from markdown code fence
    # Pattern: ```json\n...\n``` or ```\n...\n```
    fence_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    fence_match = re.search(fence_pattern, text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    
    # Try to find JSON object or array
    # Look for outermost { } or [ ]
    json_patterns = [
        (r'\{.*\}', 'object'),
        (r'\[.*\]', 'array'),
    ]
    
    for pattern, _ in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
    
    return text


def parse_json_safe(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with repair and extraction.
    
    Args:
        text: Text containing JSON
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON object or default
    """
    if not text or not text.strip():
        return default
    
    # Step 1: Extract JSON from text
    extracted = extract_json(text)
    
    # Step 2: Try standard parsing first
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass
    
    # Step 3: Try with repair
    try:
        repaired = repair_json(extracted)
        return json.loads(repaired)
    except (json.JSONDecodeError, Exception) as e:
        logger.debug("JSON parsing failed after repair: %s", e)
        return default


def parse_json_strict(text: str) -> Any:
    """
    Parse JSON with repair, raising on failure.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed JSON object
        
    Raises:
        ValueError: If JSON cannot be parsed even after repair
    """
    if not text or not text.strip():
        raise ValueError("Empty JSON input")
    
    extracted = extract_json(text)
    
    # Try standard parsing
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass
    
    # Try with repair
    try:
        repaired = repair_json(extracted)
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e


def parse_list_safe(text: str, separator: str = "\n") -> list[str]:
    """
    Parse text as a list, handling various formats.
    
    Handles:
    - JSON arrays: ["item1", "item2"]
    - Newline-separated: item1\nitem2
    - Bullet points: - item1\n- item2
    - Numbered lists: 1. item1\n2. item2
    
    Args:
        text: Text containing list items
        separator: Default separator for plain text
        
    Returns:
        List of strings
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # Try JSON array first
    if text.startswith('['):
        try:
            result = parse_json_safe(text, None)
            if isinstance(result, list):
                return [str(item) for item in result]
        except Exception:
            pass
    
    # Try bullet points
    bullet_pattern = r'^[\s]*[-*•]\s*(.+)$'
    lines = text.split('\n')
    bullet_items = []
    for line in lines:
        match = re.match(bullet_pattern, line)
        if match:
            bullet_items.append(match.group(1).strip())
    if bullet_items:
        return bullet_items
    
    # Try numbered list
    numbered_pattern = r'^[\s]*\d+[.)]\s*(.+)$'
    numbered_items = []
    for line in lines:
        match = re.match(numbered_pattern, line)
        if match:
            numbered_items.append(match.group(1).strip())
    if numbered_items:
        return numbered_items
    
    # Fall back to separator split
    items = text.split(separator)
    return [item.strip() for item in items if item.strip()]


def parse_dict_safe(text: str) -> dict[str, Any]:
    """
    Parse text as a dictionary, handling various formats.
    
    Args:
        text: Text containing key-value pairs
        
    Returns:
        Dictionary
    """
    if not text or not text.strip():
        return {}
    
    # Try JSON object
    result = parse_json_safe(text, None)
    if isinstance(result, dict):
        return result
    
    # Try key: value format
    kv_pattern = r'^[\s]*([^:]+):\s*(.+)$'
    result = {}
    for line in text.split('\n'):
        match = re.match(kv_pattern, line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            result[key] = value
    
    return result


def ensure_json_serializable(obj: Any) -> Any:
    """
    Convert an object to be JSON serializable.
    
    Handles:
    - Sets -> lists
    - Custom objects -> str
    - Nested structures
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    
    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    
    if isinstance(obj, dict):
        return {str(k): ensure_json_serializable(v) for k, v in obj.items()}
    
    if isinstance(obj, set):
        return [ensure_json_serializable(item) for item in obj]
    
    # Try common serialization methods
    if hasattr(obj, 'to_dict'):
        return ensure_json_serializable(obj.to_dict())
    
    if hasattr(obj, '__dict__'):
        return ensure_json_serializable(obj.__dict__)
    
    # Fall back to string
    return str(obj)

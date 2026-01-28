"""Shared tree-sitter utilities.

Provides thread-safe parser management for tree-sitter across modules.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Parser

logger = logging.getLogger(__name__)

# Check if tree-sitter is available
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Language module mappings
LANGUAGE_MODULES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "ruby": "tree_sitter_ruby",
    "c_sharp": "tree_sitter_c_sharp",
}

# File extension to language mapping
EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".rb": "ruby",
    ".cs": "c_sharp",
}

# Thread-safe parser cache
_PARSERS: dict[str, "Parser"] = {}
_PARSE_LOCKS: dict[str, threading.Lock] = {}
_PARSERS_LOCK = threading.Lock()


def get_parser(language: str) -> tuple["Parser | None", threading.Lock | None]:
    """Get or create a thread-safe tree-sitter parser for a language.

    Args:
        language: Programming language name (e.g., "python", "javascript")

    Returns:
        Tuple of (parser, parse_lock).
        - Both None if language not supported or tree-sitter unavailable.
        - The parse_lock MUST be held during parser.parse() calls.

    Example:
        parser, lock = get_parser("python")
        if parser and lock:
            with lock:
                tree = parser.parse(code_bytes)
    """
    if not TREE_SITTER_AVAILABLE:
        return None, None

    # Fast path: check cache without lock
    if language in _PARSERS:
        return _PARSERS[language], _PARSE_LOCKS[language]

    module_name = LANGUAGE_MODULES.get(language)
    if not module_name:
        return None, None

    # Slow path: acquire lock and create parser
    with _PARSERS_LOCK:
        # Double-check after acquiring lock
        if language in _PARSERS:
            return _PARSERS[language], _PARSE_LOCKS[language]

        try:
            import importlib
            lang_module = importlib.import_module(module_name)

            # Handle special cases for language initialization
            if language == "typescript":
                lang = Language(lang_module.language_typescript())
            elif hasattr(lang_module, f"language_{language}"):
                lang = Language(getattr(lang_module, f"language_{language}")())
            elif hasattr(lang_module, "language"):
                lang = Language(lang_module.language())
            else:
                logger.debug("No language() function found in %s", module_name)
                return None, None

            parser = Parser(lang)
            parse_lock = threading.Lock()
            _PARSERS[language] = parser
            _PARSE_LOCKS[language] = parse_lock
            return parser, parse_lock

        except ImportError as e:
            logger.debug("tree-sitter module not installed: %s", e)
            return None, None
        except Exception as e:
            logger.debug("Failed to initialize tree-sitter for %s: %s", language, e)
            return None, None


def get_parser_simple(language: str) -> "Parser | None":
    """Get a parser without thread-safety (for single-threaded use).

    This is a convenience wrapper when you don't need the lock.
    
    Args:
        language: Programming language name
        
    Returns:
        Parser or None if unavailable
    """
    parser, _ = get_parser(language)
    return parser


def language_from_extension(extension: str) -> str | None:
    """Get language name from file extension.
    
    Args:
        extension: File extension including dot (e.g., ".py")
        
    Returns:
        Language name or None
    """
    return EXTENSION_MAP.get(extension.lower())


def is_available() -> bool:
    """Check if tree-sitter is available."""
    return TREE_SITTER_AVAILABLE

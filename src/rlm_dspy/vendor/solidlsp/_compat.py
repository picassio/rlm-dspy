"""
Compatibility layer replacing sensai/serena dependencies.

This module provides replacements for utilities from the sensai and serena
packages that solidlsp depends on, allowing solidlsp to be vendored without
those dependencies.
"""
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Self

import pathspec

__all__ = [
    # sensai.util.pickle replacements
    "getstate",
    "load_pickle",
    "dump_pickle",
    # sensai.util.string replacements
    "ToStringMixin",
    # sensai.util.logging replacements
    "LogTime",
    # serena.text_utils replacements
    "LineType",
    "TextLine",
    "MatchedConsecutiveLines",
    # serena.util.file_system replacements
    "match_path",
]


# =============================================================================
# sensai.util.pickle replacements
# =============================================================================


def getstate(cls: type, obj: Any, transient_properties: list[str] | None = None) -> dict:
    """
    Get picklable state, excluding transient properties.
    
    Args:
        cls: The class of the object (for type checking)
        obj: The object to get state from
        transient_properties: List of property names to exclude from state
        
    Returns:
        Dictionary of picklable state
    """
    state = obj.__dict__.copy()
    for prop in transient_properties or []:
        state.pop(prop, None)
    return state


def load_pickle(path: str | Path) -> Any:
    """
    Load pickled object from file.
    
    Args:
        path: Path to the pickle file
        
    Returns:
        The unpickled object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj: Any, path: str | Path) -> None:
    """
    Dump object to pickle file.
    
    Args:
        obj: Object to pickle
        path: Path to save the pickle file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# =============================================================================
# sensai.util.string replacements
# =============================================================================


class ToStringMixin:
    """
    Mixin class providing a pretty __str__ representation.
    
    Generates a string like: ClassName(attr1=value1, attr2=value2)
    Excludes private attributes (those starting with _).
    """
    
    def __str__(self) -> str:
        attrs = ", ".join(
            f"{k}={v!r}" 
            for k, v in self.__dict__.items() 
            if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({attrs})"
    
    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# sensai.util.logging replacements
# =============================================================================


class LogTime:
    """
    Context manager for timing operations and logging the duration.
    
    Usage:
        with LogTime("Processing data", logger=my_logger):
            do_something()
        # Logs: "Processing data: 1.23s"
    """
    
    def __init__(self, description: str, logger: logging.Logger | None = None):
        """
        Initialize the timer.
        
        Args:
            description: Description to include in the log message
            logger: Logger to use (defaults to module logger)
        """
        self.description = description
        self.logger = logger or logging.getLogger(__name__)
        self.start: float = 0
        self.elapsed: float = 0
        
    def __enter__(self) -> "LogTime":
        self.start = time.time()
        return self
        
    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.time() - self.start
        self.logger.info(f"{self.description}: {self.elapsed:.2f}s")


# =============================================================================
# serena.text_utils replacements
# =============================================================================


class LineType(Enum):
    """Type of line relative to the matched line."""
    BEFORE_MATCH = "before"
    MATCH = "match"
    AFTER_MATCH = "after"


@dataclass
class TextLine:
    """
    A single line of text with metadata.
    
    Attributes:
        line_number: 1-indexed line number in the source file
        content: The text content of the line
        match_type: Whether this line is before, at, or after the match
    """
    line_number: int
    content: str
    match_type: LineType = LineType.MATCH
    
    def format_line(self, include_line_numbers: bool = True) -> str:
        """
        Format the line for display.
        
        Args:
            include_line_numbers: Whether to include line numbers
            
        Returns:
            Formatted line string
        """
        if include_line_numbers:
            return f"{self.line_number:4d} | {self.content}"
        return self.content


@dataclass
class MatchedConsecutiveLines:
    """
    Collection of consecutive lines from a file, centered around a match.
    
    This class represents a snippet of code with context lines before and
    after the matched line(s).
    
    Attributes:
        lines: All lines in the snippet
        source_file_path: Path to the source file (optional)
        lines_before_matched: Lines before the match
        matched_lines: The matched line(s)
        lines_after_matched: Lines after the match
    """
    lines: list[TextLine]
    source_file_path: str | None = None
    lines_before_matched: list[TextLine] = field(default_factory=list)
    matched_lines: list[TextLine] = field(default_factory=list)
    lines_after_matched: list[TextLine] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Categorize lines by their match type."""
        # Clear the default lists
        self.lines_before_matched = []
        self.matched_lines = []
        self.lines_after_matched = []
        
        for line in self.lines:
            if line.match_type == LineType.BEFORE_MATCH:
                self.lines_before_matched.append(line)
            elif line.match_type == LineType.MATCH:
                self.matched_lines.append(line)
            elif line.match_type == LineType.AFTER_MATCH:
                self.lines_after_matched.append(line)
        
        if not self.matched_lines:
            raise ValueError("At least one matched line is required")
    
    @property
    def start_line(self) -> int:
        """Get the first line number in the snippet."""
        return self.lines[0].line_number
    
    @property
    def end_line(self) -> int:
        """Get the last line number in the snippet."""
        return self.lines[-1].line_number
    
    @property
    def num_matched_lines(self) -> int:
        """Get the number of matched lines."""
        return len(self.matched_lines)
    
    def to_display_string(self, include_line_numbers: bool = True) -> str:
        """
        Format all lines for display.
        
        Args:
            include_line_numbers: Whether to include line numbers
            
        Returns:
            Formatted multi-line string
        """
        return "\n".join(
            line.format_line(include_line_numbers) 
            for line in self.lines
        )
    
    @classmethod
    def from_file_contents(
        cls, 
        file_contents: str, 
        line: int, 
        context_lines_before: int = 0, 
        context_lines_after: int = 0,
        source_file_path: str | None = None
    ) -> Self:
        """
        Create a MatchedConsecutiveLines from file contents.
        
        Args:
            file_contents: The full file contents as a string
            line: The 0-indexed line number to match
            context_lines_before: Number of context lines before the match
            context_lines_after: Number of context lines after the match
            source_file_path: Path to the source file (optional)
            
        Returns:
            A new MatchedConsecutiveLines instance
        """
        all_lines = file_contents.split("\n")
        start = max(0, line - context_lines_before)
        end = min(len(all_lines) - 1, line + context_lines_after)
        
        text_lines: list[TextLine] = []
        for i in range(start, end + 1):
            if i < line:
                match_type = LineType.BEFORE_MATCH
            elif i == line:
                match_type = LineType.MATCH
            else:
                match_type = LineType.AFTER_MATCH
            
            # Line numbers are 1-indexed for display
            text_lines.append(TextLine(
                line_number=i + 1,
                content=all_lines[i] if i < len(all_lines) else "",
                match_type=match_type
            ))
        
        return cls(lines=text_lines, source_file_path=source_file_path)


# =============================================================================
# serena.util.file_system replacements
# =============================================================================


def match_path(
    relative_path: str, 
    path_spec: pathspec.PathSpec, 
    root_path: str = ""
) -> bool:
    """
    Match a relative path against a pathspec.
    
    This function handles path normalization to ensure consistent matching
    across platforms.
    
    Args:
        relative_path: The relative path to match
        path_spec: The pathspec to match against
        root_path: The root path (unused, for API compatibility)
        
    Returns:
        True if the path matches the spec, False otherwise
    """
    # Normalize path separators
    normalized = str(relative_path).replace(os.path.sep, "/")
    
    # Pathspec expects paths relative to root, prefix with /
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    
    return path_spec.match_file(normalized)

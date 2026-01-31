"""Response types for Anthropic LM compatibility with DSPy.

These types match the OpenAI/LiteLLM format expected by DSPy's base LM.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __iter__(self):
        """Allow dict() conversion."""
        yield "prompt_tokens", self.prompt_tokens
        yield "completion_tokens", self.completion_tokens
        yield "total_tokens", self.total_tokens


@dataclass
class ToolCallFunction:
    """Tool call function details."""
    name: str
    arguments: str


@dataclass
class ToolCall:
    """A tool call from the model."""
    id: str
    type: str = "function"
    function: ToolCallFunction = None


@dataclass
class Message:
    """A message in a response choice."""
    content: str | None = None
    role: str = "assistant"
    tool_calls: list[ToolCall] | None = None


@dataclass
class Choice:
    """A choice in a completion response."""
    index: int = 0
    message: Message = None
    finish_reason: str = "stop"
    text: str = ""  # For text completion compatibility


@dataclass
class CompletionResponse:
    """Response format matching OpenAI/LiteLLM expected by DSPy."""
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[Choice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


# Claude Code version for stealth mode (from pi-mono)
CLAUDE_CODE_VERSION = "2.1.2"

# Claude Code tools for name conversion (from pi-mono)
CLAUDE_CODE_TOOLS = [
    "Read", "Write", "Edit", "Bash", "Grep", "Glob",
    "AskUserQuestion", "EnterPlanMode", "ExitPlanMode", "KillShell",
    "NotebookEdit", "Skill", "Task", "TaskOutput", "TodoWrite",
    "WebFetch", "WebSearch",
]
CC_TOOL_LOOKUP = {t.lower(): t for t in CLAUDE_CODE_TOOLS}


def to_claude_code_name(name: str) -> str:
    """Convert tool name to Claude Code canonical casing."""
    return CC_TOOL_LOOKUP.get(name.lower(), name)

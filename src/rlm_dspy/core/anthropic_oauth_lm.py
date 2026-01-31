"""Anthropic LM with OAuth token support.

This module provides a custom DSPy LM that supports:
1. Regular API keys (sk-ant-api...)
2. OAuth tokens (sk-ant-oat...) from Claude Pro/Max subscriptions

The OAuth flow mimics Claude Code's authentication to use the same
API endpoint and features.

IMPORTANT: LiteLLM doesn't properly support OAuth tokens for Anthropic.
For OAuth tokens, we use the Anthropic SDK directly with auth_token.

Based on pi-mono's implementation:
https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/anthropic.ts
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

import anthropic
import dspy
from dspy.clients.base_lm import BaseLM

from .anthropic_types import (
    CLAUDE_CODE_VERSION,
    CC_TOOL_LOOKUP,
    Choice,
    CompletionResponse,
    Message,
    ToolCall,
    ToolCallFunction,
    Usage,
    to_claude_code_name,
)

logger = logging.getLogger(__name__)


def is_oauth_token(api_key: str) -> bool:
    """Check if the API key is an OAuth token."""
    return "sk-ant-oat" in api_key


def get_anthropic_api_key() -> str | None:
    """Get the best available Anthropic API key/token.
    
    Priority:
    1. ANTHROPIC_OAUTH_TOKEN env var
    2. OAuth credentials from ~/.rlm/oauth/credentials.json
    3. ANTHROPIC_API_KEY env var
    
    Returns:
        API key/token or None
    """
    # Try OAuth token from env first
    token = os.environ.get("ANTHROPIC_OAUTH_TOKEN")
    if token:
        return token
    
    # Try OAuth credentials from storage
    try:
        from .oauth import get_anthropic_token
        token = get_anthropic_token()
        if token:
            return token
    except ImportError:
        pass
    
    # Fall back to regular API key
    return os.environ.get("ANTHROPIC_API_KEY")


class AnthropicOAuthLM(BaseLM):
    """Custom DSPy LM that uses Anthropic SDK with OAuth token support.
    
    This bypasses LiteLLM for Anthropic models because LiteLLM has a bug
    where it sends the OAuth token as x-api-key which doesn't work.
    
    The Anthropic SDK properly supports OAuth via auth_token parameter.
    
    Implements the same OAuth flow as pi-mono/Claude Code.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        auth_token: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        num_retries: int = 3,
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):
        """Create an Anthropic OAuth LM.
        
        Args:
            model: Model ID (e.g., "claude-sonnet-4-20250514")
            auth_token: OAuth token from Claude Pro/Max
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            cache: Whether to cache responses
            num_retries: Number of retries on transient errors
            model_type: Type of model ("chat" or "text")
            **kwargs: Additional arguments
        """
        # Strip "anthropic/" prefix if present
        self.model = model.split("/")[-1] if "/" in model else model
        self.model_type = model_type
        self.cache = cache
        self.num_retries = num_retries
        self.history = []
        self.callbacks = []
        self.provider = None  # For DSPy compatibility
        self.finetuning_model = None
        self.launch_kwargs = {}
        self.train_kwargs = {}
        
        # Get model's max_tokens from registry if not specified
        # We use streaming which avoids Anthropic's 10-minute timeout limit
        if max_tokens is None:
            from .models import find_model
            model_id = f"anthropic/{self.model}"
            model_info = find_model(model_id)
            if model_info:
                max_tokens = model_info.max_tokens
            else:
                max_tokens = 8192  # Default fallback
        
        # Store kwargs for DSPy compatibility
        self.kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Add any extra kwargs, removing None values
        for k, v in kwargs.items():
            if v is not None:
                self.kwargs[k] = v
        
        # Get auth token
        if auth_token is None:
            auth_token = get_anthropic_api_key()
        if not auth_token:
            raise ValueError(
                "No Anthropic OAuth token found. "
                "Run 'rlm-dspy auth login anthropic' or set ANTHROPIC_API_KEY"
            )
        
        self._is_oauth = is_oauth_token(auth_token)
        self._auth_token = auth_token
        
        # Create Anthropic client
        if self._is_oauth:
            # OAuth token - use auth_token with Claude Code headers
            beta_features = [
                "fine-grained-tool-streaming-2025-05-14",
                "interleaved-thinking-2025-05-14",
            ]
            default_headers = {
                "accept": "application/json",
                "anthropic-dangerous-direct-browser-access": "true",
                "anthropic-beta": f"claude-code-20250219,oauth-2025-04-20,{','.join(beta_features)}",
                "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
                "x-app": "cli",
            }
            
            self._client = anthropic.Anthropic(
                api_key=None,  # Must be None for OAuth
                auth_token=auth_token,
                max_retries=num_retries,
                default_headers=default_headers,
            )
            logger.info(f"Created AnthropicOAuthLM for model {self.model} with OAuth token")
        else:
            # Regular API key
            self._client = anthropic.Anthropic(
                api_key=auth_token,
                max_retries=num_retries,
            )
            logger.info(f"Created AnthropicOAuthLM for model {self.model} with API key")
    
    def __getstate__(self) -> dict:
        """Get state for pickling - exclude non-picklable client."""
        state = self.__dict__.copy()
        state.pop('_client', None)
        return state
    
    def __setstate__(self, state: dict) -> None:
        """Restore state from pickle - recreate client."""
        self.__dict__.update(state)
        # Recreate the client based on OAuth status
        if self._is_oauth:
            beta_features = [
                "fine-grained-tool-streaming-2025-05-14",
                "interleaved-thinking-2025-05-14",
            ]
            default_headers = {
                "accept": "application/json",
                "anthropic-dangerous-direct-browser-access": "true",
                "anthropic-beta": f"claude-code-20250219,oauth-2025-04-20,{','.join(beta_features)}",
                "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
                "x-app": "cli",
            }
            self._client = anthropic.Anthropic(
                api_key=None,
                auth_token=self._auth_token,
                max_retries=self.num_retries,
                default_headers=default_headers,
            )
        else:
            self._client = anthropic.Anthropic(
                api_key=self._auth_token,
                max_retries=self.num_retries,
            )

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Forward pass to generate completions.
        
        Args:
            prompt: Optional prompt string
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation kwargs
            
        Returns:
            List of response dicts with completions
        """
        # Merge kwargs
        generation_kwargs = {**self.kwargs, **kwargs}
        
        # Handle prompt vs messages
        if messages is None and prompt:
            messages = [{"role": "user", "content": prompt}]
        elif messages is None:
            raise ValueError("Either prompt or messages must be provided")
        
        # Convert messages to Anthropic format, extracting system message
        anthropic_messages, system = self._convert_messages(messages)
        
        # Handle system in kwargs (override extracted system)
        if "system" in generation_kwargs:
            system = generation_kwargs.pop("system")
        
        # Get generation parameters
        temperature = generation_kwargs.pop("temperature", None)
        max_tokens = generation_kwargs.pop("max_tokens", 4096)
        
        # Handle tools if provided
        tools = generation_kwargs.pop("tools", None)
        tool_choice = generation_kwargs.pop("tool_choice", None)
        
        # Remove unsupported kwargs
        for key in ["response_format", "n", "stop", "logprobs", "top_logprobs",
                    "presence_penalty", "frequency_penalty", "logit_bias", "user"]:
            generation_kwargs.pop(key, None)
        
        # Build API call parameters
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        
        # Add system prompt - for OAuth, must include Claude Code identity
        if self._is_oauth:
            system_blocks = [
                {
                    "type": "text",
                    "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            if system:
                system_blocks.append({
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                })
            create_kwargs["system"] = system_blocks
        elif system:
            create_kwargs["system"] = system
        
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        
        # Convert and add tools if provided
        if tools:
            create_kwargs["tools"] = self._convert_tools(tools)
            if tool_choice:
                if isinstance(tool_choice, str):
                    create_kwargs["tool_choice"] = {"type": tool_choice}
                else:
                    create_kwargs["tool_choice"] = tool_choice
        
        # Make API call with streaming to avoid 10-minute timeout for large max_tokens
        # Anthropic requires streaming for operations that may take longer than 10 minutes
        try:
            response = self._stream_response(create_kwargs)
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        
        # Convert response to DSPy format
        return self._convert_response(response)
    
    def _stream_response(self, create_kwargs: dict[str, Any]) -> anthropic.types.Message:
        """Stream response and accumulate into a Message object.
        
        This avoids Anthropic's 10-minute timeout for non-streaming requests.
        """
        # Collect streamed content
        text_parts: list[str] = []
        tool_uses: list[dict[str, Any]] = []
        current_tool: dict[str, Any] | None = None
        input_tokens = 0
        output_tokens = 0
        stop_reason = None
        message_id = ""
        model = ""
        
        with self._client.messages.stream(**create_kwargs) as stream:
            for event in stream:
                if event.type == "message_start":
                    message_id = event.message.id
                    model = event.message.model
                    input_tokens = event.message.usage.input_tokens
                elif event.type == "content_block_start":
                    if event.content_block.type == "text":
                        text_parts.append("")
                    elif event.content_block.type == "tool_use":
                        current_tool = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "input": "",
                        }
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        if text_parts:
                            text_parts[-1] += event.delta.text
                    elif event.delta.type == "input_json_delta":
                        if current_tool:
                            current_tool["input"] += event.delta.partial_json
                elif event.type == "content_block_stop":
                    if current_tool:
                        # Parse the accumulated JSON input
                        try:
                            import json
                            current_tool["input"] = json.loads(current_tool["input"]) if current_tool["input"] else {}
                        except json.JSONDecodeError:
                            current_tool["input"] = {}
                        tool_uses.append(current_tool)
                        current_tool = None
                elif event.type == "message_delta":
                    stop_reason = event.delta.stop_reason
                    output_tokens = event.usage.output_tokens
        
        # Build content blocks
        content_blocks = []
        for text in text_parts:
            if text:
                content_blocks.append(anthropic.types.TextBlock(type="text", text=text))
        for tool in tool_uses:
            content_blocks.append(anthropic.types.ToolUseBlock(
                type="tool_use",
                id=tool["id"],
                name=tool["name"],
                input=tool["input"],
            ))
        
        # Construct Message object
        return anthropic.types.Message(
            id=message_id,
            type="message",
            role="assistant",
            content=content_blocks,
            model=model,
            stop_reason=stop_reason,
            usage=anthropic.types.Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

    def _convert_messages(
        self, 
        messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Convert DSPy/OpenAI format messages to Anthropic format.
        
        Returns:
            Tuple of (anthropic_messages, system_prompt)
        """
        result = []
        system = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle system messages - extract as separate system prompt
            if role == "system":
                if system:
                    system = f"{system}\n\n{content}"
                else:
                    system = content
                continue
            
            # Map roles
            anthropic_role = "assistant" if role == "assistant" else "user"
            
            # Handle content
            if isinstance(content, str):
                if content.strip():
                    result.append({"role": anthropic_role, "content": content})
            elif isinstance(content, list):
                # Handle content blocks (text, images, tool results)
                blocks = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            blocks.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # Convert OpenAI image format to Anthropic
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                # Parse base64 data URL
                                parts = url.split(",", 1)
                                if len(parts) == 2:
                                    media_type = parts[0].split(":")[1].split(";")[0]
                                    data = parts[1]
                                    blocks.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": data,
                                        }
                                    })
                        elif item.get("type") == "tool_result":
                            blocks.append({
                                "type": "tool_result",
                                "tool_use_id": item.get("tool_use_id"),
                                "content": item.get("content", ""),
                            })
                    elif isinstance(item, str):
                        blocks.append({"type": "text", "text": item})
                
                if blocks:
                    result.append({"role": anthropic_role, "content": blocks})
        
        # Handle tool calls in assistant messages
        for i, msg in enumerate(result):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, list):
                    new_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_use":
                            # Already in correct format
                            name = item.get("name", "")
                            if self._is_oauth:
                                item["name"] = to_claude_code_name(name)
                            new_content.append(item)
                        else:
                            new_content.append(item)
                    msg["content"] = new_content
        
        return result, system

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI/DSPy tool format to Anthropic format."""
        result = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "")
                if self._is_oauth:
                    name = to_claude_code_name(name)
                result.append({
                    "name": name,
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                # Direct tool definition
                name = tool.get("name", "")
                if self._is_oauth:
                    name = to_claude_code_name(name)
                result.append({
                    "name": name,
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", tool.get("input_schema", {"type": "object", "properties": {}})),
                })
        return result

    def _convert_response(self, response: anthropic.types.Message) -> CompletionResponse:
        """Convert Anthropic response to DSPy-compatible format."""
        import time
        
        # Extract text content and tool calls
        text_parts = []
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    type="function",
                    function=ToolCallFunction(
                        name=block.name,
                        arguments=json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                    )
                ))
        
        text = "\n".join(text_parts)
        
        # Build message
        message = Message(
            content=text,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
        )
        
        # Build choice
        choice = Choice(
            index=0,
            message=message,
            finish_reason=self._map_stop_reason(response.stop_reason),
            text=text,  # For text completion compatibility
        )
        
        # Build usage
        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        
        # Build response
        return CompletionResponse(
            id=response.id,
            object="chat.completion",
            created=int(time.time()),
            model=response.model,
            choices=[choice],
            usage=usage,
        )
    
    def _map_stop_reason(self, reason: str | None) -> str:
        """Map Anthropic stop reason to OpenAI format."""
        if reason is None:
            return "stop"
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
        }
        return mapping.get(reason, "stop")

    def __repr__(self) -> str:
        token_type = "OAuth" if self._is_oauth else "API"
        return f"AnthropicOAuthLM(model={self.model}, auth={token_type})"
    
    def copy(self, **kwargs):
        """Create a copy of this LM with optional overrides."""
        new_kwargs = {
            "model": self.model,
            "auth_token": self._auth_token,
            "temperature": self.kwargs.get("temperature"),
            "max_tokens": self.kwargs.get("max_tokens"),
            "cache": self.cache,
            "num_retries": self.num_retries,
            "model_type": self.model_type,
        }
        new_kwargs.update(kwargs)
        return AnthropicOAuthLM(**new_kwargs)


def create_anthropic_lm(
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
    **kwargs,
) -> BaseLM:
    """Create a DSPy LM for Anthropic with OAuth support.
    
    Args:
        model: Model ID (e.g., "claude-sonnet-4-20250514")
        api_key: API key or OAuth token. If None, tries to get from:
                 1. ANTHROPIC_OAUTH_TOKEN env var
                 2. ANTHROPIC_API_KEY env var
        **kwargs: Additional arguments for dspy.LM
        
    Returns:
        Configured DSPy LM (AnthropicOAuthLM for OAuth, dspy.LM for API keys)
    """
    # Resolve API key
    if api_key is None:
        # Try OAuth token first
        api_key = os.environ.get("ANTHROPIC_OAUTH_TOKEN")
        
        if not api_key:
            # Fall back to regular API key
            api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable."
        )
    
    # Use custom LM for both OAuth and API keys (more consistent)
    return AnthropicOAuthLM(model, auth_token=api_key, **kwargs)


def create_lm_with_oauth_fallback(
    model: str, 
    api_key: str | None = None, 
    **kwargs
) -> BaseLM:
    """Create a DSPy LM with automatic OAuth fallback for Anthropic.
    
    For Anthropic models, this will use the custom AnthropicOAuthLM.
    For other providers, creates a standard dspy.LM.
    
    Args:
        model: Model string (e.g., "anthropic/claude-3-opus", "openrouter/...")
        api_key: Optional API key
        **kwargs: Additional LM arguments
        
    Returns:
        Configured DSPy LM
    """
    # Check if Anthropic model
    if model.startswith("anthropic/"):
        # Extract model ID
        model_id = model.split("/", 1)[1]
        return create_anthropic_lm(model_id, api_key=api_key, **kwargs)
    
    # Standard LM creation for other providers
    return dspy.LM(model, api_key=api_key, **kwargs)

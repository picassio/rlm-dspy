"""OpenCode LM for DSPy.

OpenCode provides an OpenAI-compatible API for GLM models.

GLM Models available:
- glm-4.6: 200K context, 128K output, reasoning capable
- glm-4.7: 200K context, 128K output, reasoning capable
- glm-4.7-free: Free tier with rate limits

Endpoint: https://opencode.ai/zen/v1
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from openai import OpenAI
from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)

OPENCODE_BASE_URL = "https://opencode.ai/zen/v1"


def get_opencode_api_key() -> str | None:
    """Get OpenCode API key from environment."""
    return os.environ.get("OPENCODE_API_KEY")


class OpenCodeLM(BaseLM):
    """DSPy-compatible LM for OpenCode GLM models via OpenAI-compatible API."""
    
    def __init__(
        self,
        model: str = "glm-4.7",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        num_retries: int = 3,
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):
        """Initialize OpenCode LM.
        
        Args:
            model: Model ID (e.g., "glm-4.6", "glm-4.7", "glm-4.7-free")
            api_key: OpenCode API key (or from OPENCODE_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(
            model=f"opencode/{model}" if not model.startswith("opencode/") else model,
            model_type=model_type,
            temperature=temperature or 0.0,
            max_tokens=max_tokens or 8192,
            cache=cache,
            num_retries=num_retries,
        )
        
        self._model_id = model.replace("opencode/", "") if model.startswith("opencode/") else model
        self._temperature = temperature or 0.0
        self._max_tokens = max_tokens or 8192
        
        # Get API key
        self._api_key = api_key or get_opencode_api_key()
        if not self._api_key:
            raise ValueError("No OpenCode API key found. Set OPENCODE_API_KEY environment variable.")
        
        # Create OpenAI client with OpenCode base URL
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=OPENCODE_BASE_URL,
        )
    
    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> list[dict]:
        """Call the OpenCode API.
        
        Args:
            prompt: Single prompt string
            messages: List of messages (OpenAI format)
            **kwargs: Additional parameters
            
        Returns:
            List of response dicts with 'text' key (DSPy format)
        """
        # Build messages
        if messages is None:
            messages = []
        if prompt:
            messages = [{"role": "user", "content": prompt}]
        
        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "max_completion_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
            "stream": True,  # Use streaming for long responses
        }
        
        # Call API with streaming
        try:
            text = ""
            stream = self._client.chat.completions.create(**request_kwargs)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
        except Exception as e:
            logger.error("OpenCode API error: %s", e)
            raise ValueError(f"OpenCode API error: {e}")
        
        # Update history for BaseLM
        self.history.append({
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": {"text": text},
        })
        
        # Return in DSPy format
        return [{"text": text}]


def create_opencode_lm(model: str = "glm-4.7", **kwargs) -> OpenCodeLM:
    """Create an OpenCode LM instance.
    
    Args:
        model: Model ID (without opencode/ prefix)
        **kwargs: Additional parameters
    """
    if model.startswith("opencode/"):
        model = model[9:]
    return OpenCodeLM(model=model, **kwargs)

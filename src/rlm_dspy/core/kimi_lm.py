"""Kimi LM for DSPy.

Kimi (by Moonshot AI) provides an Anthropic-compatible API for coding tasks.

Kimi Models available:
- k2p5 (Kimi K2.5): 256K context, 32K output, reasoning capable, vision
- kimi-k2-thinking: 256K context, 32K output, extended thinking

Endpoint: https://api.kimi.com/coding
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import anthropic
from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)

KIMI_BASE_URL = "https://api.kimi.com/coding"


def get_kimi_api_key() -> str | None:
    """Get Kimi API key from environment."""
    return os.environ.get("KIMI_API_KEY")


class KimiLM(BaseLM):
    """DSPy-compatible LM for Kimi models via Anthropic-compatible API.
    
    Kimi (Moonshot AI) provides an Anthropic Messages API compatible endpoint.
    """
    
    def __init__(
        self,
        model: str = "k2p5",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        num_retries: int = 3,
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):
        """Initialize Kimi LM.
        
        Args:
            model: Model ID (e.g., "k2p5", "kimi-k2-thinking")
            api_key: Kimi API key (or from KIMI_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(
            model=f"kimi/{model}" if not model.startswith("kimi/") else model,
            model_type=model_type,
            temperature=temperature or 0.0,
            max_tokens=max_tokens or 8192,
            cache=cache,
            num_retries=num_retries,
        )
        
        self._model_id = model.replace("kimi/", "") if model.startswith("kimi/") else model
        self._temperature = temperature or 0.0
        self._max_tokens = max_tokens or 8192
        
        # Get API key
        self._api_key = api_key or get_kimi_api_key()
        if not self._api_key:
            raise ValueError("No Kimi API key found. Set KIMI_API_KEY environment variable.")
        
        # Create Anthropic client with Kimi base URL
        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            base_url=KIMI_BASE_URL,
            default_headers={
                "accept": "application/json",
            },
        )
    
    def __getstate__(self) -> dict:
        """Get state for pickling - exclude non-picklable client."""
        state = self.__dict__.copy()
        # Remove the client - will be recreated on unpickle
        state.pop('_client', None)
        return state
    
    def __setstate__(self, state: dict) -> None:
        """Restore state from pickle - recreate client."""
        self.__dict__.update(state)
        # Recreate the client
        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            base_url=KIMI_BASE_URL,
            default_headers={
                "accept": "application/json",
            },
        )
    
    def _convert_messages(self, messages: list[dict]) -> tuple[list[dict], str | None]:
        """Convert to Anthropic message format."""
        converted = []
        system = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system = content
            elif role == "assistant":
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append({"role": "user", "content": content})
        
        return converted, system
    
    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> list[dict]:
        """Call the Kimi API.
        
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
        
        # Convert to Anthropic format
        converted_messages, system = self._convert_messages(messages)
        
        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": converted_messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
        }
        
        if system:
            request_kwargs["system"] = system
        
        # Call API with streaming to support large responses
        try:
            text = ""
            with self._client.messages.stream(**request_kwargs) as stream:
                for chunk in stream.text_stream:
                    text += chunk
        except anthropic.APIError as e:
            logger.error("Kimi API error: %s", e)
            raise ValueError(f"Kimi API error: {e}")
        
        # Update history for BaseLM
        self.history.append({
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": {"text": text},
        })
        
        # Return in DSPy format
        return [{"text": text}]


def create_kimi_lm(model: str = "k2p5", **kwargs) -> KimiLM:
    """Create a Kimi LM instance.
    
    Args:
        model: Model ID (without kimi/ prefix)
        **kwargs: Additional parameters
    """
    if model.startswith("kimi/"):
        model = model[5:]
    return KimiLM(model=model, **kwargs)

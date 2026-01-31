"""Z.AI LM for DSPy.

Z.AI provides an OpenAI-compatible API for Zhipu's GLM models.

GLM Models available:
- glm-4.5: 128K context, 98K output, reasoning capable
- glm-4.5-air: 128K context, 128K output, reasoning capable
- glm-4.6: 200K context, 128K output, reasoning capable
- glm-4.7: 200K context, 128K output, reasoning capable

Endpoint: https://api.z.ai/api/coding/paas/v4

Special features:
- Supports extended thinking format (zai format)
- Different from standard OpenAI thinking format
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from openai import OpenAI
from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)

ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


def get_zai_api_key() -> str | None:
    """Get Z.AI API key from environment."""
    return os.environ.get("ZAI_API_KEY")


class ZaiLM(BaseLM):
    """DSPy-compatible LM for Z.AI GLM models via OpenAI-compatible API.
    
    Z.AI is the native provider for Zhipu's GLM models with special
    support for extended thinking in the 'zai' format.
    """
    
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
        """Initialize Z.AI LM.
        
        Args:
            model: Model ID (e.g., "glm-4.5", "glm-4.6", "glm-4.7")
            api_key: Z.AI API key (or from ZAI_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(
            model=f"zai/{model}" if not model.startswith("zai/") else model,
            model_type=model_type,
            temperature=temperature or 0.0,
            max_tokens=max_tokens or 8192,
            cache=cache,
            num_retries=num_retries,
        )
        
        self._model_id = model.replace("zai/", "") if model.startswith("zai/") else model
        self._temperature = temperature or 0.0
        self._max_tokens = max_tokens or 8192
        
        # Get API key
        self._api_key = api_key or get_zai_api_key()
        if not self._api_key:
            raise ValueError("No Z.AI API key found. Set ZAI_API_KEY environment variable.")
        
        # Create OpenAI client with Z.AI base URL
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=ZAI_BASE_URL,
        )
    
    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> list[dict]:
        """Call the Z.AI API.
        
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
        # Z.AI doesn't support developer role
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "developer":
                # Convert developer role to system
                filtered_messages.append({"role": "system", "content": msg.get("content", "")})
            else:
                filtered_messages.append(msg)
        
        request_kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": filtered_messages,
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
            logger.error("Z.AI API error: %s", e)
            raise ValueError(f"Z.AI API error: {e}")
        
        # Update history for BaseLM
        self.history.append({
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": {"text": text},
        })
        
        # Return in DSPy format
        return [{"text": text}]


def create_zai_lm(model: str = "glm-4.7", **kwargs) -> ZaiLM:
    """Create a Z.AI LM instance.
    
    Args:
        model: Model ID (without zai/ prefix)
        **kwargs: Additional parameters
    """
    if model.startswith("zai/"):
        model = model[4:]
    return ZaiLM(model=model, **kwargs)

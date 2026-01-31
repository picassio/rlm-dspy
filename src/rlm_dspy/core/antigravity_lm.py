"""Antigravity LM for DSPy.

Uses Antigravity OAuth tokens to call models via the Cloud Code Assist
sandbox API (daily-cloudcode-pa.sandbox.googleapis.com).

Supports:
- Gemini 3 models (gemini-3-flash, gemini-3-pro)
- Claude models via Google (claude-sonnet-4-5, claude-opus-4-5)
- GPT-OSS models
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

import httpx
from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)

# Antigravity uses the sandbox endpoint
ANTIGRAVITY_API = "https://daily-cloudcode-pa.sandbox.googleapis.com"

# Headers required for Antigravity
ANTIGRAVITY_HEADERS = {
    "User-Agent": "antigravity/1.15.8 darwin/arm64",
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": json.dumps({
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    }),
}


def get_antigravity_api_key() -> tuple[str, str] | None:
    """Get Antigravity OAuth token and project ID."""
    from .antigravity_oauth import get_antigravity_token
    return get_antigravity_token()


class AntigravityLM(BaseLM):
    """DSPy-compatible LM using Antigravity OAuth for advanced models.
    
    Uses the Cloud Code Assist sandbox API for access to Gemini 3,
    Claude (via Google), and GPT-OSS models.
    """
    
    def __init__(
        self,
        model: str = "gemini-3-flash",
        auth_token: str | None = None,
        project_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        num_retries: int = 3,
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):
        """Initialize Antigravity LM.
        
        Args:
            model: Model ID (e.g., "gemini-3-flash", "claude-sonnet-4-5", "gpt-oss-120b")
            auth_token: OAuth access token (or fetch from stored credentials)
            project_id: Google Cloud project ID
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(
            model=f"antigravity/{model}" if not model.startswith("antigravity/") else model,
            model_type=model_type,
            temperature=temperature or 0.0,
            max_tokens=max_tokens or 8192,
            cache=cache,
            num_retries=num_retries,
        )
        
        self._model_id = model.replace("antigravity/", "") if model.startswith("antigravity/") else model
        self._auth_token = auth_token
        self._project_id = project_id
        self._temperature = temperature or 0.0
        self._max_tokens = max_tokens or 8192
    
    def _get_credentials(self) -> tuple[str, str]:
        """Get OAuth credentials (token and project ID)."""
        if self._auth_token and self._project_id:
            return self._auth_token, self._project_id
        
        result = get_antigravity_api_key()
        if not result:
            raise ValueError(
                "No Antigravity OAuth credentials found. "
                "Run 'rlm-dspy auth login antigravity' to authenticate."
            )
        
        return result
    
    def _convert_messages(self, messages: list[dict]) -> tuple[list[dict], str | None]:
        """Convert OpenAI-style messages to Gemini contents format."""
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
            else:
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
        
        return contents, system_instruction
    
    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> list[dict]:
        """Call the Antigravity API.
        
        Args:
            prompt: Single prompt string
            messages: List of messages (OpenAI format)
            **kwargs: Additional parameters
            
        Returns:
            List of response dicts with 'text' key (DSPy format)
        """
        token, project_id = self._get_credentials()
        
        # Build messages
        if messages is None:
            messages = []
        if prompt:
            messages = [{"role": "user", "content": prompt}]
        
        # Convert to Gemini format
        contents, system_instruction = self._convert_messages(messages)
        
        # Build Cloud Code Assist request format
        inner_request: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self._max_tokens),
                "temperature": kwargs.get("temperature", self._temperature),
            },
        }
        
        if system_instruction:
            inner_request["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        # Wrap in Cloud Code Assist envelope (Antigravity format)
        request_body = {
            "project": project_id,
            "model": self._model_id,
            "request": inner_request,
            "requestType": "agent",  # Antigravity uses agent request type
            "userAgent": "antigravity",
            "requestId": f"ag-{id(self)}-{len(self.history)}",
        }
        
        # Call API (non-streaming)
        url = f"{ANTIGRAVITY_API}/v1internal:generateContent"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            **ANTIGRAVITY_HEADERS,
        }
        
        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, json=request_body, headers=headers)
        
        if response.status_code != 200:
            error_text = response.text
            logger.error("Antigravity API error: %s %s", response.status_code, error_text)
            
            # Parse error message for better UX
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", error_text)
            except Exception:
                error_msg = error_text
            
            raise ValueError(f"Antigravity API error ({response.status_code}): {error_msg}")
        
        data = response.json()
        
        # Extract text from response
        response_data = data.get("response", data)
        candidates = response_data.get("candidates", [])
        if not candidates:
            raise ValueError("No response candidates from Antigravity API")
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        text = ""
        for part in parts:
            if "text" in part:
                text += part["text"]
        
        # Update history for BaseLM
        self.history.append({
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": {"text": text},
        })
        
        # Return in DSPy format
        return [{"text": text}]


def create_antigravity_lm(model: str = "gemini-3-flash", **kwargs) -> AntigravityLM:
    """Create an Antigravity LM instance."""
    if model.startswith("antigravity/"):
        model = model[12:]
    return AntigravityLM(model=model, **kwargs)

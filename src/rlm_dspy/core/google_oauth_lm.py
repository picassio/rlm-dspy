"""Google Gemini CLI OAuth LM for DSPy.

Uses Google Cloud Code Assist OAuth tokens to call Gemini models via
the Cloud Code Assist API (cloudcode-pa.googleapis.com).

This bypasses LiteLLM because it doesn't properly support OAuth tokens
for Google Cloud Code Assist.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

import httpx
from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)

# Cloud Code Assist API endpoint (NOT generativelanguage.googleapis.com)
CLOUDCODE_API = "https://cloudcode-pa.googleapis.com"

# Headers required for Gemini CLI
GEMINI_CLI_HEADERS = {
    "User-Agent": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": json.dumps({
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    }),
}


def get_google_api_key() -> tuple[str, str] | None:
    """Get Google OAuth token and project ID."""
    from .oauth import get_google_token
    return get_google_token()


class GoogleOAuthLM(BaseLM):
    """DSPy-compatible LM using Google OAuth for Gemini models via Cloud Code Assist.
    
    Uses the Cloud Code Assist API (cloudcode-pa.googleapis.com), not the
    public Generative Language API.
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        auth_token: str | None = None,
        project_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        num_retries: int = 3,
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):
        """Initialize Google OAuth LM.
        
        Args:
            model: Model ID (e.g., "gemini-2.0-flash", "gemini-2.5-pro")
            auth_token: OAuth access token (or fetch from stored credentials)
            project_id: Google Cloud project ID
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(
            model=f"google/{model}" if not model.startswith("google/") else model,
            model_type=model_type,
            temperature=temperature or 0.0,
            max_tokens=max_tokens or 8192,
            cache=cache,
            num_retries=num_retries,
        )
        
        self._model_id = model.replace("google/", "") if model.startswith("google/") else model
        self._auth_token = auth_token
        self._project_id = project_id
        self._temperature = temperature or 0.0
        self._max_tokens = max_tokens or 8192
    
    def _get_credentials(self) -> tuple[str, str]:
        """Get OAuth credentials (token and project ID)."""
        if self._auth_token and self._project_id:
            return self._auth_token, self._project_id
        
        result = get_google_api_key()
        if not result:
            raise ValueError(
                "No Google OAuth credentials found. "
                "Run 'rlm-dspy auth login google' to authenticate."
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
        """Call the Cloud Code Assist API.
        
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
        
        # Wrap in Cloud Code Assist envelope
        request_body = {
            "project": project_id,
            "model": self._model_id,
            "request": inner_request,
            "userAgent": "rlm-dspy",
            "requestId": f"rlm-{id(self)}-{len(self.history)}",
        }
        
        # Call API (non-streaming for simplicity)
        url = f"{CLOUDCODE_API}/v1internal:generateContent"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            **GEMINI_CLI_HEADERS,
        }
        
        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, json=request_body, headers=headers)
        
        if response.status_code != 200:
            error_text = response.text
            logger.error("Cloud Code Assist API error: %s %s", response.status_code, error_text)
            
            # Parse error message for better UX
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", error_text)
            except Exception:
                error_msg = error_text
            
            raise ValueError(f"Cloud Code Assist API error ({response.status_code}): {error_msg}")
        
        data = response.json()
        
        # Extract text from response (Cloud Code Assist wraps in "response")
        response_data = data.get("response", data)
        candidates = response_data.get("candidates", [])
        if not candidates:
            raise ValueError("No response candidates from Cloud Code Assist API")
        
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


def create_google_oauth_lm(model: str = "gemini-2.0-flash", **kwargs) -> GoogleOAuthLM:
    """Create a Google OAuth LM instance."""
    if model.startswith("google/"):
        model = model[7:]
    return GoogleOAuthLM(model=model, **kwargs)

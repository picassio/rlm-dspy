"""Async client for high-performance concurrent LLM requests.

Uses httpx for async HTTP and semaphores for rate limiting.
Much faster than sequential DSPy calls.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class AsyncResponse:
    """Response from async LLM call."""

    content: str
    model: str
    usage: dict[str, int]
    latency_ms: float
    error: str | None = None


class AsyncLLMClient:
    """
    Async client for concurrent LLM requests via OpenRouter.

    Uses semaphore to limit concurrent requests and avoid rate limits.
    Supports prompt caching and thinking mode control.

    Example:
        ```python
        client = AsyncLLMClient(max_concurrent=20, enable_cache=True)
        prompts = ["What is 1+1?", "What is 2+2?", "What is 3+3?"]
        results = await client.batch_complete(prompts)
        ```
    """

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        api_key: str | None = None,
        api_base: str = "https://openrouter.ai/api/v1",
        max_concurrent: int = 20,
        timeout: float = 120.0,
        disable_thinking: bool = True,
        enable_cache: bool = True,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        self.enable_cache = enable_cache
        self._semaphore: asyncio.Semaphore | None = None

    async def _ensure_semaphore(self) -> asyncio.Semaphore:
        """Lazily create semaphore in the right event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncResponse:
        """
        Single async completion.

        Args:
            prompt: User message
            system: System prompt
            max_tokens: Max response tokens
            temperature: Sampling temperature

        Returns:
            AsyncResponse with content or error
        """
        semaphore = await self._ensure_semaphore()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Disable thinking/reasoning for faster responses
        if self.disable_thinking:
            # OpenRouter provider preferences to disable extended thinking
            payload["provider"] = {
                "allow_fallbacks": True,
                "require_parameters": False,
            }
            # For models that support it, disable reasoning tokens
            if "gemini" in self.model.lower():
                payload["reasoning"] = {"effort": "none"}
            elif "claude" in self.model.lower():
                payload["thinking"] = {"type": "disabled"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rlm-dspy",
        }

        # Enable prompt caching for repeated similar requests
        if self.enable_cache:
            headers["X-Use-Prompt-Cache"] = "true"

        start = time.perf_counter()

        async with semaphore:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                try:
                    response = await client.post(
                        f"{self.api_base}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()

                    latency = (time.perf_counter() - start) * 1000

                    return AsyncResponse(
                        content=data["choices"][0]["message"]["content"],
                        model=data.get("model", self.model),
                        usage=data.get("usage", {}),
                        latency_ms=latency,
                    )
                except httpx.HTTPStatusError as e:
                    return AsyncResponse(
                        content="",
                        model=self.model,
                        usage={},
                        latency_ms=(time.perf_counter() - start) * 1000,
                        error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                    )
                except Exception as e:
                    return AsyncResponse(
                        content="",
                        model=self.model,
                        usage={},
                        latency_ms=(time.perf_counter() - start) * 1000,
                        error=str(e),
                    )

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> list[AsyncResponse]:
        """
        Batch async completions - runs all prompts concurrently.

        Args:
            prompts: List of user messages
            system: Shared system prompt
            max_tokens: Max response tokens
            temperature: Sampling temperature

        Returns:
            List of AsyncResponse in same order as prompts
        """
        tasks = [self.complete(prompt, system, max_tokens, temperature) for prompt in prompts]
        return await asyncio.gather(*tasks)


async def analyze_chunks_async(
    query: str,
    chunks: list[str],
    model: str = "google/gemini-2.5-flash",
    max_concurrent: int = 20,
    disable_thinking: bool = True,
    enable_cache: bool = True,
) -> list[dict[str, Any]]:
    """
    Analyze multiple chunks concurrently.

    Args:
        query: The question to answer
        chunks: List of text chunks
        model: Model to use
        max_concurrent: Max concurrent requests
        disable_thinking: Disable extended thinking for speed
        enable_cache: Enable prompt caching

    Returns:
        List of analysis results with relevant_info and confidence
    """
    client = AsyncLLMClient(
        model=model,
        max_concurrent=max_concurrent,
        disable_thinking=disable_thinking,
        enable_cache=enable_cache,
    )

    system = """You are analyzing a chunk of content to answer a query.
Extract ALL information relevant to the query. Be thorough and specific.
If the chunk contains relevant information, extract it completely.
If no relevant information, say "None".

Your response format:
CONFIDENCE: high|medium|low|none
RELEVANT_INFO: <extracted information or "None">"""

    prompts = [f"Query: {query}\n\nChunk {i + 1} of {len(chunks)}:\n{chunk}" for i, chunk in enumerate(chunks)]

    responses = await client.batch_complete(prompts, system=system)

    results = []
    for i, resp in enumerate(responses):
        if resp.error:
            results.append(
                {
                    "index": i,
                    "relevant_info": "",
                    "confidence": "none",
                    "error": resp.error,
                }
            )
        else:
            # Parse response - robust multi-line parsing
            content = resp.content.strip()
            info = ""
            confidence = "medium"  # Default if not specified

            # Parse CONFIDENCE
            if "CONFIDENCE:" in content:
                conf_part = content.split("CONFIDENCE:")[1]
                conf_line = conf_part.split("\n")[0].strip().lower()
                for c in ("high", "medium", "low", "none"):
                    if c in conf_line:
                        confidence = c
                        break

            # Parse RELEVANT_INFO (can be multi-line)
            if "RELEVANT_INFO:" in content:
                info_part = content.split("RELEVANT_INFO:")[1].strip()
                if info_part.lower().startswith("none"):
                    info = ""
                    confidence = "none"
                else:
                    info = info_part
            else:
                # No structured format - use full content if not "none"
                lower = content.lower()
                if any(x in lower for x in ["no relevant", "not relevant", "none found"]):
                    confidence = "none"
                else:
                    info = content

            results.append(
                {
                    "index": i,
                    "relevant_info": info,
                    "confidence": confidence,
                    "latency_ms": resp.latency_ms,
                    "usage": resp.usage,
                }
            )

    return results


async def aggregate_answers_async(
    query: str,
    partial_answers: list[str],
    model: str = "google/gemini-2.5-flash",
    disable_thinking: bool = True,
    enable_cache: bool = True,
) -> str:
    """
    Aggregate partial answers into final answer.

    Args:
        query: Original query
        partial_answers: List of partial answers from chunks
        model: Model to use
        disable_thinking: Disable extended thinking for speed
        enable_cache: Enable prompt caching

    Returns:
        Final aggregated answer
    """
    client = AsyncLLMClient(
        model=model,
        disable_thinking=disable_thinking,
        enable_cache=enable_cache,
    )

    system = """You are synthesizing multiple partial answers into a comprehensive final answer.
Combine the information coherently, remove redundancy, and provide a complete response."""

    prompt = f"""Query: {query}

Partial answers from different sources:
{chr(10).join(f"{i + 1}. {ans}" for i, ans in enumerate(partial_answers))}

Provide a comprehensive final answer:"""

    response = await client.complete(prompt, system=system)

    if response.error:
        return f"Error aggregating: {response.error}"

    return response.content

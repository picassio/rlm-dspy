"""Batch processing utilities.

Learned from modaic: JSONL batch handling, polling, streaming downloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx

logger = logging.getLogger(__name__)

# Batch file storage
BATCH_DIR = Path(os.environ.get("RLM_BATCH_DIR", tempfile.gettempdir())) / "rlm_batches"


@dataclass
class BatchRequest:
    """A single request in a batch."""

    custom_id: str
    messages: list[dict[str, Any]]
    model: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> dict[str, Any]:
        """Format for OpenAI batch API."""
        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Format for Anthropic batch API."""
        return {
            "custom_id": self.custom_id,
            "params": {
                "model": self.model,
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }


@dataclass
class BatchResult:
    """Result from a single batch request."""

    custom_id: str
    content: str
    success: bool = True
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> "BatchResult":
        """Parse OpenAI batch result format."""
        custom_id = data.get("custom_id", "")
        response = data.get("response", {})

        if response.get("status_code") != 200:
            return cls(
                custom_id=custom_id,
                content="",
                success=False,
                error=response.get("body", {}).get("error", {}).get("message", "Unknown error"),
            )

        body = response.get("body", {})
        choices = body.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""

        return cls(
            custom_id=custom_id,
            content=content,
            usage=body.get("usage", {}),
        )


@dataclass
class BatchStatus:
    """Status of a batch job."""

    id: str
    status: str  # pending, in_progress, completed, failed, cancelled, expired
    progress: float = 0.0  # 0.0 to 1.0
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    created_at: float = 0.0
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled", "expired")

    @property
    def is_success(self) -> bool:
        return self.status == "completed"


def create_jsonl(
    requests: list[BatchRequest],
    format_func: Callable[[BatchRequest], dict[str, Any]] | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Create a JSONL file from batch requests.

    Learned from modaic: standardized JSONL creation with custom formatting.

    Args:
        requests: List of batch requests
        format_func: Optional custom formatter (defaults to OpenAI format)
        output_path: Optional output path (creates temp file if None)

    Returns:
        Path to the created JSONL file
    """
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = BATCH_DIR / f"batch_{int(time.time() * 1000)}.jsonl"

    format_func = format_func or (lambda r: r.to_openai_format())

    with open(output_path, "w") as f:
        for request in requests:
            line = json.dumps(format_func(request))
            f.write(line + "\n")

    logger.debug("Created JSONL with %d requests: %s", len(requests), output_path)
    return output_path


def parse_jsonl(
    path: Path | str,
    parse_func: Callable[[dict[str, Any]], BatchResult] | None = None,
) -> Iterator[BatchResult]:
    """
    Parse a JSONL results file.

    Args:
        path: Path to JSONL file
        parse_func: Optional custom parser (defaults to OpenAI format)

    Yields:
        BatchResult objects
    """
    parse_func = parse_func or BatchResult.from_openai

    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                yield parse_func(data)


async def stream_download(
    url: str,
    output_path: Path,
    headers: dict[str, str] | None = None,
    chunk_size: int = 8192,
) -> Path:
    """
    Stream download large files in chunks.

    Learned from modaic's Together AI pattern: iter_bytes() for memory efficiency.

    Args:
        url: URL to download
        output_path: Where to save the file
        headers: Optional HTTP headers
        chunk_size: Size of each chunk

    Returns:
        Path to downloaded file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("GET", url, headers=headers or {}) as response:
            response.raise_for_status()

            with open(output_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size):
                    f.write(chunk)

    logger.debug("Downloaded %s to %s", url, output_path)
    return output_path


class BatchPoller:
    """
    Poll for batch job completion.

    Learned from modaic: standardized polling with progress callbacks.
    """

    def __init__(
        self,
        poll_interval: float = 30.0,
        max_poll_time: float = 86400.0,  # 24 hours
        on_progress: Callable[[BatchStatus], None] | None = None,
    ):
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.on_progress = on_progress

    async def poll_until_complete(
        self,
        get_status: Callable[[], BatchStatus] | Callable[[], Any],
    ) -> BatchStatus:
        """
        Poll until batch reaches terminal state.

        Args:
            get_status: Async function that returns BatchStatus

        Returns:
            Final BatchStatus
        """
        start_time = time.time()

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.max_poll_time:
                return BatchStatus(
                    id="timeout",
                    status="expired",
                    error=f"Polling timeout after {elapsed:.0f}s",
                )

            # Get status
            if asyncio.iscoroutinefunction(get_status):
                status = await get_status()
            else:
                status = get_status()

            # Notify progress
            if self.on_progress:
                self.on_progress(status)

            # Check terminal
            if status.is_terminal:
                return status

            # Wait
            await asyncio.sleep(self.poll_interval)


def sort_results_by_custom_id(results: list[BatchResult]) -> list[BatchResult]:
    """
    Sort batch results by custom_id to maintain input order.

    Learned from modaic: custom_id pattern (request-0, request-1, etc.)
    """

    def extract_index(result: BatchResult) -> int:
        # Handle formats like "request-42" or just "42"
        custom_id = result.custom_id
        if "-" in custom_id:
            try:
                return int(custom_id.split("-")[-1])
            except ValueError:
                pass
        try:
            return int(custom_id)
        except ValueError:
            return 0

    return sorted(results, key=extract_index)


def cleanup_batch_files(max_age_hours: float = 24.0) -> int:
    """
    Clean up old batch files.

    Args:
        max_age_hours: Delete files older than this

    Returns:
        Number of files deleted
    """
    if not BATCH_DIR.exists():
        return 0

    deleted = 0
    cutoff = time.time() - (max_age_hours * 3600)

    for path in BATCH_DIR.glob("*.jsonl"):
        if path.stat().st_mtime < cutoff:
            try:
                path.unlink()
                deleted += 1
            except OSError as e:
                logger.warning("Failed to delete %s: %s", path, e)

    if deleted:
        logger.info("Cleaned up %d old batch files", deleted)

    return deleted

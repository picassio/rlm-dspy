"""Progress display for batch processing.

Learned from modaic: rich-based live progress dashboard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


@dataclass
class BatchProgress:
    """
    Live progress display for batch chunk processing.

    Learned from modaic's BatchProgressDisplay:
    - Real-time terminal dashboard
    - Status, progress, timing info
    - Color-coded status indicators
    """

    total_chunks: int
    model: str = "unknown"
    status: str = "initializing"
    processed: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.time)
    current_chunk: int = 0

    def update(
        self,
        status: Optional[str] = None,
        processed: Optional[int] = None,
        failed: Optional[int] = None,
        current_chunk: Optional[int] = None,
    ) -> None:
        """Update progress state."""
        if status is not None:
            self.status = status
        if processed is not None:
            self.processed = processed
        if failed is not None:
            self.failed = failed
        if current_chunk is not None:
            self.current_chunk = current_chunk

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def progress_pct(self) -> int:
        if self.total_chunks == 0:
            return 0
        return int((self.processed + self.failed) / self.total_chunks * 100)

    @property
    def rate(self) -> float:
        """Chunks per second."""
        if self.elapsed == 0:
            return 0.0
        return (self.processed + self.failed) / self.elapsed

    def make_panel(self) -> Panel:
        """Create a rich Panel showing current progress."""
        table = Table.grid(padding=(0, 4))
        table.add_column(style="cyan", justify="right", width=15)
        table.add_column(style="white", min_width=25)

        # Status with color
        status_lower = self.status.lower()
        if status_lower == "completed":
            styled_status = "[bold green]✓ completed[/bold green]"
        elif status_lower in ("failed", "error"):
            styled_status = f"[bold red]✗ {status_lower}[/bold red]"
        elif status_lower == "processing":
            styled_status = "[bold yellow]⟳ processing[/bold yellow]"
        else:
            styled_status = f"[dim]{status_lower}[/dim]"

        # Progress bar
        pct = self.progress_pct
        bar_width = 20
        filled = int(bar_width * pct / 100)
        bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_width - filled)}[/dim]"

        table.add_row("Model:", f"[magenta]{self.model}[/magenta]")
        table.add_row("Status:", styled_status)
        table.add_row("Progress:", f"{bar} {pct}%")
        table.add_row("Chunks:", f"[bold]{self.processed}[/bold]/{self.total_chunks} ([red]{self.failed} failed[/red])")
        table.add_row("Elapsed:", f"{self.elapsed:.1f}s")
        table.add_row("Rate:", f"{self.rate:.1f} chunks/sec")

        if self.rate > 0 and self.processed < self.total_chunks:
            remaining = (self.total_chunks - self.processed - self.failed) / self.rate
            table.add_row("ETA:", f"~{remaining:.0f}s")

        return Panel(
            table,
            title="[bold blue]RLM-DSPy Batch Processing[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )


class ProgressContext:
    """
    Context manager for live progress display.

    Usage:
        with ProgressContext(total=10, model="gemini") as progress:
            for i, chunk in enumerate(chunks):
                result = process(chunk)
                progress.update(processed=i+1)
    """

    def __init__(
        self,
        total: int,
        model: str = "unknown",
        show_progress: bool = True,
        console: Optional[Console] = None,
    ):
        self.progress = BatchProgress(total_chunks=total, model=model)
        self.show_progress = show_progress
        self.console = console or Console()
        self._live: Optional[Live] = None

    def __enter__(self) -> BatchProgress:
        if self.show_progress:
            self._live = Live(
                self.progress.make_panel(),
                console=self.console,
                refresh_per_second=4,
                transient=True,
            )
            self._live.__enter__()
        return self.progress

    def __exit__(self, *args) -> None:
        if self._live:
            # Final update
            self._live.update(self.progress.make_panel())
            self._live.__exit__(*args)

    def update(self, **kwargs) -> None:
        """Update progress and refresh display."""
        self.progress.update(**kwargs)
        if self._live:
            self._live.update(self.progress.make_panel())

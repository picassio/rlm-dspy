"""CLI commands for trace management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

traces_app = typer.Typer(help="Manage execution traces", no_args_is_help=True)


@traces_app.command("list")
def traces_list(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max traces to show")] = 20,
    query_filter: Annotated[str | None, typer.Option("--filter", "-f", help="Filter by query text")] = None,
) -> None:
    """List recent execution traces."""
    from .core.trace_collector import get_trace_collector

    collector = get_trace_collector()
    traces = list(collector.traces)[-limit:]  # Get last N traces

    if query_filter:
        traces = [t for t in traces if query_filter.lower() in t.query.lower()]

    if not traces:
        console.print("[yellow]No traces found[/yellow]")
        return

    table = Table(title=f"Recent Traces (showing {len(traces)})")
    table.add_column("ID", style="cyan", max_width=16)
    table.add_column("Query", max_width=40)
    table.add_column("Type")
    table.add_column("Score", justify="right")
    table.add_column("Tools", justify="right")

    for trace in traces:
        query = trace.query[:40]
        if len(trace.query) > 40:
            query += "..."
        score_color = "green" if trace.grounded_score >= 0.8 else "yellow" if trace.grounded_score >= 0.5 else "red"

        table.add_row(
            trace.trace_id[:16] if trace.trace_id else "",
            query,
            trace.query_type,
            f"[{score_color}]{trace.grounded_score:.0%}[/{score_color}]",
            str(len(trace.tools_used)),
        )

    console.print(table)


@traces_app.command("show")
def traces_show(
    trace_id: Annotated[str, typer.Argument(help="Trace ID to show")],
    full: Annotated[bool, typer.Option("--full", "-f", help="Show full trajectory")] = False,
) -> None:
    """Show details of a specific trace."""
    from .core.trace_collector import get_trace_collector

    collector = get_trace_collector()
    trace = collector.get_trace(trace_id)

    if not trace:
        console.print(f"[red]Trace not found: {trace_id}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Trace: {trace_id}[/bold cyan]")
    console.print(f"Query: {trace.get('query', '')}")
    console.print(f"Status: {'[green]Success[/green]' if trace.get('success') else '[red]Failed[/red]'}")
    console.print(f"Iterations: {trace.get('iterations', 0)}")
    console.print(f"Time: {trace.get('elapsed_time', 0):.2f}s")

    if trace.get("error"):
        console.print(f"[red]Error: {trace['error']}[/red]")

    if full and trace.get("trajectory"):
        console.print("\n[bold]Trajectory:[/bold]")
        for i, step in enumerate(trace["trajectory"], 1):
            console.print(f"\n[cyan]Step {i}:[/cyan]")
            console.print(f"Reasoning: {step.get('reasoning', '')[:200]}...")
            if step.get("code"):
                console.print(f"Code: {step.get('code', '')[:100]}...")


@traces_app.command("stats")
def traces_stats() -> None:
    """Show trace statistics."""
    from .core.trace_collector import get_trace_collector

    collector = get_trace_collector()
    stats = collector.get_stats()

    table = Table(title="Trace Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Total Traces", str(stats.get("total", 0)))
    table.add_row("Successful", str(stats.get("successful", 0)))
    table.add_row("Failed", str(stats.get("failed", 0)))
    table.add_row("Avg Iterations", f"{stats.get('avg_iterations', 0):.1f}")
    table.add_row("Avg Time", f"{stats.get('avg_time', 0):.1f}s")
    table.add_row("Storage Used", f"{stats.get('storage_bytes', 0) / 1024:.1f} KB")

    console.print(table)


@traces_app.command("export")
def traces_export(
    output: Annotated[Path, typer.Argument(help="Output file path")],
) -> None:
    """Export traces to JSON file."""
    from .core.trace_collector import get_trace_collector

    collector = get_trace_collector()
    traces = collector.list_traces(limit=1000)

    output.write_text(json.dumps(traces, indent=2))
    console.print(f"[green]✓[/green] Exported {len(traces)} traces to {output}")


@traces_app.command("import")
def traces_import(
    input_file: Annotated[Path, typer.Argument(help="Input JSON file")],
) -> None:
    """Import traces from JSON file."""
    from .core.trace_collector import get_trace_collector

    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    collector = get_trace_collector()
    traces = json.loads(input_file.read_text())

    count = 0
    for trace in traces:
        if collector.add_trace(trace):
            count += 1

    console.print(f"[green]✓[/green] Imported {count} traces")


@traces_app.command("clear")
def traces_clear(
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
    older_than: Annotated[int | None, typer.Option("--older-than", help="Only clear traces older than N days")] = None,
) -> None:
    """Clear execution traces."""
    from .core.trace_collector import get_trace_collector

    collector = get_trace_collector()

    if not force:
        if not typer.confirm("Clear all traces?"):
            raise typer.Abort()

    count = collector.clear(older_than_days=older_than)
    console.print(f"[green]✓[/green] Cleared {count} traces")

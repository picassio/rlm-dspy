"""CLI commands for optimization management."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

optimize_app = typer.Typer(help="Manage prompt optimization", no_args_is_help=True)


@optimize_app.command("stats")
def optimize_stats() -> None:
    """Show optimization statistics."""
    from .core.instruction_optimizer import get_instruction_optimizer
    from .core.grounded_proposer import get_grounded_proposer

    optimizer = get_instruction_optimizer()
    proposer = get_grounded_proposer()

    opt_stats = optimizer.get_stats()
    prop_stats = proposer.get_stats()

    table = Table(title="Optimization Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("[bold]Instruction Optimizer[/bold]", "")
    table.add_row("Total Outcomes", str(opt_stats.get("total_outcomes", 0)))
    table.add_row("Success Rate", f"{opt_stats.get('success_rate', 0):.1%}")
    table.add_row("Active Candidates", str(opt_stats.get("active_candidates", 0)))

    table.add_row("", "")
    table.add_row("[bold]Grounded Proposer[/bold]", "")
    table.add_row("Recorded Failures", str(prop_stats.get("total_failures", 0)))
    table.add_row("Recorded Successes", str(prop_stats.get("total_successes", 0)))
    table.add_row("Current Tips", str(prop_stats.get("current_tips_count", 0)))

    console.print(table)


@optimize_app.command("tips")
def optimize_tips(
    refresh: Annotated[bool, typer.Option("--refresh", "-r", help="Force refresh tips")] = False,
    clear: Annotated[bool, typer.Option("--clear", "-c", help="Clear all tips")] = False,
) -> None:
    """Show or manage optimization tips."""
    from .core.grounded_proposer import get_grounded_proposer

    proposer = get_grounded_proposer()

    if clear:
        proposer.reset_tips()
        console.print("[green]✓[/green] Cleared all tips")
        return

    if refresh:
        tips = proposer.generate_tips()
        console.print(f"[green]✓[/green] Refreshed tips ({len(tips)} generated)")
    else:
        tips = proposer.get_tips()

    if not tips:
        console.print("[yellow]No tips available. Run some queries to generate tips.[/yellow]")
        return

    console.print("\n[bold cyan]Current Optimization Tips:[/bold cyan]\n")
    for i, tip in enumerate(tips, 1):
        console.print(f"  {i}. {tip}")


@optimize_app.command("clear")
def optimize_clear(
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Clear optimization data."""
    from .core.instruction_optimizer import get_instruction_optimizer
    from .core.grounded_proposer import get_grounded_proposer

    if not force:
        if not typer.confirm("Clear all optimization data?"):
            raise typer.Abort()

    optimizer = get_instruction_optimizer()
    proposer = get_grounded_proposer()

    opt_cleared = optimizer.clear()
    failures, successes = proposer.clear()

    console.print(f"[green]✓[/green] Cleared {opt_cleared} optimizer outcomes")
    console.print(f"[green]✓[/green] Cleared {failures} failures and {successes} successes from proposer")


@optimize_app.command("patterns")
def optimize_patterns() -> None:
    """Show learned patterns from past queries."""
    from .core.grounded_proposer import get_grounded_proposer

    proposer = get_grounded_proposer()
    stats = proposer.get_stats()

    failure_patterns = stats.get("failure_patterns", {})
    success_patterns = stats.get("success_patterns", {})

    if not failure_patterns and not success_patterns:
        console.print("[yellow]No patterns learned yet. Run more queries to learn patterns.[/yellow]")
        return

    if failure_patterns:
        console.print("\n[bold red]Failure Patterns:[/bold red]")
        for reason, count in sorted(failure_patterns.items(), key=lambda x: -x[1])[:10]:
            console.print(f"  {count}x: {reason[:60]}...")

    if success_patterns:
        console.print("\n[bold green]Success Patterns:[/bold green]")
        for pattern, count in sorted(success_patterns.items(), key=lambda x: -x[1])[:10]:
            console.print(f"  {count}x: {pattern[:60]}...")

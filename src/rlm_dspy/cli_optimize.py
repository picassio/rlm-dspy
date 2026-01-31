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


@optimize_app.command("instructions")
def optimize_instructions(
    key: Annotated[str | None, typer.Argument(help="Instruction key to show/modify")] = None,
    reset: Annotated[bool, typer.Option("--reset", help="Reset to default")] = False,
    propose: Annotated[bool, typer.Option("--propose", "-p", help="Propose improvement")] = False,
) -> None:
    """Show or modify tool instructions."""
    from .core.instruction_optimizer import get_instruction_optimizer, DEFAULT_INSTRUCTIONS

    optimizer = get_instruction_optimizer()

    if key is None:
        table = Table(title="Instruction Keys")
        table.add_column("Key")
        table.add_column("Length", justify="right")
        table.add_column("Modified")

        for k in DEFAULT_INSTRUCTIONS:
            current = optimizer.get_instruction(k)
            default = DEFAULT_INSTRUCTIONS[k]
            modified = "✓" if current != default else ""
            table.add_row(k, str(len(current)), modified)

        console.print(table)
        return

    if key not in DEFAULT_INSTRUCTIONS:
        console.print(f"[red]Unknown key: {key}[/red]")
        console.print(f"Available: {', '.join(DEFAULT_INSTRUCTIONS.keys())}")
        raise typer.Exit(1)

    if reset:
        optimizer.reset_instruction(key)
        console.print(f"[green]✓[/green] Reset '{key}' to default")
        return

    if propose:
        console.print(f"[cyan]Proposing improvement for '{key}'...[/cyan]")
        new_instruction = optimizer.propose_improvement(key)
        if new_instruction:
            console.print("\n[bold]Proposed instruction:[/bold]")
            console.print(new_instruction[:500] + "..." if len(new_instruction) > 500 else new_instruction)
        else:
            console.print("[yellow]No improvement proposed[/yellow]")
        return

    # Show current instruction
    current = optimizer.get_instruction(key)
    console.print(f"\n[bold cyan]{key}:[/bold cyan]\n")
    console.print(current)


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
        top_reasons = failure_patterns.get("top_reasons", [])
        for reason, count in top_reasons[:10]:
            console.print(f"  {count}x: {reason[:60]}{'...' if len(reason) > 60 else ''}")
        
        missing_tools = failure_patterns.get("missing_tools", [])
        if missing_tools:
            console.print("\n[bold yellow]Missing Tools:[/bold yellow]")
            for tool, count in missing_tools[:5]:
                console.print(f"  {count}x: {tool}")

    if success_patterns:
        console.print("\n[bold green]Success Patterns:[/bold green]")
        tools_used = success_patterns.get("tools_used", [])
        for tool, count in tools_used[:10]:
            console.print(f"  {count}x: {tool}")


@optimize_app.command("simba")
def optimize_simba(
    min_score: Annotated[float, typer.Option("--min-score", "-s", help="Minimum trace score")] = 0.7,
    max_examples: Annotated[int, typer.Option("--max-examples", "-n", help="Maximum training examples")] = 100,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="Mini-batch size")] = 16,
    steps: Annotated[int, typer.Option("--steps", help="Optimization steps")] = 4,
    candidates: Annotated[int, typer.Option("--candidates", "-c", help="Candidates per step")] = 4,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be optimized")] = False,
) -> None:
    """Run SIMBA self-improving optimization.

    Uses collected traces to optimize the RLM program via DSPy's SIMBA optimizer.
    """
    from .core.trace_collector import get_trace_collector
    from .core.simba_optimizer import get_simba_optimizer, create_training_example

    collector = get_trace_collector()
    # Filter traces by grounded_score
    all_traces = list(collector.traces)
    traces = [t for t in all_traces if t.grounded_score >= min_score][:max_examples]

    if not traces:
        console.print(f"[yellow]No traces found with score >= {min_score}[/yellow]")
        console.print("Run some queries first to collect training data.")
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(traces)} traces for training[/cyan]")

    if dry_run:
        console.print("\n[bold]Would train on:[/bold]")
        for trace in traces[:10]:
            console.print(f"  - {trace.query[:50]}... (score: {trace.grounded_score:.2f})")
        if len(traces) > 10:
            console.print(f"  ... and {len(traces) - 10} more")
        return

    # Convert traces to training examples
    examples = []
    for trace in traces:
        example = create_training_example(trace)
        if example:
            examples.append(example)

    if not examples:
        console.print("[red]No valid training examples could be created[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Created {len(examples)} training examples[/cyan]")
    console.print(f"[cyan]Running SIMBA optimization (steps={steps}, candidates={candidates})...[/cyan]")

    try:
        optimizer = get_simba_optimizer()
        result = optimizer.optimize(
            examples=examples,
            batch_size=batch_size,
            num_steps=steps,
            num_candidates=candidates,
        )

        console.print("\n[bold green]✓ Optimization complete![/bold green]")
        console.print(f"  Initial score: {result.initial_score:.2%}")
        console.print(f"  Final score: {result.final_score:.2%}")
        console.print(f"  Improvement: {result.improvement:.2%}")

        if result.learned_rules:
            console.print("\n[bold]Learned Rules:[/bold]")
            for rule in result.learned_rules[:5]:
                console.print(f"  - {rule}")

    except Exception as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        raise typer.Exit(1)

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
        example = create_training_example(
            query=trace.query,
            answer=trace.final_answer,
            context="",  # Context not stored in trace
        )
        if example:
            examples.append(example)

    if not examples:
        console.print("[red]No valid training examples could be created[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Created {len(examples)} training examples[/cyan]")
    console.print(f"[cyan]Running SIMBA optimization (steps={steps}, candidates={candidates})...[/cyan]")

    try:
        from .core.rlm import RLM
        from .core.simba_optimizer import save_optimized_program, save_optimization_state, OptimizationState, get_trace_count

        # Get the RLM program to optimize
        rlm = RLM()
        optimizer = get_simba_optimizer()

        # Pass the LM from RLM to the optimizer
        optimized_program, result = optimizer.optimize(
            program=rlm._rlm,
            trainset=examples,
            lm=rlm._lm,
        )

        console.print("\n[bold green]✓ Optimization complete![/bold green]")
        console.print(f"  Baseline score: {result.baseline_score:.2%}")
        console.print(f"  Optimized score: {result.optimized_score:.2%}")
        console.print(f"  Improvement: {result.improvement:.1f}%")

        if result.improved:
            # Save the optimized program
            save_optimized_program(optimized_program, result, "simba")
            console.print("\n[green]✓ Optimized program saved - will be auto-loaded on next query[/green]")

        # Update optimization state
        from datetime import datetime, UTC
        state = OptimizationState(
            last_optimization=datetime.now(UTC),
            traces_at_last_optimization=get_trace_count(),
            last_result=result,
            optimizer_type="simba",
        )
        save_optimization_state(state)

    except Exception as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        raise typer.Exit(1)


@optimize_app.command("status")
def optimize_status() -> None:
    """Show auto-optimization status."""
    from .core.user_config import OptimizationConfig, load_config
    from .core.simba_optimizer import (
        load_optimization_state, load_optimized_program, get_trace_count,
        OPTIMIZED_PROGRAM_FILE,
    )

    config = OptimizationConfig.from_user_config()
    state = load_optimization_state()
    saved = load_optimized_program()
    current_traces = get_trace_count()
    user_config = load_config()

    table = Table(title="Auto-Optimization Status")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    # Config
    status = "[green]Enabled[/green]" if config.enabled else "[red]Disabled[/red]"
    table.add_row("Status", status)
    table.add_row("Optimizer", config.optimizer)

    model = config.get_model(user_config.get("model", "default"))
    table.add_row("Model", model)
    table.add_row("Min New Traces", str(config.min_new_traces))
    table.add_row("Min Hours Between", str(config.min_hours_between))
    table.add_row("Max Budget", f"${config.max_budget:.2f}")
    table.add_row("Background Mode", "Yes" if config.run_in_background else "No")

    table.add_row("", "")
    table.add_row("[bold]Current State[/bold]", "")
    table.add_row("Current Traces", str(current_traces))
    table.add_row("Traces at Last Opt", str(state.traces_at_last_optimization))

    new_traces = current_traces - state.traces_at_last_optimization
    table.add_row("New Traces", str(new_traces))

    if state.last_optimization:
        from datetime import datetime, UTC
        hours_ago = (datetime.now(UTC) - state.last_optimization).total_seconds() / 3600
        table.add_row("Last Optimization", f"{hours_ago:.1f} hours ago")
    else:
        table.add_row("Last Optimization", "Never")

    table.add_row("", "")
    table.add_row("[bold]Saved Optimization[/bold]", "")

    if saved:
        table.add_row("Saved Program", "[green]Yes[/green]")
        table.add_row("Demos", str(len(saved.demos)))
        if saved.result:
            table.add_row("Improvement", f"+{saved.result.improvement:.1f}%")
        table.add_row("File", str(OPTIMIZED_PROGRAM_FILE))
    else:
        table.add_row("Saved Program", "[dim]None[/dim]")

    console.print(table)

    # Show when next optimization will trigger
    if config.enabled:
        needed_traces = config.min_new_traces - new_traces
        if needed_traces > 0:
            console.print(f"\n[dim]Next auto-optimization: {needed_traces} more traces needed[/dim]")
        elif state.last_optimization:
            from datetime import datetime, UTC
            hours_since = (datetime.now(UTC) - state.last_optimization).total_seconds() / 3600
            hours_left = config.min_hours_between - hours_since
            if hours_left > 0:
                console.print(f"\n[dim]Next auto-optimization: {hours_left:.1f} hours cooldown remaining[/dim]")
            else:
                console.print("\n[green]Ready to optimize on next query![/green]")
        else:
            console.print("\n[green]Ready to optimize on next query![/green]")


@optimize_app.command("enable")
def optimize_enable() -> None:
    """Enable auto-optimization."""
    from pathlib import Path
    import yaml

    config_file = Path.home() / ".rlm" / "config.yaml"

    # Load existing config
    config = {}
    if config_file.exists():
        try:
            config = yaml.safe_load(config_file.read_text()) or {}
        except Exception:
            pass

    # Update optimization.enabled
    if "optimization" not in config:
        config["optimization"] = {}
    config["optimization"]["enabled"] = True

    # Save back
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

    console.print("[green]✓ Auto-optimization enabled[/green]")


@optimize_app.command("disable")
def optimize_disable() -> None:
    """Disable auto-optimization."""
    from pathlib import Path
    import yaml

    config_file = Path.home() / ".rlm" / "config.yaml"

    # Load existing config
    config = {}
    if config_file.exists():
        try:
            config = yaml.safe_load(config_file.read_text()) or {}
        except Exception:
            pass

    # Update optimization.enabled
    if "optimization" not in config:
        config["optimization"] = {}
    config["optimization"]["enabled"] = False

    # Save back
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

    console.print("[yellow]✓ Auto-optimization disabled[/yellow]")


@optimize_app.command("reset")
def optimize_reset(
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Reset optimization - clear saved program and state."""
    from .core.simba_optimizer import clear_optimization

    if not force:
        if not typer.confirm("Clear all saved optimization data?"):
            raise typer.Abort()

    cleared = clear_optimization()

    if cleared:
        console.print("[green]✓ Cleared saved optimization data[/green]")
    else:
        console.print("[dim]No optimization data to clear[/dim]")

"""CLI for RLM-DSPy.

Provides a command-line interface for recursive language model queries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .core.rlm import RLM, RLMConfig, RLMResult

app = typer.Typer(
    name="rlm-dspy",
    help="Recursive Language Models with DSPy optimization",
    no_args_is_help=True,
)
console = Console()


def _get_config(
    model: str | None = None,
    budget: float | None = None,
    timeout: float | None = None,
    chunk_size: int | None = None,
    strategy: str | None = None,
) -> RLMConfig:
    """Build config from CLI args and environment.

    Environment variables (all optional):
    - RLM_MODEL: Model name (default: openrouter/google/gemini-3-flash-preview)
    - RLM_API_BASE: API endpoint (default: https://openrouter.ai/api/v1)
    - RLM_API_KEY or OPENROUTER_API_KEY: API key
    - RLM_MAX_BUDGET: Max cost in USD
    - RLM_MAX_TIMEOUT: Max time in seconds
    - RLM_CHUNK_SIZE: Chunk size in chars
    - RLM_PARALLEL_CHUNKS: Max concurrent chunks
    """
    # RLMConfig reads from env by default, only override if CLI args provided
    config = RLMConfig()

    if model:
        config.model = model
        config.sub_model = model
    if budget:
        config.max_budget = budget
    if timeout:
        config.max_timeout = timeout
    if chunk_size:
        config.default_chunk_size = chunk_size
    if strategy:
        config.strategy = strategy  # type: ignore

    return config


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="The question to answer")],
    paths: Annotated[
        Optional[list[Path]],
        typer.Argument(help="Files or directories to analyze"),
    ] = None,
    stdin: Annotated[
        bool,
        typer.Option("--stdin", "-", help="Read context from stdin"),
    ] = False,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model to use"),
    ] = None,
    budget: Annotated[
        Optional[float],
        typer.Option("--budget", "-b", help="Max budget in USD"),
    ] = None,
    timeout: Annotated[
        Optional[float],
        typer.Option("--timeout", "-t", help="Max timeout in seconds"),
    ] = None,
    chunk_size: Annotated[
        Optional[int],
        typer.Option("--chunk-size", "-c", help="Chunk size in characters"),
    ] = None,
    strategy: Annotated[
        Optional[str],
        typer.Option("--strategy", "-s", help="Processing strategy: auto|map_reduce|iterative|hierarchical"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed progress"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Show full debug output with API calls"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Validate config without running query"),
    ] = False,
) -> None:
    """
    Ask a question about files or piped content.

    Examples:
        rlm-dspy ask "What does main() do?" src/
        cat large_file.txt | rlm-dspy ask "Summarize this" --stdin
        rlm-dspy ask "Find bugs" src/*.py --model claude-3-opus
        rlm-dspy ask "Explain" file.py -v  # verbose
        rlm-dspy ask "Debug" file.py -d    # full debug
        rlm-dspy ask "Test" file.py -n     # dry run (validate only)
    """
    import os

    # Set debug/verbose environment before importing debug module
    if debug:
        os.environ["RLM_DEBUG"] = "1"
        os.environ["RLM_VERBOSE"] = "1"
    elif verbose:
        os.environ["RLM_VERBOSE"] = "1"

    # Import and setup debug logging
    from .core.debug import debug_summary, is_debug, setup_logging, timer
    from .core.validation import preflight_check

    if verbose or debug:
        setup_logging()

    config = _get_config(model, budget, timeout, chunk_size, strategy)
    rlm = RLM(config=config)

    # Dry run mode - validate and exit
    if dry_run:
        console.print("[bold]Dry Run Mode - Validating configuration...[/bold]\n")

    if is_debug():
        console.print(
            Panel(
                f"[bold]Model:[/bold] {config.model}\n"
                f"[bold]API:[/bold] {config.api_base}\n"
                f"[bold]Chunk Size:[/bold] {config.default_chunk_size:,}\n"
                f"[bold]Strategy:[/bold] {config.strategy}",
                title="[bold blue]Debug Mode[/bold blue]",
                border_style="blue",
            )
        )

    # Load context
    if stdin:
        context = sys.stdin.read()
        if not context.strip():
            console.print("[red]Error: No input received from stdin[/red]")
            raise typer.Exit(1)
    elif paths:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Loading files...", total=None)
            context = rlm.load_context([str(p) for p in paths])
    else:
        console.print("[red]Error: Provide paths or use --stdin[/red]")
        raise typer.Exit(1)

    if verbose or debug:
        src = f"{len(paths or [])} path(s)" if paths else "stdin"
        console.print(f"\n[dim]Context: {len(context):,} chars from {src}[/dim]")

    # Dry run - run preflight checks and exit
    if dry_run:
        preflight = preflight_check(
            api_key_required=True,
            model=config.model,
            api_base=config.api_base,
            budget=config.max_budget,
            context=context,
            chunk_size=config.default_chunk_size,
            check_network=True,
        )
        preflight.print_report()

        if preflight.passed:
            console.print("\n[green]✓ Ready to run! Remove --dry-run to execute.[/green]")
            raise typer.Exit(0)
        else:
            raise typer.Exit(1)

    # Execute query
    with timer("Total query time", log=verbose or debug):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=not (verbose or debug),
        ) as progress:
            task = progress.add_task("Analyzing...", total=None)
            result = rlm.query(query, context)
            progress.update(task, description="Done!")

    # Output
    if output_json:
        print(
            json.dumps(
                {
                    "answer": result.answer,
                    "success": result.success,
                    "tokens": result.total_tokens,
                    "cost": result.total_cost,
                    "time": result.elapsed_time,
                    "error": result.error,
                },
                indent=2,
            )
        )
    else:
        if result.success:
            console.print(
                Panel(
                    Markdown(result.answer),
                    title="Answer",
                    border_style="green",
                )
            )
            if verbose or debug:
                _print_stats(result)
            if debug:
                debug_summary(
                    chunks_processed=result.chunks_processed,
                    chunks_relevant=result.chunks_with_relevant_info,
                    total_tokens=result.total_tokens,
                    total_cost=result.total_cost,
                    elapsed=result.elapsed_time,
                )
        else:
            console.print(f"[red]Error: {result.error}[/red]")
            if result.partial_answer:
                console.print(
                    Panel(
                        Markdown(result.partial_answer),
                        title="Partial Answer",
                        border_style="yellow",
                    )
                )
            raise typer.Exit(1)


@app.command()
def analyze(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Files or directories to analyze"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file for analysis"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model to use"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: markdown|json"),
    ] = "markdown",
) -> None:
    """
    Generate a comprehensive analysis of files.

    Creates a structured summary including:
    - File structure overview
    - Key components and their purposes
    - Dependencies and relationships
    - Potential issues or improvements
    """
    config = _get_config(model)
    rlm = RLM(config=config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading files...", total=None)
        context = rlm.load_context([str(p) for p in paths])

        progress.add_task("Analyzing structure...", total=None)
        structure = rlm.query(
            "List all files and their purposes in a structured format",
            context,
        )

        progress.add_task("Identifying components...", total=None)
        components = rlm.query(
            "Identify the main components, classes, and functions. Explain their roles.",
            context,
        )

        progress.add_task("Finding issues...", total=None)
        issues = rlm.query(
            "Find potential bugs, code smells, or areas for improvement.",
            context,
        )

    # Format output
    if format == "json":
        analysis = {
            "structure": structure.answer,
            "components": components.answer,
            "issues": issues.answer,
        }
        output_text = json.dumps(analysis, indent=2)
    else:
        output_text = f"""# Code Analysis

## File Structure
{structure.answer}

## Key Components
{components.answer}

## Issues & Improvements
{issues.answer}
"""

    if output:
        output.write_text(output_text)
        console.print(f"[green]Analysis saved to {output}[/green]")
    else:
        console.print(Markdown(output_text))


@app.command()
def diff(
    query: Annotated[str, typer.Argument(help="Question about the diff")],
    diff_file: Annotated[
        Optional[Path],
        typer.Option("--file", "-f", help="Diff file to analyze"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model to use"),
    ] = None,
) -> None:
    """
    Analyze a git diff.

    Examples:
        git diff | rlm-dspy diff "What changed?"
        rlm-dspy diff "Are there any bugs?" --file changes.diff
    """
    config = _get_config(model)
    rlm = RLM(config=config)

    if diff_file:
        context = diff_file.read_text()
    else:
        context = sys.stdin.read()

    if not context.strip():
        console.print("[red]Error: No diff content provided[/red]")
        raise typer.Exit(1)

    result = rlm.query(
        f"Analyze this git diff and answer: {query}\n\nDiff:\n{context}",
        context,
    )

    if result.success:
        console.print(Panel(Markdown(result.answer), title="Diff Analysis"))
    else:
        console.print(f"[red]Error: {result.error}[/red]")
        raise typer.Exit(1)


@app.command()
def compile(
    training_data: Annotated[
        Path,
        typer.Argument(help="Path to training data JSON"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for compiled prompts"),
    ] = Path("~/.rlm-dspy/compiled").expanduser(),
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model to use for compilation"),
    ] = None,
) -> None:
    """
    Compile optimized prompts from training examples.

    Training data should be a JSON file with examples:
    [
        {"query": "...", "context": "...", "answer": "..."},
        ...
    ]
    """
    import dspy
    from dspy.teleprompt import BootstrapFewShot

    from .core.programs import ChunkedProcessor

    config = _get_config(model)

    # Load training data
    with open(training_data) as f:
        examples = json.load(f)

    trainset = [
        dspy.Example(
            query=ex["query"],
            context=ex["context"],
            answer=ex["answer"],
        ).with_inputs("query", "context")
        for ex in examples
    ]

    console.print(f"[dim]Loaded {len(trainset)} training examples[/dim]")

    # Setup DSPy
    lm = dspy.LM(model=config.model, api_key=config.api_key)
    dspy.configure(lm=lm)

    # Compile
    program = ChunkedProcessor()
    optimizer = BootstrapFewShot(metric=lambda x, y: x.answer == y.answer)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Compiling prompts...", total=None)
        compiled = optimizer.compile(program, trainset=trainset)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled.save(str(output_dir / "chunked_processor.json"))

    console.print(f"[green]Compiled prompts saved to {output_dir}[/green]")


@app.command()
def config(
    show: Annotated[
        bool,
        typer.Option("--show", "-s", help="Show current configuration"),
    ] = True,
) -> None:
    """Show or edit configuration."""
    if show:
        table = Table(title="RLM-DSPy Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Env Var", style="dim")

        # Get actual config values
        cfg = RLMConfig()

        api_key_status = "***" if cfg.api_key else "[red](not set)[/red]"

        settings = [
            ("Model", cfg.model, "RLM_MODEL"),
            ("API Base", cfg.api_base, "RLM_API_BASE"),
            ("API Key", api_key_status, "RLM_API_KEY / OPENROUTER_API_KEY"),
            ("Max Budget", f"${cfg.max_budget:.2f}", "RLM_MAX_BUDGET"),
            ("Max Timeout", f"{cfg.max_timeout:.0f}s", "RLM_MAX_TIMEOUT"),
            ("Chunk Size", f"{cfg.default_chunk_size:,}", "RLM_CHUNK_SIZE"),
            ("Parallel Chunks", str(cfg.parallel_chunks), "RLM_PARALLEL_CHUNKS"),
            ("Disable Thinking", str(cfg.disable_thinking), "RLM_DISABLE_THINKING"),
            ("Enable Cache", str(cfg.enable_cache), "RLM_ENABLE_CACHE"),
            ("Use Async", str(cfg.use_async), "RLM_USE_ASYNC"),
        ]

        for name, value, env_var in settings:
            table.add_row(name, str(value), env_var)

        console.print(table)
        console.print("\n[dim]Set environment variables to override defaults.[/dim]")


def _print_stats(result: RLMResult) -> None:
    """Print execution statistics."""
    table = Table(title="Execution Stats", show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="cyan")

    table.add_row("Tokens", f"{result.total_tokens:,}")
    table.add_row("Cost", f"${result.total_cost:.4f}")
    table.add_row("Time", f"{result.elapsed_time:.1f}s")
    table.add_row("Chunks", f"{result.chunks_processed}")
    table.add_row("Depth", f"{result.depth_reached}")

    console.print(table)


@app.command()
def preflight(
    paths: Annotated[
        Optional[list[Path]],
        typer.Argument(help="Files or directories to check"),
    ] = None,
    check_network: Annotated[
        bool,
        typer.Option("--network/--no-network", help="Check API endpoint connectivity"),
    ] = True,
) -> None:
    """
    Run preflight checks to validate configuration.

    Validates:
    - API key is set
    - Model format is valid
    - API endpoint is reachable (optional)
    - Context size estimation

    Examples:
        rlm-dspy preflight src/
        rlm-dspy preflight --no-network
    """
    from .core.validation import preflight_check

    config = RLMConfig()

    # Load context if paths provided
    context = None
    if paths:
        rlm = RLM(config=config)
        context = rlm.load_context([str(p) for p in paths])
        console.print(f"[dim]Loaded {len(context):,} chars from {len(paths)} path(s)[/dim]\n")

    # Run checks
    result = preflight_check(
        api_key_required=True,
        model=config.model,
        api_base=config.api_base,
        budget=config.max_budget,
        context=context,
        chunk_size=config.default_chunk_size,
        check_network=check_network,
    )

    result.print_report()

    if result.passed:
        raise typer.Exit(0)
    else:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

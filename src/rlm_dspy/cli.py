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
from .core.fileutils import PathTraversalError, validate_path_safety

app = typer.Typer(
    name="rlm-dspy",
    help="Recursive Language Models with DSPy optimization",
    no_args_is_help=True,
)
console = Console()


def _safe_write_output(output_file: Path, content: str) -> None:
    """Write content to output file with path validation.
    
    Args:
        output_file: Target file path
        content: Content to write
        
    Raises:
        typer.Exit: If path validation fails
    """
    try:
        # Validate path for traversal attacks
        safe_path = validate_path_safety(output_file)
        # Ensure parent directory exists
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
    except PathTraversalError as e:
        console.print(f"[red]Security Error: {e}[/red]")
        raise typer.Exit(1)


def _get_config(
    model: str | None = None,
    sub_model: str | None = None,
    budget: float | None = None,
    timeout: float | None = None,
    max_iterations: int | None = None,
    verbose: bool = False,
) -> RLMConfig:
    """Build config from CLI args and environment.

    Environment variables (all optional):
    - RLM_MODEL: Model name (default: openai/gpt-4o-mini)
    - RLM_SUB_MODEL: Model for sub-queries (defaults to RLM_MODEL)
    - RLM_API_BASE: Custom API endpoint (optional, for self-hosted)
    - RLM_API_KEY or OPENROUTER_API_KEY: API key
    - RLM_MAX_BUDGET: Max cost in USD
    - RLM_MAX_TIMEOUT: Max time in seconds
    - RLM_MAX_ITERATIONS: Max REPL iterations
    - RLM_MAX_LLM_CALLS: Max sub-LLM calls per execution
    """
    # RLMConfig reads from env by default, only override if CLI args provided
    config = RLMConfig()

    if model:
        config.model = model
        # Only override sub_model with model if sub_model not explicitly set
        if not sub_model:
            config.sub_model = model
    if sub_model:
        config.sub_model = sub_model
    if budget:
        config.max_budget = budget
    if timeout:
        config.max_timeout = timeout
    if max_iterations:
        config.max_iterations = max_iterations
    if verbose:
        config.verbose = verbose

    return config


# =============================================================================
# Ask Command Helpers (reduce function length and nesting)
# =============================================================================

def _load_context(
    rlm: RLM,
    paths: list[Path] | None,
    stdin: bool,
    verbose: bool,
) -> str:
    """Load context from stdin or file paths.
    
    Args:
        rlm: RLM instance for loading files
        paths: List of file/directory paths
        stdin: Whether to read from stdin
        verbose: Show progress
        
    Returns:
        Loaded context string
        
    Raises:
        typer.Exit: On error
    """
    if stdin:
        context = sys.stdin.read()
        if not context.strip():
            console.print("[red]Error: No input received from stdin[/red]")
            raise typer.Exit(1)
        return context
    
    if paths:
        # Check for missing paths
        missing = [p for p in paths if not p.exists()]
        for p in missing:
            console.print(f"[yellow]Warning: Path not found: {p}[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Loading files...", total=None)
            context = rlm.load_context([str(p) for p in paths])
        
        if not context.strip():
            console.print("[red]Error: No content loaded from provided paths[/red]")
            raise typer.Exit(1)
        return context
    
    console.print("[red]Error: Provide paths or use --stdin[/red]")
    raise typer.Exit(1)


def _run_dry_run(config: RLMConfig, context: str) -> None:
    """Run preflight checks for dry-run mode.
    
    Args:
        config: RLM configuration
        context: Loaded context
        
    Raises:
        typer.Exit: Always exits after checks
    """
    from .core.validation import preflight_check
    
    preflight = preflight_check(
        api_key_required=True,
        model=config.model,
        api_base=config.api_base,
        budget=config.max_budget,
        context=context,
        check_network=True,
    )
    preflight.print_report()
    
    if preflight.passed:
        console.print("\n[green]✓ Ready to run! Remove --dry-run to execute.[/green]")
        raise typer.Exit(0)
    else:
        raise typer.Exit(1)


def _output_result(
    result: RLMResult,
    output_json: bool,
    output_file: Path | None,
    verbose: bool,
    debug: bool,
) -> None:
    """Handle result output in various formats.
    
    Args:
        result: Query result
        output_json: Output as JSON
        output_file: File to write to
        verbose: Show stats
        debug: Show trajectory
        
    Raises:
        typer.Exit: On error
    """
    if output_json:
        json_output = json.dumps(
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
        if output_file:
            _safe_write_output(output_file, json_output)
            console.print(f"[green]Output written to {output_file}[/green]")
        else:
            print(json_output)
        return
    
    if output_file:
        if result.success:
            _safe_write_output(output_file, result.answer)
            console.print(f"[green]Output written to {output_file}[/green]")
            if verbose or debug:
                _print_stats(result)
        else:
            console.print(f"[red]Error: {result.error or 'Unknown error'}[/red]")
            raise typer.Exit(1)
        return
    
    # Console output
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
        if debug and result.trajectory:
            _print_trajectory(result.trajectory)
    else:
        console.print(f"[red]Error: {result.error or 'Unknown error'}[/red]")
        if debug and result.trajectory:
            _print_trajectory(result.trajectory[-3:], "Partial trajectory before failure")
        raise typer.Exit(1)


def _print_trajectory(trajectory: list, title: str = "Trajectory") -> None:
    """Print query trajectory for debugging."""
    console.print(f"\n[bold]{title}:[/bold]")
    for i, step in enumerate(trajectory, 1):
        console.print(f"[dim]Step {i}:[/dim]")
        if isinstance(step, dict):
            if "code" in step:
                console.print(f"  [cyan]Code:[/cyan] {step['code'][:200]}...")
            if "output" in step:
                console.print(f"  [green]Output:[/green] {step['output'][:200]}...")


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
        typer.Option("--model", "-m", help="Model to use for reasoning"),
    ] = None,
    sub_model: Annotated[
        Optional[str],
        typer.Option("--sub-model", "-s", help="Model for llm_query() calls (cheaper)"),
    ] = None,
    budget: Annotated[
        Optional[float],
        typer.Option("--budget", "-b", help="Max budget in USD"),
    ] = None,
    timeout: Annotated[
        Optional[float],
        typer.Option("--timeout", "-t", help="Max timeout in seconds"),
    ] = None,
    max_iterations: Annotated[
        Optional[int],
        typer.Option("--max-iterations", "-i", help="Max REPL iterations"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON"),
    ] = False,
    output_file: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Write output to file"),
    ] = None,
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

    Uses DSPy's RLM (Recursive Language Model) to explore your context
    through a Python REPL. The LLM writes code to navigate, analyze,
    and build up answers iteratively.

    HOW IT WORKS:
        1. Your files are loaded as the 'context' variable in a REPL
        2. The LLM writes Python code to explore the context
        3. It can call llm_query() for semantic analysis of sections
        4. It iterates until it has enough info, then calls SUBMIT()

    EXAMPLES:
        rlm-dspy ask "What does main() do?" src/
        rlm-dspy ask "Summarize this" --stdin < large_file.txt
        rlm-dspy ask "Find bugs" src/*.py --model anthropic/claude-sonnet-4
        rlm-dspy ask "Explain" file.py -v           # verbose progress
        rlm-dspy ask "Debug" file.py -d             # full debug output
        rlm-dspy ask "Test" file.py -n              # dry run (validate only)
        rlm-dspy ask "Analyze" src/ -o report.md    # write to file
        rlm-dspy ask "Analyze" src/ -j -o data.json # JSON output to file
    """
    import os

    # Set debug/verbose environment before importing debug module
    if debug:
        os.environ["RLM_DEBUG"] = "1"
        os.environ["RLM_VERBOSE"] = "1"
    elif verbose:
        os.environ["RLM_VERBOSE"] = "1"

    # Import and setup debug logging
    from .core.debug import is_debug, setup_logging, timer
    from .core.validation import preflight_check

    if verbose or debug:
        setup_logging()

    config = _get_config(model, sub_model, budget, timeout, max_iterations, verbose or debug)
    rlm = RLM(config=config)

    # Show debug info
    if is_debug():
        console.print(
            Panel(
                f"[bold]Model:[/bold] {config.model}\n"
                f"[bold]Sub Model:[/bold] {config.sub_model}\n"
                f"[bold]Max Iterations:[/bold] {config.max_iterations}\n"
                f"[bold]Max LLM Calls:[/bold] {config.max_llm_calls}",
                title="[bold blue]Debug Mode (RLM)[/bold blue]",
                border_style="blue",
            )
        )

    # Load context from stdin or paths
    context = _load_context(rlm, paths, stdin, verbose or debug)
    
    if verbose or debug:
        src = f"{len(paths or [])} path(s)" if paths else "stdin"
        console.print(f"\n[dim]Context: {len(context):,} chars from {src}[/dim]")

    # Dry run - validate and exit
    if dry_run:
        console.print("[bold]Dry Run Mode - Validating configuration...[/bold]\n")
        _run_dry_run(config, context)

    # Execute query with progress
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

    # Output result
    _output_result(result, output_json, output_file, verbose, debug)


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

    # Check for failures
    results = [("structure", structure), ("components", components), ("issues", issues)]
    for name, result in results:
        if not result.success:
            console.print(f"[red]Error analyzing {name}: {result.error}[/red]")
            raise typer.Exit(1)

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
        _safe_write_output(output, output_text)
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
        console.print(f"[red]Error: {result.error or 'Unknown error'}[/red]")
        raise typer.Exit(1)


@app.command()
def setup(
    env_file: Annotated[
        Optional[Path],
        typer.Option("--env-file", "-e", help="Path to .env file with API keys"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Default model (e.g., openai/gpt-4o)"),
    ] = None,
    budget: Annotated[
        Optional[float],
        typer.Option("--budget", "-b", help="Default max budget in USD"),
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option("--interactive/--no-interactive", "-i", help="Interactive setup wizard"),
    ] = True,
) -> None:
    """
    Configure RLM-DSPy settings.

    Creates ~/.rlm/config.yaml with your preferences.

    Examples:
        rlm-dspy setup                           # Interactive wizard
        rlm-dspy setup --env-file ~/.claude/.env # Set env file location
        rlm-dspy setup --model deepseek/deepseek-chat --budget 0.50
    """
    from .core.rlm import PROVIDER_API_KEYS
    from .core.user_config import (
        CONFIG_FILE,
        get_config_status,
        load_config,
        load_env_file,
        save_config,
    )

    config = load_config()
    changed = False

    if interactive and not (env_file or model or budget):
        # Interactive wizard
        console.print(Panel.fit(
            "[bold cyan]RLM-DSPy Setup Wizard[/bold cyan]\n\n"
            "This will create a config file at ~/.rlm/config.yaml",
            title="Welcome"
        ))

        # Step 1: Env file
        console.print("\n[bold]Step 1: Environment File[/bold]")
        console.print("Where are your API keys stored? (e.g., ~/.claude/.env)")
        console.print("[dim]Press Enter to skip if you use environment variables directly.[/dim]")

        env_input = typer.prompt("Env file path", default=config.get("env_file") or "", show_default=False)
        if env_input:
            env_path = Path(env_input).expanduser()
            if env_path.exists():
                config["env_file"] = str(env_path)
                changed = True
                # Load and show what keys were found
                loaded = load_env_file(env_path)
                if loaded:
                    console.print(f"[green]✓ Found {len(loaded)} environment variables[/green]")
                    for key in loaded:
                        if "KEY" in key or "TOKEN" in key:
                            console.print(f"  • {key}")
            else:
                console.print(f"[yellow]⚠ File not found: {env_path}[/yellow]")

        # Step 2: Default model
        console.print("\n[bold]Step 2: Default Model[/bold]")
        console.print("Available providers:")
        providers = sorted(set(p.rstrip("/") for p in PROVIDER_API_KEYS.keys() if p))
        for i, p in enumerate(providers, 1):
            console.print(f"  {i}. {p}/")

        model_input = typer.prompt(
            "Default model",
            default=config.get("model", "openai/gpt-4o-mini"),
        )
        if model_input != config.get("model"):
            config["model"] = model_input
            changed = True

        # Step 3: Budget
        console.print("\n[bold]Step 3: Default Budget[/bold]")
        budget_input = typer.prompt(
            "Max budget (USD)",
            default=str(config.get("max_budget", 1.0)),
        )
        try:
            budget_val = float(budget_input)
            if budget_val <= 0:
                console.print("[yellow]⚠ Budget must be positive, using default.[/yellow]")
            elif budget_val != config.get("max_budget"):
                config["max_budget"] = budget_val
                changed = True
        except ValueError:
            console.print("[yellow]⚠ Invalid number, using default.[/yellow]")

    else:
        # Non-interactive: apply provided options
        if env_file:
            env_path = Path(env_file).expanduser()
            if env_path.exists():
                config["env_file"] = str(env_path)
                changed = True
            else:
                console.print(f"[red]Error: File not found: {env_path}[/red]")
                raise typer.Exit(1)

        if model:
            if "/" not in model:
                console.print(
                    f"[yellow]⚠ Model should include provider prefix "
                    f"(e.g., openai/{model})[/yellow]"
                )
            config["model"] = model
            changed = True

        if budget is not None:
            if budget <= 0:
                console.print("[red]Error: Budget must be positive[/red]")
                raise typer.Exit(1)
            config["max_budget"] = budget
            changed = True

    # Save config
    if changed:
        save_config(config)
        console.print(f"\n[green]✓ Configuration saved to {CONFIG_FILE}[/green]")
    else:
        console.print("\n[dim]No changes made.[/dim]")

    # Show status
    status = get_config_status()

    console.print("\n[bold]Current Status:[/bold]")
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="dim")
    table.add_column("Value")

    table.add_row("Config file", str(CONFIG_FILE) if CONFIG_FILE.exists() else "[dim]not created[/dim]")
    table.add_row("Env file", status["env_file"] or "[dim]not set[/dim]")
    table.add_row("Model", status["model"])
    if status["api_key_found"]:
        table.add_row("API key", "[green]✓ found[/green]")
    else:
        table.add_row("API key", f"[red]✗ set {status['api_key_env_var']}[/red]")

    console.print(table)

    if status["is_configured"]:
        console.print("\n[green]✓ RLM-DSPy is ready to use![/green]")
        console.print("[dim]Try: rlm-dspy ask 'What does this code do?' ./src[/dim]")
    else:
        console.print(f"\n[yellow]⚠ Set {status['api_key_env_var']} to complete setup.[/yellow]")


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
            ("Sub Model", cfg.sub_model, "RLM_SUB_MODEL"),
            ("API Base", cfg.api_base or "(auto)", "RLM_API_BASE"),
            ("API Key", api_key_status, "RLM_API_KEY / OPENROUTER_API_KEY"),
            ("Max Iterations", str(cfg.max_iterations), "RLM_MAX_ITERATIONS"),
            ("Max LLM Calls", str(cfg.max_llm_calls), "RLM_MAX_LLM_CALLS"),
            ("Max Output Chars", f"{cfg.max_output_chars:,}", "RLM_MAX_OUTPUT_CHARS"),
            ("Verbose", str(cfg.verbose), "RLM_VERBOSE"),
            ("Max Budget", f"${cfg.max_budget:.2f}", "RLM_MAX_BUDGET"),
            ("Max Timeout", f"{cfg.max_timeout:.0f}s", "RLM_MAX_TIMEOUT"),
        ]

        for name, value, env_var in settings:
            table.add_row(name, str(value), env_var)

        console.print(table)
        console.print("\n[bold]RLM Mode:[/bold] REPL-based exploration (dspy.RLM)")
        console.print("[dim]The LLM writes Python code to explore your context.[/dim]")
        console.print("[dim]Set environment variables to override defaults.[/dim]")


def _print_stats(result: RLMResult) -> None:
    """Print execution statistics."""
    table = Table(title="Execution Stats", show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="cyan")

    table.add_row("Time", f"{result.elapsed_time:.1f}s")
    table.add_row("Iterations", f"{result.iterations}")
    if result.total_tokens > 0:
        table.add_row("Tokens", f"{result.total_tokens:,}")
    if result.total_cost > 0:
        table.add_row("Cost", f"${result.total_cost:.4f}")
    if result.final_reasoning:
        table.add_row("Final Reasoning", result.final_reasoning[:100] + "..." if len(result.final_reasoning) > 100 else result.final_reasoning)

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
        check_network=check_network,
    )

    result.print_report()

    if result.passed:
        raise typer.Exit(0)
    else:
        raise typer.Exit(1)


@app.command()
def index(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Files or directories to index"),
    ],
    kind: Annotated[
        Optional[str],
        typer.Option("--kind", "-k", help="Filter by kind: class, function, method"),
    ] = None,
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Filter by name (substring match)"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON"),
    ] = False,
) -> None:
    """
    Index code structure using tree-sitter (100% accurate).

    Unlike LLM queries, this uses AST parsing for exact results.
    No hallucination - perfect for finding exact line numbers.

    Examples:
        rlm-dspy index src/                     # All definitions
        rlm-dspy index src/ --kind class        # Only classes
        rlm-dspy index src/ --name "Config"     # Search by name
        rlm-dspy index src/ -k method -n init   # Methods containing "init"
    """
    from .core.ast_index import index_file

    # Collect all files
    all_files: list[Path] = []
    for p in paths:
        if p.is_file():
            all_files.append(p)
        elif p.is_dir():
            for ext in [".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".rb"]:
                all_files.extend(p.rglob(f"*{ext}"))

    # Index all files
    from .core.ast_index import Definition
    all_defs: list[Definition] = []
    for f in sorted(all_files):
        idx = index_file(f)
        all_defs.extend(idx.definitions)

    # Filter
    results = all_defs
    if kind:
        results = [d for d in results if d.kind == kind]
    if name:
        results = [d for d in results if name.lower() in d.name.lower()]

    if not results:
        console.print("[yellow]No definitions found[/yellow]")
        raise typer.Exit(0)

    if json_output:
        import json
        data = [
            {"name": d.name, "kind": d.kind, "line": d.line, "file": d.file, "parent": d.parent}
            for d in results
        ]
        console.print(json.dumps(data, indent=2))
    else:
        table = Table(title=f"Code Index ({len(results)} definitions)")
        table.add_column("Kind", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Line", style="yellow", justify="right")
        table.add_column("File", style="dim")

        for d in results:
            display_name = f"{d.parent}.{d.name}" if d.parent else d.name
            table.add_row(d.kind, display_name, str(d.line), d.file)

        console.print(table)


# Example prompts for different use cases
EXAMPLE_PROMPTS = {
    "understand": {
        "title": "🔍 Code Understanding",
        "description": "Understand what code does and how it works",
        "prompts": [
            ("Overview", "What does this codebase do? Summarize the main components."),
            ("Specific", "Explain what the {class/function} does and how to use it"),
            ("Architecture", "How is the code organized? What are the main modules?"),
            ("Data flow", "Trace the flow: what happens when a user calls {function}()?"),
        ],
    },
    "bugs": {
        "title": "🐛 Bug Finding",
        "description": "Find bugs, edge cases, and error handling issues",
        "prompts": [
            ("General", "Find potential bugs, edge cases, or error conditions"),
            (
                "Specific",
                "Check for: 1) Division by zero 2) Null dereferences "
                "3) Unhandled exceptions 4) Race conditions",
            ),
            ("Input", "Find places where user input isn't validated or sanitized"),
            ("Resources", "Find resource leaks: unclosed files, connections, memory"),
            (
                "Trace",
                "In function {X}, trace where variable {Y} comes from, "
                "what produces it, and verify if {Y}['key'] can raise KeyError",
            ),
            (
                "Evidence",
                "For each exception handler, quote the exact logging line. "
                "If no logging exists, say 'no logging found'",
            ),
        ],
    },
    "dead-code": {
        "title": "🗑️ Dead Code Detection",
        "description": "Find unused modules, functions, and exports",
        "prompts": [
            (
                "Modules",
                "For each module, check if it's actually imported and used by the main "
                "code. List any modules that are exported but never used.",
            ),
            ("Functions", "List functions that are defined but never called"),
            ("Exports", "Check __init__.py - which exports are never imported?"),
            (
                "Trace",
                "For function {X}, trace all callers. Show the call chain from "
                "entry points (main, CLI, public API) to this function.",
            ),
        ],
    },
    "security": {
        "title": "🔒 Security Review",
        "description": "Find security vulnerabilities",
        "prompts": [
            (
                "General",
                "Find security vulnerabilities: injection, hardcoded secrets, "
                "unsafe deserialization, path traversal",
            ),
            ("Auth", "Review auth. Are there any bypass vulnerabilities?"),
            ("Secrets", "Find hardcoded API keys, passwords, or secrets"),
            ("Input", "Find where external input reaches dangerous functions"),
            (
                "Trace",
                "Trace user input from {entry_point} through all functions until "
                "it reaches a dangerous sink (exec, SQL, file ops). Show full path.",
            ),
        ],
    },
    "review": {
        "title": "📝 Code Review",
        "description": "Comprehensive code review",
        "prompts": [
            (
                "Full",
                "Review for: 1) Bugs 2) Performance 3) Security "
                "4) Code smells 5) Missing error handling",
            ),
            ("Senior", "Review like a senior engineer. What would you flag?"),
            ("Best practices", "Does this follow best practices? Suggest improvements."),
        ],
    },
    "performance": {
        "title": "⚡ Performance",
        "description": "Find performance bottlenecks",
        "prompts": [
            ("Algorithms", "Find O(n²) algorithms, repeated computations, allocations"),
            ("Database", "Find N+1 queries, missing indexes, inefficient DB access"),
            ("Memory", "Find memory leaks, large allocations, objects kept alive"),
            (
                "Trace",
                "For the hot path starting at {entry_point}, trace each function call, "
                "count loop iterations, and identify the most expensive operations.",
            ),
        ],
    },
    "refactor": {
        "title": "🔄 Refactoring",
        "description": "Find code that needs refactoring",
        "prompts": [
            ("Duplicates", "Find duplicate code that could be shared functions"),
            ("Smells", "Find code smells: long functions, deep nesting, god classes"),
            ("Simplify", "What code could be simplified? Find overly complex parts."),
            (
                "Trace",
                "For class {X}, list all its dependencies (imports, calls) and all "
                "dependents (who uses it). Can it be safely refactored?",
            ),
        ],
    },
    "docs": {
        "title": "📚 Documentation",
        "description": "Find documentation gaps",
        "prompts": [
            ("Missing", "Which public functions are missing docstrings?"),
            ("Outdated", "Find docs that don't match actual code behavior"),
            ("Generate", "Generate documentation for the public API"),
        ],
    },
}


@app.command()
def example(
    case: Annotated[
        Optional[str],
        typer.Argument(help="Use case: understand, bugs, dead-code, security, review, performance, refactor, docs"),
    ] = None,
) -> None:
    """
    Show example prompts for different use cases.

    Run without arguments to see all available cases.

    Examples:
        rlm-dspy example                # List all cases
        rlm-dspy example bugs           # Show bug-finding prompts
        rlm-dspy example dead-code      # Show dead code detection prompts
        rlm-dspy example security       # Show security review prompts
    """
    if case is None:
        # List all cases
        console.print("\n[bold]Available Example Cases:[/bold]\n")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Case", style="green")
        table.add_column("Description")

        for key, data in EXAMPLE_PROMPTS.items():
            table.add_row(key, data["description"])

        console.print(table)
        console.print("\n[dim]Usage: rlm-dspy example <case>[/dim]")
        console.print("[dim]Example: rlm-dspy example bugs[/dim]\n")
        return

    if case not in EXAMPLE_PROMPTS:
        console.print(f"[red]Unknown case: {case}[/red]")
        console.print(f"[dim]Available: {', '.join(EXAMPLE_PROMPTS.keys())}[/dim]")
        raise typer.Exit(1)

    data = EXAMPLE_PROMPTS[case]
    console.print(f"\n[bold]{data['title']}[/bold]")
    console.print(f"[dim]{data['description']}[/dim]\n")

    for name, prompt in data["prompts"]:
        console.print(f"[cyan]{name}:[/cyan]")
        console.print(Panel(
            f"rlm-dspy ask \"{prompt}\" src/",
            border_style="dim",
        ))
        console.print()


if __name__ == "__main__":
    app()

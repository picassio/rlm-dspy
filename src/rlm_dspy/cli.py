"""CLI for RLM-DSPy.

Provides a command-line interface for recursive language model queries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional, TYPE_CHECKING

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Lazy imports for faster CLI startup (~3s saved by deferring DSPy import)
if TYPE_CHECKING:
    from .core.rlm import RLM, RLMConfig, RLMResult

# Import lightweight fileutils directly without going through core/__init__.py
# This avoids triggering the DSPy import chain
import importlib
_fileutils = importlib.import_module("rlm_dspy.core.fileutils")
PathTraversalError = _fileutils.PathTraversalError
validate_path_safety = _fileutils.validate_path_safety

app = typer.Typer(
    name="rlm-dspy",
    help="Recursive Language Models with DSPy optimization",
    no_args_is_help=True,
)
console = Console()


# Type aliases for common CLI options - use these in function signatures to reduce duplication
ModelOpt = Annotated[Optional[str], typer.Option("--model", "-m", help="Model to use for reasoning")]
SubModelOpt = Annotated[Optional[str], typer.Option("--sub-model", "-s", help="Model for llm_query() calls")]
BudgetOpt = Annotated[Optional[float], typer.Option("--budget", "-b", help="Max budget in USD")]
TimeoutOpt = Annotated[Optional[float], typer.Option("--timeout", "-t", help="Max timeout in seconds")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed progress")]
FormatOpt = Annotated[Optional[str], typer.Option("--format", "-f", help="Output format: text, json, markdown")]
OutputFileOpt = Annotated[Optional[Path], typer.Option("--output", "-o", help="Write output to file")]


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
    except PermissionError:
        console.print(f"[red]Permission denied: Cannot write to {output_file}[/red]")
        raise typer.Exit(1)
    except OSError as e:
        console.print(f"[red]Failed to write file: {e}[/red]")
        raise typer.Exit(1)


def _get_config(
    model: str | None = None,
    sub_model: str | None = None,
    budget: float | None = None,
    timeout: float | None = None,
    max_iterations: int | None = None,
    max_workers: int | None = None,
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
    # Validate numeric arguments
    if budget is not None and budget <= 0:
        console.print("[red]Error: --budget must be positive[/red]")
        raise typer.Exit(1)
    if timeout is not None and timeout <= 0:
        console.print("[red]Error: --timeout must be positive[/red]")
        raise typer.Exit(1)
    if max_iterations is not None and max_iterations <= 0:
        console.print("[red]Error: --max-iterations must be positive[/red]")
        raise typer.Exit(1)
    if max_workers is not None and max_workers <= 0:
        console.print("[red]Error: --max-workers must be positive[/red]")
        raise typer.Exit(1)

    # Lazy import to avoid 3s DSPy startup for non-query commands
    from .core.rlm import RLMConfig

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
    if max_workers:
        config.max_workers = max_workers
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
    max_tokens: int | None = None,
    use_cache: bool = True,
) -> str:
    """Load context from stdin or file paths.

    Args:
        rlm: RLM instance for loading files
        paths: List of file/directory paths
        stdin: Whether to read from stdin
        verbose: Show progress
        max_tokens: Optional token limit for context
        use_cache: Whether to use context caching

    Returns:
        Loaded context string

    Raises:
        typer.Exit: On error
    """
    if stdin:
        # Warn if stdin appears to be a TTY (interactive, not piped)
        if sys.stdin.isatty():
            console.print("[yellow]Reading from stdin (press Ctrl+D when done, or Ctrl+C to cancel)...[/yellow]")

        # Read with size limit to prevent memory exhaustion (50MB max)
        MAX_STDIN_SIZE = 50 * 1024 * 1024  # 50MB
        context = sys.stdin.read(MAX_STDIN_SIZE + 1)
        if len(context) > MAX_STDIN_SIZE:
            console.print(f"[red]Error: stdin input too large (>{MAX_STDIN_SIZE // 1024 // 1024}MB)[/red]")
            raise typer.Exit(1)

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
            task_desc = "Loading files..."
            if max_tokens:
                task_desc = f"Loading files (max {max_tokens:,} tokens)..."
            progress.add_task(task_desc, total=None)
            context = rlm.load_context(
                [str(p) for p in paths],
                max_tokens=max_tokens,
                use_cache=use_cache,
            )

        if not context.strip():
            console.print("[red]Error: No content loaded from provided paths[/red]")
            raise typer.Exit(1)

        # Set current project for semantic_search tool (auto-detect from first path)
        from .tools import set_current_project
        first_path = paths[0].resolve()
        if first_path.is_file():
            first_path = first_path.parent
        set_current_project(str(first_path))

        if verbose and max_tokens:
            from .core.fileutils import estimate_tokens
            actual_tokens = estimate_tokens(context)
            console.print(f"[dim]Context: ~{actual_tokens:,} tokens (limit: {max_tokens:,})[/dim]")

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
        console.print("\n[red]✗ Preflight checks failed. Fix the issues above and try again.[/red]")
        raise typer.Exit(1)


def _output_result(
    result: RLMResult,
    output_format: str,
    output_file: Path | None,
    verbose: bool,
    debug: bool,
) -> None:
    """Handle result output in various formats.

    Args:
        result: Query result
        output_format: Output format (text, json, markdown)
        output_file: File to write to
        verbose: Show stats
        debug: Show trajectory

    Raises:
        typer.Exit: On error
    """
    if output_format == "json":
        output_data = {
            "answer": result.answer,
            "success": result.success,
            "tokens": result.total_tokens,
            "cost": result.total_cost,
            "time": result.elapsed_time,
            "error": result.error,
        }
        # Include structured outputs if present (custom signatures)
        if result.outputs:
            output_data["outputs"] = result.outputs
        json_output = json.dumps(output_data, indent=2)
        if output_file:
            _safe_write_output(output_file, json_output)
            console.print(f"[green]Output written to {output_file}[/green]")
        else:
            print(json_output)
        return

    # Markdown format
    if output_format == "markdown":
        md_output = _format_as_markdown(result)
        if output_file:
            _safe_write_output(output_file, md_output)
            console.print(f"[green]Markdown written to {output_file}[/green]")
        else:
            print(md_output)
        return

    # Text format (default) to file
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
        if result.outputs:
            # Structured output from custom signature
            table = Table(title="Structured Output", show_header=True, header_style="bold cyan")
            table.add_column("Field", style="cyan")
            table.add_column("Value")

            for key, value in result.outputs.items():
                if isinstance(value, list):
                    value_str = "\n".join(f"• {v}" for v in value) if value else "(empty)"
                elif isinstance(value, bool):
                    value_str = "[green]Yes[/green]" if value else "[red]No[/red]"
                else:
                    value_str = str(value)
                table.add_row(key, value_str)

            console.print(table)
        else:
            # Standard string answer
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
        error_msg = result.error or "Unknown error (use --debug for more details)"
        console.print(f"[red]Error: {error_msg}[/red]")
        if debug and result.trajectory:
            _print_trajectory(result.trajectory[-3:], "Partial trajectory before failure")
        raise typer.Exit(1)


def _format_as_markdown(result: RLMResult) -> str:
    """Format result as Markdown."""
    lines = ["# Analysis Result\n"]

    if not result.success:
        lines.append(f"**Error:** {result.error or 'Unknown error'}\n")
        return "\n".join(lines)

    if result.outputs:
        # Structured output from custom signature
        # Include summary answer if present and different from outputs
        if result.answer and result.answer.strip():
            lines.append("## Summary\n")
            lines.append(result.answer)
            lines.append("")

        for key, value in result.outputs.items():
            # Convert key from snake_case to Title Case
            title = key.replace("_", " ").title()
            lines.append(f"## {title}\n")

            if isinstance(value, list):
                for item in value:
                    lines.append(f"- {item}")
                lines.append("")
            elif isinstance(value, bool):
                lines.append(f"{'Yes' if value else 'No'}\n")
            else:
                lines.append(f"{value}\n")
    else:
        lines.append(result.answer)

    # Add stats
    lines.append("\n---\n")
    lines.append(f"*Time: {result.elapsed_time:.1f}s | Iterations: {result.iterations}*")

    return "\n".join(lines)


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
    model: ModelOpt = None,
    sub_model: SubModelOpt = None,
    budget: BudgetOpt = None,
    timeout: TimeoutOpt = None,
    max_iterations: Annotated[
        Optional[int],
        typer.Option("--max-iterations", "-i", help="Max REPL iterations (default 20, min 20, max 100)"),
    ] = None,
    signature: Annotated[
        Optional[str],
        typer.Option(
            "--signature", "-S",
            help="Output signature: security, bugs, review, architecture, performance, diff"
        ),
    ] = None,
    output_format: FormatOpt = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", "-j", help="Shorthand for --format json"),
    ] = False,
    output_file: OutputFileOpt = None,
    verbose: VerboseOpt = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Show full debug output with API calls"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Validate config without running query"),
    ] = False,
    validate: Annotated[
        bool,
        typer.Option("--validate/--no-validate", "-V", help="Check output for hallucinations (enabled by default)"),
    ] = True,
    no_tools: Annotated[
        bool,
        typer.Option("--no-tools", help="Disable code analysis tools (ripgrep, tree-sitter)"),
    ] = False,
    max_tokens: Annotated[
        Optional[int],
        typer.Option("--max-tokens", "-T", help="Truncate context to fit token limit"),
    ] = None,
    max_workers: Annotated[
        Optional[int],
        typer.Option("--max-workers", "-w", help="Max parallel workers for batch ops"),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable context caching"),
    ] = False,
) -> None:
    """
    Ask a question about files or piped content.

    Uses DSPy's RLM (Recursive Language Model) to explore your context
    through a Python REPL. The LLM writes code to navigate, analyze,
    and build up answers iteratively.

    BEST FOR LARGE CONTEXTS:
        RLM shines when analyzing codebases too large to fit in a single
        LLM context window. For small files (<50KB), a direct LLM query
        may be faster. For large codebases (>200KB), RLM is essential.

    HOW IT WORKS:
        1. Your files are loaded as the 'context' variable in a REPL
        2. The LLM writes Python code to explore the context
        3. It uses tools (ripgrep, tree-sitter) for accurate code search
        4. It can call llm_query() for semantic analysis of sections
        5. It iterates until it has enough info, then calls SUBMIT()

    BUILT-IN TOOLS (enabled by default):
        The LLM automatically uses these tools for accurate results:
        - index_code, find_classes, find_functions: AST-based search (exact line numbers)
        - ripgrep, find_calls: Fast pattern search across all files
        - read_file: Read specific file sections
        Use --no-tools to disable if needed.

    EXAMPLES:
        rlm-dspy ask "Describe the architecture" src/          # Best for large codebases
        rlm-dspy ask "Find all security issues" src/           # Cross-file analysis
        rlm-dspy ask "Trace the data flow from input to output" src/
        rlm-dspy ask "Summarize this" --stdin < large_file.txt
        rlm-dspy ask "Find bugs" src/*.py --model anthropic/claude-sonnet-4
        rlm-dspy ask "Explain" file.py -v           # verbose progress
        rlm-dspy ask "Debug" file.py -d             # full debug output
        rlm-dspy ask "Test" file.py -n              # dry run (validate only)
        rlm-dspy ask "Analyze" src/ -o report.md    # write to file
        rlm-dspy ask "Analyze" src/ -j -o data.json # JSON output to file
        rlm-dspy ask "Simple query" src/ --no-tools # disable tools
        rlm-dspy ask "Analyze" src/ -T 50000        # limit to 50K tokens
        rlm-dspy ask "Query" src/ --no-cache        # disable context caching
        rlm-dspy ask "Batch" src/ -w 8              # 8 parallel workers
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

    if verbose or debug:
        setup_logging()

    config = _get_config(model, sub_model, budget, timeout, max_iterations, max_workers, verbose or debug)

    # Resolve signature
    sig = "context, query -> answer"  # default
    if signature:
        from .signatures import get_signature, list_signatures
        sig_class = get_signature(signature)
        if sig_class is None:
            console.print(f"[red]Unknown signature: {signature}[/red]")
            console.print(f"[dim]Available: {', '.join(list_signatures())}[/dim]")
            raise typer.Exit(1)
        sig = sig_class

    # Lazy import RLM to avoid 3s DSPy startup for non-query commands
    from .core.rlm import RLM
    rlm = RLM(config=config, signature=sig, use_tools=not no_tools)

    # Show debug info
    if is_debug():
        debug_info = [
            f"[bold]Model:[/bold] {config.model}",
            f"[bold]Sub Model:[/bold] {config.sub_model}",
            f"[bold]Max Iterations:[/bold] {config.max_iterations}",
            f"[bold]Max Workers:[/bold] {config.max_workers}",
        ]
        if max_tokens:
            debug_info.append(f"[bold]Max Tokens:[/bold] {max_tokens:,}")
        if no_cache:
            debug_info.append("[bold]Cache:[/bold] disabled")
        console.print(
            Panel(
                "\n".join(debug_info),
                title="[bold blue]Debug Mode (RLM)[/bold blue]",
                border_style="blue",
            )
        )

    # Load context from stdin or paths
    context = _load_context(
        rlm, paths, stdin, verbose or debug,
        max_tokens=max_tokens, use_cache=not no_cache
    )

    if verbose or debug:
        src = f"{len(paths or [])} path(s)" if paths else "stdin"
        cache_status = "" if not no_cache else " (no cache)"
        console.print(f"\n[dim]Context: {len(context):,} chars from {src}{cache_status}[/dim]")

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
            try:
                result = rlm.query(query, context)
            except KeyboardInterrupt:
                console.print("\n[yellow]Query interrupted by user[/yellow]")
                raise typer.Exit(130)  # Standard exit code for SIGINT
            except Exception as e:
                console.print(f"\n[red]Query failed: {e}[/red]")
                if debug:
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
                raise typer.Exit(1)
            progress.update(task, description="Done!")

    # Validate output for hallucinations if requested (uses LLM-as-judge)
    if validate and result.success:
        from .guards import validate_groundedness

        console.print("\n[dim]Validating output with LLM-as-judge...[/dim]")
        validation = validate_groundedness(result.answer, context, query)

        # Record outcome for optimization learning
        try:
            from .core.trace_collector import get_trace_collector
            from .core.grounded_proposer import get_grounded_proposer

            collector = get_trace_collector()
            proposer = get_grounded_proposer()

            # Extract trace info from result metadata if available
            reasoning_steps = result.metadata.get("reasoning_steps", []) if result.metadata else []
            code_blocks = result.metadata.get("code_blocks", []) if result.metadata else []
            outputs = result.metadata.get("outputs", []) if result.metadata else []

            if validation.is_grounded:
                # Record successful trace for bootstrapping
                collector.record(
                    query=query,
                    reasoning_steps=reasoning_steps,
                    code_blocks=code_blocks,
                    outputs=outputs,
                    final_answer=result.answer,
                    grounded_score=validation.score,
                    query_type=signature if signature else None,
                )
                proposer.record_success(
                    query=query,
                    query_type=signature if signature else "general",
                    grounded_score=validation.score,
                    tools_used=result.metadata.get("tools_used", []) if result.metadata else [],
                )
            else:
                # Record failure for tip generation
                proposer.record_failure(
                    query=query,
                    query_type=signature if signature else "general",
                    failure_reason=validation.discussion[:200] if validation.discussion else "ungrounded",
                    grounded_score=validation.score,
                    ungrounded_claims=[],  # Could extract from validation
                    tools_used=result.metadata.get("tools_used", []) if result.metadata else [],
                )
        except Exception as e:
            # Don't fail the query if optimization recording fails
            if verbose:
                console.print(f"[dim]Warning: Failed to record optimization data: {e}[/dim]")

        if not validation.is_grounded:
            console.print(f"\n[yellow]⚠ Potential hallucinations detected ({validation.score:.0%} grounded)[/yellow]")
            console.print(f"[dim]{validation.discussion[:500]}[/dim]")
        else:
            console.print(f"\n[green]✓ Output validated ({validation.score:.0%} grounded)[/green]")

    # Resolve output format (--json is shorthand for --format json)
    resolved_format = output_format or ("json" if output_json else "text")

    # Output result
    _output_result(result, resolved_format, output_file, verbose, debug)


@app.command()
def analyze(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Files or directories to analyze"),
    ],
    output: OutputFileOpt = None,
    model: ModelOpt = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: markdown|json"),
    ] = "markdown",
    sequential: Annotated[
        bool,
        typer.Option("--sequential", "-s", help="Run queries sequentially (slower)"),
    ] = False,
) -> None:
    """
    Generate a comprehensive analysis of files.

    Creates a structured summary including:
    - File structure overview
    - Key components and their purposes
    - Dependencies and relationships
    - Potential issues or improvements

    By default, runs 3 analysis queries in parallel for ~3x faster results.
    Use --sequential for debugging or if you encounter issues.
    """
    from .core.rlm import RLM
    config = _get_config(model)
    rlm = RLM(config=config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading files...", total=None)
        context = rlm.load_context([str(p) for p in paths])

        if sequential:
            # Sequential mode (old behavior)
            try:
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

                results_list = [structure, components, issues]
            except KeyboardInterrupt:
                console.print("\n[yellow]Analysis interrupted by user[/yellow]")
                raise typer.Exit(130)
            except Exception as e:
                console.print(f"\n[red]Analysis failed: {e}[/red]")
                raise typer.Exit(1)
        else:
            # Parallel mode (default, ~3x faster)
            progress.add_task("Analyzing in parallel (structure, components, issues)...", total=None)
            results_list = rlm.batch([
                {"query": "List all files and their purposes in a structured format"},
                {"query": "Identify the main components, classes, and functions. Explain their roles."},
                {"query": "Find potential bugs, code smells, or areas for improvement."},
            ], context=context, num_threads=3, return_failed=True)

    # Unpack results
    if len(results_list) != 3:
        console.print(f"[red]Error: Expected 3 results, got {len(results_list)}[/red]")
        raise typer.Exit(1)

    structure, components, issues = results_list

    # Check for failures
    named_results = [("structure", structure), ("components", components), ("issues", issues)]
    for name, result in named_results:
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
    model: ModelOpt = None,
) -> None:
    """
    Analyze a git diff.

    Examples:
        git diff | rlm-dspy diff "What changed?"
        rlm-dspy diff "Are there any bugs?" --file changes.diff
    """
    from .core.rlm import RLM
    config = _get_config(model)
    rlm = RLM(config=config)

    if diff_file:
        try:
            context = diff_file.read_text()
        except FileNotFoundError:
            console.print(f"[red]Error: File not found: {diff_file}[/red]")
            raise typer.Exit(1)
        except PermissionError:
            console.print(f"[red]Error: Permission denied: {diff_file}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            raise typer.Exit(1)
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
    install_lsp: Annotated[
        bool,
        typer.Option("--install-lsp/--no-lsp", help="Install all available LSP servers"),
    ] = False,
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
        rlm-dspy setup --install-lsp             # Also install all LSP servers
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
    
    # Install LSP servers if requested
    if install_lsp:
        console.print("\n[bold]Installing LSP servers...[/bold]")
        from .core.lsp_installer import SERVERS, install_server, check_requirements
        
        success = 0
        failed = 0
        skipped = 0
        
        for server_id, info in SERVERS.items():
            reqs_met, missing = check_requirements(server_id)
            if not reqs_met:
                console.print(f"  [dim]○ {info.name} - skipped (missing: {', '.join(missing)})[/dim]")
                skipped += 1
                continue
            
            console.print(f"  Installing {info.name}...", end=" ")
            if install_server(server_id):
                console.print("[green]✓[/green]")
                success += 1
            else:
                console.print("[red]✗[/red]")
                failed += 1
        
        console.print(f"\n[green]LSP servers installed: {success}[/green]")
        if failed:
            console.print(f"[red]Failed: {failed}[/red]")
        if skipped:
            console.print(f"[dim]Skipped (missing requirements): {skipped}[/dim]")


@app.command()
def config(
    show: Annotated[
        bool,
        typer.Option("--show", "-s", help="Show current configuration"),
    ] = True,
) -> None:
    """Show or edit configuration."""
    if show:
        from .core.rlm import RLMConfig
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
        reasoning_display = (
            result.final_reasoning[:100] + "..."
            if len(result.final_reasoning) > 100
            else result.final_reasoning
        )
        table.add_row("Final Reasoning", reasoning_display)

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
    from .core.rlm import RLM, RLMConfig

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


# =============================================================================
# Index Commands (for semantic search)
# =============================================================================

index_app = typer.Typer(
    name="index",
    help="Manage vector indexes for semantic search",
    no_args_is_help=True,
)
app.add_typer(index_app, name="index")


@index_app.command("build")
def index_build(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Directories to index"),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force full rebuild"),
    ] = False,
) -> None:
    """Build or update vector index for semantic search.

    Examples:
        rlm-dspy index build src/
        rlm-dspy index build . --force
    """
    from .core.vector_index import get_index_manager

    manager = get_index_manager()

    for path in paths:
        if not path.exists():
            console.print(f"[red]Path not found: {path}[/red]")
            continue

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"Indexing {path}...", total=None)

            try:
                count = manager.build(path, force=force)
                console.print(f"[green]✓[/green] Indexed {path}: {count} code snippets")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to index {path}: {e}")


@index_app.command("status")
def index_status(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Directories to check"),
    ],
) -> None:
    """Show index status for directories.

    Examples:
        rlm-dspy index status src/
    """
    from .core.vector_index import get_index_manager

    manager = get_index_manager()

    for path in paths:
        status = manager.get_status(path)

        if not status["indexed"]:
            console.print(f"[yellow]○[/yellow] {path}: Not indexed")
            console.print("  [dim]Run: rlm-dspy index build {path}[/dim]")
        else:
            console.print(f"[green]●[/green] {path}: Indexed")
            console.print(f"  Snippets: {status['snippet_count']}")
            console.print(f"  Files: {status['file_count']}")
            if status["needs_update"]:
                changed = status['new_or_modified']
                deleted = status['deleted']
                console.print(f"  [yellow]Needs update: {changed} changed, {deleted} deleted[/yellow]")
            else:
                console.print("  [green]Up to date[/green]")


@index_app.command("clear")
def index_clear(
    path: Annotated[
        Optional[Path],
        typer.Argument(help="Specific directory to clear (or all if not specified)"),
    ] = None,
    confirm: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation"),
    ] = False,
) -> None:
    """Clear vector indexes.

    Examples:
        rlm-dspy index clear           # Clear all indexes
        rlm-dspy index clear src/      # Clear specific index
        rlm-dspy index clear -y        # Skip confirmation
    """
    from .core.vector_index import get_index_manager

    if not confirm:
        if path:
            msg = f"Clear index for {path}?"
        else:
            msg = "Clear ALL indexes?"

        if not typer.confirm(msg):
            console.print("[dim]Cancelled[/dim]")
            return

    manager = get_index_manager()
    count = manager.clear(path)

    if count > 0:
        console.print(f"[green]✓[/green] Cleared {count} index(es)")
    else:
        console.print("[dim]No indexes to clear[/dim]")


@index_app.command("search")
def index_search(
    query: Annotated[
        str,
        typer.Argument(help="Search query"),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", "-p", help="Directory to search"),
    ] = Path("."),
    k: Annotated[
        int,
        typer.Option("--results", "-k", help="Number of results"),
    ] = 5,
) -> None:
    """Search code semantically.

    Examples:
        rlm-dspy index search "authentication logic"
        rlm-dspy index search "error handling" -p src/ -k 10
    """
    from .core.vector_index import get_index_manager

    manager = get_index_manager()

    # Build index if needed
    status = manager.get_status(path)
    if not status["indexed"]:
        console.print(f"[dim]Building index for {path}...[/dim]")
        manager.build(path)

    # Search
    results = manager.search(path, query, k=k)

    if not results:
        console.print(f"[yellow]No results found for: {query}[/yellow]")
        return

    console.print(f"\n[bold]Results for:[/bold] {query}\n")

    for i, r in enumerate(results, 1):
        console.print(f"[bold cyan]{i}. {r.snippet.file}:{r.snippet.line}[/bold cyan]")
        console.print(f"   [dim]{r.snippet.type}[/dim] [green]{r.snippet.name}[/green]")

        # Show snippet preview
        preview = r.snippet.text[:200].replace('\n', '\n   ')
        if len(r.snippet.text) > 200:
            preview += "..."
        console.print(f"   {preview}")
        console.print()


@index_app.command("compress")
def index_compress(
    path: Annotated[
        Optional[Path],
        typer.Argument(help="Project path to compress (or all if not specified)"),
    ] = None,
    decompress: Annotated[
        bool,
        typer.Option("--decompress", "-d", help="Decompress instead of compress"),
    ] = False,
) -> None:
    """Compress indexes to reduce disk usage.

    Uses float16 quantization + gzip for ~4x compression ratio.

    Examples:
        rlm-dspy index compress           # Compress all indexes
        rlm-dspy index compress .         # Compress current project
        rlm-dspy index compress -d .      # Decompress current project
    """
    from .core.index_compression import (
        compress_index,
        decompress_index,
        get_index_size,
        is_compressed,
        CompressionStats,
    )
    from .core.project_registry import get_project_registry

    registry = get_project_registry()

    if path:
        # Single project - find by path
        resolved = path.resolve()
        # Find project by matching path
        project = None
        for p in registry.list():
            if Path(p.path).resolve() == resolved:
                project = p
                break
        if project:
            projects = [(resolved, registry.get_index_path(project.name))]
        else:
            console.print(f"[yellow]Project not found in registry: {path}[/yellow]")
            console.print("[dim]Use 'rlm-dspy index build' first[/dim]")
            return
    else:
        # All projects
        projects = [
            (Path(p.path), registry.get_index_path(p.name))
            for p in registry.list()
        ]

    if not projects:
        console.print("[dim]No indexed projects found[/dim]")
        return

    total_original = 0
    total_compressed = 0

    for project_path, index_path in projects:
        if not index_path.exists():
            continue

        name = project_path.name

        if decompress:
            if not is_compressed(index_path):
                console.print(f"[dim]{name}: not compressed[/dim]")
                continue

            console.print(f"[dim]Decompressing {name}...[/dim]")
            count = decompress_index(index_path)
            console.print(f"[green]✓[/green] {name}: decompressed {count} files")
        else:
            if is_compressed(index_path):
                console.print(f"[dim]{name}: already compressed[/dim]")
                continue

            original_size = get_index_size(index_path)
            console.print(f"[dim]Compressing {name}...[/dim]")
            stats = compress_index(index_path)
            
            total_original += stats.original_size
            total_compressed += stats.compressed_size
            
            console.print(f"[green]✓[/green] {name}: {stats}")

    if not decompress and total_original > 0:
        total_saved = total_original - total_compressed
        total_ratio = total_original / total_compressed if total_compressed > 0 else 1
        console.print(f"\n[bold]Total:[/bold] {CompressionStats._format_size(total_saved)} saved ({total_ratio:.1f}x ratio)")


# =============================================================================
# Project Commands (for multi-project management)
# =============================================================================

project_app = typer.Typer(
    name="project",
    help="Manage indexed projects",
    no_args_is_help=True,
)
app.add_typer(project_app, name="project")


@project_app.command("add")
def project_add(
    name: Annotated[
        str,
        typer.Argument(help="Project name"),
    ],
    path: Annotated[
        Path,
        typer.Argument(help="Path to project directory"),
    ],
    alias: Annotated[
        Optional[str],
        typer.Option("--alias", "-a", help="Short alias for the project"),
    ] = None,
    tags: Annotated[
        Optional[str],
        typer.Option("--tags", "-t", help="Comma-separated tags"),
    ] = None,
) -> None:
    """Register a project for indexing.

    Examples:
        rlm-dspy project add my-app ~/projects/my-app
        rlm-dspy project add backend ./backend --alias be --tags python,api
    """
    from .core.project_registry import get_project_registry

    registry = get_project_registry()

    try:
        tag_list = [t.strip() for t in tags.split(",")] if tags else []
        project = registry.add(name, path, alias=alias, tags=tag_list)
        console.print(f"[green]✓[/green] Registered project '{name}' at {project.path}")

        if alias:
            console.print(f"  Alias: {alias}")
        if tag_list:
            console.print(f"  Tags: {', '.join(tag_list)}")

        console.print(f"\n[dim]Run 'rlm-dspy index build {path}' to index the project.[/dim]")

    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)


@project_app.command("list")
def project_list(
    tags: Annotated[
        Optional[str],
        typer.Option("--tags", "-t", help="Filter by tags (comma-separated)"),
    ] = None,
    sort: Annotated[
        str,
        typer.Option("--sort", "-s", help="Sort by: name, indexed_at, snippet_count"),
    ] = "name",
) -> None:
    """List all registered projects.

    Examples:
        rlm-dspy project list
        rlm-dspy project list --tags python
        rlm-dspy project list --sort snippet_count
    """
    from .core.project_registry import get_project_registry

    registry = get_project_registry()
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    projects = registry.list(tags=tag_list, sort_by=sort)

    if not projects:
        console.print("[dim]No projects registered.[/dim]")
        console.print("[dim]Use 'rlm-dspy project add <name> <path>' to register a project.[/dim]")
        return

    default = registry.get_default()

    table = Table(title="Registered Projects")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Snippets", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Tags", style="green")
    table.add_column("Indexed", style="dim")

    for p in projects:
        name = f"* {p.name}" if default and p.name == default.name else p.name
        indexed = ""
        if p.indexed_at:
            from datetime import datetime
            dt = datetime.fromisoformat(p.indexed_at)
            indexed = dt.strftime("%Y-%m-%d %H:%M")

        table.add_row(
            name,
            p.path,
            str(p.snippet_count) if p.snippet_count else "-",
            str(p.file_count) if p.file_count else "-",
            ", ".join(p.tags) if p.tags else "-",
            indexed or "-",
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(projects)} project(s)[/dim]")
    if default:
        console.print("[dim]* = default project[/dim]")


@project_app.command("remove")
def project_remove(
    name: Annotated[
        str,
        typer.Argument(help="Project name to remove"),
    ],
    delete_index: Annotated[
        bool,
        typer.Option("--delete-index", "-d", help="Also delete the index files"),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation"),
    ] = False,
) -> None:
    """Remove a project from the registry.

    Examples:
        rlm-dspy project remove my-app
        rlm-dspy project remove my-app --delete-index
    """
    from .core.project_registry import get_project_registry

    registry = get_project_registry()

    project = registry.get(name)
    if not project:
        console.print(f"[red]✗[/red] Project '{name}' not found")
        raise typer.Exit(1)

    if not confirm:
        msg = f"Remove project '{name}'"
        if delete_index:
            msg += " and delete index"
        msg += "?"

        if not typer.confirm(msg):
            console.print("[dim]Cancelled[/dim]")
            return

    registry.remove(name, delete_index=delete_index)
    console.print(f"[green]✓[/green] Removed project '{name}'")
    if delete_index:
        console.print("  Index files deleted")


@project_app.command("default")
def project_default(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Project name to set as default"),
    ] = None,
) -> None:
    """Set or show the default project.

    Examples:
        rlm-dspy project default           # Show current default
        rlm-dspy project default my-app    # Set default
    """
    from .core.project_registry import get_project_registry

    registry = get_project_registry()

    if name is None:
        default = registry.get_default()
        if default:
            console.print(f"Default project: [cyan]{default.name}[/cyan]")
            console.print(f"  Path: {default.path}")
        else:
            console.print("[dim]No default project set[/dim]")
        return

    try:
        registry.set_default(name)
        console.print(f"[green]✓[/green] Set default project to '{name}'")
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)


@project_app.command("tag")
def project_tag(
    name: Annotated[
        str,
        typer.Argument(help="Project name"),
    ],
    tags: Annotated[
        str,
        typer.Argument(help="Tags to add (comma-separated)"),
    ],
) -> None:
    """Add tags to a project.

    Examples:
        rlm-dspy project tag my-app python,web,api
    """
    from .core.project_registry import get_project_registry

    registry = get_project_registry()
    tag_list = [t.strip() for t in tags.split(",")]

    try:
        registry.tag(name, tag_list)
        console.print(f"[green]✓[/green] Added tags to '{name}': {', '.join(tag_list)}")
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)


@project_app.command("cleanup")
def project_cleanup(
    confirm: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation"),
    ] = False,
) -> None:
    """Remove orphaned index directories.

    Finds index directories that don't have a registered project
    and offers to delete them.

    Examples:
        rlm-dspy project cleanup
        rlm-dspy project cleanup -y
    """
    from .core.project_registry import get_project_registry

    registry = get_project_registry()
    orphaned = registry.find_orphaned()

    if not orphaned:
        console.print("[green]✓[/green] No orphaned indexes found")
        return

    console.print(f"Found {len(orphaned)} orphaned index(es):\n")
    for path in orphaned:
        console.print(f"  [dim]{path}[/dim]")

    if not confirm:
        console.print()
        if not typer.confirm("Delete these orphaned indexes?"):
            console.print("[dim]Cancelled[/dim]")
            return

    registry.cleanup_orphaned(dry_run=False)
    console.print(f"\n[green]✓[/green] Removed {len(orphaned)} orphaned index(es)")


@project_app.command("migrate")
def project_migrate() -> None:
    """Migrate legacy hash-based indexes to named projects.

    Scans for index directories with hash names and registers
    them as projects using the directory name from their manifest.

    Examples:
        rlm-dspy project migrate
    """
    from .core.project_registry import get_project_registry

    registry = get_project_registry()
    migrated = registry.migrate_legacy()

    if not migrated:
        console.print("[dim]No legacy indexes to migrate[/dim]")
        return

    console.print(f"[green]✓[/green] Migrated {len(migrated)} index(es):\n")
    for old_hash, new_name in migrated:
        console.print(f"  {old_hash} → [cyan]{new_name}[/cyan]")


# =============================================================================
# Daemon Commands (for automatic background indexing)
# =============================================================================

daemon_app = typer.Typer(
    name="daemon",
    help="Manage background index daemon",
    no_args_is_help=True,
)
app.add_typer(daemon_app, name="daemon")


@daemon_app.command("start")
def daemon_start(
    foreground: Annotated[
        bool,
        typer.Option("--foreground", "-f", help="Run in foreground (don't daemonize)"),
    ] = False,
) -> None:
    """Start the index daemon.

    The daemon watches registered projects for file changes and
    automatically updates their indexes.

    Examples:
        rlm-dspy daemon start              # Start in background
        rlm-dspy daemon start --foreground # Run in foreground
    """
    from .core.daemon import IndexDaemon, is_daemon_running, get_daemon_pid

    if is_daemon_running():
        pid = get_daemon_pid()
        console.print(f"[yellow]Daemon already running (PID: {pid})[/yellow]")
        return

    if foreground:
        console.print("[dim]Starting daemon in foreground (Ctrl+C to stop)...[/dim]")
        daemon = IndexDaemon()
        daemon.run_forever()
    else:
        console.print("[dim]Starting daemon in background...[/dim]")
        daemon = IndexDaemon()
        daemon.start(daemonize=True)
        console.print(f"[green]✓[/green] Daemon started (PID: {get_daemon_pid()})")
        console.print(f"[dim]Log file: {daemon.config.log_file}[/dim]")


@daemon_app.command("stop")
def daemon_stop() -> None:
    """Stop the index daemon.

    Examples:
        rlm-dspy daemon stop
    """
    from .core.daemon import stop_daemon, is_daemon_running, get_daemon_pid

    if not is_daemon_running():
        console.print("[dim]Daemon not running[/dim]")
        return

    pid = get_daemon_pid()
    console.print(f"[dim]Stopping daemon (PID: {pid})...[/dim]")

    if stop_daemon():
        console.print("[green]✓[/green] Daemon stopped")
    else:
        console.print("[red]✗[/red] Failed to stop daemon")
        raise typer.Exit(1)


@daemon_app.command("status")
def daemon_status() -> None:
    """Show daemon status.

    Examples:
        rlm-dspy daemon status
    """
    from .core.daemon import is_daemon_running, get_daemon_pid, DaemonConfig

    if not is_daemon_running():
        console.print("[yellow]○[/yellow] Daemon not running")
        console.print("[dim]Run 'rlm-dspy daemon start' to start[/dim]")
        return

    pid = get_daemon_pid()
    config = DaemonConfig.from_user_config()

    console.print(f"[green]●[/green] Daemon running (PID: {pid})")
    console.print(f"  Log file: {config.log_file}")
    console.print(f"  PID file: {config.pid_file}")

    # Show watched projects
    from .core.project_registry import get_project_registry
    registry = get_project_registry()
    watched = [p for p in registry.list() if p.auto_watch]

    if watched:
        console.print(f"\n  Watching {len(watched)} project(s):")
        for p in watched:
            console.print(f"    - {p.name}")


@daemon_app.command("watch")
def daemon_watch(
    project: Annotated[
        str,
        typer.Argument(help="Project name to watch"),
    ],
) -> None:
    """Add a project to the daemon watch list.

    The daemon must be running for this to take effect immediately.
    Otherwise, the project will be watched when the daemon starts.

    Examples:
        rlm-dspy daemon watch my-app
    """
    from .core.project_registry import get_project_registry
    from .core.daemon import is_daemon_running

    registry = get_project_registry()
    proj = registry.get(project)

    if not proj:
        console.print(f"[red]✗[/red] Project '{project}' not found")
        console.print("[dim]Use 'rlm-dspy project add' to register it first[/dim]")
        raise typer.Exit(1)

    # Set auto_watch flag
    proj.auto_watch = True
    registry._save()

    console.print(f"[green]✓[/green] Project '{project}' will be watched")

    if not is_daemon_running():
        console.print("[dim]Note: Daemon not running. Start with 'rlm-dspy daemon start'[/dim]")
    else:
        console.print("[dim]Daemon will pick up the change automatically[/dim]")


@daemon_app.command("unwatch")
def daemon_unwatch(
    project: Annotated[
        str,
        typer.Argument(help="Project name to stop watching"),
    ],
) -> None:
    """Remove a project from the daemon watch list.

    Examples:
        rlm-dspy daemon unwatch my-app
    """
    from .core.project_registry import get_project_registry

    registry = get_project_registry()
    proj = registry.get(project)

    if not proj:
        console.print(f"[red]✗[/red] Project '{project}' not found")
        raise typer.Exit(1)

    proj.auto_watch = False
    registry._save()

    console.print(f"[green]✓[/green] Project '{project}' will no longer be watched")


@daemon_app.command("list")
def daemon_list() -> None:
    """List projects being watched by the daemon.

    Examples:
        rlm-dspy daemon list
    """
    from .core.project_registry import get_project_registry
    from .core.daemon import is_daemon_running

    registry = get_project_registry()
    watched = [p for p in registry.list() if p.auto_watch]

    if not watched:
        console.print("[dim]No projects configured for watching[/dim]")
        console.print("[dim]Use 'rlm-dspy daemon watch <project>' to add one[/dim]")
        return

    running = is_daemon_running()
    status = "[green]●[/green] running" if running else "[yellow]○[/yellow] stopped"

    console.print(f"Daemon: {status}\n")

    table = Table(title="Watched Projects")
    table.add_column("Project", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Snippets", justify="right")

    for p in watched:
        table.add_row(p.name, p.path, str(p.snippet_count) if p.snippet_count else "-")

    console.print(table)


@daemon_app.command("log")
def daemon_log(
    lines: Annotated[
        int,
        typer.Option("--lines", "-n", help="Number of lines to show"),
    ] = 50,
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="Follow log output (like tail -f)"),
    ] = False,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="Clear the log file"),
    ] = False,
) -> None:
    """View the daemon log file.

    Examples:
        rlm-dspy daemon log              # Show last 50 lines
        rlm-dspy daemon log -n 100       # Show last 100 lines
        rlm-dspy daemon log -f           # Follow log output
        rlm-dspy daemon log --clear      # Clear the log
    """
    from .core.daemon import DaemonConfig

    config = DaemonConfig.from_user_config()
    log_file = config.log_file

    if clear:
        if log_file.exists():
            log_file.write_text("")
            console.print(f"[green]✓[/green] Cleared log file: {log_file}")
        else:
            console.print("[dim]Log file doesn't exist[/dim]")
        return

    if not log_file.exists():
        console.print(f"[dim]No log file found at {log_file}[/dim]")
        console.print("[dim]Start the daemon to create logs[/dim]")
        return

    if follow:
        import subprocess
        console.print(f"[dim]Following {log_file} (Ctrl+C to stop)...[/dim]\n")
        try:
            subprocess.run(["tail", "-f", str(log_file)])
        except KeyboardInterrupt:
            pass
        return

    # Read last N lines
    content = log_file.read_text()
    all_lines = content.splitlines()

    if not all_lines:
        console.print("[dim]Log file is empty[/dim]")
        return

    # Show last N lines
    display_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

    console.print(f"[dim]Showing last {len(display_lines)} of {len(all_lines)} lines from {log_file}[/dim]\n")

    for line in display_lines:
        # Colorize based on log level
        if " - ERROR - " in line or "ERROR" in line:
            console.print(f"[red]{line}[/red]")
        elif " - WARNING - " in line or "WARNING" in line:
            console.print(f"[yellow]{line}[/yellow]")
        elif " - INFO - " in line:
            console.print(line)
        else:
            console.print(f"[dim]{line}[/dim]")


# =============================================================================
# Traces commands (trace collection for bootstrapping)
# =============================================================================

traces_app = typer.Typer(
    name="traces",
    help="Manage collected REPL traces for few-shot bootstrapping",
    no_args_is_help=True,
)
app.add_typer(traces_app, name="traces")


@traces_app.command("list")
def traces_list(
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Max traces to show"),
    ] = 20,
    query_type: Annotated[
        Optional[str],
        typer.Option("--type", "-t", help="Filter by query type"),
    ] = None,
) -> None:
    """List collected traces."""
    from .core.trace_collector import get_trace_collector

    collector = get_trace_collector()
    traces = collector.traces

    if query_type:
        traces = [t for t in traces if t.query_type == query_type]

    if not traces:
        console.print("[dim]No traces collected yet[/dim]")
        return

    table = Table(title=f"Collected Traces ({len(traces)} total)")
    table.add_column("ID", style="dim")
    table.add_column("Type")
    table.add_column("Query", max_width=40)
    table.add_column("Score", justify="right")
    table.add_column("Tools")

    for trace in traces[-limit:]:
        table.add_row(
            trace.trace_id[:12],
            trace.query_type,
            trace.query[:40] + "..." if len(trace.query) > 40 else trace.query,
            f"{trace.grounded_score:.0%}",
            ", ".join(trace.tools_used[:3]) + ("..." if len(trace.tools_used) > 3 else ""),
        )

    console.print(table)


@traces_app.command("show")
def traces_show(
    trace_id: Annotated[str, typer.Argument(help="Trace ID (or prefix)")],
) -> None:
    """Show details of a specific trace."""
    from .core.trace_collector import get_trace_collector

    collector = get_trace_collector()

    # Find trace by ID prefix
    matches = [t for t in collector.traces if t.trace_id.startswith(trace_id)]

    if not matches:
        console.print(f"[red]No trace found with ID starting with '{trace_id}'[/red]")
        raise typer.Exit(1)

    if len(matches) > 1:
        console.print(f"[yellow]Multiple matches, showing first:[/yellow]")

    trace = matches[0]

    console.print(Panel(
        f"[bold]Query:[/bold] {trace.query}\n"
        f"[bold]Type:[/bold] {trace.query_type}\n"
        f"[bold]Score:[/bold] {trace.grounded_score:.0%}\n"
        f"[bold]Tools:[/bold] {', '.join(trace.tools_used)}\n"
        f"[bold]Timestamp:[/bold] {trace.timestamp}",
        title=f"Trace {trace.trace_id}",
    ))

    console.print("\n[bold]Formatted Demo:[/bold]")
    console.print(trace.format_as_demo())


@traces_app.command("stats")
def traces_stats() -> None:
    """Show trace collection statistics."""
    from .core.trace_collector import get_trace_collector

    collector = get_trace_collector()
    stats = collector.get_stats()

    if stats["total"] == 0:
        console.print("[dim]No traces collected yet[/dim]")
        return

    table = Table(title="Trace Statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Total Traces", str(stats["total"]))
    table.add_row("Average Score", f"{stats['avg_score']:.1%}")
    table.add_row("Min Score", f"{stats['min_score']:.1%}")
    table.add_row("Max Score", f"{stats['max_score']:.1%}")

    console.print(table)

    if stats["by_type"]:
        type_table = Table(title="By Query Type")
        type_table.add_column("Type")
        type_table.add_column("Count", justify="right")

        for qtype, count in sorted(stats["by_type"].items(), key=lambda x: x[1], reverse=True):
            type_table.add_row(qtype, str(count))

        console.print(type_table)


@traces_app.command("export")
def traces_export(
    output: Annotated[Path, typer.Argument(help="Output file path")],
) -> None:
    """Export traces to a JSON file."""
    from .core.trace_collector import get_trace_collector

    collector = get_trace_collector()
    count = collector.export(output)

    console.print(f"[green]✓[/green] Exported {count} traces to {output}")


@traces_app.command("import")
def traces_import(
    input_file: Annotated[Path, typer.Argument(help="Input file path")],
) -> None:
    """Import traces from a JSON file."""
    from .core.trace_collector import get_trace_collector

    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    collector = get_trace_collector()
    count = collector.import_traces(input_file)

    console.print(f"[green]✓[/green] Imported {count} traces from {input_file}")


@traces_app.command("clear")
def traces_clear(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation"),
    ] = False,
) -> None:
    """Clear all collected traces."""
    from .core.trace_collector import get_trace_collector

    collector = get_trace_collector()
    count = len(collector.traces)

    if count == 0:
        console.print("[dim]No traces to clear[/dim]")
        return

    if not force:
        confirm = typer.confirm(f"Delete {count} traces?")
        if not confirm:
            raise typer.Abort()

    deleted = collector.clear()
    console.print(f"[green]✓[/green] Cleared {deleted} traces")


# =============================================================================
# Optimize commands (instruction optimization)
# =============================================================================

optimize_app = typer.Typer(
    name="optimize",
    help="Optimize tool instructions and prompts",
    no_args_is_help=True,
)
app.add_typer(optimize_app, name="optimize")


@optimize_app.command("stats")
def optimize_stats() -> None:
    """Show optimization statistics."""
    from .core.instruction_optimizer import get_instruction_optimizer
    from .core.grounded_proposer import get_grounded_proposer

    optimizer = get_instruction_optimizer()
    proposer = get_grounded_proposer()

    opt_stats = optimizer.get_stats()
    prop_stats = proposer.get_stats()

    # Optimizer stats
    table = Table(title="Instruction Optimizer")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Total Outcomes", str(opt_stats["total_outcomes"]))

    console.print(table)

    if opt_stats["by_key"]:
        key_table = Table(title="By Instruction Key")
        key_table.add_column("Key")
        key_table.add_column("Total", justify="right")
        key_table.add_column("Success Rate", justify="right")
        key_table.add_column("Avg Score", justify="right")

        for key, stats in opt_stats["by_key"].items():
            key_table.add_row(
                key,
                str(stats["total"]),
                f"{stats['success_rate']:.0%}",
                f"{stats['avg_score']:.1%}",
            )

        console.print(key_table)

    # Proposer stats
    console.print()
    prop_table = Table(title="Grounded Proposer")
    prop_table.add_column("Metric")
    prop_table.add_column("Value", justify="right")

    prop_table.add_row("Total Failures", str(prop_stats["total_failures"]))
    prop_table.add_row("Total Successes", str(prop_stats["total_successes"]))
    prop_table.add_row("Current Tips", str(prop_stats["current_tips_count"]))
    prop_table.add_row("Queries Since Refresh", str(prop_stats["queries_since_refresh"]))

    console.print(prop_table)


@optimize_app.command("tips")
def optimize_tips(
    regenerate: Annotated[
        bool,
        typer.Option("--regenerate", "-r", help="Regenerate tips from history"),
    ] = False,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Reset to default tips"),
    ] = False,
) -> None:
    """Show or regenerate optimization tips."""
    from .core.grounded_proposer import get_grounded_proposer

    proposer = get_grounded_proposer()

    if reset:
        proposer.reset_tips()
        console.print("[green]✓[/green] Reset to default tips")

    if regenerate:
        console.print("[dim]Regenerating tips from history...[/dim]")
        tips = proposer.generate_tips()
        proposer.current_tips = tips
        proposer._save_state()
        console.print(f"[green]✓[/green] Generated {len(tips)} tips")

    tips = proposer.get_tips()

    console.print(Panel(
        "\n".join(f"• {tip}" for tip in tips),
        title=f"Current Tips ({len(tips)})",
    ))


@optimize_app.command("instructions")
def optimize_instructions(
    key: Annotated[
        Optional[str],
        typer.Argument(help="Instruction key to show/modify"),
    ] = None,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Reset to default"),
    ] = False,
    propose: Annotated[
        bool,
        typer.Option("--propose", "-p", help="Propose improvement"),
    ] = False,
) -> None:
    """Show or modify tool instructions."""
    from .core.instruction_optimizer import get_instruction_optimizer, DEFAULT_INSTRUCTIONS

    optimizer = get_instruction_optimizer()

    if key is None:
        # List all keys
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
        console.print("\n[dim]Use 'rlm-dspy optimize instructions <key>' to view details[/dim]")
        return

    if key not in DEFAULT_INSTRUCTIONS:
        console.print(f"[red]Unknown key: {key}[/red]")
        console.print(f"[dim]Available: {', '.join(DEFAULT_INSTRUCTIONS.keys())}[/dim]")
        raise typer.Exit(1)

    if reset:
        optimizer.reset_to_defaults(key)
        console.print(f"[green]✓[/green] Reset '{key}' to default")

    if propose:
        console.print(f"[dim]Proposing improvement for '{key}'...[/dim]")
        proposed = optimizer.propose_improvement(key)
        if proposed:
            console.print(Panel(proposed, title="Proposed Improvement"))
        else:
            console.print("[yellow]Not enough data to propose improvement[/yellow]")
        return

    instruction = optimizer.get_instruction(key)
    console.print(Panel(instruction, title=f"Instruction: {key}"))


@optimize_app.command("clear")
def optimize_clear(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation"),
    ] = False,
) -> None:
    """Clear optimization history."""
    from .core.instruction_optimizer import get_instruction_optimizer
    from .core.grounded_proposer import get_grounded_proposer

    if not force:
        confirm = typer.confirm("Clear all optimization history?")
        if not confirm:
            raise typer.Abort()

    optimizer = get_instruction_optimizer()
    proposer = get_grounded_proposer()

    opt_count = optimizer.clear_history()
    fail_count, success_count = proposer.clear()

    console.print(f"[green]✓[/green] Cleared {opt_count} optimizer records")
    console.print(f"[green]✓[/green] Cleared {fail_count} failures, {success_count} successes")


@optimize_app.command("simba")
def optimize_simba(
    min_score: Annotated[
        float,
        typer.Option("--min-score", "-s", help="Minimum trace score to include"),
    ] = 0.7,
    max_examples: Annotated[
        int,
        typer.Option("--max-examples", "-n", help="Maximum training examples"),
    ] = 100,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Mini-batch size"),
    ] = 16,
    steps: Annotated[
        int,
        typer.Option("--steps", help="Optimization steps"),
    ] = 4,
    candidates: Annotated[
        int,
        typer.Option("--candidates", "-c", help="Candidates per step"),
    ] = 4,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be optimized without running"),
    ] = False,
) -> None:
    """Run SIMBA self-improving optimization.

    Uses collected traces to optimize the RLM program via DSPy's SIMBA optimizer.
    SIMBA analyzes performance on challenging examples and generates improvement rules.

    Examples:
        rlm-dspy optimize simba                    # Run with defaults
        rlm-dspy optimize simba --min-score 0.5   # Include lower-scoring traces
        rlm-dspy optimize simba --steps 8         # More optimization steps
        rlm-dspy optimize simba --dry-run         # Preview without running
    """
    import json
    from pathlib import Path
    from .core.simba_optimizer import SIMBAOptimizer, grounded_metric

    traces_dir = Path.home() / ".rlm" / "traces"

    if not traces_dir.exists():
        console.print("[yellow]No traces directory found[/yellow]")
        console.print("[dim]Run queries with validation to collect traces first[/dim]")
        raise typer.Exit(1)

    # Load traces from traces.json (new format) or individual files (legacy)
    traces_file = traces_dir / "traces.json"
    all_traces = []
    
    if traces_file.exists():
        # New format: all traces in single file
        try:
            data = json.loads(traces_file.read_text())
            all_traces = data.get("traces", [])
        except Exception:
            pass
    
    # Also check for individual trace files (legacy format)
    for f in traces_dir.glob("trace_*.json"):
        try:
            all_traces.append(json.loads(f.read_text()))
        except Exception:
            pass

    if not all_traces:
        console.print("[yellow]No traces found[/yellow]")
        console.print("[dim]Run queries with validation to collect traces[/dim]")
        raise typer.Exit(1)

    # Filter and count qualifying traces
    qualifying = 0
    for trace in all_traces:
        # Support both grounded_score (new) and validation_score (old)
        score = trace.get("grounded_score", trace.get("validation_score", trace.get("score", 0)))
        if isinstance(score, (int, float)) and score >= min_score:
            qualifying += 1

    console.print(f"[bold]SIMBA Optimization[/bold]")
    console.print(f"  Traces found: {len(all_traces)}")
    console.print(f"  Qualifying (score >= {min_score}): {qualifying}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Steps: {steps}")
    console.print(f"  Candidates/step: {candidates}")
    console.print()

    if qualifying < batch_size:
        console.print(f"[yellow]Not enough qualifying traces ({qualifying} < {batch_size})[/yellow]")
        console.print("[dim]Lower --min-score or collect more traces[/dim]")
        raise typer.Exit(1)

    if dry_run:
        console.print("[dim]Dry run - would optimize with above settings[/dim]")
        return

    # Create optimizer
    optimizer = SIMBAOptimizer(
        metric=grounded_metric,
        batch_size=batch_size,
        num_candidates=candidates,
        max_steps=steps,
    )

    # Create a simple RLM program for optimization
    console.print("[dim]Loading RLM program...[/dim]")

    try:
        import dspy
        from .core.rlm import RLMClient

        # Get current RLM client
        client = RLMClient()

        # Run optimization
        console.print("[dim]Running SIMBA optimization (this may take a while)...[/dim]")

        optimized, result = optimizer.optimize_from_traces(
            program=client._rlm,  # The underlying dspy.RLM
            traces_dir=traces_dir,
            min_score=min_score,
            max_examples=max_examples,
        )

        # Display result
        console.print()
        if result.improved:
            console.print(f"[green]✓ Optimization successful![/green]")
            console.print(f"  Baseline score: {result.baseline_score:.2%}")
            console.print(f"  Optimized score: {result.optimized_score:.2%}")
            console.print(f"  Improvement: +{result.improvement:.1f}%")

            # Save the optimized program
            save_path = Path.home() / ".rlm" / "optimized_program.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save optimization result
            save_path.write_text(json.dumps(result.to_dict(), indent=2))
            console.print(f"\n[dim]Result saved to: {save_path}[/dim]")
        else:
            console.print(f"[yellow]No improvement found[/yellow]")
            console.print(f"  Score: {result.baseline_score:.2%}")

    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        console.print("[dim]Ensure dspy>=2.5 is installed[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# LSP Commands
# =============================================================================

lsp_app = typer.Typer(
    help="LSP (Language Server Protocol) management.",
    no_args_is_help=True,
)
app.add_typer(lsp_app, name="lsp")


@lsp_app.command("status")
def lsp_status():
    """Show LSP server status for all languages."""
    from .core.lsp_installer import get_server_status, LANGUAGE_TO_SERVER
    
    status = get_server_status()
    
    console.print("[bold]LSP Server Status[/bold]\n")
    
    # Group by installed status
    installed = []
    not_installed = []
    
    for server_id, info in status.items():
        if info["installed"]:
            installed.append((server_id, info))
        else:
            not_installed.append((server_id, info))
    
    if installed:
        console.print("[green]✓ Installed:[/green]")
        for server_id, info in installed:
            langs = ", ".join(info["languages"])
            console.print(f"  [green]●[/green] {info['name']} ({langs})")
        console.print()
    
    if not_installed:
        console.print("[yellow]○ Not Installed:[/yellow]")
        for server_id, info in not_installed:
            langs = ", ".join(info["languages"])
            console.print(f"  [dim]○[/dim] {info['name']} ({langs})")
            
            if info["missing_requirements"]:
                console.print(f"    [red]Missing: {', '.join(info['missing_requirements'])}[/red]")
            else:
                console.print(f"    [dim]Install: {info['install_cmd']}[/dim]")
        console.print()
    
    # Summary
    total = len(status)
    num_installed = len(installed)
    console.print(f"[dim]{num_installed}/{total} servers installed[/dim]")


@lsp_app.command("install")
def lsp_install(
    language: str = typer.Argument(..., help="Language to install server for (e.g., python, go, rust)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall"),
):
    """Install LSP server for a language."""
    from .core.lsp_installer import (
        install_for_language, 
        LANGUAGE_TO_SERVER, 
        SERVERS,
        check_requirements,
    )
    
    language = language.lower()
    
    # Check if we support this language
    server_id = LANGUAGE_TO_SERVER.get(language)
    if not server_id:
        console.print(f"[red]No LSP server configured for '{language}'[/red]")
        console.print("\n[dim]Supported languages:[/dim]")
        for lang in sorted(LANGUAGE_TO_SERVER.keys()):
            console.print(f"  - {lang}")
        raise typer.Exit(1)
    
    info = SERVERS[server_id]
    
    # Check requirements
    reqs_met, missing = check_requirements(server_id)
    if not reqs_met:
        console.print(f"[red]Missing requirements: {', '.join(missing)}[/red]")
        console.print(f"\n[dim]Install the missing tools first:[/dim]")
        for req in missing:
            if req == "npm":
                console.print("  npm: Install Node.js from https://nodejs.org")
            elif req == "go":
                console.print("  go: Install Go from https://go.dev/dl/")
            elif req == "rustup":
                console.print("  rustup: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
            elif req == "gem":
                console.print("  gem: Install Ruby from https://www.ruby-lang.org")
            elif req == "java":
                console.print("  java: Install JDK from https://adoptium.net")
        raise typer.Exit(1)
    
    console.print(f"Installing {info.name} for {language}...")
    
    from .core.lsp_installer import install_server
    
    if install_server(server_id, force=force):
        console.print(f"[green]✓ Successfully installed {info.name}[/green]")
    else:
        console.print(f"[red]✗ Failed to install {info.name}[/red]")
        raise typer.Exit(1)


@lsp_app.command("install-all")
def lsp_install_all(
    skip_missing: bool = typer.Option(True, "--skip-missing/--no-skip-missing", 
                                       help="Skip servers with missing requirements"),
):
    """Install all available LSP servers."""
    from .core.lsp_installer import SERVERS, install_server, check_requirements
    
    console.print("[bold]Installing all LSP servers...[/bold]\n")
    
    success = 0
    failed = 0
    skipped = 0
    
    for server_id, info in SERVERS.items():
        # Check requirements
        reqs_met, missing = check_requirements(server_id)
        if not reqs_met:
            if skip_missing:
                console.print(f"[dim]○ {info.name} - skipped (missing: {', '.join(missing)})[/dim]")
                skipped += 1
                continue
        
        console.print(f"  Installing {info.name}...", end=" ")
        
        if install_server(server_id):
            console.print("[green]✓[/green]")
            success += 1
        else:
            console.print("[red]✗[/red]")
            failed += 1
    
    console.print()
    console.print(f"[green]Installed: {success}[/green]")
    if failed:
        console.print(f"[red]Failed: {failed}[/red]")
    if skipped:
        console.print(f"[dim]Skipped: {skipped}[/dim]")


@lsp_app.command("test")
def lsp_test(
    file: Path = typer.Argument(..., help="File to test LSP on"),
    line: int = typer.Option(1, "--line", "-l", help="Line number (1-indexed)"),
    column: int = typer.Option(0, "--column", "-c", help="Column number (0-indexed)"),
):
    """Test LSP functionality on a file."""
    from .core.lsp import get_lsp_manager
    
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Testing LSP on {file}[/bold]\n")
    
    manager = get_lsp_manager()
    
    # Test find references
    console.print("[dim]Finding references...[/dim]")
    refs = manager.find_references(str(file), line, column)
    if refs:
        console.print(f"[green]✓ Found {len(refs)} references[/green]")
        for ref in refs[:5]:
            console.print(f"  {ref['file']}:{ref['line']}:{ref['column']}")
        if len(refs) > 5:
            console.print(f"  ... and {len(refs) - 5} more")
    else:
        console.print("[yellow]○ No references found[/yellow]")
    
    console.print()
    
    # Test go to definition
    console.print("[dim]Going to definition...[/dim]")
    defn = manager.go_to_definition(str(file), line, column)
    if defn:
        console.print(f"[green]✓ Definition found[/green]")
        console.print(f"  {defn['file']}:{defn['line']}:{defn['column']}")
    else:
        console.print("[yellow]○ No definition found[/yellow]")
    
    console.print()
    
    # Test hover
    console.print("[dim]Getting hover info...[/dim]")
    hover = manager.get_hover_info(str(file), line, column)
    if hover:
        console.print(f"[green]✓ Hover info found[/green]")
        # Truncate if too long
        if len(hover) > 200:
            hover = hover[:200] + "..."
        console.print(f"  {hover}")
    else:
        console.print("[yellow]○ No hover info found[/yellow]")
    
    console.print()
    
    # Test document symbols
    console.print("[dim]Getting document symbols...[/dim]")
    symbols = manager.get_document_symbols(str(file))
    if symbols:
        console.print(f"[green]✓ Found {len(symbols)} symbols[/green]")
        for sym in symbols[:10]:
            console.print(f"  {sym['kind']}: {sym['name']} (line {sym['line']})")
        if len(symbols) > 10:
            console.print(f"  ... and {len(symbols) - 10} more")
    else:
        console.print("[yellow]○ No symbols found[/yellow]")


if __name__ == "__main__":
    app()

"""Main CLI commands - ask, analyze, diff, setup, config, preflight, example."""

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

if TYPE_CHECKING:
    pass

import importlib
_fileutils = importlib.import_module("rlm_dspy.core.fileutils")
PathTraversalError = _fileutils.PathTraversalError
validate_path_safety = _fileutils.validate_path_safety

console = Console()

# Type aliases
ModelOpt = Annotated[Optional[str], typer.Option("--model", "-m", help="Model to use")]
SubModelOpt = Annotated[Optional[str], typer.Option("--sub-model", "-s", help="Model for llm_query()")]
BudgetOpt = Annotated[Optional[float], typer.Option("--budget", "-b", help="Max budget in USD")]
TimeoutOpt = Annotated[Optional[float], typer.Option("--timeout", "-t", help="Max timeout in seconds")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed progress")]
FormatOpt = Annotated[Optional[str], typer.Option("--format", "-f", help="Output format")]
OutputFileOpt = Annotated[Optional[Path], typer.Option("--output", "-o", help="Write output to file")]


def _safe_write_output(output_file: Path, content: str) -> None:
    """Write content to output file with path validation."""
    try:
        safe_path = validate_path_safety(output_file)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
    except PathTraversalError as e:
        console.print(f"[red]Security Error: {e}[/red]")
        raise typer.Exit(1)
    except (PermissionError, OSError) as e:
        console.print(f"[red]Failed to write file: {e}[/red]")
        raise typer.Exit(1)


def _get_config(model=None, sub_model=None, budget=None, timeout=None, 
                max_iterations=None, max_workers=None, verbose=False):
    """Build config from CLI args and environment."""
    for name, val in [("budget", budget), ("timeout", timeout), 
                      ("max_iterations", max_iterations), ("max_workers", max_workers)]:
        if val is not None and val <= 0:
            console.print(f"[red]Error: --{name.replace('_', '-')} must be positive[/red]")
            raise typer.Exit(1)

    from .core.rlm import RLMConfig
    config = RLMConfig()

    if model:
        config.model = model
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


def _load_context(rlm, paths, stdin, verbose, max_tokens=None, use_cache=True):
    """Load context from stdin or file paths."""
    if stdin:
        if sys.stdin.isatty():
            console.print("[yellow]Reading from stdin (Ctrl+D when done)...[/yellow]")
        MAX_SIZE = 50 * 1024 * 1024
        context = sys.stdin.read(MAX_SIZE + 1)
        if len(context) > MAX_SIZE:
            console.print("[red]Error: stdin too large[/red]")
            raise typer.Exit(1)
        if not context.strip():
            console.print("[red]Error: No input from stdin[/red]")
            raise typer.Exit(1)
        return context

    if paths:
        for p in paths:
            if not p.exists():
                console.print(f"[yellow]Warning: Path not found: {p}[/yellow]")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console, transient=True) as progress:
            progress.add_task("Loading files...", total=None)
            context = rlm.load_context([str(p) for p in paths], max_tokens=max_tokens, use_cache=use_cache)

        if not context.strip():
            console.print("[red]Error: No content loaded[/red]")
            raise typer.Exit(1)

        from .tools import set_current_project
        first_path = paths[0].resolve()
        if first_path.is_file():
            first_path = first_path.parent
        set_current_project(str(first_path))
        return context

    console.print("[red]Error: Provide paths or use --stdin[/red]")
    raise typer.Exit(1)


def _output_result(result, output_format, output_file, verbose, debug):
    """Handle result output in various formats."""
    if output_format == "json":
        data = {"answer": result.answer, "success": result.success, "time": result.elapsed_time, "error": result.error}
        if result.outputs:
            data["outputs"] = result.outputs
        json_output = json.dumps(data, indent=2)
        if output_file:
            _safe_write_output(output_file, json_output)
            console.print(f"[green]Output written to {output_file}[/green]")
        else:
            print(json_output)
        return

    if output_format == "markdown":
        lines = ["# Analysis Result\n"]
        if result.success:
            lines.append(result.answer)
        else:
            lines.append(f"**Error:** {result.error}\n")
        md = "\n".join(lines)
        if output_file:
            _safe_write_output(output_file, md)
        else:
            print(md)
        return

    if output_file:
        if result.success:
            _safe_write_output(output_file, result.answer)
            console.print(f"[green]Output written to {output_file}[/green]")
        else:
            console.print(f"[red]Error: {result.error}[/red]")
            raise typer.Exit(1)
        return

    if result.success:
        console.print(Panel(Markdown(result.answer), title="Answer", border_style="green"))
        if verbose or debug:
            table = Table(title="Stats", show_header=False)
            table.add_column("Metric", style="dim")
            table.add_column("Value", style="cyan")
            table.add_row("Time", f"{result.elapsed_time:.1f}s")
            table.add_row("Iterations", str(result.iterations))
            console.print(table)
    else:
        console.print(f"[red]Error: {result.error or 'Unknown error'}[/red]")
        raise typer.Exit(1)


def register_commands(app: typer.Typer) -> None:
    """Register main commands on the app."""
    
    @app.command()
    def ask(
        query: Annotated[str, typer.Argument(help="The question to answer")],
        paths: Annotated[Optional[list[Path]], typer.Argument(help="Files or directories")] = None,
        stdin: Annotated[bool, typer.Option("--stdin", "-", help="Read from stdin")] = False,
        model: ModelOpt = None,
        sub_model: SubModelOpt = None,
        budget: BudgetOpt = None,
        timeout: TimeoutOpt = None,
        max_iterations: Annotated[Optional[int], typer.Option("--max-iterations", "-i")] = None,
        signature: Annotated[Optional[str], typer.Option("--signature", "-S")] = None,
        output_format: FormatOpt = None,
        output_json: Annotated[bool, typer.Option("--json", "-j")] = False,
        output_file: OutputFileOpt = None,
        verbose: VerboseOpt = False,
        debug: Annotated[bool, typer.Option("--debug", "-d")] = False,
        dry_run: Annotated[bool, typer.Option("--dry-run", "-n")] = False,
        validate: Annotated[bool, typer.Option("--validate/--no-validate", "-V")] = True,
        no_tools: Annotated[bool, typer.Option("--no-tools")] = False,
        max_tokens: Annotated[Optional[int], typer.Option("--max-tokens", "-T")] = None,
        max_workers: Annotated[Optional[int], typer.Option("--max-workers", "-w")] = None,
        no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    ) -> None:
        """Ask a question about files or piped content."""
        import os
        if debug:
            os.environ["RLM_DEBUG"] = "1"

        config = _get_config(model, sub_model, budget, timeout, max_iterations, max_workers, verbose or debug)

        sig = "context, query -> answer"
        if signature:
            from .signatures import get_signature
            sig_class = get_signature(signature)
            if not sig_class:
                console.print(f"[red]Unknown signature: {signature}[/red]")
                raise typer.Exit(1)
            sig = sig_class

        from .core.rlm import RLM
        rlm = RLM(config=config, signature=sig, use_tools=not no_tools)
        context = _load_context(rlm, paths, stdin, verbose or debug, max_tokens, not no_cache)

        if dry_run:
            from .core.validation import preflight_check
            result = preflight_check(api_key_required=True, model=config.model, context=context)
            result.print_report()
            raise typer.Exit(0 if result.passed else 1)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console, transient=True) as progress:
            progress.add_task("Analyzing...", total=None)
            try:
                result = rlm.query(query, context)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                raise typer.Exit(130)
            except Exception as e:
                console.print(f"\n[red]Query failed: {e}[/red]")
                raise typer.Exit(1)

        if validate and result.success:
            from .guards import validate_groundedness
            console.print("\n[dim]Validating output...[/dim]")
            validation = validate_groundedness(result.answer, context, query)
            if validation.is_grounded:
                console.print(f"[green]✓ Output validated ({validation.score:.0%} grounded)[/green]")
            else:
                console.print(f"[yellow]⚠ Potential hallucinations ({validation.score:.0%} grounded)[/yellow]")

        resolved_format = output_format or ("json" if output_json else "text")
        _output_result(result, resolved_format, output_file, verbose, debug)

    @app.command()
    def analyze(
        paths: Annotated[list[Path], typer.Argument(help="Files or directories")],
        output: OutputFileOpt = None,
        model: ModelOpt = None,
        format: Annotated[str, typer.Option("--format", "-f")] = "markdown",
    ) -> None:
        """Generate a comprehensive analysis of files."""
        from .core.rlm import RLM
        config = _get_config(model)
        rlm = RLM(config=config)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console) as progress:
            progress.add_task("Loading...", total=None)
            context = rlm.load_context([str(p) for p in paths])

            progress.add_task("Analyzing...", total=None)
            # Use num_threads=1 to avoid Deno subprocess conflicts in parallel execution
            results = rlm.batch([
                {"query": "List all files and their purposes"},
                {"query": "Identify main components and their roles"},
                {"query": "Find potential bugs or improvements"},
            ], context=context, num_threads=1, return_failed=True)

        if len(results) != 3 or not all(r.success for r in results):
            console.print("[red]Analysis failed[/red]")
            raise typer.Exit(1)

        structure, components, issues = results
        if format == "json":
            out = json.dumps({"structure": structure.answer, "components": components.answer, "issues": issues.answer}, indent=2)
        else:
            out = f"# Analysis\n\n## Structure\n{structure.answer}\n\n## Components\n{components.answer}\n\n## Issues\n{issues.answer}"

        if output:
            _safe_write_output(output, out)
            console.print(f"[green]Saved to {output}[/green]")
        else:
            console.print(Markdown(out) if format != "json" else out)

    @app.command()
    def diff(
        query: Annotated[str, typer.Argument(help="Question about the diff")],
        diff_file: Annotated[Optional[Path], typer.Option("--file", "-f")] = None,
        model: ModelOpt = None,
    ) -> None:
        """Analyze a git diff."""
        from .core.rlm import RLM
        config = _get_config(model)
        rlm = RLM(config=config)

        context = diff_file.read_text() if diff_file else sys.stdin.read()
        if not context.strip():
            console.print("[red]No diff content[/red]")
            raise typer.Exit(1)

        result = rlm.query(f"Analyze this diff: {query}\n\n{context}", context)
        if result.success:
            console.print(Panel(Markdown(result.answer), title="Diff Analysis"))
        else:
            console.print(f"[red]Error: {result.error}[/red]")
            raise typer.Exit(1)

    @app.command()
    def config(
        show: Annotated[bool, typer.Option("--show", "-s")] = True,
    ) -> None:
        """Show configuration."""
        from .core.rlm import RLMConfig
        cfg = RLMConfig()

        table = Table(title="Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        table.add_row("Model", cfg.model)
        table.add_row("Sub Model", cfg.sub_model)
        table.add_row("API Key", "***" if cfg.api_key else "[red]not set[/red]")
        table.add_row("Max Iterations", str(cfg.max_iterations))
        table.add_row("Max Budget", f"${cfg.max_budget:.2f}")
        console.print(table)

    @app.command()
    def preflight(
        paths: Annotated[Optional[list[Path]], typer.Argument(help="Files to check")] = None,
        check_network: Annotated[bool, typer.Option("--network/--no-network")] = True,
    ) -> None:
        """Run preflight checks."""
        from .core.validation import preflight_check
        from .core.rlm import RLM, RLMConfig

        config = RLMConfig()
        context = None
        if paths:
            rlm = RLM(config=config)
            context = rlm.load_context([str(p) for p in paths])

        result = preflight_check(api_key_required=True, model=config.model, 
                                 api_base=config.api_base, context=context, check_network=check_network)
        result.print_report()
        raise typer.Exit(0 if result.passed else 1)

    @app.command()
    def setup(
        env_file: Annotated[Optional[Path], typer.Option("--env-file", "-e")] = None,
        model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
        budget: Annotated[Optional[float], typer.Option("--budget", "-b")] = None,
    ) -> None:
        """Configure RLM-DSPy settings."""
        from .core.user_config import CONFIG_FILE, load_config, save_config

        config = load_config()
        changed = False

        if env_file:
            if env_file.exists():
                config["env_file"] = str(env_file)
                changed = True
            else:
                console.print(f"[red]File not found: {env_file}[/red]")
                raise typer.Exit(1)
        if model:
            config["model"] = model
            changed = True
        if budget is not None:
            config["max_budget"] = budget
            changed = True

        if changed:
            save_config(config)
            console.print(f"[green]✓ Saved to {CONFIG_FILE}[/green]")
        else:
            console.print("[dim]No changes made[/dim]")

    @app.command()
    def example(
        case: Annotated[Optional[str], typer.Argument(help="Use case")] = None,
    ) -> None:
        """Show example prompts."""
        examples = {
            "bugs": ("Bug Finding", ["Find potential bugs", "Check for edge cases"]),
            "security": ("Security", ["Find vulnerabilities", "Check for injection"]),
            "understand": ("Understanding", ["What does this do?", "Explain the architecture"]),
        }
        if not case:
            table = Table(title="Examples")
            table.add_column("Case")
            table.add_column("Description")
            for k, (desc, _) in examples.items():
                table.add_row(k, desc)
            console.print(table)
            return

        if case not in examples:
            console.print(f"[red]Unknown: {case}[/red]")
            raise typer.Exit(1)

        title, prompts = examples[case]
        console.print(f"\n[bold]{title}[/bold]")
        for p in prompts:
            console.print(f'  rlm-dspy ask "{p}" src/')

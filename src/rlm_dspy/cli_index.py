"""CLI commands for index management."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

index_app = typer.Typer(help="Manage vector indexes for semantic search", no_args_is_help=True)


@index_app.command("build")
def index_build(
    paths: Annotated[list[Path], typer.Argument(help="Directories to index")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Force full rebuild")] = False,
) -> None:
    """Build or update vector index for semantic search."""
    from .core.vector_index import get_index_manager

    manager = get_index_manager()

    for path in paths:
        if not path.exists():
            console.print(f"[red]Path not found: {path}[/red]")
            continue

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console, transient=True) as progress:
            progress.add_task(f"Indexing {path}...", total=None)
            try:
                count = manager.build(path, force=force)
                console.print(f"[green]✓[/green] Indexed {path}: {count} code snippets")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to index {path}: {e}")


@index_app.command("status")
def index_status(
    paths: Annotated[list[Path], typer.Argument(help="Directories to check")],
) -> None:
    """Show index status for directories."""
    from .core.vector_index import get_index_manager

    manager = get_index_manager()

    for path in paths:
        if not path.exists():
            console.print(f"[red]Path not found: {path}[/red]")
            continue

        status = manager.get_status(path)
        table = Table(title=f"Index Status: {path}")
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Indexed", "Yes" if status["indexed"] else "No")
        if status["indexed"]:
            table.add_row("Snippets", str(status.get("snippet_count", 0)))
            table.add_row("Files", str(status.get("file_count", 0)))
            if status.get("indexed_at"):
                table.add_row("Last Updated", status["indexed_at"])
            if status.get("needs_update"):
                table.add_row("Status", "[yellow]Needs update[/yellow]")

        console.print(table)


@index_app.command("clear")
def index_clear(
    paths: Annotated[list[Path] | None, typer.Argument(help="Directories to clear (all if none)")] = None,
    all_indexes: Annotated[bool, typer.Option("--all", help="Clear all indexes")] = False,
) -> None:
    """Clear vector indexes."""
    from .core.vector_index import get_index_manager

    if not paths and not all_indexes:
        console.print("[yellow]Specify paths or use --all to clear all indexes[/yellow]")
        raise typer.Exit(1)

    manager = get_index_manager()

    if all_indexes:
        count = manager.clear(None)
        console.print(f"[green]✓[/green] Cleared all indexes ({count} removed)")
    else:
        for path in paths or []:
            try:
                count = manager.clear(path)
                console.print(f"[green]✓[/green] Cleared index for {path}")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to clear {path}: {e}")


@index_app.command("search")
def index_search(
    query: Annotated[str, typer.Argument(help="Search query")],
    path: Annotated[Path | None, typer.Option("--path", "-p", help="Directory to search")] = None,
    k: Annotated[int, typer.Option("--top", "-k", help="Number of results")] = 5,
) -> None:
    """Search code semantically."""
    from .core.vector_index import get_index_manager

    manager = get_index_manager()
    search_path = path or Path.cwd()

    if not search_path.exists():
        console.print(f"[red]Path not found: {search_path}[/red]")
        raise typer.Exit(1)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console, transient=True) as progress:
        progress.add_task("Searching...", total=None)
        try:
            results = manager.search(search_path, query, k=k)
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")
            raise typer.Exit(1)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    for i, result in enumerate(results, 1):
        snippet = result.snippet
        console.print(f"\n[bold cyan]Result {i}:[/bold cyan] {snippet.file}:{snippet.line}")
        console.print(f"[dim]Type: {snippet.type} | Name: {snippet.name}[/dim]")
        console.print(snippet.text[:500] + ("..." if len(snippet.text) > 500 else ""))


@index_app.command("compress")
def index_compress(
    paths: Annotated[list[Path] | None, typer.Argument(help="Directories to compress")] = None,
    all_indexes: Annotated[bool, typer.Option("--all", help="Compress all indexes")] = False,
    decompress: Annotated[bool, typer.Option("--decompress", "-d", help="Decompress instead")] = False,
) -> None:
    """Compress indexes to reduce disk usage."""
    from .core.index_compression import compress_index, decompress_index
    from .core.project_registry import get_project_registry

    if not paths and not all_indexes:
        console.print("[yellow]Specify paths or use --all[/yellow]")
        raise typer.Exit(1)

    registry = get_project_registry()
    index_dir = registry.config.index_dir

    if all_indexes:
        paths_to_process = [index_dir / p.name for p in registry.list()]
    else:
        paths_to_process = [index_dir / p.name for p in (paths or []) if (index_dir / p.name).exists()]

    action = "Decompressing" if decompress else "Compressing"
    func = decompress_index if decompress else compress_index

    for idx_path in paths_to_process:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console, transient=True) as progress:
            progress.add_task(f"{action} {idx_path.name}...", total=None)
            try:
                result = func(idx_path)
                if result:
                    console.print(f"[green]✓[/green] {action} {idx_path.name}")
                else:
                    console.print(f"[dim]Skipped {idx_path.name} (already processed)[/dim]")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed: {e}")


@index_app.command("cache")
def index_cache(
    clear: Annotated[bool, typer.Option("--clear", "-c", help="Clear AST cache")] = False,
) -> None:
    """Show or clear AST index cache statistics."""
    from .core.ast_index import get_cache_stats, clear_index_cache

    if clear:
        cleared = clear_index_cache()
        console.print(f"[green]✓[/green] Cleared AST cache ({cleared} entries)")
        return

    stats = get_cache_stats()
    table = Table(title="AST Index Cache")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Entries", str(stats.get("entries", 0)))
    table.add_row("Hits", str(stats.get("hits", 0)))
    table.add_row("Misses", str(stats.get("misses", 0)))
    hit_rate = stats.get("hits", 0) / max(stats.get("hits", 0) + stats.get("misses", 0), 1) * 100
    table.add_row("Hit Rate", f"{hit_rate:.1f}%")
    console.print(table)

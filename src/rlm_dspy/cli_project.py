"""CLI commands for project management."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

project_app = typer.Typer(help="Manage registered projects", no_args_is_help=True)


@project_app.command("add")
def project_add(
    path: Annotated[Path, typer.Argument(help="Path to project")],
    name: Annotated[str | None, typer.Option("--name", "-n", help="Project name")] = None,
    alias: Annotated[str | None, typer.Option("--alias", "-a", help="Short alias")] = None,
    default: Annotated[bool, typer.Option("--default", "-d", help="Set as default")] = False,
    watch: Annotated[bool, typer.Option("--watch", "-w", help="Auto-watch for changes")] = False,
) -> None:
    """Register a project for indexing and search."""
    from .core.project_registry import get_project_registry

    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    registry = get_project_registry()
    project_name = name or path.resolve().name

    try:
        project = registry.add(str(path.resolve()), name=project_name, alias=alias, auto_watch=watch)
        console.print(f"[green]✓[/green] Added project: {project.name}")

        if default:
            registry.set_default(project.name)
            console.print(f"[dim]Set as default project[/dim]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@project_app.command("list")
def project_list(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show more details")] = False,
) -> None:
    """List registered projects."""
    from .core.project_registry import get_project_registry

    registry = get_project_registry()
    projects = registry.list()

    if not projects:
        console.print("[yellow]No projects registered. Use 'rlm-dspy project add <path>' to add one.[/yellow]")
        return

    default_project = registry.get_default()
    default_name = default_project.name if default_project else None

    table = Table(title="Registered Projects")
    table.add_column("Name", style="cyan")
    table.add_column("Path")
    table.add_column("Snippets", justify="right")
    if verbose:
        table.add_column("Files", justify="right")
        table.add_column("Watch")
        table.add_column("Tags")

    for project in projects:
        name = project.name
        if name == default_name:
            name = f"[bold]{name}[/bold] ★"

        if verbose:
            table.add_row(
                name, project.path, str(project.snippet_count),
                str(project.file_count),
                "✓" if project.auto_watch else "",
                ", ".join(project.tags) if project.tags else "",
            )
        else:
            table.add_row(name, project.path, str(project.snippet_count))

    console.print(table)


@project_app.command("remove")
def project_remove(
    name: Annotated[str, typer.Argument(help="Project name to remove")],
    delete_index: Annotated[bool, typer.Option("--delete-index", "-d", help="Also delete index")] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Remove a registered project."""
    from .core.project_registry import get_project_registry

    registry = get_project_registry()
    project = registry.get(name)

    if not project:
        console.print(f"[red]Project not found: {name}[/red]")
        raise typer.Exit(1)

    if not force:
        msg = f"Remove project '{name}'"
        if delete_index:
            msg += " and its index"
        if not typer.confirm(f"{msg}?"):
            raise typer.Abort()

    if registry.remove(name, delete_index=delete_index):
        console.print(f"[green]✓[/green] Removed project: {name}")
    else:
        console.print(f"[red]Failed to remove project[/red]")


@project_app.command("default")
def project_default(
    name: Annotated[str | None, typer.Argument(help="Project to set as default")] = None,
) -> None:
    """Get or set the default project."""
    from .core.project_registry import get_project_registry

    registry = get_project_registry()

    if name is None:
        default = registry.get_default()
        if default:
            console.print(f"Default project: [cyan]{default.name}[/cyan] ({default.path})")
        else:
            console.print("[yellow]No default project set[/yellow]")
        return

    project = registry.get(name)
    if not project:
        console.print(f"[red]Project not found: {name}[/red]")
        raise typer.Exit(1)

    registry.set_default(name)
    console.print(f"[green]✓[/green] Set default project: {name}")


@project_app.command("tag")
def project_tag(
    name: Annotated[str, typer.Argument(help="Project name")],
    tags: Annotated[list[str], typer.Argument(help="Tags to add")],
    remove: Annotated[bool, typer.Option("--remove", "-r", help="Remove tags instead")] = False,
) -> None:
    """Add or remove tags from a project."""
    from .core.project_registry import get_project_registry

    registry = get_project_registry()
    project = registry.get(name)

    if not project:
        console.print(f"[red]Project not found: {name}[/red]")
        raise typer.Exit(1)

    if remove:
        registry.untag(name, tags)
        console.print(f"[green]✓[/green] Removed tags: {', '.join(tags)}")
    else:
        registry.tag(name, tags)
        console.print(f"[green]✓[/green] Added tags: {', '.join(tags)}")


@project_app.command("cleanup")
def project_cleanup(
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Show what would be cleaned")] = True,
) -> None:
    """Clean up orphaned index directories."""
    from .core.project_registry import get_project_registry

    registry = get_project_registry()
    orphaned = registry.cleanup_orphaned(dry_run=dry_run)

    if not orphaned:
        console.print("[green]No orphaned indexes found[/green]")
        return

    if dry_run:
        console.print(f"[yellow]Found {len(orphaned)} orphaned indexes (use --no-dry-run to remove):[/yellow]")
        for path in orphaned:
            console.print(f"  {path}")
    else:
        console.print(f"[green]✓[/green] Cleaned up {len(orphaned)} orphaned indexes")


@project_app.command("migrate")
def project_migrate() -> None:
    """Migrate legacy hash-based indexes to named projects."""
    from .core.project_registry import get_project_registry

    registry = get_project_registry()
    migrated = registry.migrate_legacy()

    if not migrated:
        console.print("[green]No legacy indexes to migrate[/green]")
        return

    console.print(f"[green]✓[/green] Migrated {len(migrated)} indexes:")
    for old_hash, new_name in migrated:
        console.print(f"  {old_hash} → {new_name}")

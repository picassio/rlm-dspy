"""CLI commands for daemon management."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

daemon_app = typer.Typer(help="Manage background index daemon", no_args_is_help=True)


@daemon_app.command("start")
def daemon_start(
    watch_all: Annotated[bool, typer.Option("--watch-all", "-a", help="Watch all auto-watch projects")] = True,
    foreground: Annotated[bool, typer.Option("--foreground", "-f", help="Run in foreground")] = False,
) -> None:
    """Start the background index daemon."""
    from .core.daemon import IndexDaemon, is_daemon_running

    if is_daemon_running():
        console.print("[yellow]Daemon is already running[/yellow]")
        raise typer.Exit(1)

    daemon = IndexDaemon()

    if foreground:
        console.print("[dim]Starting daemon in foreground (Ctrl+C to stop)...[/dim]")
        daemon.start(daemonize=False)
        if watch_all:
            count = daemon.watch_all()
            console.print(f"[dim]Watching {count} projects[/dim]")
        daemon.run_forever()
    else:
        daemon.start(daemonize=True)
        if watch_all:
            daemon.watch_all()
        console.print("[green]✓[/green] Daemon started in background")


@daemon_app.command("stop")
def daemon_stop() -> None:
    """Stop the background daemon."""
    from .core.daemon import stop_daemon, is_daemon_running

    if not is_daemon_running():
        console.print("[yellow]Daemon is not running[/yellow]")
        return

    if stop_daemon():
        console.print("[green]✓[/green] Daemon stopped")
    else:
        console.print("[red]Failed to stop daemon[/red]")


@daemon_app.command("status")
def daemon_status() -> None:
    """Show daemon status."""
    from .core.daemon import get_daemon_pid, is_daemon_running, IndexDaemon

    pid = get_daemon_pid()
    running = is_daemon_running()

    table = Table(title="Daemon Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Running", "[green]Yes[/green]" if running else "[red]No[/red]")
    if pid:
        table.add_row("PID", str(pid))

    config = IndexDaemon().config
    table.add_row("PID File", str(config.pid_file))
    table.add_row("Log File", str(config.log_file))

    console.print(table)


@daemon_app.command("watch")
def daemon_watch(
    names: Annotated[list[str], typer.Argument(help="Project names to watch")],
) -> None:
    """Add projects to daemon watch list."""
    from .core.daemon import is_daemon_running

    if not is_daemon_running():
        console.print("[yellow]Daemon is not running. Start with 'rlm-dspy daemon start'[/yellow]")
        raise typer.Exit(1)

    # For now, just update the project registry to set auto_watch
    from .core.project_registry import get_project_registry

    registry = get_project_registry()

    for name in names:
        project = registry.get(name)
        if not project:
            console.print(f"[red]Project not found: {name}[/red]")
            continue

        registry.add(project.path, name=project.name, auto_watch=True)
        console.print(f"[green]✓[/green] Added to watch list: {name}")

    console.print("[dim]Restart daemon to apply changes[/dim]")


@daemon_app.command("unwatch")
def daemon_unwatch(
    names: Annotated[list[str], typer.Argument(help="Project names to unwatch")],
) -> None:
    """Remove projects from daemon watch list."""
    from .core.project_registry import get_project_registry

    registry = get_project_registry()

    for name in names:
        project = registry.get(name)
        if not project:
            console.print(f"[red]Project not found: {name}[/red]")
            continue

        registry.add(project.path, name=project.name, auto_watch=False)
        console.print(f"[green]✓[/green] Removed from watch list: {name}")


@daemon_app.command("list")
def daemon_list() -> None:
    """List projects being watched by daemon."""
    from .core.project_registry import get_project_registry

    registry = get_project_registry()
    watched = [p for p in registry.list() if p.auto_watch]

    if not watched:
        console.print("[yellow]No projects are being watched[/yellow]")
        return

    table = Table(title="Watched Projects")
    table.add_column("Name", style="cyan")
    table.add_column("Path")

    for project in watched:
        table.add_row(project.name, project.path)

    console.print(table)


@daemon_app.command("log")
def daemon_log(
    lines: Annotated[int, typer.Option("--lines", "-n", help="Number of lines")] = 50,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Follow log output")] = False,
) -> None:
    """Show daemon log."""
    import subprocess

    from .core.daemon import IndexDaemon

    config = IndexDaemon().config
    log_file = config.log_file

    if not log_file.exists():
        console.print("[yellow]No log file found[/yellow]")
        return

    if follow:
        try:
            subprocess.run(["tail", "-f", str(log_file)])
        except KeyboardInterrupt:
            pass
    else:
        try:
            result = subprocess.run(["tail", f"-n{lines}", str(log_file)], capture_output=True, text=True)
            console.print(result.stdout)
        except Exception as e:
            console.print(f"[red]Error reading log: {e}[/red]")

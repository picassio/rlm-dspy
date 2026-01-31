"""CLI commands for OAuth authentication."""

from __future__ import annotations

from datetime import datetime, UTC
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

auth_app = typer.Typer(
    name="auth",
    help="OAuth authentication for LLM providers",
    no_args_is_help=True,
)


@auth_app.command("login")
def auth_login(
    provider: Annotated[str, typer.Argument(help="Provider (anthropic)")] = "anthropic",
    no_browser: Annotated[bool, typer.Option("--no-browser", help="Don't open browser")] = False,
) -> None:
    """Login with OAuth (opens browser for authentication)."""
    from .core.oauth import oauth_login
    
    supported = ["anthropic"]
    if provider not in supported:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        console.print(f"Supported providers: {', '.join(supported)}")
        raise typer.Exit(1)
    
    try:
        credentials = oauth_login(provider, open_browser=not no_browser)
        console.print(f"\n[green]✓ Logged in to {provider}[/green]")
        
        expires = datetime.fromtimestamp(credentials.expires_at, UTC)
        console.print(f"  Token expires: {expires.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
    except ValueError as e:
        console.print(f"[red]Login failed: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Login cancelled[/yellow]")
        raise typer.Exit(130)


@auth_app.command("logout")
def auth_logout(
    provider: Annotated[str, typer.Argument(help="Provider (anthropic)")] = "anthropic",
) -> None:
    """Logout and delete stored credentials."""
    from .core.oauth import oauth_logout
    
    if oauth_logout(provider):
        console.print(f"[green]✓ Logged out from {provider}[/green]")
    else:
        console.print(f"[dim]No credentials found for {provider}[/dim]")


@auth_app.command("status")
def auth_status(
    provider: Annotated[str, typer.Argument(help="Provider (anthropic, or 'all')")] = "all",
) -> None:
    """Show OAuth authentication status."""
    from .core.oauth import oauth_status
    
    providers = ["anthropic"] if provider != "all" else ["anthropic"]
    
    table = Table(title="OAuth Authentication Status")
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Expires")
    table.add_column("Created")
    
    for p in providers:
        status = oauth_status(p)
        
        if status["authenticated"]:
            if status["is_expired"]:
                status_str = "[yellow]Expired (will refresh)[/yellow]"
            else:
                status_str = "[green]Authenticated[/green]"
            
            expires = datetime.fromtimestamp(status["expires_at"], UTC)
            expires_str = expires.strftime("%Y-%m-%d %H:%M")
            
            created = datetime.fromtimestamp(status["created_at"], UTC)
            created_str = created.strftime("%Y-%m-%d %H:%M")
        else:
            status_str = "[dim]Not authenticated[/dim]"
            expires_str = "-"
            created_str = "-"
        
        table.add_row(p, status_str, expires_str, created_str)
    
    console.print(table)
    
    # Show hint
    console.print("\n[dim]Use 'rlm-dspy auth login <provider>' to authenticate[/dim]")


@auth_app.command("refresh")
def auth_refresh(
    provider: Annotated[str, typer.Argument(help="Provider (anthropic)")] = "anthropic",
) -> None:
    """Manually refresh OAuth token."""
    from .core.oauth import _load_credentials, anthropic_refresh_token
    
    credentials = _load_credentials(provider)
    if not credentials:
        console.print(f"[red]Not authenticated with {provider}[/red]")
        console.print(f"Run: rlm-dspy auth login {provider}")
        raise typer.Exit(1)
    
    try:
        if provider == "anthropic":
            new_creds = anthropic_refresh_token(credentials)
        else:
            console.print(f"[red]Unknown provider: {provider}[/red]")
            raise typer.Exit(1)
        
        expires = datetime.fromtimestamp(new_creds.expires_at, UTC)
        console.print(f"[green]✓ Token refreshed[/green]")
        console.print(f"  New expiry: {expires.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
    except Exception as e:
        console.print(f"[red]Refresh failed: {e}[/red]")
        console.print("You may need to login again: rlm-dspy auth login anthropic")
        raise typer.Exit(1)


def register_auth_commands(app: typer.Typer) -> None:
    """Register auth subcommand group."""
    app.add_typer(auth_app)

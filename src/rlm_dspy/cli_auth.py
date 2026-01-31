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


def _get_supported_providers() -> list[str]:
    """Get list of supported OAuth providers."""
    from .core.oauth import list_providers
    return list_providers()


@auth_app.command("login")
def auth_login(
    provider: Annotated[str, typer.Argument(help="Provider (google-gemini, antigravity)")] = "google-gemini",
) -> None:
    """Login with OAuth (opens browser for authentication).
    
    Supported providers:
    - google-gemini: Gemini models via Google Cloud Code Assist OAuth
    - antigravity: Gemini 3, Claude, GPT-OSS via Antigravity OAuth
    """
    from .core.oauth import authenticate, list_providers, AuthenticationError
    
    providers = list_providers()
    if provider not in providers and provider not in ["gemini"]:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        console.print(f"Supported providers: {', '.join(providers)}")
        raise typer.Exit(1)
    
    try:
        console.print(f"[dim]Authenticating with {provider}...[/dim]")
        credentials = authenticate(provider)
        
        console.print(f"\n[green]✓ Logged in to {provider}[/green]")
        
        if credentials.email:
            console.print(f"  Email: {credentials.email}")
        if credentials.project_id:
            console.print(f"  Project: {credentials.project_id}")
        
        expires = datetime.fromtimestamp(credentials.expires_at, UTC)
        console.print(f"  Expires: {expires.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
    except AuthenticationError as e:
        console.print(f"[red]Login failed: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Login cancelled[/yellow]")
        raise typer.Exit(130)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@auth_app.command("logout")
def auth_logout(
    provider: Annotated[str, typer.Argument(help="Provider (google-gemini, antigravity)")] = "google-gemini",
) -> None:
    """Logout and delete stored credentials."""
    from .core.oauth import revoke_credentials
    
    if revoke_credentials(provider):
        console.print(f"[green]✓ Logged out from {provider}[/green]")
    else:
        console.print(f"[dim]No credentials found for {provider}[/dim]")


@auth_app.command("status")
def auth_status(
    provider: Annotated[str, typer.Argument(help="Provider or 'all'")] = "all",
) -> None:
    """Show OAuth authentication status."""
    from .core.oauth import list_providers, get_credentials, is_authenticated
    
    providers = list_providers() if provider == "all" else [provider]
    
    table = Table(title="OAuth Authentication Status")
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Email")
    table.add_column("Project")
    table.add_column("Expires")
    
    for p in providers:
        creds = get_credentials(p)
        
        if creds:
            if creds.is_expired:
                status_str = "[yellow]Expired[/yellow]"
            else:
                status_str = "[green]Authenticated[/green]"
            
            email = creds.email or "-"
            project = creds.project_id or "-"
            expires = datetime.fromtimestamp(creds.expires_at, UTC)
            expires_str = expires.strftime("%Y-%m-%d %H:%M")
        else:
            status_str = "[dim]Not authenticated[/dim]"
            email = "-"
            project = "-"
            expires_str = "-"
        
        table.add_row(p, status_str, email, project, expires_str)
    
    console.print(table)
    
    # Show hint
    supported = list_providers()
    console.print(f"\n[dim]Use 'rlm-dspy auth login <provider>' to authenticate[/dim]")
    console.print(f"[dim]Supported providers: {', '.join(supported)}[/dim]")


@auth_app.command("refresh")
def auth_refresh(
    provider: Annotated[str, typer.Argument(help="Provider (google-gemini, antigravity)")] = "google-gemini",
) -> None:
    """Manually refresh OAuth token."""
    from .core.oauth import refresh_credentials, get_credentials, TokenRefreshError
    
    creds = get_credentials(provider)
    if not creds:
        console.print(f"[red]Not authenticated with {provider}[/red]")
        console.print(f"Run: rlm-dspy auth login {provider}")
        raise typer.Exit(1)
    
    try:
        new_creds = refresh_credentials(provider)
        expires = datetime.fromtimestamp(new_creds.expires_at, UTC)
        console.print(f"[green]✓ Token refreshed[/green]")
        console.print(f"  New expiry: {expires.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    except TokenRefreshError as e:
        console.print(f"[red]Refresh failed: {e}[/red]")
        console.print(f"You may need to login again: rlm-dspy auth login {provider}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@auth_app.command("list")
def auth_list() -> None:
    """List all authenticated providers."""
    from .core.oauth import list_authenticated, list_providers
    
    authenticated = list_authenticated()
    all_providers = list_providers()
    
    if authenticated:
        console.print("[green]Authenticated providers:[/green]")
        for p in authenticated:
            console.print(f"  • {p}")
    else:
        console.print("[dim]No authenticated providers[/dim]")
    
    unauthenticated = [p for p in all_providers if p not in authenticated]
    if unauthenticated:
        console.print(f"\n[dim]Available: {', '.join(unauthenticated)}[/dim]")


def register_auth_commands(app: typer.Typer) -> None:
    """Register auth subcommand group."""
    app.add_typer(auth_app)

"""CLI commands for OAuth authentication."""

from __future__ import annotations

from datetime import datetime, UTC
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

# Supported OAuth providers
SUPPORTED_PROVIDERS = ["anthropic", "google", "antigravity"]

auth_app = typer.Typer(
    name="auth",
    help="OAuth authentication for LLM providers",
    no_args_is_help=True,
)


@auth_app.command("login")
def auth_login(
    provider: Annotated[str, typer.Argument(help="Provider (anthropic, google, antigravity)")] = "anthropic",
    no_browser: Annotated[bool, typer.Option("--no-browser", help="Manual mode: paste redirect URL (for SSH/headless)")] = False,
) -> None:
    """Login with OAuth (opens browser for authentication).
    
    Supported providers:
    - anthropic: Claude Pro/Max via claude.ai OAuth
    - google: Gemini models via Google Cloud Code Assist OAuth
    - antigravity: Gemini 3, Claude, GPT-OSS via Antigravity OAuth
    
    For SSH/headless servers, use --no-browser to manually paste the redirect URL.
    """
    if provider not in SUPPORTED_PROVIDERS:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        console.print(f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}")
        raise typer.Exit(1)
    
    try:
        if provider == "google":
            from .core.google_oauth import google_login
            credentials = google_login(open_browser=not no_browser, manual=no_browser)
        elif provider == "antigravity":
            from .core.antigravity_oauth import antigravity_login
            credentials = antigravity_login(open_browser=not no_browser, manual=no_browser)
        else:
            from .core.oauth import oauth_login
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
    provider: Annotated[str, typer.Argument(help="Provider (anthropic, google)")] = "anthropic",
) -> None:
    """Logout and delete stored credentials."""
    from .core.oauth import oauth_logout
    
    if oauth_logout(provider):
        console.print(f"[green]✓ Logged out from {provider}[/green]")
    else:
        console.print(f"[dim]No credentials found for {provider}[/dim]")


@auth_app.command("status")
def auth_status(
    provider: Annotated[str, typer.Argument(help="Provider (anthropic, google, or 'all')")] = "all",
) -> None:
    """Show OAuth authentication status."""
    from .core.oauth import oauth_status, OAUTH_PROVIDERS
    
    providers = [provider] if provider != "all" else OAUTH_PROVIDERS
    
    table = Table(title="OAuth Authentication Status")
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Details")
    table.add_column("Expires")
    
    for p in providers:
        status = oauth_status(p)
        
        if status["authenticated"]:
            if status["is_expired"]:
                status_str = "[yellow]Expired (will refresh)[/yellow]"
            else:
                status_str = "[green]Authenticated[/green]"
            
            expires = datetime.fromtimestamp(status["expires_at"], UTC)
            expires_str = expires.strftime("%Y-%m-%d %H:%M")
            
            # Provider-specific details
            if p == "google":
                details = status.get("email") or status.get("project_id", "-")
            else:
                details = "-"
        else:
            status_str = "[dim]Not authenticated[/dim]"
            expires_str = "-"
            details = "-"
        
        table.add_row(p, status_str, details, expires_str)
    
    console.print(table)
    
    # Show hint
    console.print("\n[dim]Use 'rlm-dspy auth login <provider>' to authenticate[/dim]")
    console.print(f"[dim]Supported providers: {', '.join(OAUTH_PROVIDERS)}[/dim]")


@auth_app.command("refresh")
def auth_refresh(
    provider: Annotated[str, typer.Argument(help="Provider (anthropic, google, antigravity)")] = "anthropic",
) -> None:
    """Manually refresh OAuth token."""
    from .core.oauth import _load_credentials, anthropic_refresh_token
    
    if provider == "google":
        from .core.google_oauth import _load, google_refresh_token
        credentials = _load()
        if not credentials:
            console.print(f"[red]Not authenticated with {provider}[/red]")
            console.print(f"Run: rlm-dspy auth login {provider}")
            raise typer.Exit(1)
        
        try:
            new_creds = google_refresh_token(credentials)
            expires = datetime.fromtimestamp(new_creds.expires_at, UTC)
            console.print(f"[green]✓ Token refreshed[/green]")
            console.print(f"  New expiry: {expires.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        except Exception as e:
            console.print(f"[red]Refresh failed: {e}[/red]")
            console.print("You may need to login again: rlm-dspy auth login google")
            raise typer.Exit(1)
        return
    
    if provider == "antigravity":
        from .core.antigravity_oauth import _load as _load_ag, antigravity_refresh_token
        credentials = _load_ag()
        if not credentials:
            console.print(f"[red]Not authenticated with {provider}[/red]")
            console.print(f"Run: rlm-dspy auth login {provider}")
            raise typer.Exit(1)
        
        try:
            new_creds = antigravity_refresh_token(credentials)
            expires = datetime.fromtimestamp(new_creds.expires_at, UTC)
            console.print(f"[green]✓ Token refreshed[/green]")
            console.print(f"  New expiry: {expires.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        except Exception as e:
            console.print(f"[red]Refresh failed: {e}[/red]")
            console.print("You may need to login again: rlm-dspy auth login antigravity")
            raise typer.Exit(1)
        return
    
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
        console.print(f"You may need to login again: rlm-dspy auth login {provider}")
        raise typer.Exit(1)


def register_auth_commands(app: typer.Typer) -> None:
    """Register auth subcommand group."""
    app.add_typer(auth_app)

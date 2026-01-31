"""CLI commands for model listing and selection."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

models_app = typer.Typer(
    name="models",
    help="List and manage available models",
    no_args_is_help=False,
)


def _format_tokens(count: int) -> str:
    """Format token count as human-readable (e.g., 200K, 1M)."""
    if count >= 1_000_000:
        m = count / 1_000_000
        return f"{m:.0f}M" if m == int(m) else f"{m:.1f}M"
    if count >= 1_000:
        k = count / 1_000
        return f"{k:.0f}K" if k == int(k) else f"{k:.1f}K"
    return str(count)


def _format_cost(cost: float) -> str:
    """Format cost per million tokens."""
    if cost == 0:
        return "-"
    if cost < 0.01:
        return f"${cost:.3f}"
    if cost < 1:
        return f"${cost:.2f}"
    return f"${cost:.1f}"


@models_app.command("list")
def models_list(
    search: Annotated[str | None, typer.Argument(help="Search pattern (fuzzy)")] = None,
    all_models: Annotated[bool, typer.Option("--all", "-a", help="Show all models (not just available)")] = False,
    provider: Annotated[str | None, typer.Option("--provider", "-p", help="Filter by provider")] = None,
    reasoning: Annotated[bool | None, typer.Option("--reasoning/--no-reasoning", "-r/-R", help="Filter by reasoning support")] = None,
) -> None:
    """List available models."""
    
    from .core.models import get_model_registry
    
    registry = get_model_registry()
    
    # Get models
    if all_models:
        models = registry.get_all()
    else:
        models = registry.get_available()
    
    if not models:
        if all_models:
            console.print("[dim]No models registered[/dim]")
        else:
            console.print("[yellow]No models available.[/yellow]")
            console.print("\nTo use models, configure authentication:")
            console.print("  • [cyan]rlm-dspy auth login anthropic[/cyan] - Login with Claude Pro/Max")
            console.print("  • Set [cyan]OPENROUTER_API_KEY[/cyan] environment variable")
            console.print("  • Set [cyan]ANTHROPIC_API_KEY[/cyan] environment variable")
            console.print("  • Set [cyan]OPENAI_API_KEY[/cyan] environment variable")
            console.print("\nRun [cyan]rlm-dspy models --all[/cyan] to see all models")
        return
    
    # Apply filters
    if provider:
        models = [m for m in models if m.provider.lower() == provider.lower()]
    
    if reasoning is not None:
        models = [m for m in models if m.reasoning == reasoning]
    
    if search:
        models = registry.search(search)
        if all_models:
            # search() returns all matching, filter to just available if not --all
            pass
        else:
            available_ids = {m.id for m in registry.get_available()}
            models = [m for m in models if m.id in available_ids]
    
    if not models:
        if search:
            console.print(f"[yellow]No models matching '{search}'[/yellow]")
        else:
            console.print("[yellow]No models match the filters[/yellow]")
        return
    
    # Sort by provider, then ID
    models.sort(key=lambda m: (m.provider, m.id))
    
    # Build table
    table = Table(title="Available Models" if not all_models else "All Models")
    table.add_column("Provider", style="cyan")
    table.add_column("Model ID")
    table.add_column("Context", justify="right")
    table.add_column("Max Out", justify="right")
    table.add_column("Thinking", justify="center")
    table.add_column("Images", justify="center")
    table.add_column("Cost (in/out)", justify="right")
    
    for m in models:
        cost_str = f"{_format_cost(m.cost.input)}/{_format_cost(m.cost.output)}"
        table.add_row(
            m.provider,
            m.id.split("/", 1)[-1] if "/" in m.id else m.id,
            _format_tokens(m.context_window),
            _format_tokens(m.max_tokens),
            "✓" if m.reasoning else "",
            "✓" if m.supports_images else "",
            cost_str,
        )
    
    console.print(table)
    console.print(f"\n[dim]Showing {len(models)} models[/dim]")
    
    if not all_models:
        all_count = len(registry.get_all())
        if all_count > len(models):
            console.print(f"[dim]Use --all to see all {all_count} models[/dim]")


@models_app.command("providers")
def models_providers(
    show_all: Annotated[bool, typer.Option("--all", "-a", help="Show all providers")] = False,
) -> None:
    """List available providers and their authentication status."""
    from .core.models import get_model_registry, has_provider_auth, PROVIDER_ENV_VARS
    from .core.oauth import is_anthropic_authenticated
    
    registry = get_model_registry()
    
    table = Table(title="Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Auth Method")
    table.add_column("Models")
    
    providers = registry.get_providers()
    
    for provider in providers:
        has_auth = has_provider_auth(provider)
        model_count = len(registry.get_by_provider(provider))
        
        if provider == "anthropic" and is_anthropic_authenticated():
            auth_method = "OAuth (Claude Pro/Max)"
            status = "[green]✓ Authenticated[/green]"
        elif has_auth:
            env_vars = PROVIDER_ENV_VARS.get(provider, [])
            auth_method = f"API Key ({env_vars[0]})" if env_vars else "API Key"
            status = "[green]✓ Configured[/green]"
        else:
            env_vars = PROVIDER_ENV_VARS.get(provider, [])
            auth_method = f"Set {env_vars[0]}" if env_vars else "API Key required"
            status = "[dim]Not configured[/dim]"
            
            if not show_all:
                continue
        
        table.add_row(provider, status, auth_method, str(model_count))
    
    console.print(table)
    
    if not show_all:
        console.print("\n[dim]Use --all to see unconfigured providers[/dim]")


@models_app.command("info")
def models_info(
    model_id: Annotated[str, typer.Argument(help="Model ID (fuzzy search)")],
) -> None:
    """Show detailed information about a model."""
    from .core.models import get_model_registry, has_provider_auth
    
    registry = get_model_registry()
    
    # Try exact match first
    model = registry.find(model_id)
    
    if not model:
        # Try search
        matches = registry.search(model_id)
        if len(matches) == 1:
            model = matches[0]
        elif len(matches) > 1:
            console.print(f"[yellow]Multiple models match '{model_id}':[/yellow]")
            for m in matches[:10]:
                console.print(f"  • {m.id}")
            if len(matches) > 10:
                console.print(f"  ... and {len(matches) - 10} more")
            return
        else:
            console.print(f"[red]No model found matching '{model_id}'[/red]")
            return
    
    # Display model info
    table = Table(title=f"Model: {model.name}", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    
    table.add_row("ID", model.id)
    table.add_row("Name", model.name)
    table.add_row("Provider", model.provider)
    table.add_row("API", model.api)
    table.add_row("", "")
    table.add_row("Context Window", _format_tokens(model.context_window))
    table.add_row("Max Output", _format_tokens(model.max_tokens))
    table.add_row("Reasoning/Thinking", "Yes" if model.reasoning else "No")
    table.add_row("Image Input", "Yes" if model.supports_images else "No")
    table.add_row("", "")
    table.add_row("Cost (input)", f"${model.cost.input:.2f}/M tokens")
    table.add_row("Cost (output)", f"${model.cost.output:.2f}/M tokens")
    if model.cost.cache_read > 0:
        table.add_row("Cost (cache read)", f"${model.cost.cache_read:.2f}/M tokens")
    if model.cost.cache_write > 0:
        table.add_row("Cost (cache write)", f"${model.cost.cache_write:.2f}/M tokens")
    table.add_row("", "")
    
    has_auth = has_provider_auth(model.provider)
    if has_auth:
        table.add_row("Status", "[green]✓ Available[/green]")
    else:
        table.add_row("Status", "[yellow]⚠ Auth required[/yellow]")
    
    console.print(table)
    
    if not has_auth:
        console.print(f"\n[yellow]To use this model, configure authentication for {model.provider}[/yellow]")


@models_app.command("set")
def models_set(
    model_id: Annotated[str, typer.Argument(help="Model ID to set as default")],
) -> None:
    """Set the default model in config."""
    from .core.models import get_model_registry, has_provider_auth
    from .core.user_config import load_config, save_config
    
    registry = get_model_registry()
    
    # Find model
    model = registry.find(model_id)
    if not model:
        matches = registry.search(model_id)
        if len(matches) == 1:
            model = matches[0]
        elif len(matches) > 1:
            console.print(f"[yellow]Multiple models match '{model_id}':[/yellow]")
            for m in matches[:5]:
                console.print(f"  • {m.id}")
            return
        else:
            console.print(f"[red]No model found matching '{model_id}'[/red]")
            return
    
    # Check auth
    if not has_provider_auth(model.provider):
        console.print(f"[yellow]Warning: {model.provider} authentication not configured[/yellow]")
    
    # Update config
    config = load_config()
    config["model"] = model.id
    save_config(config, use_template=False)
    
    console.print(f"[green]✓ Default model set to: {model.id}[/green]")


def register_models_commands(app: typer.Typer) -> None:
    """Register models subcommand group."""
    app.add_typer(models_app)

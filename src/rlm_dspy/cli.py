"""CLI for RLM-DSPy - Recursive Language Models with DSPy optimization."""

from __future__ import annotations

import typer

# Create main app
app = typer.Typer(
    name="rlm-dspy",
    help="Recursive Language Models with DSPy optimization",
    no_args_is_help=True,
)

# Register main commands (ask, analyze, diff, setup, config, preflight, example)
from .cli_main import register_commands
register_commands(app)

# Register subcommand groups
from .cli_index import index_app
from .cli_project import project_app
from .cli_daemon import daemon_app
from .cli_traces import traces_app
from .cli_optimize import optimize_app
from .cli_auth import auth_app

app.add_typer(index_app, name="index")
app.add_typer(project_app, name="project")
app.add_typer(daemon_app, name="daemon")
app.add_typer(traces_app, name="traces")
app.add_typer(optimize_app, name="optimize")
app.add_typer(auth_app, name="auth")


if __name__ == "__main__":
    app()

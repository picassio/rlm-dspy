"""Validation utilities for pre-flight checks and config validation.

Learned from modaic: validate before expensive operations.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

console = Console(stderr=True)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    suggestion: str | None = None


@dataclass
class PreflightResult:
    """Result of all preflight checks."""

    checks: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    @property
    def errors(self) -> list[ValidationResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[ValidationResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    def add(self, result: ValidationResult) -> None:
        self.checks.append(result)

    def print_report(self) -> None:
        """Print a formatted report of all checks."""
        table = Table(title="Preflight Checks", show_header=True)
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Message")

        for check in self.checks:
            if check.passed:
                status = "[green]✓[/green]"
            elif check.severity == "warning":
                status = "[yellow]⚠[/yellow]"
            else:
                status = "[red]✗[/red]"

            msg = check.message
            if check.suggestion and not check.passed:
                msg += f"\n[dim]{check.suggestion}[/dim]"

            table.add_row(check.name, status, msg)

        console.print(table)

        if self.passed:
            console.print("\n[green]✓ All preflight checks passed![/green]")
        else:
            console.print(f"\n[red]✗ {len(self.errors)} error(s), {len(self.warnings)} warning(s)[/red]")


def check_api_key(
    env_vars: list[str] | None = None,
    required: bool = True,
) -> ValidationResult:
    """Check if API key is configured."""
    env_vars = env_vars or ["RLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]

    for var in env_vars:
        if os.environ.get(var):
            return ValidationResult(
                name="API Key",
                passed=True,
                message=f"Found in ${var}",
            )

    return ValidationResult(
        name="API Key",
        passed=not required,
        message="No API key found",
        severity="error" if required else "warning",
        suggestion=f"Set one of: {', '.join(env_vars)}",
    )


def check_model_format(model: str) -> ValidationResult:
    """Check if model name is valid."""
    # Valid patterns: provider/model or just model
    valid_providers = ["openrouter", "openai", "anthropic", "together", "google", "azure"]

    if "/" in model:
        provider = model.split("/")[0]
        if provider in valid_providers:
            return ValidationResult(
                name="Model Format",
                passed=True,
                message=f"Valid provider: {provider}",
            )
        # Could be openrouter/google/model format
        if model.count("/") >= 2:
            return ValidationResult(
                name="Model Format",
                passed=True,
                message=f"OpenRouter model: {model}",
            )

    return ValidationResult(
        name="Model Format",
        passed=True,  # Allow any format, just warn
        message=f"Model: {model}",
        severity="info",
    )


def check_api_endpoint(api_base: str) -> ValidationResult:
    """Check if API endpoint is reachable."""
    import httpx

    try:
        # Just check if the host responds (don't need auth for this)
        with httpx.Client(timeout=5) as client:
            response = client.get(api_base.rstrip("/") + "/models", headers={"Authorization": "Bearer test"})
            # Even 401 means the endpoint is reachable
            if response.status_code in (200, 401, 403):
                return ValidationResult(
                    name="API Endpoint",
                    passed=True,
                    message=f"Reachable: {api_base}",
                )
    except httpx.ConnectError:
        pass
    except Exception as e:
        return ValidationResult(
            name="API Endpoint",
            passed=False,
            message=f"Connection failed: {str(e)[:50]}",
            severity="error",
            suggestion="Check network connection and API URL",
        )

    return ValidationResult(
        name="API Endpoint",
        passed=False,
        message=f"Cannot reach: {api_base}",
        severity="error",
        suggestion="Check RLM_API_BASE environment variable",
    )


def check_budget(budget: float, context_size: int) -> ValidationResult:
    """Check if budget is sufficient for context size."""
    # Rough estimate: $0.50 per 1M tokens input for Gemini Flash
    estimated_tokens = context_size / 4  # ~4 chars per token
    estimated_cost = (estimated_tokens / 1_000_000) * 0.50

    if budget < estimated_cost:
        return ValidationResult(
            name="Budget",
            passed=False,
            message=f"Budget ${budget:.2f} may be insufficient",
            severity="warning",
            suggestion=f"Estimated cost: ${estimated_cost:.4f} for {estimated_tokens:,.0f} tokens",
        )

    return ValidationResult(
        name="Budget",
        passed=True,
        message=f"Budget ${budget:.2f} (estimated: ${estimated_cost:.4f})",
    )


def check_context_size(context: str, max_chunk_size: int) -> ValidationResult:
    """Check context size and chunking."""
    size = len(context)
    chunks_needed = max(1, size // max_chunk_size + 1)

    if size == 0:
        return ValidationResult(
            name="Context Size",
            passed=False,
            message="Empty context",
            severity="error",
            suggestion="Provide files or stdin input",
        )

    if chunks_needed > 100:
        return ValidationResult(
            name="Context Size",
            passed=True,
            message=f"{size:,} chars → {chunks_needed} chunks",
            severity="warning",
            suggestion="Large context may be slow. Consider filtering files.",
        )

    return ValidationResult(
        name="Context Size",
        passed=True,
        message=f"{size:,} chars → {chunks_needed} chunk(s)",
    )


def preflight_check(
    api_key_required: bool = True,
    model: str | None = None,
    api_base: str | None = None,
    budget: float | None = None,
    context: str | None = None,
    chunk_size: int = 100_000,
    check_network: bool = True,
) -> PreflightResult:
    """
    Run all preflight checks before an expensive operation.

    Learned from modaic: validate config before batch jobs.

    Args:
        api_key_required: Whether API key is required
        model: Model name to validate
        api_base: API endpoint to check
        budget: Budget to validate
        context: Context string to check size
        chunk_size: Chunk size for estimation
        check_network: Whether to check API endpoint

    Returns:
        PreflightResult with all check results
    """
    result = PreflightResult()

    # API Key
    result.add(check_api_key(required=api_key_required))

    # Model format
    if model:
        result.add(check_model_format(model))

    # API endpoint (optional, can be slow)
    if check_network and api_base:
        result.add(check_api_endpoint(api_base))

    # Context size
    if context is not None:
        result.add(check_context_size(context, chunk_size))

        # Budget
        if budget is not None:
            result.add(check_budget(budget, len(context)))

    return result


def validate_project_name(name: str) -> ValidationResult:
    """
    Validate a project/model name.

    Learned from modaic: enforce naming conventions.
    """
    # Valid: lowercase, numbers, hyphens, underscores
    pattern = r"^[a-z][a-z0-9_-]*$"

    if not name:
        return ValidationResult(
            name="Project Name",
            passed=False,
            message="Name cannot be empty",
            severity="error",
        )

    if len(name) > 64:
        return ValidationResult(
            name="Project Name",
            passed=False,
            message=f"Name too long ({len(name)} > 64 chars)",
            severity="error",
        )

    if not re.match(pattern, name):
        return ValidationResult(
            name="Project Name",
            passed=False,
            message=f"Invalid name: {name}",
            severity="error",
            suggestion="Use lowercase letters, numbers, hyphens, underscores. Start with letter.",
        )

    return ValidationResult(
        name="Project Name",
        passed=True,
        message=f"Valid: {name}",
    )


def validate_jsonl_file(path: str) -> ValidationResult:
    """
    Validate a JSONL file for batch processing.

    Learned from modaic: validate before submission.
    """
    import json
    from pathlib import Path

    p = Path(path)

    if not p.exists():
        return ValidationResult(
            name="JSONL File",
            passed=False,
            message=f"File not found: {path}",
            severity="error",
        )

    try:
        lines = p.read_text().strip().split("\n")
        valid_lines = 0

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "custom_id" not in data:
                    return ValidationResult(
                        name="JSONL File",
                        passed=False,
                        message=f"Line {i + 1} missing custom_id",
                        severity="error",
                    )
                valid_lines += 1
            except json.JSONDecodeError as e:
                return ValidationResult(
                    name="JSONL File",
                    passed=False,
                    message=f"Invalid JSON on line {i + 1}: {e}",
                    severity="error",
                )

        return ValidationResult(
            name="JSONL File",
            passed=True,
            message=f"Valid: {valid_lines} requests",
        )

    except Exception as e:
        return ValidationResult(
            name="JSONL File",
            passed=False,
            message=f"Error reading file: {e}",
            severity="error",
        )

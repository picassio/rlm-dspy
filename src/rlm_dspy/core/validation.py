"""Validation utilities for pre-flight checks and config validation.

Learned from modaic: validate before expensive operations.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
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
    except httpx.ConnectError as e:
        logger.debug("API endpoint connection failed: %s", e)
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


def check_budget(budget: float, context_size: int, model: str = "gemini-2.0-flash") -> ValidationResult:
    """Check if budget is sufficient for context size."""
    from .token_stats import estimate_cost

    # Estimate tokens (~4 chars per token)
    estimated_tokens = int(context_size / 4)
    # Assume output is ~20% of input for estimation
    estimated_output = int(estimated_tokens * 0.2)
    estimated_cost = estimate_cost(estimated_tokens, estimated_output, model)

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


def check_context_size(context: str) -> ValidationResult:
    """Check context size is valid."""
    size = len(context)

    if size == 0:
        return ValidationResult(
            name="Context Size",
            passed=False,
            message="Empty context",
            severity="error",
            suggestion="Provide files or stdin input",
        )

    if size > 10_000_000:
        return ValidationResult(
            name="Context Size",
            passed=True,
            message=f"{size:,} chars",
            severity="warning",
            suggestion="Very large context may be slow. Consider filtering files.",
        )

    return ValidationResult(
        name="Context Size",
        passed=True,
        message=f"{size:,} chars",
    )


def preflight_check(
    api_key_required: bool = True,
    model: str | None = None,
    api_base: str | None = None,
    budget: float | None = None,
    context: str | None = None,
    check_network: bool = True,
) -> PreflightResult:
    """
    Run all preflight checks before an expensive operation.

    Args:
        api_key_required: Whether API key is required
        model: Model name to validate
        api_base: API endpoint to check
        budget: Budget to validate
        context: Context string to check size
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
        result.add(check_context_size(context))

        # Budget
        if budget is not None:
            result.add(check_budget(budget, len(context)))

    return result

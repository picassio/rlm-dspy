"""CLI commands for optimization management."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

optimize_app = typer.Typer(help="Manage prompt optimization", no_args_is_help=True)


@optimize_app.command("stats")
def optimize_stats() -> None:
    """Show optimization statistics."""
    from .core.instruction_optimizer import get_instruction_optimizer
    from .core.grounded_proposer import get_grounded_proposer

    optimizer = get_instruction_optimizer()
    proposer = get_grounded_proposer()

    opt_stats = optimizer.get_stats()
    prop_stats = proposer.get_stats()

    table = Table(title="Optimization Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("[bold]Instruction Optimizer[/bold]", "")
    table.add_row("Total Outcomes", str(opt_stats.get("total_outcomes", 0)))
    table.add_row("Success Rate", f"{opt_stats.get('success_rate', 0):.1%}")
    table.add_row("Active Candidates", str(opt_stats.get("active_candidates", 0)))

    table.add_row("", "")
    table.add_row("[bold]Grounded Proposer[/bold]", "")
    table.add_row("Recorded Failures", str(prop_stats.get("total_failures", 0)))
    table.add_row("Recorded Successes", str(prop_stats.get("total_successes", 0)))
    table.add_row("Current Tips", str(prop_stats.get("current_tips_count", 0)))

    console.print(table)


@optimize_app.command("tips")
def optimize_tips(
    refresh: Annotated[bool, typer.Option("--refresh", "-r", help="Force refresh tips")] = False,
    clear: Annotated[bool, typer.Option("--clear", "-c", help="Clear all tips")] = False,
) -> None:
    """Show or manage optimization tips."""
    from .core.grounded_proposer import get_grounded_proposer

    proposer = get_grounded_proposer()

    if clear:
        proposer.reset_tips()
        console.print("[green]✓[/green] Cleared all tips")
        return

    if refresh:
        tips = proposer.generate_tips()
        console.print(f"[green]✓[/green] Refreshed tips ({len(tips)} generated)")
    else:
        tips = proposer.get_tips()

    if not tips:
        console.print("[yellow]No tips available. Run some queries to generate tips.[/yellow]")
        return

    console.print("\n[bold cyan]Current Optimization Tips:[/bold cyan]\n")
    for i, tip in enumerate(tips, 1):
        console.print(f"  {i}. {tip}")


@optimize_app.command("clear")
def optimize_clear(
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Clear optimization data."""
    from .core.instruction_optimizer import get_instruction_optimizer
    from .core.grounded_proposer import get_grounded_proposer

    if not force:
        if not typer.confirm("Clear all optimization data?"):
            raise typer.Abort()

    optimizer = get_instruction_optimizer()
    proposer = get_grounded_proposer()

    opt_cleared = optimizer.clear()
    failures, successes = proposer.clear()

    console.print(f"[green]✓[/green] Cleared {opt_cleared} optimizer outcomes")
    console.print(f"[green]✓[/green] Cleared {failures} failures and {successes} successes from proposer")


@optimize_app.command("instructions")
def optimize_instructions(
    key: Annotated[str | None, typer.Argument(help="Instruction key to show/modify")] = None,
    reset: Annotated[bool, typer.Option("--reset", help="Reset to default")] = False,
    propose: Annotated[bool, typer.Option("--propose", "-p", help="Propose improvement")] = False,
) -> None:
    """Show or modify tool instructions."""
    from .core.instruction_optimizer import get_instruction_optimizer, DEFAULT_INSTRUCTIONS

    optimizer = get_instruction_optimizer()

    if key is None:
        table = Table(title="Instruction Keys")
        table.add_column("Key")
        table.add_column("Length", justify="right")
        table.add_column("Modified")

        for k in DEFAULT_INSTRUCTIONS:
            current = optimizer.get_instruction(k)
            default = DEFAULT_INSTRUCTIONS[k]
            modified = "✓" if current != default else ""
            table.add_row(k, str(len(current)), modified)

        console.print(table)
        return

    if key not in DEFAULT_INSTRUCTIONS:
        console.print(f"[red]Unknown key: {key}[/red]")
        console.print(f"Available: {', '.join(DEFAULT_INSTRUCTIONS.keys())}")
        raise typer.Exit(1)

    if reset:
        optimizer.reset_instruction(key)
        console.print(f"[green]✓[/green] Reset '{key}' to default")
        return

    if propose:
        console.print(f"[cyan]Proposing improvement for '{key}'...[/cyan]")
        new_instruction = optimizer.propose_improvement(key)
        if new_instruction:
            console.print("\n[bold]Proposed instruction:[/bold]")
            console.print(new_instruction[:500] + "..." if len(new_instruction) > 500 else new_instruction)
        else:
            console.print("[yellow]No improvement proposed[/yellow]")
        return

    # Show current instruction
    current = optimizer.get_instruction(key)
    console.print(f"\n[bold cyan]{key}:[/bold cyan]\n")
    console.print(current)


@optimize_app.command("patterns")
def optimize_patterns() -> None:
    """Show learned patterns from past queries."""
    from .core.grounded_proposer import get_grounded_proposer

    proposer = get_grounded_proposer()
    stats = proposer.get_stats()

    failure_patterns = stats.get("failure_patterns", {})
    success_patterns = stats.get("success_patterns", {})

    if not failure_patterns and not success_patterns:
        console.print("[yellow]No patterns learned yet. Run more queries to learn patterns.[/yellow]")
        return

    if failure_patterns:
        console.print("\n[bold red]Failure Patterns:[/bold red]")
        top_reasons = failure_patterns.get("top_reasons", [])
        for reason, count in top_reasons[:10]:
            console.print(f"  {count}x: {reason[:60]}{'...' if len(reason) > 60 else ''}")
        
        missing_tools = failure_patterns.get("missing_tools", [])
        if missing_tools:
            console.print("\n[bold yellow]Missing Tools:[/bold yellow]")
            for tool, count in missing_tools[:5]:
                console.print(f"  {count}x: {tool}")

    if success_patterns:
        console.print("\n[bold green]Success Patterns:[/bold green]")
        tools_used = success_patterns.get("tools_used", [])
        for tool, count in tools_used[:10]:
            console.print(f"  {count}x: {tool}")


@optimize_app.command("simba")
def optimize_simba(
    min_score: Annotated[float, typer.Option("--min-score", "-s", help="Minimum trace score")] = 0.7,
    max_examples: Annotated[int, typer.Option("--max-examples", "-n", help="Maximum training examples")] = 100,
    batch_size: Annotated[int | None, typer.Option("--batch-size", "-b", help="Mini-batch size (smaller=faster)")] = None,
    steps: Annotated[int | None, typer.Option("--steps", help="Optimization steps (fewer=faster)")] = None,
    candidates: Annotated[int | None, typer.Option("--candidates", "-c", help="Candidates per step (fewer=faster)")] = None,
    threads: Annotated[int | None, typer.Option("--threads", "-t", help="Parallel threads (default: 2 to avoid rate limits)")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use (overrides optimization.model in config)")] = None,
    fast: Annotated[bool, typer.Option("--fast", "-f", help="Fast proxy mode (50x faster, like GEPA --fast)")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be optimized")] = False,
) -> None:
    """Run SIMBA self-improving optimization.

    Uses collected traces to optimize the RLM program via DSPy's SIMBA optimizer.

    \b
    Model priority:
      1. --model CLI option (highest)
      2. optimization.model in ~/.rlm/config.yaml
      3. model in ~/.rlm/config.yaml (default)

    \b
    Performance:
      --fast:   Proxy mode (1 LLM call per eval, ~1-3 min)
      Default:  Full RLM mode (10-50 LLM calls per eval, ~1-2 hours)

    \b
    Or customize individually:
      --steps 2 --candidates 3 --threads 4
    """
    # Apply presets
    # Note: threads default to 2 to avoid rate limiting on most APIs
    if fast:
        steps = steps or 1
        candidates = candidates or 2
        threads = threads or 2
        batch_size = batch_size or 8
    else:
        steps = steps or 4
        candidates = candidates or 4
        threads = threads or 2
        batch_size = batch_size or 16
    from .core.trace_collector import get_trace_collector
    from .core.simba_optimizer import get_simba_optimizer, create_training_example

    collector = get_trace_collector()
    # Filter traces by grounded_score
    all_traces = list(collector.traces)
    traces = [t for t in all_traces if t.grounded_score >= min_score][:max_examples]

    if not traces:
        console.print(f"[yellow]No traces found with score >= {min_score}[/yellow]")
        console.print("Run some queries first to collect training data.")
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(traces)} traces for training[/cyan]")

    if dry_run:
        from .core.user_config import OptimizationConfig, load_config
        user_cfg = load_config()
        opt_cfg = OptimizationConfig.from_user_config()
        default_model = user_cfg.get("model", "openai/gpt-4o-mini")
        model_name = model or opt_cfg.get_model(default_model)
        
        console.print("\n[bold]Would train on:[/bold]")
        for trace in traces[:10]:
            console.print(f"  - {trace.query[:50]}... (score: {trace.grounded_score:.2f})")
        if len(traces) > 10:
            console.print(f"  ... and {len(traces) - 10} more")
        console.print(f"\n[dim]Model: {model_name}[/dim]")
        console.print(f"[dim]Settings: steps={steps}, candidates={candidates}, batch_size={batch_size}[/dim]")
        return

    # Convert traces to training examples
    examples = []
    for trace in traces:
        example = create_training_example(
            query=trace.query,
            answer=trace.final_answer,
            context="",  # Context not stored in trace
        )
        if example:
            examples.append(example)

    if not examples:
        console.print("[red]No valid training examples could be created[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Created {len(examples)} training examples[/cyan]")

    try:
        from .core.rlm import RLM
        from .core.simba_optimizer import save_optimized_program, save_optimization_state, OptimizationState, get_trace_count
        from .core.user_config import OptimizationConfig, load_config, load_env_file
        from .core.optimization_state import OptimizationResult
        import dspy

        # Load env file for API keys
        load_env_file()

        # Get model from CLI, config, or default
        user_cfg = load_config()
        opt_cfg = OptimizationConfig.from_user_config()
        default_model = user_cfg.get("model", "openai/gpt-4o-mini")
        
        # Priority: CLI --model > optimization.model > main model
        model_name = model or opt_cfg.get_model(default_model)

        # Configure DSPy with the optimization model (handle custom providers)
        if model_name.startswith("zai/"):
            from .core.zai_lm import ZaiLM
            lm = ZaiLM(model_name.replace("zai/", ""))
        elif model_name.startswith("kimi/"):
            from .core.kimi_lm import KimiLM
            model_id = model_name.replace("kimi/", "")
            if model_id == "k2p5":
                model_id = "k2-0130-8k"
            lm = KimiLM(model_id)
        elif model_name.startswith("anthropic/"):
            # Use OAuth for Anthropic models
            from .core.anthropic_oauth_lm import get_anthropic_api_key, is_oauth_token, AnthropicOAuthLM
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY") or get_anthropic_api_key()
            if api_key and is_oauth_token(api_key):
                lm = AnthropicOAuthLM(
                    model=model_name.replace("anthropic/", ""),
                    api_key=api_key,
                )
            else:
                lm = dspy.LM(model_name, api_key=api_key)
        elif model_name.startswith("google/"):
            # Use OAuth for Google models
            from .core.oauth import get_google_token
            from .core.google_oauth_lm import GoogleOAuthLM
            token, project_id = get_google_token()
            if token:
                lm = GoogleOAuthLM(
                    model=model_name.replace("google/", ""),
                    access_token=token,
                    project_id=project_id,
                )
            else:
                lm = dspy.LM(model_name)
        else:
            lm = dspy.LM(model_name)
        dspy.configure(lm=lm)

        if fast:
            # FAST PROXY MODE: Use lightweight proxy instead of full RLM
            from .core.gepa_proxy import RLMProxy, create_proxy_metric, extract_proxy_instructions
            from dspy.teleprompt import SIMBA
            
            console.print(f"[cyan]Running SIMBA in FAST PROXY mode (steps={steps}, candidates={candidates})...[/cyan]")
            console.print(f"  [dim]Model: {model_name}[/dim]")
            console.print(f"  [dim]Proxy mode: 1 LLM call per eval instead of 10-50 (50x faster)[/dim]")
            
            # Get the RLM program and create proxy from it
            rlm = RLM()
            proxy = RLMProxy.from_rlm(rlm._rlm)
            
            # Adjust batch_size if trainset is smaller
            effective_batch_size = min(batch_size, len(examples))
            if effective_batch_size < batch_size:
                console.print(f"  [dim]Adjusted batch_size: {batch_size} -> {effective_batch_size} (trainset size)[/dim]")
            
            # Create proxy metric that returns float (SIMBA expects float, not Prediction)
            def simba_proxy_metric(example, pred):
                result = create_proxy_metric()(example, pred)
                return result.score if hasattr(result, 'score') else float(result)
            
            # Create SIMBA optimizer with proxy metric
            optimizer = SIMBA(
                metric=simba_proxy_metric,
                bsize=effective_batch_size,
                num_candidates=candidates,
                max_steps=steps,
                num_threads=threads,
            )
            
            # Run SIMBA on proxy
            optimized_proxy = optimizer.compile(proxy, trainset=examples)
            
            # Get scores from SIMBA
            if hasattr(optimizer, "best_score"):
                optimized_score = optimizer.best_score
                baseline_score = getattr(optimizer, "baseline_score", 0.0)
            else:
                optimized_score = 0.0
                baseline_score = 0.0
            
            improvement = ((optimized_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0.0
            
            result = OptimizationResult(
                improved=optimized_score > baseline_score,
                baseline_score=baseline_score,
                optimized_score=optimized_score,
                improvement=improvement,
                num_steps=steps,
                num_candidates=candidates,
                best_program_idx=0,
            )
            
            console.print("\n[bold green]✓ SIMBA (fast proxy) optimization complete![/bold green]")
            console.print(f"  Baseline score: {result.baseline_score:.2%}")
            console.print(f"  Optimized score: {result.optimized_score:.2%}")
            console.print(f"  Improvement: {result.improvement:.1f}%")
            
            # Extract and save instructions from proxy
            instructions = extract_proxy_instructions(optimized_proxy)
            
            # Extract demos if available
            demos = []
            if hasattr(optimized_proxy, 'demos') and optimized_proxy.demos:
                demos = optimized_proxy.demos
            elif hasattr(optimized_proxy, 'predict') and hasattr(optimized_proxy.predict, 'demos'):
                demos = optimized_proxy.predict.demos or []
            
            if result.improved or demos or instructions:
                save_optimized_program(
                    optimized_proxy,
                    result,
                    "simba",
                    instructions=instructions,
                )
                console.print("\n[green]✓ Optimized program saved - will be auto-loaded on next query[/green]")
                if demos:
                    console.print(f"  [dim]Saved {len(demos)} demos[/dim]")
                if instructions:
                    console.print(f"  [dim]Saved {len(instructions)} instruction keys[/dim]")
        else:
            # FULL RLM MODE
            console.print(f"[cyan]Running SIMBA optimization (steps={steps}, candidates={candidates}, threads={threads})...[/cyan]")
            console.print(f"  [dim]Model: {model_name}[/dim]")
            console.print(f"  [dim]Full RLM mode: 10-50 LLM calls per eval (use --fast for 50x speedup)[/dim]")
            
            # Get the RLM program to optimize
            rlm = RLM()
            optimizer = get_simba_optimizer(
                batch_size=batch_size,
                num_candidates=candidates,
                max_steps=steps,
                num_threads=threads,
            )

            optimized_program, result = optimizer.optimize(
                program=rlm._rlm,
                trainset=examples,
                lm=None,  # Use dspy.settings.lm
            )

            console.print("\n[bold green]✓ Optimization complete![/bold green]")
            console.print(f"  Baseline score: {result.baseline_score:.2%}")
            console.print(f"  Optimized score: {result.optimized_score:.2%}")
            console.print(f"  Improvement: {result.improvement:.1f}%")

            if result.improved:
                # Save the optimized program
                save_optimized_program(optimized_program, result, "simba")
                console.print("\n[green]✓ Optimized program saved - will be auto-loaded on next query[/green]")

        # Update optimization state
        from datetime import datetime, UTC
        state = OptimizationState(
            last_optimization=datetime.now(UTC),
            traces_at_last_optimization=get_trace_count(),
            last_result=result,
            optimizer_type="simba",
        )
        save_optimization_state(state)

    except Exception as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@optimize_app.command("gepa")
def optimize_gepa(
    min_score: Annotated[float, typer.Option("--min-score", "-s", help="Min trace score for training")] = 0.7,
    max_examples: Annotated[int, typer.Option("--max-examples", "-n", help="Max training examples")] = 50,
    auto: Annotated[str | None, typer.Option("--auto", "-a", help="Budget preset: test, light, medium, heavy (default: from config)")] = None,
    max_evals: Annotated[int | None, typer.Option("--max-evals", "-e", help="Max full evaluations (overrides --auto)")] = None,
    threads: Annotated[int | None, typer.Option("--threads", "-t", help="Number of parallel threads (default: from config)")] = None,
    teacher: Annotated[str | None, typer.Option("--teacher", help="Teacher/reflection model (overrides config)")] = None,
    tool_optimization: Annotated[bool, typer.Option("--tools", help="Enable tool optimization")] = False,
    fast_proxy: Annotated[bool | None, typer.Option("--fast", "-f", help="Use fast proxy mode (default: from config)")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be optimized")] = False,
) -> None:
    """Run GEPA reflective prompt evolution.
    
    GEPA uses execution traces and textual feedback to evolve prompts.
    Better for complex multi-step agents like RLM.
    
    Modes:
      Default    - Full RLM evaluation (slow but accurate, 30-120s per eval)
      --fast     - Proxy mode (50x faster, 1-2s per eval, instructions only)
    
    Budget options (use ONE):
      --auto test    - Quick testing (~50 evals)
      --auto light   - Quick experimentation (~100-200 evals, default)
      --auto medium  - Balanced optimization (~300-500 evals)
      --auto heavy   - Thorough optimization (~800+ evals)
      --max-evals N  - Exactly N full evaluations (faster, explicit control)
    
    Teacher Model:
      GEPA uses a teacher model for reflection. Set via:
      - --teacher CLI option
      - optimization.teacher_model in ~/.rlm/config.yaml
      - Defaults to the main model if not set
    
    Examples:
        rlm-dspy optimize gepa --fast                       # Fast proxy mode (recommended)
        rlm-dspy optimize gepa --fast --auto medium         # Fast + medium budget
        rlm-dspy optimize gepa --fast --max-evals 2         # Explicit budget control
        rlm-dspy optimize gepa --fast -n 10 -e 3            # 10 examples, 3 evals
        rlm-dspy optimize gepa --teacher openai/gpt-4o     # Use GPT-4o as teacher
        rlm-dspy optimize gepa --dry-run                    # Preview only
        rlm-dspy optimize gepa                              # Full RLM mode (slow, 1-6h)
    """
    from .core.trace_collector import get_trace_collector
    from .core.simba_optimizer import create_training_example
    from .core.gepa_optimizer import GEPAOptimizer, GEPAConfig
    from .core.user_config import OptimizationConfig

    # Load defaults from config
    opt_cfg = OptimizationConfig.from_user_config()
    
    # Apply config defaults if not specified on CLI
    if auto is None:
        auto = opt_cfg.gepa.auto if (max_evals is None and opt_cfg.gepa.max_evals is None) else None
    if max_evals is None and opt_cfg.gepa.max_evals is not None:
        max_evals = opt_cfg.gepa.max_evals
    if threads is None:
        threads = opt_cfg.threads
    if fast_proxy is None:
        fast_proxy = opt_cfg.fast

    console.print("[bold cyan]GEPA Reflective Prompt Evolution[/bold cyan]\n")

    collector = get_trace_collector()
    traces = [t for t in collector.traces if t.grounded_score >= min_score][:max_examples]

    console.print(f"Total traces: {len(collector.traces)}")
    console.print(f"High quality (>= {min_score}): {len(traces)}")

    if len(traces) < 4:
        console.print(f"\n[yellow]Not enough traces ({len(traces)} < 4)[/yellow]")
        console.print("Run some queries first to collect training data.")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Found {len(traces)} traces for training[/cyan]")

    if dry_run:
        console.print("\n[bold]Would train on:[/bold]")
        for trace in traces[:10]:
            console.print(f"  - {trace.query[:50]}... (score: {trace.grounded_score:.2f})")
        if len(traces) > 10:
            console.print(f"  ... and {len(traces) - 10} more")
        
        # Show settings
        from .core.user_config import load_config
        user_cfg = load_config()
        teacher_model_name = teacher or opt_cfg.get_teacher_model(user_cfg.get("model", "openai/gpt-4o-mini"))
        
        budget_str = f"max_evals={max_evals}" if max_evals else f"auto={auto}"
        console.print(f"\n[dim]Mode: {'fast proxy' if fast_proxy else 'full RLM'}[/dim]")
        console.print(f"[dim]Budget: {budget_str}, Threads: {threads}, Tools: {tool_optimization}[/dim]")
        console.print(f"[dim]Teacher model: {teacher_model_name}[/dim]")
        return

    # Convert traces to training examples
    examples = []
    for trace in traces:
        example = create_training_example(
            query=trace.query,
            answer=trace.final_answer,
            context="",
        )
        if example:
            examples.append(example)

    if not examples:
        console.print("[red]No valid training examples could be created[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Created {len(examples)} training examples[/cyan]")
    
    # Show mode info
    if fast_proxy:
        console.print(f"[cyan]Running GEPA in FAST PROXY mode (auto={auto}, threads={threads})...[/cyan]")
        console.print(f"  [dim]Proxy mode: 1 LLM call per eval instead of 10-50 (50x faster)[/dim]")
    else:
        console.print(f"[cyan]Running GEPA optimization (auto={auto}, threads={threads})...[/cyan]")
        console.print(f"  [dim]Full RLM mode: 10-50 LLM calls per eval (use --fast for 50x speedup)[/dim]")

    try:
        from .core.rlm import RLM
        from .core.user_config import OptimizationConfig, load_config
        from .core.simba_optimizer import save_optimized_program, save_optimization_state, OptimizationState, get_trace_count
        import dspy

        # Get the RLM program to optimize
        rlm = RLM()
        
        # Get teacher model from CLI or config
        opt_cfg = OptimizationConfig.from_user_config()
        user_cfg = load_config()
        default_model = user_cfg.get("model", "openai/gpt-4o-mini")
        teacher_model_name = teacher or opt_cfg.get_teacher_model(default_model)
        
        # Create teacher LM for reflection
        # Use the same LM creation logic as RLM (supporting OAuth for Anthropic/Google)
        reflection_lm = None
        if teacher_model_name:
            if teacher_model_name.startswith("zai/"):
                from .core.zai_lm import ZaiLM
                reflection_lm = ZaiLM(teacher_model_name.replace("zai/", ""))
            elif teacher_model_name.startswith("kimi/"):
                from .core.kimi_lm import KimiLM
                model_id = teacher_model_name.replace("kimi/", "")
                if model_id == "k2p5":
                    model_id = "k2-0130-8k"
                reflection_lm = KimiLM(model_id)
            elif teacher_model_name.startswith("anthropic/"):
                # Use OAuth for Anthropic models
                from .core.anthropic_oauth_lm import get_anthropic_api_key, is_oauth_token, AnthropicOAuthLM
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY") or get_anthropic_api_key()
                if api_key and is_oauth_token(api_key):
                    reflection_lm = AnthropicOAuthLM(
                        model=teacher_model_name.replace("anthropic/", ""),
                        api_key=api_key,
                    )
                else:
                    reflection_lm = dspy.LM(teacher_model_name, api_key=api_key)
            elif teacher_model_name.startswith("google/"):
                # Use OAuth for Google models
                from .core.oauth import get_google_token
                from .core.google_oauth_lm import GoogleOAuthLM
                token, project_id = get_google_token()
                if token:
                    reflection_lm = GoogleOAuthLM(
                        model=teacher_model_name.replace("google/", ""),
                        access_token=token,
                        project_id=project_id,
                    )
                else:
                    reflection_lm = dspy.LM(teacher_model_name)
            else:
                reflection_lm = dspy.LM(teacher_model_name)
            console.print(f"  [dim]Teacher model: {teacher_model_name}[/dim]")
        
        # Configure GEPA budget - max_evals overrides auto
        if max_evals:
            config = GEPAConfig(
                auto=None,
                max_full_evals=max_evals,
                num_threads=threads,
                enable_tool_optimization=tool_optimization,
            )
            console.print(f"  [dim]Budget: {max_evals} full evaluations[/dim]")
        else:
            config = GEPAConfig(
                auto=auto,
                num_threads=threads,
                enable_tool_optimization=tool_optimization,
            )
            console.print(f"  [dim]Budget: auto={auto}[/dim]")

        # Configure DSPy with our LM
        dspy.configure(lm=rlm._lm)

        if fast_proxy:
            # FAST MODE: Use lightweight proxy instead of full RLM
            from .core.gepa_proxy import run_fast_gepa
            
            gepa_instructions, result = run_fast_gepa(
                rlm=rlm,
                trainset=examples,
                config=config,
                reflection_lm=reflection_lm,
            )
            
            console.print("\n[bold green]✓ GEPA (fast proxy) optimization complete![/bold green]")
            console.print(f"  Baseline score: {result.baseline_score:.2%}")
            console.print(f"  Optimized score: {result.optimized_score:.2%}")
            console.print(f"  Improvement: {result.improvement:.1f}%")
            
            if gepa_instructions:
                # Save instructions (no full program in proxy mode)
                save_optimized_program(
                    None,  # No optimized program in proxy mode
                    result, 
                    optimizer_type="gepa",
                    instructions=gepa_instructions,
                )
                console.print("\n[green]✓ Optimized instructions saved - will be auto-loaded on next query[/green]")
                console.print(f"  [dim]Saved {len(gepa_instructions)} evolved instructions[/dim]")
        else:
            # FULL MODE: Run GEPA on actual RLM (slow but accurate)
            optimizer = GEPAOptimizer(config=config, reflection_lm=reflection_lm)

            optimized_program, result = optimizer.optimize(
                program=rlm._rlm,
                trainset=examples,
                lm=rlm._lm,
            )

            console.print("\n[bold green]✓ GEPA optimization complete![/bold green]")
            console.print(f"  Baseline score: {result.baseline_score:.2%}")
            console.print(f"  Optimized score: {result.optimized_score:.2%}")
            console.print(f"  Improvement: {result.improvement:.1f}%")

            if result.improved:
                # Extract GEPA instructions from optimized program
                from .core.gepa_optimizer import extract_gepa_instructions
                gepa_instructions = extract_gepa_instructions(optimized_program)
                
                # Save the optimized program with instructions
                save_optimized_program(
                    optimized_program, 
                    result, 
                    optimizer_type="gepa",
                    instructions=gepa_instructions,
                )
                console.print("\n[green]✓ Optimized program saved - will be auto-loaded on next query[/green]")
                if gepa_instructions:
                    console.print(f"  [dim]Saved {len(gepa_instructions)} evolved instructions[/dim]")

        # Update optimization state
        from datetime import datetime, UTC
        state = OptimizationState(
            last_optimization=datetime.now(UTC),
            traces_at_last_optimization=get_trace_count(),
            last_result=result,
            optimizer_type="gepa",
        )
        save_optimization_state(state)

    except Exception as e:
        console.print(f"[red]GEPA optimization failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@optimize_app.command("run")
def optimize_run(
    min_score: Annotated[float, typer.Option("--min-score", "-s", help="Minimum trace score")] = 0.7,
    max_examples: Annotated[int, typer.Option("--max-examples", "-n", help="Maximum training examples")] = 100,
    fast: Annotated[bool, typer.Option("--fast", help="Fast preset: 1 step, 2 candidates")] = False,
    target: Annotated[str | None, typer.Option("--target", "-t", help="Target: all, demos, tips (default: all)")] = None,
    optimizer: Annotated[str | None, typer.Option("--optimizer", "-o", help="Optimizer: gepa or simba (default: from config)")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be optimized")] = False,
) -> None:
    """Run unified optimization (demos + tips + rules).

    Runs full optimization pipeline:
    1. GEPA/SIMBA optimizer for instruction/demo optimization
    2. Generates tips from failure patterns
    3. Extracts rules for instructions

    Optimizers:
      gepa  - Reflective Prompt Evolution (recommended for RLM)
      simba - Stochastic Mini-Batch Ascent (legacy, demos only)

    Default optimizer is read from ~/.rlm/config.yaml (optimization.optimizer).

    All components are saved and auto-loaded on next query.

    \b
    Examples:
      rlm-dspy optimize run            # Use optimizer from config
      rlm-dspy optimize run --fast     # Fast mode (~5-15 min)
      rlm-dspy optimize run -o simba   # Use SIMBA optimizer
      rlm-dspy optimize run -o gepa    # Use GEPA optimizer
      rlm-dspy optimize run --target tips  # Only regenerate tips
    """
    from .core.trace_collector import get_trace_collector
    from .core.simba_optimizer import (
        get_simba_optimizer, create_training_example,
        save_optimized_program, save_optimization_state,
        OptimizationState, get_trace_count,
        _generate_optimized_tips, _extract_simba_rules,
    )
    from .core.grounded_proposer import get_grounded_proposer
    from .core.instruction_optimizer import get_instruction_optimizer
    from .core.user_config import OptimizationConfig

    # Get optimizer from config if not specified
    optimizer_source = "CLI"
    if optimizer is None:
        opt_cfg = OptimizationConfig.from_user_config()
        optimizer = opt_cfg.optimizer or "gepa"
        optimizer_source = "config"
    
    if optimizer not in ("gepa", "simba"):
        console.print(f"[red]Invalid optimizer: {optimizer}. Use: gepa, simba[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Optimizer: {optimizer} (from {optimizer_source})[/cyan]")

    target = target or "all"
    if target not in ("all", "demos", "tips"):
        console.print(f"[red]Invalid target: {target}. Use: all, demos, tips[/red]")
        raise typer.Exit(1)

    collector = get_trace_collector()
    all_traces = list(collector.traces)

    # Show stats
    console.print(f"[cyan]Total traces: {len(all_traces)}[/cyan]")
    console.print(f"[cyan]High quality (>= {min_score}): {len([t for t in all_traces if t.grounded_score >= min_score])}[/cyan]")
    console.print(f"[cyan]Failures (<= 0.6): {len([t for t in all_traces if t.grounded_score <= 0.6])}[/cyan]")

    if dry_run:
        console.print("\n[bold]Would optimize:[/bold]")
        if target in ("all", "demos"):
            traces = [t for t in all_traces if t.grounded_score >= min_score][:max_examples]
            console.print(f"  - Demos: {len(traces)} training examples")
        if target in ("all", "tips"):
            failures = collector.to_failure_patterns(max_score=0.6)
            console.print(f"  - Tips: from {len(failures)} failure patterns")
        return

    results = {"demos": 0, "tips": 0, "rules": 0}
    optimized_program = None
    optimization_result = None

    # Step 1: Optimization for demos (if target includes demos)
    if target in ("all", "demos"):
        # Limit traces - each takes 20-60s to evaluate
        # For fast mode, use fewer traces for quicker optimization
        max_opt_traces = 8 if fast else 16
        traces = [t for t in all_traces if t.grounded_score >= min_score][:min(max_examples, max_opt_traces)]
        
        if len(traces) < 4:
            console.print(f"[yellow]Not enough traces ({len(traces)} < 4). Skipping demos.[/yellow]")
        else:
            opt_name = optimizer.upper()
            console.print(f"\n[bold cyan]Step 1: {opt_name} optimization ({len(traces)} traces)...[/bold cyan]")
            
            examples = []
            for trace in traces:
                example = create_training_example(
                    query=trace.query,
                    answer=trace.final_answer,
                    context="",
                )
                if example:
                    examples.append(example)
            
            if examples:
                try:
                    from .core.rlm import RLM
                    from .core.rlm_types import RLMConfig
                    import dspy
                    
                    # Create RLM with higher iteration limits for optimization
                    # Use model from user config (~/.rlm/config.yaml)
                    from .core.user_config import load_config
                    user_cfg = load_config()
                    opt_config = RLMConfig(
                        model=user_cfg.get("model", "openai/gpt-4o-mini"),
                        sub_model=user_cfg.get("sub_model", user_cfg.get("model", "openai/gpt-4o-mini")),
                        max_iterations=50,   # Higher than default 30
                        max_llm_calls=150,   # Higher than default 100
                    )
                    rlm = RLM(config=opt_config)
                    console.print(f"  [dim]Using model: {opt_config.model}, sub: {opt_config.sub_model}[/dim]")
                    
                    dspy.configure(lm=rlm._lm)
                    
                    if optimizer == "gepa":
                        # Use GEPA optimizer
                        from .core.gepa_optimizer import GEPAOptimizer, GEPAConfig
                        from .core.user_config import OptimizationConfig
                        
                        # Get teacher model from config
                        opt_cfg = OptimizationConfig.from_user_config()
                        teacher_model_name = opt_cfg.get_teacher_model(user_cfg.get("model", "openai/gpt-4o-mini"))
                        
                        # Create teacher LM for reflection
                        reflection_lm = None
                        if teacher_model_name:
                            if teacher_model_name.startswith("zai/"):
                                from .core.zai_lm import ZaiLM
                                reflection_lm = ZaiLM(teacher_model_name.replace("zai/", ""))
                            elif teacher_model_name.startswith("kimi/"):
                                from .core.kimi_lm import KimiLM
                                model_id = teacher_model_name.replace("kimi/", "")
                                if model_id == "k2p5":
                                    model_id = "k2-0130-8k"
                                reflection_lm = KimiLM(model_id)
                            else:
                                reflection_lm = dspy.LM(teacher_model_name)
                            console.print(f"  [dim]Teacher model: {teacher_model_name}[/dim]")
                        
                        auto_budget = "light" if fast else "medium"
                        gepa_config = GEPAConfig(
                            auto=auto_budget,
                            num_threads=2,
                        )
                        opt = GEPAOptimizer(config=gepa_config, reflection_lm=reflection_lm)
                        console.print(f"  [dim]auto={auto_budget}, threads=2[/dim]")
                        
                        optimized_program, optimization_result = opt.optimize(
                            program=rlm._rlm,
                            trainset=examples,
                            lm=rlm._lm,
                        )
                    else:
                        # Use SIMBA optimizer (default)
                        # Apply presets
                        # Each example takes 30-60s with RLM due to interpreter loop
                        if fast:
                            steps, candidates, threads, batch_size = 1, 2, 2, 2
                        else:
                            steps, candidates, threads, batch_size = 1, 2, 2, min(4, len(examples))
                        
                        opt = get_simba_optimizer(
                            batch_size=batch_size,
                            num_candidates=candidates,
                            max_steps=steps,
                            num_threads=threads,
                        )
                        console.print(f"  [dim]steps={steps}, candidates={candidates}, threads={threads}[/dim]")
                        
                        optimized_program, optimization_result = opt.optimize(
                            program=rlm._rlm,
                            trainset=examples,
                            lm=rlm._lm,
                        )
                    
                    if optimized_program and hasattr(optimized_program, "demos"):
                        results["demos"] = len(optimized_program.demos) if optimized_program.demos else 0
                    
                    console.print(f"  [green]✓ {opt_name} complete: {results['demos']} demos, {optimization_result.improvement:.1f}% improvement[/green]")
                    
                except Exception as e:
                    console.print(f"  [red]✗ {opt_name} failed: {e}[/red]")

    # Step 2: Generate tips from failures
    if target in ("all", "tips"):
        console.print("\n[bold cyan]Step 2: Generating tips from failures...[/bold cyan]")
        
        try:
            tips = _generate_optimized_tips()
            if tips:
                results["tips"] = len(tips)
                # Apply tips immediately
                proposer = get_grounded_proposer()
                proposer.set_optimized_tips(tips)
                console.print(f"  [green]✓ Generated {len(tips)} tips[/green]")
                for tip in tips[:3]:
                    console.print(f"    - {tip[:60]}{'...' if len(tip) > 60 else ''}")
                if len(tips) > 3:
                    console.print(f"    ... and {len(tips) - 3} more")
            else:
                console.print("  [dim]No tips generated (need more failure data)[/dim]")
        except Exception as e:
            console.print(f"  [red]✗ Tip generation failed: {e}[/red]")

    # Step 3: Extract rules (if we ran optimization)
    if target in ("all", "demos") and optimized_program:
        console.print("\n[bold cyan]Step 3: Extracting rules...[/bold cyan]")
        
        try:
            rules = _extract_simba_rules(optimized_program)
            if rules:
                results["rules"] = len(rules)
                # Apply rules to instruction optimizer
                inst_optimizer = get_instruction_optimizer()
                inst_optimizer.add_rules(rules)
                console.print(f"  [green]✓ Extracted {len(rules)} rules[/green]")
                for rule in rules[:3]:
                    console.print(f"    - {rule[:60]}{'...' if len(rule) > 60 else ''}")
            else:
                console.print("  [dim]No rules extracted[/dim]")
        except Exception as e:
            console.print(f"  [red]✗ Rule extraction failed: {e}[/red]")

    # Save everything
    if results["demos"] > 0 or results["tips"] > 0 or results["rules"] > 0:
        console.print("\n[bold cyan]Saving optimization...[/bold cyan]")
        
        try:
            # Get current tips and instructions
            proposer = get_grounded_proposer()
            inst_optimizer = get_instruction_optimizer()
            
            tips = proposer.get_tips()
            instructions = inst_optimizer.get_all_instructions()
            
            # If GEPA, also extract evolved instructions
            if optimizer == "gepa" and optimized_program:
                from .core.gepa_optimizer import extract_gepa_instructions
                gepa_instructions = extract_gepa_instructions(optimized_program)
                if gepa_instructions:
                    instructions.update(gepa_instructions)
                    console.print(f"  [dim]Extracted {len(gepa_instructions)} GEPA-evolved instructions[/dim]")
            
            # Save with all components
            if optimized_program and optimization_result:
                save_optimized_program(
                    optimized_program,
                    optimization_result,
                    optimizer,  # Use the selected optimizer type
                    instructions=instructions,
                    tips=tips,
                )
            
            # Update state
            from datetime import datetime, UTC
            state = OptimizationState(
                last_optimization=datetime.now(UTC),
                traces_at_last_optimization=get_trace_count(),
                last_result=optimization_result,
                optimizer_type=optimizer,  # Use the selected optimizer type
            )
            save_optimization_state(state)
            
            console.print("[green]✓ Saved to ~/.rlm/optimization/[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to save: {e}[/red]")

    # Summary
    console.print("\n[bold]Optimization Summary:[/bold]")
    console.print(f"  Demos: {results['demos']}")
    console.print(f"  Tips: {results['tips']}")
    console.print(f"  Rules: {results['rules']}")
    
    if optimization_result and optimization_result.improved:
        console.print(f"\n[green]✓ Score improvement: +{optimization_result.improvement:.1f}%[/green]")
    
    console.print("\n[dim]Changes will be auto-loaded on next query.[/dim]")


@optimize_app.command("status")
def optimize_status() -> None:
    """Show auto-optimization status."""
    from .core.user_config import OptimizationConfig, load_config
    from .core.simba_optimizer import (
        load_optimization_state, load_optimized_program, get_trace_count,
        OPTIMIZED_PROGRAM_FILE,
    )

    config = OptimizationConfig.from_user_config()
    state = load_optimization_state()
    saved = load_optimized_program()
    current_traces = get_trace_count()
    user_config = load_config()

    table = Table(title="Auto-Optimization Status")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    # Config
    status = "[green]Enabled[/green]" if config.enabled else "[red]Disabled[/red]"
    table.add_row("Status", status)
    table.add_row("Optimizer", config.optimizer)

    model = config.get_model(user_config.get("model", "default"))
    table.add_row("Model", model)
    table.add_row("Min New Traces", str(config.min_new_traces))
    table.add_row("Min Hours Between", str(config.min_hours_between))
    table.add_row("Max Budget", f"${config.max_budget:.2f}")
    table.add_row("Background Mode", "Yes" if config.run_in_background else "No")

    table.add_row("", "")
    table.add_row("[bold]Current State[/bold]", "")
    table.add_row("Current Traces", str(current_traces))
    table.add_row("Traces at Last Opt", str(state.traces_at_last_optimization))

    new_traces = current_traces - state.traces_at_last_optimization
    table.add_row("New Traces", str(new_traces))

    if state.last_optimization:
        from datetime import datetime, UTC
        hours_ago = (datetime.now(UTC) - state.last_optimization).total_seconds() / 3600
        table.add_row("Last Optimization", f"{hours_ago:.1f} hours ago")
    else:
        table.add_row("Last Optimization", "Never")

    table.add_row("", "")
    table.add_row("[bold]Saved Optimization[/bold]", "")

    if saved:
        table.add_row("Saved Program", "[green]Yes[/green]")
        table.add_row("Demos", str(len(saved.demos)))
        table.add_row("Tips", str(len(saved.tips)))
        table.add_row("Rules", str(len(saved.rules)))
        if saved.instructions:
            table.add_row("Instructions", f"{len(saved.instructions)} keys")
        if saved.result:
            table.add_row("Improvement", f"+{saved.result.improvement:.1f}%")
        table.add_row("File", str(OPTIMIZED_PROGRAM_FILE))
    else:
        table.add_row("Saved Program", "[dim]None[/dim]")

    console.print(table)

    # Show when next optimization will trigger
    if config.enabled:
        needed_traces = config.min_new_traces - new_traces
        if needed_traces > 0:
            console.print(f"\n[dim]Next auto-optimization: {needed_traces} more traces needed[/dim]")
        elif state.last_optimization:
            from datetime import datetime, UTC
            hours_since = (datetime.now(UTC) - state.last_optimization).total_seconds() / 3600
            hours_left = config.min_hours_between - hours_since
            if hours_left > 0:
                console.print(f"\n[dim]Next auto-optimization: {hours_left:.1f} hours cooldown remaining[/dim]")
            else:
                console.print("\n[green]Ready to optimize on next query![/green]")
        else:
            console.print("\n[green]Ready to optimize on next query![/green]")


@optimize_app.command("enable")
def optimize_enable(
    background: Annotated[bool | None, typer.Option("--background/--no-background", "-b/-B", help="Enable/disable background mode")] = None,
) -> None:
    """Enable auto-optimization.
    
    Examples:
        rlm-dspy optimize enable              # Enable auto-optimization
        rlm-dspy optimize enable --background # Enable with background mode
        rlm-dspy optimize enable -B           # Enable without background mode
    """
    from pathlib import Path
    import yaml

    config_file = Path.home() / ".rlm" / "config.yaml"

    # Load existing config
    config = {}
    if config_file.exists():
        try:
            config = yaml.safe_load(config_file.read_text()) or {}
        except Exception:
            pass

    # Update optimization settings
    if "optimization" not in config:
        config["optimization"] = {}
    config["optimization"]["enabled"] = True
    
    if background is not None:
        config["optimization"]["background"] = background

    # Save back
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

    bg_status = config["optimization"].get("background", True)
    console.print("[green]✓ Auto-optimization enabled[/green]")
    console.print(f"  Background mode: {'Yes' if bg_status else 'No'}")


@optimize_app.command("disable")
def optimize_disable() -> None:
    """Disable auto-optimization."""
    from pathlib import Path
    import yaml

    config_file = Path.home() / ".rlm" / "config.yaml"

    # Load existing config
    config = {}
    if config_file.exists():
        try:
            config = yaml.safe_load(config_file.read_text()) or {}
        except Exception:
            pass

    # Update optimization.enabled
    if "optimization" not in config:
        config["optimization"] = {}
    config["optimization"]["enabled"] = False

    # Save back
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

    console.print("[yellow]✓ Auto-optimization disabled[/yellow]")


@optimize_app.command("config")
def optimize_config(
    background: Annotated[bool | None, typer.Option("--background/--no-background", "-b/-B", help="Enable/disable background mode")] = None,
    min_traces: Annotated[int | None, typer.Option("--min-traces", help="Min new traces before optimization")] = None,
    min_hours: Annotated[int | None, typer.Option("--min-hours", help="Min hours between optimizations")] = None,
    max_budget: Annotated[float | None, typer.Option("--max-budget", help="Max budget per optimization ($)")] = None,
) -> None:
    """Configure optimization settings.
    
    Examples:
        rlm-dspy optimize config                    # Show current config
        rlm-dspy optimize config --background       # Enable background mode
        rlm-dspy optimize config -B                 # Disable background mode
        rlm-dspy optimize config --min-traces 20    # Set min traces to 20
        rlm-dspy optimize config --min-hours 12     # Set min hours to 12
    """
    from pathlib import Path
    import yaml

    config_file = Path.home() / ".rlm" / "config.yaml"

    # Load existing config
    config = {}
    if config_file.exists():
        try:
            config = yaml.safe_load(config_file.read_text()) or {}
        except Exception:
            pass

    if "optimization" not in config:
        config["optimization"] = {}
    
    # Check if any updates requested
    has_updates = any(v is not None for v in [background, min_traces, min_hours, max_budget])
    
    if has_updates:
        # Apply updates
        if background is not None:
            config["optimization"]["background"] = background
        if min_traces is not None:
            config["optimization"]["min_new_traces"] = min_traces
        if min_hours is not None:
            config["optimization"]["min_hours_between"] = min_hours
        if max_budget is not None:
            config["optimization"]["max_budget"] = max_budget
        
        # Save back
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
        console.print("[green]✓ Configuration updated[/green]")
    
    # Show current config
    opt_config = config.get("optimization", {})
    console.print("\n[bold]Optimization Settings:[/bold]")
    console.print(f"  Enabled:         {opt_config.get('enabled', True)}")
    console.print(f"  Background:      {opt_config.get('background', True)}")
    console.print(f"  Min New Traces:  {opt_config.get('min_new_traces', 50)}")
    console.print(f"  Min Hours:       {opt_config.get('min_hours_between', 24)}")
    console.print(f"  Max Budget:      ${opt_config.get('max_budget', 0.50):.2f}")


@optimize_app.command("reset")
def optimize_reset(
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
    target: Annotated[str, typer.Option("--target", "-t", help="What to reset: all, traces, optimization, optimizer, proposer")] = "all",
) -> None:
    """Reset optimization - clear saved data.
    
    Targets:
      all          - Clear everything (traces + optimization + optimizer + proposer)
      traces       - Clear trace history (~/.rlm/traces/)
      optimization - Clear SIMBA saved program (~/.rlm/optimization/)
      optimizer    - Clear instruction optimizer state (~/.rlm/optimizer/)
      proposer     - Clear grounded proposer state (~/.rlm/proposer/)
    """
    import shutil
    from pathlib import Path
    
    rlm_dir = Path.home() / ".rlm"
    
    targets_map = {
        "traces": rlm_dir / "traces",
        "optimization": rlm_dir / "optimization", 
        "optimizer": rlm_dir / "optimizer",
        "proposer": rlm_dir / "proposer",
    }
    
    if target == "all":
        targets = list(targets_map.keys())
    elif target in targets_map:
        targets = [target]
    else:
        console.print(f"[red]Unknown target: {target}[/red]")
        console.print(f"[dim]Valid targets: all, {', '.join(targets_map.keys())}[/dim]")
        raise typer.Abort()
    
    # Show what will be cleared
    console.print(f"[bold]Will clear:[/bold]")
    for t in targets:
        path = targets_map[t]
        if path.exists():
            files = list(path.glob("*"))
            size = sum(f.stat().st_size for f in files if f.is_file())
            console.print(f"  • {t}: {len(files)} files ({size / 1024:.1f} KB)")
        else:
            console.print(f"  • {t}: [dim]not found[/dim]")
    
    if not force:
        if not typer.confirm(f"\nClear {target} data?"):
            raise typer.Abort()
    
    cleared = []
    for t in targets:
        path = targets_map[t]
        if path.exists():
            try:
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
                cleared.append(t)
            except Exception as e:
                console.print(f"[red]Failed to clear {t}: {e}[/red]")
    
    if cleared:
        console.print(f"[green]✓ Cleared: {', '.join(cleared)}[/green]")
    else:
        console.print("[dim]Nothing to clear[/dim]")

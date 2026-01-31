"""Core RLM class using DSPy's native RLM module."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import dspy

from .rlm_types import (
    RLMConfig, RLMResult, ProgressCallback, DspyProgressCallback,
    get_provider_env_var, sanitize_trajectory, extract_trace_metadata,
)
from .secrets import sanitize_text as _sanitize_secrets

_logger = logging.getLogger(__name__)


class RLM:
    """Recursive Language Model using DSPy's native RLM module."""

    def __init__(
        self,
        config: RLMConfig | None = None,
        tools: dict[str, Callable[..., str]] | None = None,
        signature: str | type = "context, query -> answer",
        interpreter: Any | None = None,
        use_tools: bool | str = True,
        progress_callback: ProgressCallback | None = None,
    ):
        self.config = config or RLMConfig()
        self._interpreter = interpreter
        self._persistent_interpreter = None
        self._progress_callback = progress_callback
        self._tool_call_counts: dict[str, int] = {}

        # Initialize tools
        self._tools = tools.copy() if tools else {}
        if use_tools:
            from ..tools import BUILTIN_TOOLS, SAFE_TOOLS
            builtin = BUILTIN_TOOLS if use_tools == "all" else SAFE_TOOLS
            for name, func in builtin.items():
                if name not in self._tools:
                    self._tools[name] = func

        self._tools = self._wrap_tools_with_counter(self._tools)
        self._validate_tools(self._tools)
        self._is_structured = not isinstance(signature, str)
        self._signature = self._wrap_signature_with_tool_instructions(signature, use_tools)

        requires_api_key = not self.config.model.lower().startswith("ollama/")
        if requires_api_key and not self.config.api_key:
            env_var = get_provider_env_var(self.config.model) or "PROVIDER_API_KEY"
            raise ValueError(f"No API key for '{self.config.model}'. Set RLM_API_KEY or {env_var}.")

        self._setup_dspy()
        self._rlm = self._create_rlm()
        self._rlm_dirty = False
        self._start_time: float | None = None
        self._metrics_callback: Any = None
        self._setup_callbacks()

        # Load saved optimization and maybe trigger background optimization
        self._load_and_apply_optimization()
        self._maybe_trigger_background_optimization()

    def _setup_callbacks(self) -> None:
        from .callbacks import get_callback_manager, LoggingCallback, MetricsCallback
        manager = get_callback_manager()
        if self.config.enable_logging:
            import logging
            manager.add(LoggingCallback(level=logging.DEBUG if self.config.verbose else logging.INFO))
        if self.config.enable_metrics:
            self._metrics_callback = MetricsCallback()
            manager.add(self._metrics_callback)

    def _load_and_apply_optimization(self) -> None:
        """Load and apply saved SIMBA optimization if available."""
        try:
            from .simba_optimizer import load_optimized_program

            saved = load_optimized_program()
            if saved is None:
                return

            # Apply demos to the RLM module
            if saved.demos and hasattr(self._rlm, "demos"):
                self._rlm.demos = saved.demos
                _logger.debug("Applied %d saved demos from optimization", len(saved.demos))

            # Note: instructions are already handled by _get_optimized_instructions()
            # which loads from InstructionOptimizer

            if saved.result and saved.result.improved:
                _logger.info(
                    "Loaded saved optimization: +%.1f%% improvement (%s)",
                    saved.result.improvement,
                    saved.optimizer_type,
                )

        except Exception as e:
            _logger.debug("Failed to load optimization: %s", e)

    def _maybe_trigger_background_optimization(self) -> None:
        """Trigger background SIMBA optimization if conditions are met."""
        try:
            from .user_config import OptimizationConfig
            from .simba_optimizer import should_optimize, run_background_optimization

            config = OptimizationConfig.from_user_config()

            if not config.enabled:
                return

            if should_optimize(config):
                _logger.debug("Triggering background %s optimization", config.optimizer)
                # Get the model to use for optimization
                model = config.get_model(self.config.model)
                run_background_optimization(config, model)

        except Exception as e:
            _logger.debug("Failed to check/trigger optimization: %s", e)

    def get_metrics(self) -> dict[str, Any] | None:
        return self._metrics_callback.get_summary() if self._metrics_callback else None

    def _wrap_tools_with_counter(self, tools: dict[str, Callable]) -> dict[str, Callable]:
        import functools
        wrapped = {}
        for name, func in tools.items():
            @functools.wraps(func)
            def wrapper(*args, _tool_name=name, _original=func, **kwargs):
                self._tool_call_counts[_tool_name] = self._tool_call_counts.get(_tool_name, 0) + 1
                return _original(*args, **kwargs)
            wrapped[name] = wrapper
        return wrapped

    def get_tool_call_counts(self) -> dict[str, int]:
        return dict(self._tool_call_counts)

    def _get_optimized_instructions(self) -> str:
        try:
            from .instruction_optimizer import get_instruction_optimizer
            from .grounded_proposer import get_grounded_proposer
            optimizer = get_instruction_optimizer()
            proposer = get_grounded_proposer()
            tool_inst = optimizer.get_instruction("tool_instructions")
            verify_rules = optimizer.get_instruction("verification_rules")
            iter_guide = optimizer.get_instruction("iteration_guidance")
            tips = proposer.get_tips()
            tips_text = "\n\nLEARNED TIPS:\n" + "\n".join(f"- {t}" for t in tips[:5]) if tips else ""
            return f"{tool_inst}\n\n{verify_rules}\n\n{iter_guide}{tips_text}\n\n"
        except Exception:
            return self._default_instructions()

    def _default_instructions(self) -> str:
        return """CONTEXT: You are exploring a LARGE CODEBASE.
Use tools strategically to find what matters.

EXPLORATION STRATEGY:
1. GET THE BIG PICTURE: file_stats("."), find_files("*.py"), semantic_search("topic")
2. NARROW DOWN: index_code("src/"), find_usages("file.py")
3. READ SPECIFIC CODE: read_file(path, start, end)

CRITICAL: Extract exact symbol names before searching (use find_usages or index_code).

VERIFICATION RULES:
1. NEVER claim issues without read_file() to see actual code
2. ALWAYS verify line numbers by reading the file
3. Quote actual code in your findings

"""

    def _wrap_signature_with_tool_instructions(self, signature: str | type, use_tools: bool | str) -> str | type:
        if not use_tools or not self._tools:
            return signature

        tool_instructions = self._get_optimized_instructions()
        if isinstance(signature, str):
            base_sig = dspy.Signature(signature)
            class ToolFirstSignature(base_sig):
                pass
            ToolFirstSignature.__doc__ = tool_instructions
            return ToolFirstSignature
        else:
            class WrappedSignature(signature):
                pass
            WrappedSignature.__doc__ = tool_instructions + (signature.__doc__ or "")
            WrappedSignature.__name__ = signature.__name__
            return WrappedSignature

    def _setup_dspy(self) -> None:
        lm_kwargs: dict[str, Any] = {"model": self.config.model, "api_key": self.config.api_key}
        if self.config.api_base:
            lm_kwargs["api_base"] = self.config.api_base
        self._lm = dspy.LM(**lm_kwargs)
        dspy.settings.configure(async_max_workers=self.config.max_workers, num_threads=self.config.max_workers)

    def _create_sub_lm(self) -> dspy.LM | None:
        if self.config.sub_model == self.config.model:
            return None
        lm_kwargs: dict[str, Any] = {"model": self.config.sub_model, "api_key": self.config.api_key}
        if self.config.api_base:
            lm_kwargs["api_base"] = self.config.api_base
        return dspy.LM(**lm_kwargs)

    def _get_or_create_interpreter(self):
        if self._interpreter is not None:
            return self._interpreter
        if self._persistent_interpreter is None:
            try:
                from dspy.primitives.python_interpreter import PythonInterpreter
                self._persistent_interpreter = PythonInterpreter()
            except Exception as e:
                _logger.warning("Failed to create interpreter: %s", e)
                return None
        return self._persistent_interpreter

    def _create_rlm(self, use_persistent_interpreter: bool = True) -> dspy.RLM:
        kwargs: dict[str, Any] = {
            "signature": self._signature,
            "max_iterations": self.config.max_iterations,
            "max_llm_calls": self.config.max_llm_calls,
            "max_output_chars": self.config.max_output_chars,
            "verbose": self.config.verbose,
            "tools": self._tools,
            "sub_lm": self._create_sub_lm(),
        }
        if use_persistent_interpreter:
            if (interp := self._get_or_create_interpreter()) is not None:
                kwargs["interpreter"] = interp
        elif self._interpreter is not None:
            kwargs["interpreter"] = self._interpreter
        return dspy.RLM(**kwargs)

    def shutdown(self) -> None:
        if self._persistent_interpreter is not None:
            try:
                self._persistent_interpreter.shutdown()
            except Exception:
                pass
            self._persistent_interpreter = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def load_context(self, paths: list, gitignore: bool = True, use_cache: bool = True,
                     max_tokens: int | None = None) -> str:
        from pathlib import Path
        from .fileutils import load_context_from_paths, load_context_from_paths_cached, smart_truncate_context
        loader = load_context_from_paths_cached if use_cache else load_context_from_paths
        context = loader(paths=[Path(p) for p in paths], gitignore=gitignore, add_line_numbers=True)
        if max_tokens is not None:
            context, was_truncated = smart_truncate_context(context, max_tokens)
            if was_truncated:
                _logger.warning("Context truncated to fit %d token limit", max_tokens)
        return context

    def _build_result(self, prediction: Any, elapsed: float) -> RLMResult:
        raw_trajectory = getattr(prediction, "trajectory", [])
        raw_reasoning = getattr(prediction, "final_reasoning", "")
        extra_secrets = [self.config.api_key] if self.config.api_key else None

        outputs: dict[str, Any] = {}
        answer = ""

        if self._is_structured:
            sig = self._signature
            if hasattr(sig, "output_fields"):
                fields = sig.output_fields
                output_field_names = list(fields.keys()) if isinstance(fields, dict) else []
            else:
                output_field_names = []

            for field_name in output_field_names:
                value = getattr(prediction, field_name, None)
                if value is not None:
                    if isinstance(value, str):
                        value = _sanitize_secrets(value, extra_secrets)
                    elif isinstance(value, list):
                        value = [_sanitize_secrets(v, extra_secrets) if isinstance(v, str) else v for v in value]
                    outputs[field_name] = value

            if outputs:
                first_val = next(iter(outputs.values()))
                answer = "\n".join(str(v) for v in first_val) if isinstance(first_val, list) else str(first_val)
        else:
            answer = _sanitize_secrets(getattr(prediction, "answer", ""), extra_secrets)

        sanitized_trajectory = sanitize_trajectory(raw_trajectory, extra_secrets)
        metadata = extract_trace_metadata(sanitized_trajectory)

        return RLMResult(
            answer=answer, success=True, elapsed_time=elapsed,
            trajectory=sanitized_trajectory, final_reasoning=_sanitize_secrets(raw_reasoning, extra_secrets),
            iterations=len(raw_trajectory), outputs=outputs, metadata=metadata,
        )

    def query(self, query: str, context: str) -> RLMResult:
        import concurrent.futures
        from .fileutils import estimate_tokens

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not context or not context.strip():
            raise ValueError("Context cannot be empty")

        if self.config.validate:
            from .validation import preflight_check
            result = preflight_check(api_key_required=True, model=self.config.model,
                                     budget=self.config.max_budget, context=context, check_network=False)
            if not result.passed:
                errors = [e.message for e in result.errors]
                raise ValueError(f"Preflight check failed: {'; '.join(errors)}")

        if self._rlm_dirty:
            self._rlm = self._create_rlm()
            self._rlm_dirty = False

        self._tool_call_counts.clear()
        self._start_time = time.time()

        if self._progress_callback:
            self._progress_callback.on_start(query, estimate_tokens(context))

        def _execute_rlm() -> Any:
            dspy_callback = DspyProgressCallback(self._progress_callback)
            with dspy.settings.context(lm=self._lm, callbacks=[dspy_callback]):
                result = self._rlm(context=context, query=query)
                if self.config.verbose:
                    dspy_stats = dspy_callback.get_stats()
                    _logger.info("DSPy stats: %d LLM calls, %d tool calls",
                                 dspy_stats["lm_calls"], sum(self._tool_call_counts.values()))
                return result

        try:
            if self.config.max_timeout and self.config.max_timeout > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_execute_rlm)
                    try:
                        prediction = future.result(timeout=self.config.max_timeout)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutExceededError(self.config.max_timeout, self.config.max_timeout)
            else:
                prediction = _execute_rlm()

            elapsed = time.time() - self._start_time
            result = self._build_result(prediction, elapsed)
            if self._progress_callback:
                self._progress_callback.on_complete(result)
            return result

        except TimeoutExceededError:
            elapsed = time.time() - self._start_time if self._start_time else 0
            _logger.warning("RLM timed out after %.1fs", elapsed)
            if self._progress_callback:
                self._progress_callback.on_error(TimeoutExceededError(self.config.max_timeout, elapsed))
            return RLMResult(answer="", success=False,
                             error=f"Timed out after {elapsed:.1f}s", elapsed_time=elapsed)
        except Exception as e:
            elapsed = time.time() - self._start_time if self._start_time else 0
            if self._progress_callback:
                self._progress_callback.on_error(e)
            _logger.exception("RLM execution failed")
            return RLMResult(answer="", success=False, error=str(e), elapsed_time=elapsed)

    async def query_async(self, query: str, context: str) -> RLMResult:
        self._start_time = time.time()
        try:
            with dspy.settings.context(lm=self._lm):
                prediction = await self._rlm.aforward(context=context, query=query)
            return self._build_result(prediction, time.time() - self._start_time)
        except Exception as e:
            elapsed = time.time() - self._start_time if self._start_time else 0
            return RLMResult(answer="", success=False, error=str(e), elapsed_time=elapsed)

    def add_tool(self, name: str, func: Callable[..., str]) -> None:
        self._validate_tools({name: func})
        self._tools[name] = func
        self._rlm_dirty = True

    def _validate_tools(self, tools: dict[str, Callable]) -> None:
        RESERVED = {'print', 'len', 'range', 'str', 'int', 'float', 'bool', 'list', 'dict', 'SUBMIT', 'FINAL'}
        for name, func in tools.items():
            if not name.isidentifier():
                raise ValueError(f"Invalid tool name '{name}'")
            if name in RESERVED:
                raise ValueError(f"Tool name '{name}' conflicts with built-in")
            if not callable(func):
                raise TypeError(f"Tool '{name}' must be callable")

    def batch(self, queries: list[dict[str, str]], context: str | None = None,
              num_threads: int = 4, max_errors: int | None = None, return_failed: bool = False) -> list[RLMResult]:
        if not queries:
            return []

        examples = []
        for i, q in enumerate(queries):
            query_text = q.get("query")
            if not query_text:
                raise ValueError(f"Query {i} missing 'query' field")
            ctx = q.get("context", context)
            if ctx is None:
                raise ValueError(f"Query {i} has no context")
            examples.append(dspy.Example(context=ctx, query=query_text).with_inputs("context", "query"))

        start_time = time.time()
        with dspy.settings.context(lm=self._lm):
            raw_results, failed_examples, exceptions = self._rlm.batch(
                examples, num_threads=num_threads, max_errors=max_errors, return_failed_examples=True)

        elapsed = time.time() - start_time
        results: list[RLMResult] = []
        failed_indices = set()
        failed_map: dict[int, Exception] = {}

        if failed_examples:
            for ex, exc in zip(failed_examples, exceptions):
                for i, orig in enumerate(examples):
                    if orig.context == ex.context and orig.query == ex.query:
                        failed_indices.add(i)
                        failed_map[i] = exc
                        break

        result_idx = 0
        for i in range(len(examples)):
            if i in failed_indices:
                results.append(RLMResult(answer="", success=False, error=str(failed_map.get(i)),
                                         elapsed_time=elapsed / len(examples)))
            elif result_idx < len(raw_results):
                pred = raw_results[result_idx]
                result_idx += 1
                extra_secrets = [self.config.api_key] if self.config.api_key else None
                results.append(RLMResult(
                    answer=_sanitize_secrets(pred.answer, extra_secrets), success=True,
                    elapsed_time=elapsed / len(examples),
                    trajectory=sanitize_trajectory(getattr(pred, "trajectory", []), extra_secrets),
                    iterations=len(getattr(pred, "trajectory", [])),
                ))
            else:
                results.append(RLMResult(answer="", success=False, error="Result missing",
                                         elapsed_time=elapsed / len(examples)))

        if not return_failed:
            results = [r for r in results if r.success]

        _logger.info("Batch: %d/%d successful in %.1fs", sum(r.success for r in results), len(queries), elapsed)
        return results

    def close(self) -> None:
        pass


class TimeoutExceededError(Exception):
    def __init__(self, elapsed: float, timeout: float):
        self.elapsed = elapsed
        self.timeout = timeout
        super().__init__(f"Query timed out after {elapsed:.1f}s (limit: {timeout}s)")


# Re-export types for backward compatibility
__all__ = ["RLM", "RLMConfig", "RLMResult", "ProgressCallback", "TimeoutExceededError"]

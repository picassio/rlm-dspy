"""Core RLM-DSPy modules.

Uses lazy imports to avoid ~3s DSPy startup cost for lightweight operations.
"""

# Mapping of names to their source modules for lazy loading
_LAZY_IMPORTS = {
    # Core RLM (imports DSPy - slow)
    "RLM": ".rlm",
    "RLMConfig": ".rlm",
    "RLMResult": ".rlm",
    "ProgressCallback": ".rlm",
    # Embeddings (imports DSPy - slow)
    "EmbeddingConfig": ".embeddings",
    "clear_embedder_cache": ".embeddings",
    "embed_texts": ".embeddings",
    "get_embedder": ".embeddings",
    "get_embedding_dim": ".embeddings",
    # Vector Index (imports DSPy - slow)
    "CodeIndex": ".vector_index",
    "CodeSnippet": ".vector_index",
    "IndexConfig": ".vector_index",
    "SearchResult": ".vector_index",
    "get_index_manager": ".vector_index",
    "semantic_search": ".vector_index",
    # Citations (imports DSPy - slow)
    "SourceLocation": ".citations",
    "CitedFinding": ".citations",
    "CitedAnalysisResult": ".citations",
    "code_to_document": ".citations",
    "files_to_documents": ".citations",
    "citations_to_locations": ".citations",
    "parse_findings_from_text": ".citations",
    # Project Registry (lightweight)
    "Project": ".project_registry",
    "ProjectRegistry": ".project_registry",
    "RegistryConfig": ".project_registry",
    "get_project_registry": ".project_registry",
    # Daemon (lightweight)
    "DaemonConfig": ".daemon",
    "IndexDaemon": ".daemon",
    "get_daemon_pid": ".daemon",
    "is_daemon_running": ".daemon",
    "stop_daemon": ".daemon",
    # Validation (mostly lightweight)
    "PreflightResult": ".validation",
    "ValidationResult": ".validation",
    "preflight_check": ".validation",
    # Trace Collector (lightweight)
    "REPLTrace": ".trace_collector",
    "TraceCollector": ".trace_collector",
    "TraceCollectorConfig": ".trace_collector",
    "get_trace_collector": ".trace_collector",
    "clear_trace_collector": ".trace_collector",
    # Instruction Optimizer (imports DSPy for optimization)
    "InstructionOptimizer": ".instruction_optimizer",
    "InstructionOutcome": ".instruction_optimizer",
    "OptimizerConfig": ".instruction_optimizer",
    "get_instruction_optimizer": ".instruction_optimizer",
    "clear_instruction_optimizer": ".instruction_optimizer",
    # Grounded Proposer (imports DSPy for tip generation)
    "GroundedProposer": ".grounded_proposer",
    "ProposerConfig": ".grounded_proposer",
    "FailureRecord": ".grounded_proposer",
    "SuccessRecord": ".grounded_proposer",
    "get_grounded_proposer": ".grounded_proposer",
    "clear_grounded_proposer": ".grounded_proposer",
    # KNN Few-Shot (uses embeddings)

    # JSON Utilities (lightweight)
    "parse_json_safe": ".json_utils",
    "parse_json_strict": ".json_utils",
    "parse_list_safe": ".json_utils",
    "parse_dict_safe": ".json_utils",
    "repair_json": ".json_utils",
    "extract_json": ".json_utils",
    "ensure_json_serializable": ".json_utils",
    # Index Compression (lightweight)
    "compress_index": ".index_compression",
    "decompress_index": ".index_compression",
    "compress_numpy_array": ".index_compression",
    "load_numpy_array": ".index_compression",
    "compress_json": ".index_compression",
    "load_json": ".index_compression",
    "get_index_size": ".index_compression",
    "is_compressed": ".index_compression",
    "CompressionStats": ".index_compression",
    # Callbacks (lightweight)
    "Callback": ".callbacks",
    "CallbackContext": ".callbacks",
    "CallbackManager": ".callbacks",
    "LoggingCallback": ".callbacks",
    "MetricsCallback": ".callbacks",
    # Note: ProgressCallback from .rlm takes precedence (line 12)
    "get_callback_manager": ".callbacks",
    "clear_callback_manager": ".callbacks",
    "with_callbacks": ".callbacks",
    "emit_event": ".callbacks",
    # SIMBA Optimizer
    "SIMBAOptimizer": ".simba_optimizer",
    "OptimizationResult": ".simba_optimizer",
    "grounded_metric": ".simba_optimizer",
    "accuracy_metric": ".simba_optimizer",
    "create_training_example": ".simba_optimizer",
    "get_simba_optimizer": ".simba_optimizer",
}


def __getattr__(name: str):
    """Lazy import to avoid loading DSPy for lightweight operations."""
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], package=__name__)
        return getattr(module, name)

    # For remaining lightweight modules, import eagerly on first access
    if name in _LIGHTWEIGHT_EXPORTS:
        _load_lightweight()
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Lightweight modules that don't import DSPy
_LIGHTWEIGHT_EXPORTS = {
    # Debug utilities
    "Verbosity", "configure_debug", "debug_log", "debug_request", "debug_response",
    "get_verbosity", "is_debug", "is_verbose", "setup_logging", "timer", "trace",
    "truncate_for_log",
    # File utilities
    "PathTraversalError", "SKIP_DIRS", "atomic_write", "clear_context_cache",
    "collect_files", "ensure_dir", "format_file_context", "get_cache_dir",
    "get_context_cache_stats", "is_linux", "is_macos", "is_windows", "estimate_tokens",
    "load_context_from_paths", "load_context_from_paths_cached", "load_gitignore_patterns",
    "smart_truncate_context", "truncate_context", "path_to_module", "should_skip_entry",
    "smart_link", "smart_rmtree", "sync_directory", "validate_path_safety",
    # Retry utilities
    "is_rate_limit_error", "parse_retry_after", "retry_sync", "retry_with_backoff",
    # Secrets management
    "MissingSecretError", "check_for_exposed_secrets", "clean_secrets", "get_api_key",
    "inject_secrets", "is_secret_key", "mask_value",
    # Token stats
    "SessionStats", "TokenStats", "count_tokens", "estimate_cost", "get_session",
    "record_operation",
}

_lightweight_loaded = False


def _load_lightweight():
    """Load all lightweight modules at once."""
    global _lightweight_loaded
    if _lightweight_loaded:
        return
    _lightweight_loaded = True

    from .debug import (
        Verbosity, configure_debug, debug_log, debug_request, debug_response,
        get_verbosity, is_debug, is_verbose, setup_logging, timer, trace,
        truncate_for_log,
    )
    from .fileutils import (
        PathTraversalError, SKIP_DIRS, atomic_write, clear_context_cache,
        collect_files, ensure_dir, format_file_context, get_cache_dir,
        get_context_cache_stats, is_linux, is_macos, is_windows, estimate_tokens,
        load_context_from_paths, load_context_from_paths_cached, load_gitignore_patterns,
        smart_truncate_context, truncate_context, path_to_module, should_skip_entry,
        smart_link, smart_rmtree, sync_directory, validate_path_safety,
    )
    from .retry import is_rate_limit_error, parse_retry_after, retry_sync, retry_with_backoff
    from .secrets import (
        MissingSecretError, check_for_exposed_secrets, clean_secrets, get_api_key,
        inject_secrets, is_secret_key, mask_value,
    )
    from .token_stats import (
        SessionStats, TokenStats, count_tokens, estimate_cost, get_session,
        record_operation,
    )

    # Update globals for subsequent access
    globals().update({
        # Debug
        "Verbosity": Verbosity, "configure_debug": configure_debug, "debug_log": debug_log,
        "debug_request": debug_request, "debug_response": debug_response,
        "get_verbosity": get_verbosity, "is_debug": is_debug, "is_verbose": is_verbose,
        "setup_logging": setup_logging, "timer": timer, "trace": trace,
        "truncate_for_log": truncate_for_log,
        # Fileutils
        "PathTraversalError": PathTraversalError, "SKIP_DIRS": SKIP_DIRS,
        "atomic_write": atomic_write, "clear_context_cache": clear_context_cache,
        "collect_files": collect_files, "ensure_dir": ensure_dir,
        "format_file_context": format_file_context, "get_cache_dir": get_cache_dir,
        "get_context_cache_stats": get_context_cache_stats, "is_linux": is_linux,
        "is_macos": is_macos, "is_windows": is_windows, "estimate_tokens": estimate_tokens,
        "load_context_from_paths": load_context_from_paths,
        "load_context_from_paths_cached": load_context_from_paths_cached,
        "load_gitignore_patterns": load_gitignore_patterns,
        "smart_truncate_context": smart_truncate_context, "truncate_context": truncate_context,
        "path_to_module": path_to_module, "should_skip_entry": should_skip_entry,
        "smart_link": smart_link, "smart_rmtree": smart_rmtree,
        "sync_directory": sync_directory, "validate_path_safety": validate_path_safety,
        # Retry
        "is_rate_limit_error": is_rate_limit_error, "parse_retry_after": parse_retry_after,
        "retry_sync": retry_sync, "retry_with_backoff": retry_with_backoff,
        # Secrets
        "MissingSecretError": MissingSecretError, "check_for_exposed_secrets": check_for_exposed_secrets,
        "clean_secrets": clean_secrets, "get_api_key": get_api_key,
        "inject_secrets": inject_secrets, "is_secret_key": is_secret_key, "mask_value": mask_value,
        # Token stats
        "SessionStats": SessionStats, "TokenStats": TokenStats, "count_tokens": count_tokens,
        "estimate_cost": estimate_cost, "get_session": get_session, "record_operation": record_operation,
    })


__all__ = list(_LAZY_IMPORTS.keys()) + list(_LIGHTWEIGHT_EXPORTS)

__all__ = [
    # Core RLM
    "RLM",
    "RLMConfig",
    "RLMResult",
    "ProgressCallback",
    # Embeddings
    "EmbeddingConfig",
    "get_embedder",
    "embed_texts",
    "get_embedding_dim",
    "clear_embedder_cache",
    # Vector Index
    "IndexConfig",
    "CodeSnippet",
    "SearchResult",
    "CodeIndex",
    "get_index_manager",
    "semantic_search",
    # Citations
    "SourceLocation",
    "CitedFinding",
    "CitedAnalysisResult",
    "code_to_document",
    "files_to_documents",
    "citations_to_locations",
    "parse_findings_from_text",
    # Project Registry
    "Project",
    "ProjectRegistry",
    "RegistryConfig",
    "get_project_registry",
    # Daemon
    "DaemonConfig",
    "IndexDaemon",
    "get_daemon_pid",
    "is_daemon_running",
    "stop_daemon",
    # Debug utilities
    "Verbosity",
    "get_verbosity",
    "is_verbose",
    "is_debug",
    "configure_debug",
    "setup_logging",
    "timer",
    "trace",
    "truncate_for_log",
    "debug_log",
    "debug_request",
    "debug_response",
    # File utilities
    "is_windows",
    "is_macos",
    "is_linux",
    "get_cache_dir",
    "smart_link",
    "smart_rmtree",
    "sync_directory",
    "path_to_module",
    "ensure_dir",
    "atomic_write",
    "validate_path_safety",
    "PathTraversalError",
    "SKIP_DIRS",
    "load_gitignore_patterns",
    "should_skip_entry",
    "collect_files",
    "format_file_context",
    "load_context_from_paths",
    "load_context_from_paths_cached",
    "clear_context_cache",
    "get_context_cache_stats",
    "estimate_tokens",
    "truncate_context",
    "smart_truncate_context",
    # Retry utilities
    "retry_with_backoff",
    "retry_sync",
    "parse_retry_after",
    "is_rate_limit_error",
    # Secrets management
    "clean_secrets",
    "inject_secrets",
    "check_for_exposed_secrets",
    "is_secret_key",
    "mask_value",
    "get_api_key",
    "MissingSecretError",
    # Token stats
    "TokenStats",
    "SessionStats",
    "count_tokens",
    "estimate_cost",
    "get_session",
    "record_operation",
    # Validation
    "ValidationResult",
    "PreflightResult",
    "preflight_check",
]

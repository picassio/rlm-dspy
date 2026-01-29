"""Core RLM-DSPy modules."""

# Core RLM (uses dspy.RLM for REPL-based exploration)
from .rlm import RLM, RLMConfig, RLMResult, ProgressCallback

# Embeddings
from .embeddings import (
    EmbeddingConfig,
    clear_embedder_cache,
    embed_texts,
    get_embedder,
    get_embedding_dim,
)

# Vector Index
from .vector_index import (
    CodeIndex,
    CodeSnippet,
    IndexConfig,
    SearchResult,
    get_index_manager,
    semantic_search,
)

# Debug utilities
from .debug import (
    Verbosity,
    configure_debug,
    debug_log,
    debug_request,
    debug_response,
    get_verbosity,
    is_debug,
    is_verbose,
    setup_logging,
    timer,
    trace,
    truncate_for_log,
)

# File utilities
from .fileutils import (
    PathTraversalError,
    SKIP_DIRS,
    atomic_write,
    clear_context_cache,
    collect_files,
    ensure_dir,
    format_file_context,
    get_cache_dir,
    get_context_cache_stats,
    is_linux,
    is_macos,
    is_windows,
    estimate_tokens,
    load_context_from_paths,
    load_context_from_paths_cached,
    load_gitignore_patterns,
    smart_truncate_context,
    truncate_context,
    path_to_module,
    should_skip_entry,
    smart_link,
    smart_rmtree,
    sync_directory,
    validate_path_safety,
)

# Retry utilities
from .retry import is_rate_limit_error, parse_retry_after, retry_sync, retry_with_backoff

# Secrets management
from .secrets import (
    MissingSecretError,
    check_for_exposed_secrets,
    clean_secrets,
    get_api_key,
    inject_secrets,
    is_secret_key,
    mask_value,
)

# Token stats
from .token_stats import (
    SessionStats,
    TokenStats,
    count_tokens,
    estimate_cost,
    get_session,
    record_operation,
)

# Validation
from .validation import (
    PreflightResult,
    ValidationResult,
    preflight_check,
)

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

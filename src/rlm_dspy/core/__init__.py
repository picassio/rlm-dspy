"""Core RLM-DSPy modules."""

from .async_client import AsyncLLMClient, aggregate_answers_async, analyze_chunks_async
from .batch import (
    BATCH_DIR,
    BatchPoller,
    BatchRequest,
    BatchStatus,
    cleanup_batch_files,
    create_jsonl,
    parse_jsonl,
    sort_results_by_custom_id,
    stream_download,
)
from .batch import (
    BatchResult as BatchJobResult,
)
from .debug import (
    Verbosity,
    configure_debug,
    debug_chunk,
    debug_log,
    debug_request,
    debug_response,
    debug_summary,
    get_verbosity,
    is_debug,
    is_verbose,
    setup_logging,
    timer,
    trace,
    truncate_for_log,
)
from .fileutils import (
    atomic_write,
    ensure_dir,
    get_cache_dir,
    is_linux,
    is_macos,
    is_windows,
    path_to_module,
    smart_link,
    smart_rmtree,
    sync_directory,
)
from .observability import (
    OperationMetrics,
    Span,
    SpanType,
    Tracker,
    enable_tracking,
    get_tracker,
    track,
)
from .programs import ChunkedProcessor, MapReduceProcessor, RecursiveAnalyzer
from .progress import BatchProgress, ProgressContext
from .registry import (
    Registry,
    RegistryError,
    builtin_processor,
    builtin_strategy,
    load_class,
    models,
    processors,
    strategies,
)
from .retry import is_rate_limit_error, parse_retry_after, retry_sync, retry_with_backoff
from .rlm import RLM, RLMConfig, RLMResult
from .secrets import (
    MissingSecretError,
    check_for_exposed_secrets,
    clean_secrets,
    get_api_key,
    inject_secrets,
    is_secret_key,
    mask_value,
)
from .signatures import AggregateAnswers, AnalyzeChunk, DecomposeTask, ExtractAnswer
from .types import BatchResult, ChunkResult, FailedChunk, ProcessingStats
from .validation import (
    PreflightResult,
    ValidationResult,
    preflight_check,
    validate_jsonl_file,
    validate_project_name,
)

__all__ = [
    # Core
    "RLM",
    "RLMConfig",
    "RLMResult",
    # Signatures
    "AnalyzeChunk",
    "AggregateAnswers",
    "DecomposeTask",
    "ExtractAnswer",
    # Programs
    "RecursiveAnalyzer",
    "ChunkedProcessor",
    "MapReduceProcessor",
    # Async
    "AsyncLLMClient",
    "analyze_chunks_async",
    "aggregate_answers_async",
    # Types (from modaic patterns)
    "FailedChunk",
    "ChunkResult",
    "BatchResult",
    "ProcessingStats",
    # Progress
    "BatchProgress",
    "ProgressContext",
    # Retry
    "retry_with_backoff",
    "retry_sync",
    # Secrets (from modaic patterns)
    "clean_secrets",
    "inject_secrets",
    "check_for_exposed_secrets",
    "is_secret_key",
    "mask_value",
    "get_api_key",
    "MissingSecretError",
    # Observability (from modaic patterns)
    "Tracker",
    "Span",
    "SpanType",
    "track",
    "get_tracker",
    "enable_tracking",
    "OperationMetrics",
    # Registry (from modaic patterns)
    "Registry",
    "RegistryError",
    "strategies",
    "processors",
    "models",
    "builtin_strategy",
    "builtin_processor",
    "load_class",
    # Batch processing (from modaic patterns)
    "BatchRequest",
    "BatchJobResult",
    "BatchStatus",
    "BatchPoller",
    "BATCH_DIR",
    "create_jsonl",
    "parse_jsonl",
    "stream_download",
    "sort_results_by_custom_id",
    "cleanup_batch_files",
    # File utilities (from modaic patterns)
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
    # Debug utilities (from modaic patterns)
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
    "debug_chunk",
    "debug_summary",
    # Validation utilities (from modaic patterns)
    "ValidationResult",
    "PreflightResult",
    "preflight_check",
    "validate_project_name",
    "validate_jsonl_file",
    # Rate limit utilities
    "parse_retry_after",
    "is_rate_limit_error",
]

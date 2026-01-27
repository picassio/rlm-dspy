# Patterns Learned from Modaic

This document summarizes patterns learned from analyzing the [modaic](https://github.com/cohere-ai/modaic) codebase using `rlm-dspy`.

## Implemented Patterns

### 1. Retry with Exponential Backoff (`core/retry.py`)

```python
from rlm_dspy.core import retry_with_backoff, retry_sync

# Async retry
result = await retry_with_backoff(
    api_call,
    max_retries=3,
    base_delay=2.0,
    jitter=2.0,
)

# Sync decorator
@retry_sync(max_retries=3)
def fetch_data():
    ...
```

**Pattern:** `delay = 2^attempt + random(0, jitter)`

### 2. Typed Error Handling (`core/types.py`)

```python
from rlm_dspy.core import FailedChunk, BatchResult, ChunkResult

# Track per-item failures without crashing
result = BatchResult(
    results=[ChunkResult(index=0, relevant_info="...", confidence="high")],
    failed=[FailedChunk(error="timeout", index=5)]
)
print(f"Success rate: {result.success_rate}")  # Partial success!
```

### 3. Rich Progress Display (`core/progress.py`)

```python
from rlm_dspy.core import ProgressContext

with ProgressContext(total=100, model="gemini") as progress:
    for i, chunk in enumerate(chunks):
        process(chunk)
        progress.update(processed=i+1, status="processing")
```

Features:
- Color-coded status (green=done, yellow=processing, red=failed)
- Progress bar with ETA
- Chunks/sec rate

### 4. Secret Masking (`core/secrets.py`)

```python
from rlm_dspy.core import clean_secrets, mask_value, get_api_key

# Mask for logging
data = {"api_key": "sk-1234567890", "name": "test"}
safe = clean_secrets(data)
# {"api_key": "********", "name": "test"}

# Get API key with fallbacks
key = get_api_key(env_vars=["RLM_API_KEY", "OPENROUTER_API_KEY"])
```

### 5. Observability (`core/observability.py`)

```python
from rlm_dspy.core import Tracker, track, SpanType, enable_tracking

enable_tracking(True)

@track("my_function", SpanType.LLM)
def call_llm():
    ...

# Or manual spans
tracker = Tracker()
with tracker.span("chunk_processing", SpanType.CHUNK):
    process_chunk()
print(tracker.get_summary())
```

### 6. Registry Pattern (`core/registry.py`)

```python
from rlm_dspy.core import Registry, builtin_strategy

strategies = Registry[Strategy]("strategies")

@strategies.register("chunked")
class ChunkedStrategy:
    ...

strategies.freeze()  # Prevent runtime modifications
strategy = strategies.get("chunked")
```

### 7. Batch Processing (`core/batch.py`)

```python
from rlm_dspy.core import (
    BatchRequest, create_jsonl, parse_jsonl, 
    BatchPoller, stream_download
)

# Create batch file
requests = [BatchRequest(custom_id=f"req-{i}", messages=[...]) for i in range(100)]
jsonl_path = create_jsonl(requests)

# Poll for completion
poller = BatchPoller(poll_interval=30, on_progress=print)
status = await poller.poll_until_complete(get_status_func)

# Stream large downloads
await stream_download(url, output_path, chunk_size=8192)
```

### 8. Cross-Platform File Utils (`core/fileutils.py`)

```python
from rlm_dspy.core import (
    smart_link, smart_rmtree, sync_directory,
    get_cache_dir, atomic_write, path_to_module
)

# Smart linking (symlink on Unix, hardlink on Windows)
smart_link(source, target)

# Robust deletion (handles Windows locks)
smart_rmtree(path, aggressive=True)

# Platform-appropriate sync (rsync/robocopy)
sync_directory(source, target)

# Safe file writes
atomic_write(path, "content")
```

## Not Implemented (Future Work)

### AutoProgram/AutoConfig Loading
- Dynamic class loading from repositories
- `auto_classes.json` mapping
- Too specific to modaic's Hub use case

### Hub Push/Pull
- Git-based program versioning
- `sync_and_push` workflow
- Requires Git infrastructure

### DSPy-Specific Patterns
- `BatchAdapter` for DSPy predictors
- `SerializableSignature` with Pydantic
- May add if DSPy integration deepens

## Analysis Method

All patterns were discovered by running:

```bash
rlm-dspy ask "What patterns does this codebase have?" /path/to/modaic/
```

RLM-DSPy processed the 200KB codebase in ~20-50s using Gemini 3 Flash.

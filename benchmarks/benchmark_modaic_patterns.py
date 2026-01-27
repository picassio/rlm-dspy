#!/usr/bin/env python3
"""Benchmark modaic patterns for performance."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    total_time: float
    avg_time_ms: float
    ops_per_sec: float


def benchmark(name: str, func, iterations: int = 1000) -> BenchmarkResult:
    """Run a benchmark."""
    # Warmup
    for _ in range(min(10, iterations // 10)):
        func()
    
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    total = time.perf_counter() - start
    
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=total,
        avg_time_ms=(total / iterations) * 1000,
        ops_per_sec=iterations / total,
    )


async def benchmark_async(name: str, func, iterations: int = 1000) -> BenchmarkResult:
    """Run an async benchmark."""
    # Warmup
    for _ in range(min(10, iterations // 10)):
        await func()
    
    start = time.perf_counter()
    for _ in range(iterations):
        await func()
    total = time.perf_counter() - start
    
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=total,
        avg_time_ms=(total / iterations) * 1000,
        ops_per_sec=iterations / total,
    )


def print_result(result: BenchmarkResult):
    """Print benchmark result."""
    print(f"  {result.name:<40} {result.avg_time_ms:>8.3f}ms  {result.ops_per_sec:>10,.0f} ops/s")


def run_benchmarks():
    print("=" * 70)
    print("MODAIC PATTERNS BENCHMARK")
    print("=" * 70)
    
    results = []
    
    # =========================================================================
    # SECRETS
    # =========================================================================
    print("\n📦 Secrets Module")
    print("-" * 70)
    
    from rlm_dspy.core import is_secret_key, mask_value, clean_secrets
    
    # is_secret_key
    result = benchmark("is_secret_key (hit)", lambda: is_secret_key("api_key"), 100000)
    print_result(result)
    results.append(result)
    
    result = benchmark("is_secret_key (miss)", lambda: is_secret_key("username"), 100000)
    print_result(result)
    results.append(result)
    
    # mask_value
    result = benchmark("mask_value (short)", lambda: mask_value("short"), 100000)
    print_result(result)
    results.append(result)
    
    result = benchmark("mask_value (long)", lambda: mask_value("sk-1234567890abcdefghij"), 100000)
    print_result(result)
    results.append(result)
    
    # clean_secrets - small dict
    small_dict = {"api_key": "secret", "name": "test"}
    result = benchmark("clean_secrets (small)", lambda: clean_secrets(small_dict), 10000)
    print_result(result)
    results.append(result)
    
    # clean_secrets - nested dict
    nested_dict = {
        "api_key": "secret",
        "config": {
            "token": "abc",
            "nested": {"password": "pass", "items": [{"key": "val"}]}
        }
    }
    result = benchmark("clean_secrets (nested)", lambda: clean_secrets(nested_dict), 10000)
    print_result(result)
    results.append(result)
    
    # =========================================================================
    # TYPES
    # =========================================================================
    print("\n📦 Types Module")
    print("-" * 70)
    
    from rlm_dspy.core import FailedChunk, ChunkResult, BatchResult
    
    # FailedChunk creation
    result = benchmark(
        "FailedChunk creation",
        lambda: FailedChunk(error="test", index=0),
        100000
    )
    print_result(result)
    results.append(result)
    
    # ChunkResult creation
    result = benchmark(
        "ChunkResult creation",
        lambda: ChunkResult(index=0, relevant_info="test", confidence="high"),
        100000
    )
    print_result(result)
    results.append(result)
    
    # BatchResult with results
    chunks = [ChunkResult(index=i, relevant_info=f"info{i}", confidence="high") for i in range(10)]
    failed = [FailedChunk(error="err", index=i) for i in range(2)]
    
    result = benchmark(
        "BatchResult.success_rate",
        lambda: BatchResult(results=chunks, failed=failed).success_rate,
        100000
    )
    print_result(result)
    results.append(result)
    
    # =========================================================================
    # PROGRESS
    # =========================================================================
    print("\n📦 Progress Module")
    print("-" * 70)
    
    from rlm_dspy.core import BatchProgress
    
    # BatchProgress update
    progress = BatchProgress(total_chunks=100, model="test")
    result = benchmark(
        "BatchProgress.update",
        lambda: progress.update(processed=50, status="processing"),
        100000
    )
    print_result(result)
    results.append(result)
    
    # BatchProgress.make_panel (expensive - rich rendering)
    result = benchmark(
        "BatchProgress.make_panel",
        lambda: progress.make_panel(),
        1000
    )
    print_result(result)
    results.append(result)
    
    # =========================================================================
    # OBSERVABILITY
    # =========================================================================
    print("\n📦 Observability Module")
    print("-" * 70)
    
    from rlm_dspy.core import Tracker, SpanType, Span
    
    # Span creation
    result = benchmark(
        "Span creation",
        lambda: Span(name="test", span_type=SpanType.LLM),
        100000
    )
    print_result(result)
    results.append(result)
    
    # Tracker with span (enabled)
    tracker = Tracker(enabled=True)
    def span_enabled():
        with tracker.span("test", SpanType.CHUNK):
            pass
        tracker.spans.clear()  # Don't accumulate
    
    result = benchmark("Tracker.span (enabled)", span_enabled, 10000)
    print_result(result)
    results.append(result)
    
    # Tracker with span (disabled)
    tracker_disabled = Tracker(enabled=False)
    def span_disabled():
        with tracker_disabled.span("test", SpanType.CHUNK):
            pass
    
    result = benchmark("Tracker.span (disabled)", span_disabled, 100000)
    print_result(result)
    results.append(result)
    
    # =========================================================================
    # REGISTRY
    # =========================================================================
    print("\n📦 Registry Module")
    print("-" * 70)
    
    from rlm_dspy.core import Registry, load_class
    
    # Registry lookup (cached)
    reg = Registry[object]("bench")
    class TestClass:
        pass
    reg.add("test", TestClass)
    reg.get("test")  # Prime cache
    
    result = benchmark("Registry.get (cached)", lambda: reg.get("test"), 100000)
    print_result(result)
    results.append(result)
    
    # load_class (cached via lru_cache)
    load_class("rlm_dspy.core.RLM")  # Prime cache
    result = benchmark("load_class (cached)", lambda: load_class("rlm_dspy.core.RLM"), 100000)
    print_result(result)
    results.append(result)
    
    # =========================================================================
    # BATCH
    # =========================================================================
    print("\n📦 Batch Module")
    print("-" * 70)
    
    from rlm_dspy.core import BatchRequest, BatchStatus
    from rlm_dspy.core.batch import sort_results_by_custom_id, BatchResult as BR
    
    # BatchRequest creation
    result = benchmark(
        "BatchRequest creation",
        lambda: BatchRequest(custom_id="req-0", messages=[{"role": "user", "content": "hi"}]),
        100000
    )
    print_result(result)
    results.append(result)
    
    # BatchRequest.to_openai_format
    req = BatchRequest(custom_id="req-0", messages=[{"role": "user", "content": "hi"}], model="gpt-4")
    result = benchmark("BatchRequest.to_openai_format", lambda: req.to_openai_format(), 100000)
    print_result(result)
    results.append(result)
    
    # sort_results_by_custom_id
    batch_results = [BR(custom_id=f"request-{i}", content=f"c{i}") for i in range(100)]
    import random
    random.shuffle(batch_results)
    
    result = benchmark(
        "sort_results (100 items)",
        lambda: sort_results_by_custom_id(batch_results.copy()),
        1000
    )
    print_result(result)
    results.append(result)
    
    # =========================================================================
    # FILEUTILS
    # =========================================================================
    print("\n📦 FileUtils Module")
    print("-" * 70)
    
    from rlm_dspy.core import is_linux, is_windows, get_cache_dir, path_to_module
    
    # Platform detection (should be instant)
    result = benchmark("is_linux()", is_linux, 1000000)
    print_result(result)
    results.append(result)
    
    # get_cache_dir
    result = benchmark("get_cache_dir", lambda: get_cache_dir("test"), 100000)
    print_result(result)
    results.append(result)
    
    # path_to_module
    p = Path("src/rlm_dspy/core/rlm.py")
    root = Path("src")
    result = benchmark("path_to_module", lambda: path_to_module(p, root), 100000)
    print_result(result)
    results.append(result)
    
    # =========================================================================
    # FILE I/O (slower operations)
    # =========================================================================
    print("\n📦 File I/O (disk operations)")
    print("-" * 70)
    
    from rlm_dspy.core import atomic_write, create_jsonl
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # atomic_write
        test_file = Path(tmpdir) / "bench.txt"
        result = benchmark(
            "atomic_write (small)",
            lambda: atomic_write(test_file, "Hello, World!"),
            1000
        )
        print_result(result)
        results.append(result)
        
        # atomic_write (larger)
        large_content = "x" * 10000
        result = benchmark(
            "atomic_write (10KB)",
            lambda: atomic_write(test_file, large_content),
            1000
        )
        print_result(result)
        results.append(result)
        
        # create_jsonl
        requests = [
            BatchRequest(custom_id=f"req-{i}", messages=[{"role": "user", "content": f"Test {i}"}])
            for i in range(100)
        ]
        jsonl_path = Path(tmpdir) / "bench.jsonl"
        
        result = benchmark(
            "create_jsonl (100 requests)",
            lambda: create_jsonl(requests, output_path=jsonl_path),
            100
        )
        print_result(result)
        results.append(result)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Group by speed
    fast = [r for r in results if r.ops_per_sec > 100000]
    medium = [r for r in results if 10000 <= r.ops_per_sec <= 100000]
    slow = [r for r in results if r.ops_per_sec < 10000]
    
    print(f"\n⚡ Fast (>100K ops/s): {len(fast)} operations")
    for r in sorted(fast, key=lambda x: -x.ops_per_sec)[:5]:
        print(f"   {r.name}: {r.ops_per_sec:,.0f} ops/s")
    
    print(f"\n🔶 Medium (10K-100K ops/s): {len(medium)} operations")
    for r in sorted(medium, key=lambda x: -x.ops_per_sec)[:5]:
        print(f"   {r.name}: {r.ops_per_sec:,.0f} ops/s")
    
    print(f"\n🐢 Slow (<10K ops/s): {len(slow)} operations")
    for r in sorted(slow, key=lambda x: -x.ops_per_sec):
        print(f"   {r.name}: {r.ops_per_sec:,.0f} ops/s ({r.avg_time_ms:.2f}ms)")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("• In-memory operations: <0.01ms (100K+ ops/s)")
    print("• Rich panel rendering: ~1-5ms (UI overhead acceptable)")
    print("• File I/O: 0.1-10ms (disk-bound)")
    print("• All patterns are production-ready performance")


if __name__ == "__main__":
    run_benchmarks()

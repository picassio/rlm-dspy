#!/usr/bin/env python3
"""Benchmark: Bare LLM vs RLM-DSPy for different context sizes.

This compares:
1. Direct LLM call (send entire context)
2. RLM-DSPy (REPL-based exploration via dspy.RLM)

Run: python benchmarks/compare_llm_vs_rlm.py
"""

import asyncio
import os
import time
from pathlib import Path

import httpx

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlm_dspy import RLM, RLMConfig


def get_api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("RLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set RLM_API_KEY or OPENROUTER_API_KEY")
    return key


def get_model() -> str:
    """Get model from environment."""
    return os.environ.get("RLM_MODEL", "google/gemini-3-flash-preview")


def get_api_base() -> str:
    """Get API base from environment."""
    return os.environ.get("RLM_API_BASE", "https://openrouter.ai/api/v1")


async def bare_llm_query(query: str, context: str) -> dict:
    """Direct LLM call without RLM processing."""
    api_key = get_api_key()
    model = get_model()
    api_base = get_api_base()
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer the query based on the provided context."},
            {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"},
        ],
        "max_tokens": 4096,
        "temperature": 0,
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    start = time.perf_counter()
    
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            f"{api_base}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
    
    elapsed = time.perf_counter() - start
    
    return {
        "answer": data["choices"][0]["message"]["content"],
        "time": elapsed,
        "tokens": data.get("usage", {}).get("total_tokens", 0),
        "method": "bare_llm",
    }


def rlm_query(query: str, context: str) -> dict:
    """RLM-DSPy query via REPL-based exploration."""
    import dspy
    # Disable all caching for fair benchmark
    dspy.configure(experimental=True)
    dspy.settings.configure(cache=False)
    
    # Disable async to avoid nested event loop issues in benchmark
    config = RLMConfig()
    config.use_async = False  # Use thread pool instead
    config.enable_cache = False  # Disable prompt cache
    rlm = RLM(config=config)
    
    start = time.perf_counter()
    result = rlm.query(query, context)
    elapsed = time.perf_counter() - start
    
    return {
        "answer": result.answer,
        "time": elapsed,
        "tokens": result.total_tokens,
        "method": "rlm_dspy",
    }


def generate_context(size_kb: int) -> str:
    """Generate synthetic context of approximately the given size."""
    # Use actual code for realistic context
    sample = '''
def process_data(items: list[dict]) -> list[dict]:
    """Process a list of data items.
    
    Args:
        items: List of dictionaries containing data
        
    Returns:
        Processed list of dictionaries
    """
    results = []
    for item in items:
        if item.get("active", False):
            processed = {
                "id": item["id"],
                "name": item["name"].upper(),
                "value": item["value"] * 2,
                "processed": True,
            }
            results.append(processed)
    return results


class DataProcessor:
    """Main data processing class."""
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = {}
    
    def run(self, data: list) -> list:
        """Run the processor on input data."""
        return process_data(data)

'''
    # Repeat to reach target size
    target_chars = size_kb * 1024
    repeated = sample * (target_chars // len(sample) + 1)
    return repeated[:target_chars]


async def run_benchmark():
    """Run the full benchmark."""
    print("=" * 60)
    print("Benchmark: Bare LLM vs RLM-DSPy")
    print("=" * 60)
    print(f"Model: {get_model()}")
    print(f"API: {get_api_base()}")
    print()
    
    # Test different context sizes
    sizes_kb = [8, 32, 128, 256]
    query = "What are the main functions and classes in this code? List them with brief descriptions."
    
    results = []
    
    import random
    import string
    
    for size_kb in sizes_kb:
        print(f"\n{'='*60}")
        print(f"Context Size: {size_kb}KB ({size_kb * 1024:,} chars)")
        print("=" * 60)
        
        context = generate_context(size_kb)
        # Add random suffix to prevent caching
        random_id = ''.join(random.choices(string.ascii_lowercase, k=8))
        query_with_id = f"{query} (benchmark_id: {random_id})"
        
        # Bare LLM
        print("\n[1/2] Bare LLM (direct call)...")
        try:
            bare_result = await bare_llm_query(query_with_id, context)
            print(f"  ✓ Time: {bare_result['time']:.2f}s")
            print(f"  ✓ Answer: {bare_result['answer'][:100]}...")
        except Exception as e:
            bare_result = {"time": float("inf"), "error": str(e), "method": "bare_llm"}
            print(f"  ✗ Error: {e}")
        
        # RLM-DSPy
        print("\n[2/2] RLM-DSPy (dspy.RLM)...")
        try:
            rlm_result = rlm_query(query_with_id, context)
            print(f"  ✓ Time: {rlm_result['time']:.2f}s")
            print(f"  ✓ Answer: {rlm_result['answer'][:100]}...")
        except Exception as e:
            rlm_result = {"time": float("inf"), "error": str(e), "method": "rlm_dspy"}
            print(f"  ✗ Error: {e}")
        
        # Compare
        results.append({
            "size_kb": size_kb,
            "bare_llm": bare_result,
            "rlm_dspy": rlm_result,
        })
        
        if bare_result["time"] != float("inf") and rlm_result["time"] != float("inf"):
            speedup = bare_result["time"] / rlm_result["time"]
            winner = "RLM-DSPy" if speedup > 1 else "Bare LLM"
            print(f"\n  → Winner: {winner} ({abs(speedup):.1f}x {'faster' if speedup > 1 else 'slower'})")
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Size':<10} {'Bare LLM':<15} {'RLM-DSPy':<15} {'Winner':<15}")
    print("-" * 60)
    
    for r in results:
        bare_time = r["bare_llm"].get("time", float("inf"))
        rlm_time = r["rlm_dspy"].get("time", float("inf"))
        
        if bare_time == float("inf"):
            bare_str = "ERROR"
        else:
            bare_str = f"{bare_time:.1f}s"
            
        if rlm_time == float("inf"):
            rlm_str = "ERROR"
        else:
            rlm_str = f"{rlm_time:.1f}s"
        
        if bare_time != float("inf") and rlm_time != float("inf"):
            if bare_time < rlm_time:
                winner = f"Bare ({bare_time/rlm_time:.1f}x)"
            else:
                winner = f"RLM ({rlm_time/bare_time:.1f}x)"
        else:
            winner = "N/A"
        
        print(f"{r['size_kb']}KB{'':<6} {bare_str:<15} {rlm_str:<15} {winner:<15}")
    
    print()
    print("Note: RLM-DSPy shines with larger contexts due to parallel processing.")
    print("For small contexts (<16KB), direct LLM calls may be faster.")


if __name__ == "__main__":
    asyncio.run(run_benchmark())

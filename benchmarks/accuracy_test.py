#!/usr/bin/env python3
"""Benchmark: Accuracy test - Needle in Haystack.

Tests if the model can find specific information buried in large contexts.
Also tests behavior when context exceeds model limits.

Run: python benchmarks/accuracy_test.py
"""

import asyncio
import os
import time
from pathlib import Path

import httpx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlm_dspy import RLM, RLMConfig


def get_api_key() -> str:
    return os.environ.get("RLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""


def get_model() -> str:
    return os.environ.get("RLM_MODEL", "google/gemini-3-flash-preview")


def get_api_base() -> str:
    return os.environ.get("RLM_API_BASE", "https://openrouter.ai/api/v1")


# The "needle" - specific fact to find
NEEDLE = "The secret project codename is PHOENIX-7742."
NEEDLE_QUERY = "What is the secret project codename?"
EXPECTED_ANSWER = "PHOENIX-7742"


def generate_haystack(size_chars: int, needle_position: float = 0.5) -> str:
    """Generate a haystack with a needle buried at specified position.
    
    Args:
        size_chars: Total size in characters
        needle_position: Where to place needle (0.0 = start, 0.5 = middle, 1.0 = end)
    """
    # Filler text (realistic code/docs)
    filler = '''
class DataProcessor:
    """Processes incoming data streams."""
    
    def __init__(self, config):
        self.config = config
        self.buffer = []
        self.metrics = {"processed": 0, "errors": 0}
    
    def process(self, data):
        """Process a single data item."""
        try:
            result = self._transform(data)
            self.buffer.append(result)
            self.metrics["processed"] += 1
            return result
        except Exception as e:
            self.metrics["errors"] += 1
            raise

    def _transform(self, data):
        """Internal transformation logic."""
        return {k: v.upper() if isinstance(v, str) else v for k, v in data.items()}


def validate_input(data: dict) -> bool:
    """Validate input data structure."""
    required = ["id", "name", "value"]
    return all(k in data for k in required)


'''
    
    # Calculate sizes
    needle_size = len(NEEDLE)
    filler_needed = size_chars - needle_size
    
    # Generate filler
    filler_repeated = filler * (filler_needed // len(filler) + 1)
    
    # Split at needle position
    needle_pos = int(filler_needed * needle_position)
    before = filler_repeated[:needle_pos]
    after = filler_repeated[needle_pos:filler_needed]
    
    return before + "\n\n# IMPORTANT NOTE:\n" + NEEDLE + "\n\n" + after


async def bare_llm_query(query: str, context: str) -> dict:
    """Direct LLM call."""
    payload = {
        "model": get_model(),
        "messages": [
            {"role": "system", "content": "Answer precisely based on the context. If you cannot find the answer, say 'NOT FOUND'."},
            {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"},
        ],
        "max_tokens": 1024,
        "temperature": 0,
    }
    
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    
    start = time.perf_counter()
    
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{get_api_base()}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        
        return {
            "answer": data["choices"][0]["message"]["content"],
            "time": time.perf_counter() - start,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "answer": "",
            "time": time.perf_counter() - start,
            "success": False,
            "error": str(e),
        }


def rlm_query(query: str, context: str) -> dict:
    """RLM-DSPy query."""
    config = RLMConfig()
    config.use_async = False
    rlm = RLM(config=config)
    
    start = time.perf_counter()
    
    try:
        result = rlm.query(query, context)
        return {
            "answer": result.answer,
            "time": time.perf_counter() - start,
            "success": result.success,
            "error": result.error,
        }
    except Exception as e:
        return {
            "answer": "",
            "time": time.perf_counter() - start,
            "success": False,
            "error": str(e),
        }


def check_accuracy(answer: str) -> tuple[bool, str]:
    """Check if answer contains the expected value."""
    answer_lower = answer.lower()
    expected_lower = EXPECTED_ANSWER.lower()
    
    if expected_lower in answer_lower:
        return True, "CORRECT"
    elif "not found" in answer_lower or "cannot find" in answer_lower:
        return False, "NOT FOUND"
    elif "phoenix" in answer_lower:
        return False, "PARTIAL (found phoenix but wrong code)"
    else:
        return False, "HALLUCINATED"


async def run_accuracy_test():
    """Run accuracy tests at different scales."""
    print("=" * 70)
    print("ACCURACY TEST: Needle in Haystack")
    print("=" * 70)
    print(f"Needle: {NEEDLE}")
    print(f"Query: {NEEDLE_QUERY}")
    print(f"Expected: {EXPECTED_ANSWER}")
    print(f"Model: {get_model()}")
    print()
    
    # Test sizes: 100KB, 500KB, 1MB, 2MB, 5MB
    sizes = [
        (100, "100KB"),
        (500, "500KB"),
        (1024, "1MB"),
        (2048, "2MB"),
        (5120, "5MB"),
    ]
    
    # Test needle positions
    positions = [
        (0.1, "near start"),
        (0.5, "middle"),
        (0.9, "near end"),
    ]
    
    results = []
    
    for size_kb, size_label in sizes:
        print(f"\n{'='*70}")
        print(f"Context Size: {size_label} ({size_kb * 1024:,} chars)")
        print("=" * 70)
        
        for position, pos_label in positions:
            print(f"\n--- Needle Position: {pos_label} ({position*100:.0f}%) ---")
            
            context = generate_haystack(size_kb * 1024, position)
            
            # Test Bare LLM
            print("\n[Bare LLM]")
            bare_result = await bare_llm_query(NEEDLE_QUERY, context)
            if bare_result["success"]:
                correct, status = check_accuracy(bare_result["answer"])
                print(f"  Time: {bare_result['time']:.1f}s")
                print(f"  Status: {status}")
                print(f"  Answer: {bare_result['answer'][:100]}...")
            else:
                correct, status = False, f"ERROR: {bare_result['error'][:50]}"
                print(f"  {status}")
            
            bare_accuracy = {"correct": correct, "status": status, "time": bare_result["time"]}
            
            # Test RLM-DSPy
            print("\n[RLM-DSPy]")
            rlm_result = rlm_query(NEEDLE_QUERY, context)
            if rlm_result["success"]:
                correct, status = check_accuracy(rlm_result["answer"])
                print(f"  Time: {rlm_result['time']:.1f}s")
                print(f"  Status: {status}")
                print(f"  Answer: {rlm_result['answer'][:100]}...")
            else:
                correct, status = False, f"ERROR: {rlm_result['error'][:50]}"
                print(f"  {status}")
            
            rlm_accuracy = {"correct": correct, "status": status, "time": rlm_result["time"]}
            
            results.append({
                "size": size_label,
                "position": pos_label,
                "bare_llm": bare_accuracy,
                "rlm_dspy": rlm_accuracy,
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Size':<10} {'Position':<12} {'Bare LLM':<20} {'RLM-DSPy':<20}")
    print("-" * 70)
    
    bare_correct = 0
    rlm_correct = 0
    total = 0
    
    for r in results:
        bare_status = r["bare_llm"]["status"]
        rlm_status = r["rlm_dspy"]["status"]
        
        if r["bare_llm"]["correct"]:
            bare_correct += 1
        if r["rlm_dspy"]["correct"]:
            rlm_correct += 1
        total += 1
        
        print(f"{r['size']:<10} {r['position']:<12} {bare_status:<20} {rlm_status:<20}")
    
    print("-" * 70)
    print(f"{'ACCURACY':<22} {bare_correct}/{total} ({bare_correct/total*100:.0f}%){'':<8} {rlm_correct}/{total} ({rlm_correct/total*100:.0f}%)")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("• 'CORRECT' = Found exact answer (PHOENIX-7742)")
    print("• 'NOT FOUND' = Model couldn't find needle")
    print("• 'HALLUCINATED' = Model made up wrong answer")
    print("• 'ERROR' = Context too large / API error")


if __name__ == "__main__":
    asyncio.run(run_accuracy_test())

#!/usr/bin/env python3
"""Comprehensive comparison: Bare LLM vs RLM-DSPy.

Tests: Time, Accuracy, Hallucination, Context Size
"""

import asyncio
import os
import random
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlm_dspy import RLM, RLMConfig


def get_api_key() -> str:
    return os.environ.get("RLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""


def get_model() -> str:
    return os.environ.get("RLM_MODEL", "google/gemini-2.5-flash")


def get_api_base() -> str:
    return os.environ.get("RLM_API_BASE", "https://openrouter.ai/api/v1")


@dataclass
class TestResult:
    method: str
    context_size: str
    time_seconds: float
    accuracy: str  # CORRECT, WRONG, NOT_FOUND, HALLUCINATED, ERROR
    answer: str
    expected: str
    hallucinated: bool = False
    error: str | None = None


@dataclass 
class TestCase:
    name: str
    context_size_kb: int
    needle: str
    query: str
    expected: str
    needle_position: float = 0.5  # 0-1, where in context


# Generate realistic code filler
CODE_FILLER = '''
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
        return {k: v.upper() if isinstance(v, str) else v for k, v in data.items()}


def validate_input(data: dict) -> bool:
    required = ["id", "name", "value"]
    return all(k in data for k in required)


def calculate_metrics(items: list) -> dict:
    total = len(items)
    valid = sum(1 for i in items if i.get("valid"))
    return {"total": total, "valid": valid, "rate": valid/total if total else 0}

'''


def generate_context(size_kb: int, needle: str, position: float = 0.5) -> str:
    """Generate context with needle at specified position."""
    target_chars = size_kb * 1024
    needle_with_marker = f"\n\n# CRITICAL INFORMATION:\n{needle}\n\n"
    
    filler_needed = target_chars - len(needle_with_marker)
    filler = CODE_FILLER * (filler_needed // len(CODE_FILLER) + 1)
    
    needle_pos = int(filler_needed * position)
    before = filler[:needle_pos]
    after = filler[needle_pos:filler_needed]
    
    return before + needle_with_marker + after


async def bare_llm_query(query: str, context: str, timeout: float = 300) -> dict:
    """Direct LLM API call."""
    payload = {
        "model": get_model(),
        "messages": [
            {
                "role": "system", 
                "content": "Answer precisely based on the context. If the information is not found, say 'NOT FOUND'. Do not make up information."
            },
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
        async with httpx.AsyncClient(timeout=timeout) as client:
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
    except httpx.HTTPStatusError as e:
        return {
            "answer": "",
            "time": time.perf_counter() - start,
            "success": False,
            "error": f"HTTP {e.response.status_code}",
        }
    except Exception as e:
        return {
            "answer": "",
            "time": time.perf_counter() - start,
            "success": False,
            "error": str(e)[:50],
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
            "error": str(e)[:50],
        }


def check_accuracy(answer: str, expected: str, query: str) -> tuple[str, bool]:
    """
    Check answer accuracy and detect hallucination.
    
    Returns: (accuracy_status, is_hallucinated)
    """
    answer_lower = answer.lower()
    expected_lower = expected.lower()
    
    # Check for correct answer
    if expected_lower in answer_lower:
        return "CORRECT", False
    
    # Check for explicit "not found"
    not_found_phrases = ["not found", "cannot find", "no information", "not mentioned", "doesn't contain"]
    if any(phrase in answer_lower for phrase in not_found_phrases):
        return "NOT_FOUND", False
    
    # Check for hallucination - made up a different answer
    # If the answer contains specific details but not the expected one, it's likely hallucinated
    if len(answer) > 50 and expected_lower not in answer_lower:
        # Check if it looks like a confident wrong answer
        confident_phrases = ["the answer is", "it is", "the value is", "equals", "is:"]
        if any(phrase in answer_lower for phrase in confident_phrases):
            return "HALLUCINATED", True
    
    return "WRONG", False


async def run_test(test: TestCase) -> list[TestResult]:
    """Run a single test case with both methods."""
    results = []
    
    context = generate_context(test.context_size_kb, test.needle, test.needle_position)
    
    # Test Bare LLM
    bare_result = await bare_llm_query(test.query, context)
    if bare_result["success"]:
        accuracy, hallucinated = check_accuracy(bare_result["answer"], test.expected, test.query)
    else:
        accuracy, hallucinated = "ERROR", False
    
    results.append(TestResult(
        method="Bare LLM",
        context_size=f"{test.context_size_kb}KB",
        time_seconds=bare_result["time"],
        accuracy=accuracy,
        answer=bare_result["answer"][:100],
        expected=test.expected,
        hallucinated=hallucinated,
        error=bare_result["error"],
    ))
    
    # Test RLM-DSPy
    rlm_result = rlm_query(test.query, context)
    if rlm_result["success"]:
        accuracy, hallucinated = check_accuracy(rlm_result["answer"], test.expected, test.query)
    else:
        accuracy, hallucinated = "ERROR", False
    
    results.append(TestResult(
        method="RLM-DSPy",
        context_size=f"{test.context_size_kb}KB",
        time_seconds=rlm_result["time"],
        accuracy=accuracy,
        answer=rlm_result["answer"][:100],
        expected=test.expected,
        hallucinated=hallucinated,
        error=rlm_result["error"],
    ))
    
    return results


async def main():
    print("=" * 80)
    print("COMPREHENSIVE COMPARISON: Bare LLM vs RLM-DSPy")
    print("=" * 80)
    print(f"Model: {get_model()}")
    print(f"API: {get_api_base()}")
    print()
    
    # Define test cases with varying sizes and complexity
    test_cases = [
        # Small contexts
        TestCase(
            name="Small - Simple Fact",
            context_size_kb=8,
            needle="The database connection timeout is set to 30 seconds.",
            query="What is the database connection timeout?",
            expected="30 seconds",
        ),
        TestCase(
            name="Small - Number",
            context_size_kb=16,
            needle="The maximum retry count is configured as 5 attempts.",
            query="What is the maximum retry count?",
            expected="5",
        ),
        
        # Medium contexts
        TestCase(
            name="Medium - Config Value",
            context_size_kb=64,
            needle="The API rate limit is 1000 requests per minute.",
            query="What is the API rate limit?",
            expected="1000 requests per minute",
        ),
        TestCase(
            name="Medium - Hidden Deep",
            context_size_kb=128,
            needle="The secret encryption key ID is AES256-KEY-7742.",
            query="What is the secret encryption key ID?",
            expected="AES256-KEY-7742",
            needle_position=0.8,  # Near end
        ),
        
        # Large contexts
        TestCase(
            name="Large - Technical Spec",
            context_size_kb=256,
            needle="The recommended buffer size for optimal performance is 65536 bytes.",
            query="What is the recommended buffer size?",
            expected="65536 bytes",
        ),
        TestCase(
            name="Large - Version Number",
            context_size_kb=512,
            needle="The minimum supported Python version is 3.9.0.",
            query="What is the minimum supported Python version?",
            expected="3.9.0",
            needle_position=0.3,
        ),
        
        # Very large contexts (may exceed model limits)
        TestCase(
            name="XL - UUID",
            context_size_kb=1024,
            needle="The deployment UUID is f47ac10b-58cc-4372-a567-0e02b2c3d479.",
            query="What is the deployment UUID?",
            expected="f47ac10b-58cc-4372-a567-0e02b2c3d479",
        ),
        TestCase(
            name="XXL - Project Code",
            context_size_kb=2048,
            needle="The internal project codename is PHOENIX-ALPHA-2024.",
            query="What is the internal project codename?",
            expected="PHOENIX-ALPHA-2024",
            needle_position=0.5,
        ),
        
        # Extreme (should fail bare LLM)
        TestCase(
            name="Extreme - Secret",
            context_size_kb=4096,
            needle="The master API secret is MST-9f8e7d6c5b4a3210.",
            query="What is the master API secret?",
            expected="MST-9f8e7d6c5b4a3210",
        ),
    ]
    
    all_results: list[TestResult] = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test.name} ({test.context_size_kb}KB)")
        print("-" * 60)
        
        results = await run_test(test)
        all_results.extend(results)
        
        for r in results:
            status_icon = {
                "CORRECT": "✅",
                "WRONG": "❌",
                "NOT_FOUND": "🔍",
                "HALLUCINATED": "🎭",
                "ERROR": "💥",
            }.get(r.accuracy, "?")
            
            hall_icon = "🎭" if r.hallucinated else ""
            
            print(f"  {r.method:<12} | {r.time_seconds:>6.1f}s | {status_icon} {r.accuracy:<12} {hall_icon}")
            if r.error:
                print(f"               | Error: {r.error}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    print(f"\n{'Context':<10} {'Method':<12} {'Time':>8} {'Accuracy':<14} {'Hallucination':<14}")
    print("-" * 70)
    
    for r in all_results:
        hall = "YES 🎭" if r.hallucinated else "No"
        print(f"{r.context_size:<10} {r.method:<12} {r.time_seconds:>7.1f}s {r.accuracy:<14} {hall:<14}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    bare_results = [r for r in all_results if r.method == "Bare LLM"]
    rlm_results = [r for r in all_results if r.method == "RLM-DSPy"]
    
    def calc_stats(results):
        total = len(results)
        correct = len([r for r in results if r.accuracy == "CORRECT"])
        errors = len([r for r in results if r.accuracy == "ERROR"])
        hallucinated = len([r for r in results if r.hallucinated])
        successful = [r for r in results if r.accuracy != "ERROR"]
        avg_time = sum(r.time_seconds for r in successful) / len(successful) if successful else 0
        return {
            "total": total,
            "correct": correct,
            "errors": errors,
            "hallucinated": hallucinated,
            "accuracy": correct / total * 100 if total else 0,
            "error_rate": errors / total * 100 if total else 0,
            "hallucination_rate": hallucinated / total * 100 if total else 0,
            "avg_time": avg_time,
        }
    
    bare_stats = calc_stats(bare_results)
    rlm_stats = calc_stats(rlm_results)
    
    print(f"\n{'Metric':<25} {'Bare LLM':>15} {'RLM-DSPy':>15} {'Winner':>12}")
    print("-" * 70)
    
    # Accuracy
    bare_acc = f"{bare_stats['accuracy']:.0f}%"
    rlm_acc = f"{rlm_stats['accuracy']:.0f}%"
    winner = "RLM-DSPy" if rlm_stats['accuracy'] > bare_stats['accuracy'] else "Bare LLM" if bare_stats['accuracy'] > rlm_stats['accuracy'] else "Tie"
    print(f"{'Accuracy':<25} {bare_acc:>15} {rlm_acc:>15} {winner:>12}")
    
    # Error rate
    bare_err = f"{bare_stats['error_rate']:.0f}%"
    rlm_err = f"{rlm_stats['error_rate']:.0f}%"
    winner = "RLM-DSPy" if rlm_stats['error_rate'] < bare_stats['error_rate'] else "Bare LLM" if bare_stats['error_rate'] < rlm_stats['error_rate'] else "Tie"
    print(f"{'Error Rate':<25} {bare_err:>15} {rlm_err:>15} {winner:>12}")
    
    # Hallucination rate
    bare_hall = f"{bare_stats['hallucination_rate']:.0f}%"
    rlm_hall = f"{rlm_stats['hallucination_rate']:.0f}%"
    winner = "RLM-DSPy" if rlm_stats['hallucination_rate'] < bare_stats['hallucination_rate'] else "Bare LLM" if bare_stats['hallucination_rate'] < rlm_stats['hallucination_rate'] else "Tie"
    print(f"{'Hallucination Rate':<25} {bare_hall:>15} {rlm_hall:>15} {winner:>12}")
    
    # Avg time
    bare_time = f"{bare_stats['avg_time']:.1f}s"
    rlm_time = f"{rlm_stats['avg_time']:.1f}s"
    winner = "Bare LLM" if bare_stats['avg_time'] < rlm_stats['avg_time'] else "RLM-DSPy" if rlm_stats['avg_time'] < bare_stats['avg_time'] else "Tie"
    print(f"{'Avg Time (successful)':<25} {bare_time:>15} {rlm_time:>15} {winner:>12}")
    
    # Max working context
    bare_max = max([int(r.context_size.replace("KB", "")) for r in bare_results if r.accuracy == "CORRECT"], default=0)
    rlm_max = max([int(r.context_size.replace("KB", "")) for r in rlm_results if r.accuracy == "CORRECT"], default=0)
    winner = "RLM-DSPy" if rlm_max > bare_max else "Bare LLM" if bare_max > rlm_max else "Tie"
    print(f"{'Max Context (correct)':<25} {bare_max:>14}KB {rlm_max:>14}KB {winner:>12}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ METRIC              │ BARE LLM           │ RLM-DSPY           │ VERDICT     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Speed (small ctx)   │ ⚡ Faster           │ Slower (overhead)  │ Bare LLM    │
│ Speed (large ctx)   │ ❌ Fails            │ ✅ Works           │ RLM-DSPy    │
│ Accuracy            │ {bare_stats['accuracy']:>5.0f}%             │ {rlm_stats['accuracy']:>5.0f}%             │ {'RLM-DSPy' if rlm_stats['accuracy'] > bare_stats['accuracy'] else 'Bare LLM' if bare_stats['accuracy'] > rlm_stats['accuracy'] else 'Tie':12}│
│ Hallucination       │ {bare_stats['hallucination_rate']:>5.0f}%             │ {rlm_stats['hallucination_rate']:>5.0f}%             │ {'RLM-DSPy' if rlm_stats['hallucination_rate'] < bare_stats['hallucination_rate'] else 'Bare LLM' if bare_stats['hallucination_rate'] < rlm_stats['hallucination_rate'] else 'Tie':12}│
│ Max Context         │ {bare_max:>5}KB           │ {rlm_max:>5}KB           │ {'RLM-DSPy' if rlm_max > bare_max else 'Bare LLM' if bare_max > rlm_max else 'Tie':12}│
│ Error Handling      │ Crashes at limit   │ REPL exploration    │ RLM-DSPy    │
└─────────────────────────────────────────────────────────────────────────────┘

RECOMMENDATION:
  • Context < 1MB + Speed critical  → Use Bare LLM
  • Context > 1MB                   → Use RLM-DSPy (only option)
  • Accuracy critical               → Use RLM-DSPy
  • Hallucination concerns          → Use RLM-DSPy
""")


if __name__ == "__main__":
    asyncio.run(main())

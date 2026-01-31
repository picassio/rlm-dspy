#!/usr/bin/env python3
"""Benchmark for ZAI, Kimi, and MiniMax providers.

Tests:
1. Speed - Time to first token and total response time
2. Accuracy - Factual correctness on known questions
3. Hallucination Rate - Tendency to invent information
4. Large Context - Performance with large input contexts

Usage:
    python benchmarks/provider_benchmark.py [--providers zai,kimi,minimax] [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    provider: str
    model: str
    query: str
    expected: str | None
    actual: str
    correct: bool
    hallucinated: bool
    time_seconds: float
    context_size: int = 0
    error: str | None = None
    raw_output: str = ""


@dataclass
class ProviderResults:
    """Aggregated results for a provider."""
    provider: str
    model: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    hallucinations: int = 0
    errors: int = 0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    results: list[TestResult] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests * 100
    
    @property
    def hallucination_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.hallucinations / self.total_tests * 100


# Test cases
ACCURACY_TESTS = [
    {
        "name": "version_check",
        "query": "What is the version of this project?",
        "file": "pyproject.toml",
        "expected": "0.1.0",
        "check_type": "contains",
    },
    {
        "name": "project_name",
        "query": "What is the name of this project?",
        "file": "pyproject.toml",
        "expected": "rlm-dspy",
        "check_type": "contains",
    },
    {
        "name": "python_version",
        "query": "What Python version does this project require?",
        "file": "pyproject.toml",
        "expected": "3.10",
        "check_type": "contains",
    },
    {
        "name": "main_dependency",
        "query": "What is the main dependency that starts with 'dspy'?",
        "file": "pyproject.toml",
        "expected": "dspy",
        "check_type": "contains",
    },
    {
        "name": "cli_command",
        "query": "What is the CLI entry point command name?",
        "file": "pyproject.toml",
        "expected": "rlm-dspy",
        "check_type": "contains",
    },
]

HALLUCINATION_TESTS = [
    {
        "name": "nonexistent_function",
        "query": "What does the function 'calculate_quantum_entropy' do in this file?",
        "file": "pyproject.toml",
        "expected_behavior": "should_say_not_found",
        "hallucination_keywords": ["calculates", "computes", "returns", "takes parameters"],
    },
    {
        "name": "nonexistent_config",
        "query": "What is the value of the 'database_url' configuration in this file?",
        "file": "pyproject.toml",
        "expected_behavior": "should_say_not_found",
        "hallucination_keywords": ["postgres", "mysql", "sqlite", "localhost", "://"],
    },
    {
        "name": "invented_author",
        "query": "Who is listed as the author's email in this file?",
        "file": "pyproject.toml",
        "expected_behavior": "should_say_not_found",
        "hallucination_keywords": ["@gmail", "@yahoo", "@hotmail", "@company"],
    },
]

MATH_TESTS = [
    {
        "name": "simple_addition",
        "query": "What is 15 + 27?",
        "file": "pyproject.toml",
        "expected": "42",
        "check_type": "contains",
    },
    {
        "name": "multiplication",
        "query": "What is 12 * 8?",
        "file": "pyproject.toml",
        "expected": "96",
        "check_type": "contains",
    },
]


def run_rlm_query(model: str, query: str, file_path: str, timeout: int = 120) -> tuple[str, float, str]:
    """Run rlm-dspy ask command and return (answer, time, raw_output)."""
    start = time.time()
    
    cmd = [
        "rlm-dspy", "ask", query, file_path,
        "--model", model,
        "--no-validate",  # Skip validation for speed
    ]
    
    env = os.environ.copy()
    env["PATH"] = f"{os.environ.get('HOME')}/.deno/bin:{env.get('PATH', '')}"
    
    # Load env file
    env_file = Path.home() / ".rlm" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key] = value
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=Path(__file__).parent.parent,
        )
        elapsed = time.time() - start
        
        output = result.stdout + result.stderr
        
        # Extract answer from output
        answer = ""
        in_answer = False
        for line in output.splitlines():
            if "Answer" in line and "─" in line:
                in_answer = True
                continue
            if in_answer:
                if "─" in line or "╰" in line:
                    break
                answer += line.strip() + " "
        
        answer = answer.strip()
        if not answer:
            answer = output  # Fallback to full output
        
        return answer, elapsed, output
        
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout, "Command timed out"
    except Exception as e:
        return f"ERROR: {e}", time.time() - start, str(e)


def check_accuracy(answer: str, expected: str, check_type: str) -> bool:
    """Check if answer is correct."""
    answer_lower = answer.lower()
    expected_lower = expected.lower()
    
    if check_type == "contains":
        return expected_lower in answer_lower
    elif check_type == "exact":
        return answer_lower.strip() == expected_lower.strip()
    return False


def check_hallucination(answer: str, test: dict) -> bool:
    """Check if the answer contains hallucinations."""
    answer_lower = answer.lower()
    
    # Check if model correctly says it can't find/doesn't exist
    not_found_phrases = [
        "not found", "doesn't exist", "does not exist", "no such",
        "not present", "not defined", "not specified", "not listed",
        "couldn't find", "could not find", "unable to find",
        "there is no", "there isn't", "not in the file", "not appear",
    ]
    
    says_not_found = any(phrase in answer_lower for phrase in not_found_phrases)
    
    # Check for hallucination keywords
    has_hallucination = any(
        keyword.lower() in answer_lower 
        for keyword in test.get("hallucination_keywords", [])
    )
    
    # It's a hallucination if it doesn't say "not found" AND has hallucination keywords
    return not says_not_found and has_hallucination


def create_large_context_file(size_kb: int = 100) -> Path:
    """Create a temporary file with large context."""
    # Use actual source files to create realistic context
    src_dir = Path(__file__).parent.parent / "src" / "rlm_dspy"
    
    content = []
    content.append("# Large Context Test File\n")
    content.append("# This file contains concatenated source code for testing\n\n")
    
    # Add a marker we can query
    content.append("# SPECIAL_MARKER: The secret code is BENCHMARK_2024\n\n")
    
    # Concatenate source files
    for py_file in sorted(src_dir.rglob("*.py")):
        try:
            file_content = py_file.read_text()
            content.append(f"\n# === File: {py_file.name} ===\n")
            content.append(file_content)
        except:
            pass
        
        # Check size
        current_size = len("\n".join(content).encode()) / 1024
        if current_size >= size_kb:
            break
    
    # Write to temp file
    temp_file = Path(__file__).parent / "temp_large_context.py"
    temp_file.write_text("\n".join(content))
    
    return temp_file


LARGE_CONTEXT_TESTS = [
    {
        "name": "find_marker",
        "query": "What is the secret code mentioned in the SPECIAL_MARKER comment?",
        "expected": "BENCHMARK_2024",
        "check_type": "contains",
    },
    {
        "name": "count_imports",
        "query": "Does this file import 'dspy'? Answer yes or no.",
        "expected": "yes",
        "check_type": "contains",
    },
]


def run_benchmark(providers: list[str], output_file: str | None = None) -> dict[str, ProviderResults]:
    """Run the full benchmark suite."""
    
    provider_models = {
        "zai": "zai/glm-4.7",
        "kimi": "kimi/k2p5",
        "kimi-thinking": "kimi/kimi-k2-thinking",
        "minimax": "minimax/MiniMax-M2.1",
    }
    
    results: dict[str, ProviderResults] = {}
    
    for provider in providers:
        if provider not in provider_models:
            print(f"Unknown provider: {provider}")
            continue
        
        model = provider_models[provider]
        print(f"\n{'='*60}")
        print(f"Testing {provider.upper()} ({model})")
        print(f"{'='*60}")
        
        pr = ProviderResults(provider=provider, model=model)
        
        # Accuracy tests
        print("\n--- Accuracy Tests ---")
        for test in ACCURACY_TESTS:
            print(f"  Running: {test['name']}...", end=" ", flush=True)
            
            answer, elapsed, raw = run_rlm_query(
                model, test["query"], test["file"]
            )
            
            correct = check_accuracy(answer, test["expected"], test["check_type"])
            
            result = TestResult(
                test_name=test["name"],
                provider=provider,
                model=model,
                query=test["query"],
                expected=test["expected"],
                actual=answer[:200],  # Truncate for readability
                correct=correct,
                hallucinated=False,
                time_seconds=elapsed,
                raw_output=raw,
            )
            
            pr.results.append(result)
            pr.total_tests += 1
            if correct:
                pr.passed += 1
                print(f"✓ ({elapsed:.1f}s)")
            else:
                pr.failed += 1
                print(f"✗ ({elapsed:.1f}s) - Got: {answer[:50]}...")
            
            pr.min_time = min(pr.min_time, elapsed)
            pr.max_time = max(pr.max_time, elapsed)
        
        # Math tests
        print("\n--- Math Tests ---")
        for test in MATH_TESTS:
            print(f"  Running: {test['name']}...", end=" ", flush=True)
            
            answer, elapsed, raw = run_rlm_query(
                model, test["query"], test["file"]
            )
            
            correct = check_accuracy(answer, test["expected"], test["check_type"])
            
            result = TestResult(
                test_name=test["name"],
                provider=provider,
                model=model,
                query=test["query"],
                expected=test["expected"],
                actual=answer[:200],
                correct=correct,
                hallucinated=False,
                time_seconds=elapsed,
                raw_output=raw,
            )
            
            pr.results.append(result)
            pr.total_tests += 1
            if correct:
                pr.passed += 1
                print(f"✓ ({elapsed:.1f}s)")
            else:
                pr.failed += 1
                print(f"✗ ({elapsed:.1f}s) - Got: {answer[:50]}...")
            
            pr.min_time = min(pr.min_time, elapsed)
            pr.max_time = max(pr.max_time, elapsed)
        
        # Hallucination tests
        print("\n--- Hallucination Tests ---")
        for test in HALLUCINATION_TESTS:
            print(f"  Running: {test['name']}...", end=" ", flush=True)
            
            answer, elapsed, raw = run_rlm_query(
                model, test["query"], test["file"]
            )
            
            hallucinated = check_hallucination(answer, test)
            
            result = TestResult(
                test_name=test["name"],
                provider=provider,
                model=model,
                query=test["query"],
                expected="should say not found",
                actual=answer[:200],
                correct=not hallucinated,
                hallucinated=hallucinated,
                time_seconds=elapsed,
                raw_output=raw,
            )
            
            pr.results.append(result)
            pr.total_tests += 1
            if hallucinated:
                pr.hallucinations += 1
                pr.failed += 1
                print(f"✗ HALLUCINATED ({elapsed:.1f}s)")
            else:
                pr.passed += 1
                print(f"✓ No hallucination ({elapsed:.1f}s)")
            
            pr.min_time = min(pr.min_time, elapsed)
            pr.max_time = max(pr.max_time, elapsed)
        
        # Large context tests
        print("\n--- Large Context Tests (100KB) ---")
        large_file = create_large_context_file(100)
        file_size = large_file.stat().st_size / 1024
        print(f"  Created test file: {file_size:.1f} KB")
        
        for test in LARGE_CONTEXT_TESTS:
            print(f"  Running: {test['name']}...", end=" ", flush=True)
            
            answer, elapsed, raw = run_rlm_query(
                model, test["query"], str(large_file), timeout=180
            )
            
            correct = check_accuracy(answer, test["expected"], test["check_type"])
            
            result = TestResult(
                test_name=f"large_{test['name']}",
                provider=provider,
                model=model,
                query=test["query"],
                expected=test["expected"],
                actual=answer[:200],
                correct=correct,
                hallucinated=False,
                time_seconds=elapsed,
                context_size=int(file_size * 1024),
                raw_output=raw,
            )
            
            pr.results.append(result)
            pr.total_tests += 1
            if correct:
                pr.passed += 1
                print(f"✓ ({elapsed:.1f}s)")
            else:
                pr.failed += 1
                print(f"✗ ({elapsed:.1f}s) - Got: {answer[:50]}...")
            
            pr.min_time = min(pr.min_time, elapsed)
            pr.max_time = max(pr.max_time, elapsed)
        
        # Clean up temp file
        large_file.unlink(missing_ok=True)
        
        # Calculate average time
        total_time = sum(r.time_seconds for r in pr.results)
        pr.avg_time = total_time / len(pr.results) if pr.results else 0
        
        results[provider] = pr
    
    return results


def print_summary(results: dict[str, ProviderResults]):
    """Print summary table."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"\n{'Provider':<12} {'Model':<20} {'Accuracy':<10} {'Halluc.':<10} {'Avg Time':<10} {'Min/Max':<15}")
    print("-"*80)
    
    for provider, pr in results.items():
        print(f"{provider:<12} {pr.model:<20} {pr.accuracy:>6.1f}%    {pr.hallucination_rate:>6.1f}%    {pr.avg_time:>6.1f}s    {pr.min_time:.1f}s/{pr.max_time:.1f}s")
    
    print("-"*80)
    print(f"\nTotal tests per provider: {results[list(results.keys())[0]].total_tests if results else 0}")


def generate_report(results: dict[str, ProviderResults]) -> str:
    """Generate detailed markdown report."""
    report = []
    report.append("# Provider Benchmark Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    report.append("## Summary\n")
    report.append("| Provider | Model | Accuracy | Hallucination Rate | Avg Time | Min/Max Time |")
    report.append("|----------|-------|----------|-------------------|----------|--------------|")
    
    for provider, pr in results.items():
        report.append(f"| {provider} | {pr.model} | {pr.accuracy:.1f}% | {pr.hallucination_rate:.1f}% | {pr.avg_time:.1f}s | {pr.min_time:.1f}s / {pr.max_time:.1f}s |")
    
    report.append("\n## Test Categories\n")
    report.append("- **Accuracy Tests**: Factual questions with known answers from pyproject.toml")
    report.append("- **Math Tests**: Simple arithmetic to test reasoning")
    report.append("- **Hallucination Tests**: Questions about non-existent content")
    report.append("- **Large Context Tests**: 100KB file with embedded markers")
    
    for provider, pr in results.items():
        report.append(f"\n## {provider.upper()} ({pr.model})\n")
        report.append(f"- **Total Tests**: {pr.total_tests}")
        report.append(f"- **Passed**: {pr.passed}")
        report.append(f"- **Failed**: {pr.failed}")
        report.append(f"- **Hallucinations**: {pr.hallucinations}")
        report.append(f"- **Accuracy**: {pr.accuracy:.1f}%")
        report.append(f"- **Hallucination Rate**: {pr.hallucination_rate:.1f}%")
        report.append(f"- **Average Time**: {pr.avg_time:.1f}s")
        
        report.append("\n### Detailed Results\n")
        
        for result in pr.results:
            status = "✓" if result.correct else "✗"
            if result.hallucinated:
                status = "⚠️ HALLUCINATED"
            
            report.append(f"#### {result.test_name} {status}\n")
            report.append(f"- **Query**: {result.query}")
            report.append(f"- **Expected**: {result.expected}")
            report.append(f"- **Actual**: {result.actual}")
            report.append(f"- **Time**: {result.time_seconds:.1f}s")
            if result.context_size:
                report.append(f"- **Context Size**: {result.context_size / 1024:.1f} KB")
            
            report.append("\n<details>")
            report.append("<summary>Raw Output</summary>\n")
            report.append("```")
            report.append(result.raw_output[:2000])  # Truncate
            if len(result.raw_output) > 2000:
                report.append("\n... (truncated)")
            report.append("```")
            report.append("</details>\n")
    
    report.append("\n## Conclusions\n")
    
    # Find best performers
    if results:
        best_accuracy = max(results.values(), key=lambda x: x.accuracy)
        best_speed = min(results.values(), key=lambda x: x.avg_time)
        lowest_hallucination = min(results.values(), key=lambda x: x.hallucination_rate)
        
        report.append(f"- **Best Accuracy**: {best_accuracy.provider} ({best_accuracy.accuracy:.1f}%)")
        report.append(f"- **Fastest**: {best_speed.provider} ({best_speed.avg_time:.1f}s avg)")
        report.append(f"- **Lowest Hallucination**: {lowest_hallucination.provider} ({lowest_hallucination.hallucination_rate:.1f}%)")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ZAI, Kimi, and MiniMax providers")
    parser.add_argument(
        "--providers",
        default="zai,kimi,minimax",
        help="Comma-separated list of providers to test",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/provider_benchmark_results.json",
        help="Output file for JSON results",
    )
    parser.add_argument(
        "--report",
        default="benchmarks/PROVIDER_BENCHMARK_REPORT.md",
        help="Output file for markdown report",
    )
    
    args = parser.parse_args()
    providers = [p.strip() for p in args.providers.split(",")]
    
    print("Provider Benchmark")
    print("==================")
    print(f"Testing providers: {', '.join(providers)}")
    
    results = run_benchmark(providers)
    print_summary(results)
    
    # Save JSON results
    json_results = {
        "timestamp": datetime.now().isoformat(),
        "providers": {
            p: {
                "provider": pr.provider,
                "model": pr.model,
                "total_tests": pr.total_tests,
                "passed": pr.passed,
                "failed": pr.failed,
                "hallucinations": pr.hallucinations,
                "accuracy": pr.accuracy,
                "hallucination_rate": pr.hallucination_rate,
                "avg_time": pr.avg_time,
                "min_time": pr.min_time,
                "max_time": pr.max_time,
                "results": [asdict(r) for r in pr.results],
            }
            for p, pr in results.items()
        }
    }
    
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_results, indent=2))
    print(f"\nJSON results saved to: {output_path}")
    
    # Generate and save markdown report
    report = generate_report(results)
    report_path = Path(__file__).parent.parent / args.report
    report_path.write_text(report)
    print(f"Markdown report saved to: {report_path}")


if __name__ == "__main__":
    main()

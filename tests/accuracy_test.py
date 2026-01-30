#!/usr/bin/env python3
"""Test rlm-dspy accuracy and hallucination rate.

This script asks rlm-dspy specific, verifiable questions about the codebase
and checks if the answers are correct.
"""

import subprocess
import json
import re
from pathlib import Path

# Ground truth questions with verifiable answers
TEST_CASES = [
    # ===== Existence checks =====
    {
        "question": "Does the file src/rlm_dspy/core/rlm.py exist? Answer YES or NO only.",
        "expected": "YES",
        "category": "existence",
    },
    {
        "question": "Does the file src/rlm_dspy/core/nonexistent.py exist? Answer YES or NO only.",
        "expected": "NO",
        "category": "existence",
    },
    {
        "question": "Is there a class named RLMConfig in the codebase? Answer YES or NO only.",
        "expected": "YES",
        "category": "existence",
    },
    {
        "question": "Is there a class named FakeClassName123 in the codebase? Answer YES or NO only.",
        "expected": "NO",
        "category": "existence",
    },
    
    # ===== Line number verification =====
    {
        "question": "Use read_file to check line 1645 in src/rlm_dspy/cli.py. Does it contain 'registry.get'? Answer YES or NO only.",
        "expected": "YES",
        "category": "line_verification",
    },
    {
        "question": "Use read_file to check line 1646 in src/rlm_dspy/cli.py. Does it contain 'if not project'? Answer YES or NO only.",
        "expected": "YES",
        "category": "line_verification",
    },
    {
        "question": "Use read_file to check lines 1645-1650 in src/rlm_dspy/cli.py. Is there a null check after registry.get()? Answer YES or NO only.",
        "expected": "YES",
        "category": "line_verification",
    },
    
    # ===== Function/class location =====
    {
        "question": "Use index_code or find_functions to find the function 'validate_path_safety'. What file is it in? Just give the filename.",
        "expected": "fileutils.py",
        "category": "location",
    },
    {
        "question": "Use index_code to find the class 'RLM'. What file is it defined in? Just give the filename.",
        "expected": "rlm.py",
        "category": "location",
    },
    {
        "question": "Use find_classes to check: Is there a class named 'PathTraversalError'? Answer YES or NO only.",
        "expected": "YES",
        "category": "location",
    },
    
    # ===== Code pattern verification =====
    {
        "question": "Use ripgrep to search for 'def _sanitize_trajectory' in src/rlm_dspy/. Does this function exist? Answer YES or NO only.",
        "expected": "YES",
        "category": "pattern",
    },
    {
        "question": "Use ripgrep to search for 'import requests' in src/rlm_dspy/core/. Is requests imported anywhere? Answer YES or NO only.",
        "expected": "NO",
        "category": "pattern",
    },
    {
        "question": "Use ripgrep to search for '@lru_cache' in src/rlm_dspy/. Is lru_cache decorator used? Answer YES or NO only.",
        "expected": "YES",
        "category": "pattern",
    },
    
    # ===== Counting =====
    {
        "question": "Use find_classes to count classes in src/rlm_dspy/core/rlm.py. Are there more than 2 classes? Answer YES or NO only.",
        "expected": "YES",
        "category": "counting",
    },
    {
        "question": "Use index_code to find all functions in src/rlm_dspy/core/fileutils.py. Are there more than 10 functions? Answer YES or NO only.",
        "expected": "YES",
        "category": "counting",
    },
]


def run_rlm_query(question: str, max_iterations: int = 3) -> str:
    """Run an rlm-dspy query and return the answer."""
    cmd = [
        "rlm-dspy", "ask", question,
        "src/rlm_dspy/",
        "--max-iterations", str(max_iterations),
        "--budget", "0.10",
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        env={**subprocess.os.environ, "PATH": f"{Path.home()}/.deno/bin:{subprocess.os.environ.get('PATH', '')}"},
    )
    
    # Extract answer from output (between the box characters)
    output = result.stdout
    
    # Find content between the rich panel borders
    lines = output.split('\n')
    answer_lines = []
    in_answer = False
    for line in lines:
        if '─ Answer ─' in line:
            in_answer = True
            continue
        if in_answer:
            if '╰─' in line or '└─' in line:
                break
            # Strip the box characters
            clean = line.strip('│ \n')
            if clean:
                answer_lines.append(clean)
    
    return ' '.join(answer_lines)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.upper().strip()
    # Extract YES/NO from longer responses
    if 'YES' in answer and 'NO' not in answer:
        return 'YES'
    if 'NO' in answer and 'YES' not in answer:
        return 'NO'
    # For filename answers
    for word in answer.split():
        if '.py' in word.lower():
            return word.lower().strip('.,;:()[]')
    return answer


def check_answer(got: str, expected: str) -> bool:
    """Check if answer matches expected."""
    got_norm = normalize_answer(got)
    expected_norm = expected.upper() if expected in ('YES', 'NO') else expected.lower()
    
    if expected in ('YES', 'NO'):
        return got_norm == expected_norm
    else:
        return expected_norm in got_norm.lower()


def main():
    print("=" * 70)
    print("RLM-DSPy Accuracy Test")
    print("=" * 70)
    print()
    
    results = {
        "correct": 0,
        "incorrect": 0,
        "by_category": {},
    }
    
    for i, test in enumerate(TEST_CASES, 1):
        category = test["category"]
        if category not in results["by_category"]:
            results["by_category"][category] = {"correct": 0, "total": 0}
        
        print(f"Test {i}/{len(TEST_CASES)}: [{category}]")
        print(f"  Q: {test['question'][:70]}...")
        
        try:
            answer = run_rlm_query(test["question"])
            is_correct = check_answer(answer, test["expected"])
            
            print(f"  Expected: {test['expected']}")
            print(f"  Got: {normalize_answer(answer)}")
            print(f"  Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            
            if is_correct:
                results["correct"] += 1
                results["by_category"][category]["correct"] += 1
            else:
                results["incorrect"] += 1
                print(f"  Full answer: {answer[:200]}")
            
            results["by_category"][category]["total"] += 1
            
        except subprocess.TimeoutExpired:
            print(f"  Result: ✗ TIMEOUT")
            results["incorrect"] += 1
            results["by_category"][category]["total"] += 1
        except Exception as e:
            print(f"  Result: ✗ ERROR: {e}")
            results["incorrect"] += 1
            results["by_category"][category]["total"] += 1
        
        print()
    
    # Summary
    total = results["correct"] + results["incorrect"]
    accuracy = results["correct"] / total * 100 if total > 0 else 0
    hallucination_rate = results["incorrect"] / total * 100 if total > 0 else 0
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tests: {total}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Hallucination rate: {hallucination_rate:.1f}%")
    print()
    print("By category:")
    for cat, stats in results["by_category"].items():
        cat_acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({cat_acc:.0f}%)")
    
    return results


if __name__ == "__main__":
    main()

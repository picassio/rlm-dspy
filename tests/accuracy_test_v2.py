#!/usr/bin/env python3
"""Test rlm-dspy accuracy with natural questions (not forcing tool usage)."""

import subprocess
from pathlib import Path

TEST_CASES = [
    # ===== Factual verification =====
    {
        "question": "In cli.py around line 1645, after calling registry.get(name), is there a null check? Answer YES or NO.",
        "expected": "YES",
        "category": "verification",
    },
    {
        "question": "Does the codebase use the 'requests' library directly in core modules? Answer YES or NO.",
        "expected": "NO",
        "category": "verification",
    },
    {
        "question": "Does the daemon module use file locking to prevent race conditions? Answer YES or NO.",
        "expected": "YES",
        "category": "verification",
    },
    {
        "question": "Is there a function called validate_path_safety that protects against path traversal? Answer YES or NO.",
        "expected": "YES",
        "category": "verification",
    },
    {
        "question": "Does _sanitize_trajectory handle nested dictionaries recursively? Answer YES or NO.",
        "expected": "YES",
        "category": "verification",
    },
    
    # ===== Negative cases (should answer NO) =====
    {
        "question": "Is there a class called DatabaseConnection in the codebase? Answer YES or NO.",
        "expected": "NO",
        "category": "negative",
    },
    {
        "question": "Does the codebase use MongoDB? Answer YES or NO.",
        "expected": "NO",
        "category": "negative",
    },
    {
        "question": "Is there a function called send_email in the core modules? Answer YES or NO.",
        "expected": "NO",
        "category": "negative",
    },
    
    # ===== Location questions =====
    {
        "question": "Which file contains the RLM class definition? Just the filename.",
        "expected": "rlm.py",
        "category": "location",
    },
    {
        "question": "Which file contains PathTraversalError? Just the filename.",
        "expected": "fileutils.py",
        "category": "location",
    },
    
    # ===== Code understanding =====
    {
        "question": "In rlm.py, does _env_get log actual values when casting fails, or just the key name? Answer VALUES or KEY_ONLY.",
        "expected": "KEY_ONLY",
        "category": "understanding",
    },
    {
        "question": "Are secrets masked with '[REDACTED]' in the output? Answer YES or NO.",
        "expected": "YES",
        "category": "understanding",
    },
]


def run_rlm_query(question: str) -> str:
    """Run an rlm-dspy query and return the answer."""
    cmd = [
        "rlm-dspy", "ask", question,
        "src/rlm_dspy/",
        "--max-iterations", "4",
        "--budget", "0.15",
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        env={**subprocess.os.environ, "PATH": f"{Path.home()}/.deno/bin:{subprocess.os.environ.get('PATH', '')}"},
    )
    
    output = result.stdout
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
            clean = line.strip('│ \n')
            if clean:
                answer_lines.append(clean)
    
    return ' '.join(answer_lines)


def check_answer(answer: str, expected: str) -> bool:
    """Check if answer matches expected."""
    answer_upper = answer.upper()
    expected_upper = expected.upper()
    
    # Direct containment
    if expected_upper in answer_upper:
        return True
    
    # YES/NO logic
    if expected == "YES":
        has_yes = "YES" in answer_upper
        has_no = "NO" in answer_upper and "NOT" not in answer_upper.replace("NO", "XX", 1)
        return has_yes and not has_no
    
    if expected == "NO":
        # Check for negative indicators
        if any(x in answer_upper for x in ["NO,", "NO.", "NO ", "DOES NOT", "NOT FOUND", "NOT EXIST", "ISN'T", "AREN'T", "DOESN'T"]):
            return True
        if "NO" in answer_upper and "YES" not in answer_upper:
            return True
        return False
    
    # Filename matching
    if ".py" in expected:
        return expected.lower() in answer.lower()
    
    return False


def main():
    print("=" * 70)
    print("RLM-DSPy Accuracy Test v2 (Natural Questions)")
    print("=" * 70)
    print()
    
    results = {"correct": 0, "incorrect": 0, "by_category": {}}
    
    for i, test in enumerate(TEST_CASES, 1):
        cat = test["category"]
        if cat not in results["by_category"]:
            results["by_category"][cat] = {"correct": 0, "total": 0}
        
        print(f"Test {i}/{len(TEST_CASES)}: [{cat}]")
        print(f"  Q: {test['question']}")
        
        try:
            answer = run_rlm_query(test["question"])
            is_correct = check_answer(answer, test["expected"])
            
            print(f"  Expected: {test['expected']}")
            print(f"  Got: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            
            if is_correct:
                results["correct"] += 1
                results["by_category"][cat]["correct"] += 1
            else:
                results["incorrect"] += 1
            
            results["by_category"][cat]["total"] += 1
            
        except Exception as e:
            print(f"  Result: ✗ ERROR: {e}")
            results["incorrect"] += 1
            results["by_category"][cat]["total"] += 1
        
        print()
    
    total = results["correct"] + results["incorrect"]
    accuracy = results["correct"] / total * 100 if total > 0 else 0
    
    print("=" * 70)
    print(f"RESULTS: {results['correct']}/{total} correct ({accuracy:.1f}%)")
    print(f"Hallucination rate: {100-accuracy:.1f}%")
    print("=" * 70)
    print("By category:")
    for cat, stats in results["by_category"].items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({acc:.0f}%)")


if __name__ == "__main__":
    main()

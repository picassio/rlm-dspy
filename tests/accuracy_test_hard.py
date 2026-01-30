#!/usr/bin/env python3
"""Hard accuracy tests - questions that previously caused hallucinations."""

import subprocess
from pathlib import Path

# These are the questions that previously produced false positives
HARD_TEST_CASES = [
    # Previously hallucinated: "registry.get() missing null check at line 1645"
    {
        "question": "Use read_file to read lines 1643-1650 in src/rlm_dspy/cli.py. After the registry.get(name) call, is there an 'if not project' check? Answer YES or NO.",
        "expected": "YES",
        "description": "Null check after registry.get()",
    },
    # Previously hallucinated: "requests.get at line 1032 without timeout"
    {
        "question": "Use read_file to read lines 1028-1040 in src/rlm_dspy/core/rlm.py. Is this actual code or a docstring example? Answer DOCSTRING or CODE.",
        "expected": "DOCSTRING",
        "description": "requests.get in docstring vs code",
    },
    # Previously hallucinated: "daemon has race condition, no file locking"
    {
        "question": "Use ripgrep to search for 'fcntl.LOCK_EX' in src/rlm_dspy/core/daemon.py. Does the daemon use file locking? Answer YES or NO.",
        "expected": "YES",
        "description": "Daemon file locking",
    },
    # Previously hallucinated: "path traversal not handled"
    {
        "question": "Use find_functions to search for 'validate_path_safety' in src/rlm_dspy/core/fileutils.py. Does this function exist? Answer YES or NO.",
        "expected": "YES",
        "description": "Path traversal protection exists",
    },
    # Previously hallucinated: "dspy.settings.history accumulates"
    {
        "question": "Use ripgrep to search for 'dspy.settings.context' in src/rlm_dspy/. Is settings used via context manager? Answer YES or NO.",
        "expected": "YES",
        "description": "Settings context manager usage",
    },
    # Test if it correctly identifies MISSING things
    {
        "question": "Use ripgrep to search for 'class DatabaseConnection' in src/rlm_dspy/. Does this class exist? Answer YES or NO.",
        "expected": "NO",
        "description": "Non-existent class correctly identified",
    },
    {
        "question": "Use ripgrep to search for 'def connect_to_redis' in src/rlm_dspy/. Does this function exist? Answer YES or NO.",
        "expected": "NO",
        "description": "Non-existent function correctly identified",
    },
    # Test nested structure verification
    {
        "question": "Use read_file to read the _sanitize_value function in src/rlm_dspy/core/rlm.py (around lines 123-135). Does it handle nested dicts recursively? Answer YES or NO.",
        "expected": "YES",
        "description": "Recursive sanitization verification",
    },
    # Test security pattern verification
    {
        "question": "Use ripgrep to search for 'REDACTED' in src/rlm_dspy/core/rlm.py. Are secrets being masked? Answer YES or NO.",
        "expected": "YES",
        "description": "Secret masking verification",
    },
    # Verify actual bug we fixed
    {
        "question": "Use read_file to check src/rlm_dspy/core/rlm.py around line 173. Does _env_get log the actual value on error, or just the key name? Answer VALUE or KEY_ONLY.",
        "expected": "KEY_ONLY",
        "description": "No sensitive values in logs",
    },
]


def run_rlm_query(question: str, max_iterations: int = 20) -> str:
    """Run an rlm-dspy query and return the answer."""
    cmd = [
        "rlm-dspy", "ask", question,
        "src/rlm_dspy/",
        "--max-iterations", str(max_iterations),
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


def normalize_answer(answer: str, expected: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.upper().strip()
    expected_upper = expected.upper()
    
    # Direct match
    if expected_upper in answer:
        return expected_upper
    
    # YES/NO extraction
    if expected in ('YES', 'NO'):
        if 'YES' in answer and 'NO' not in answer:
            return 'YES'
        if 'NO' in answer and 'YES' not in answer:
            return 'NO'
        # Handle "does not" as NO
        if 'DOES NOT' in answer or 'NOT EXIST' in answer or 'NOT FOUND' in answer:
            return 'NO'
        if 'DOES EXIST' in answer or 'EXISTS' in answer or 'FOUND' in answer:
            return 'YES'
    
    # DOCSTRING/CODE
    if expected in ('DOCSTRING', 'CODE'):
        if 'DOCSTRING' in answer or 'DOCUMENTATION' in answer or 'EXAMPLE' in answer:
            return 'DOCSTRING'
        if 'ACTUAL CODE' in answer or 'REAL CODE' in answer:
            return 'CODE'
    
    # KEY_ONLY/VALUE
    if expected in ('KEY_ONLY', 'VALUE'):
        if 'KEY' in answer and 'VALUE' not in answer:
            return 'KEY_ONLY'
        if 'ONLY THE KEY' in answer or 'JUST THE KEY' in answer or 'KEY NAME' in answer:
            return 'KEY_ONLY'
        if 'DOES NOT LOG THE VALUE' in answer or 'NOT LOG THE ACTUAL VALUE' in answer:
            return 'KEY_ONLY'
    
    return answer[:50]


def main():
    print("=" * 70)
    print("RLM-DSPy HARD Accuracy Test (Previously Hallucinated Questions)")
    print("=" * 70)
    print()
    
    correct = 0
    incorrect = 0
    details = []
    
    for i, test in enumerate(HARD_TEST_CASES, 1):
        print(f"Test {i}/{len(HARD_TEST_CASES)}: {test['description']}")
        print(f"  Q: {test['question'][:65]}...")
        
        try:
            answer = run_rlm_query(test["question"])
            got = normalize_answer(answer, test["expected"])
            is_correct = got == test["expected"].upper()
            
            print(f"  Expected: {test['expected']}")
            print(f"  Got: {got}")
            print(f"  Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            
            if is_correct:
                correct += 1
            else:
                incorrect += 1
                print(f"  Full answer: {answer[:300]}")
            
            details.append({
                "test": test["description"],
                "correct": is_correct,
                "expected": test["expected"],
                "got": got,
            })
            
        except subprocess.TimeoutExpired:
            print(f"  Result: ✗ TIMEOUT")
            incorrect += 1
        except Exception as e:
            print(f"  Result: ✗ ERROR: {e}")
            incorrect += 1
        
        print()
    
    # Summary
    total = correct + incorrect
    accuracy = correct / total * 100 if total > 0 else 0
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tests: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Hallucination rate: {100 - accuracy:.1f}%")
    
    if incorrect > 0:
        print("\nFailed tests:")
        for d in details:
            if not d["correct"]:
                print(f"  - {d['test']}: expected {d['expected']}, got {d['got']}")


if __name__ == "__main__":
    main()

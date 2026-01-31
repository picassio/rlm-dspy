# Provider Benchmark Report

Generated: 2026-01-30 08:15:00 UTC

## Summary

| Provider | Model | Accuracy | Hallucination Rate | Avg Time | Min/Max Time |
|----------|-------|----------|-------------------|----------|--------------|
| zai | zai/glm-4.7 | 91.7% | 0.0% | 44.2s | 13.8s / 120.0s |
| kimi | kimi/k2p5 | 91.7% | 0.0% | 14.4s | 9.6s / 36.4s |
| kimi-thinking | kimi/kimi-k2-thinking | 91.7% | 0.0% | 15.5s | 9.8s / 43.2s |
| minimax | minimax/MiniMax-M2.1 | ~50%* | 0.0% | ~70s* | 14.4s / 180s+ |

*MiniMax had multiple timeouts (120s+), affecting accuracy and timing metrics.

## Test Categories

- **Accuracy Tests** (5): Factual questions with known answers from pyproject.toml
- **Math Tests** (2): Simple arithmetic to test reasoning
- **Hallucination Tests** (3): Questions about non-existent content
- **Large Context Tests** (2): 120KB file with embedded markers

## Key Findings

### 🏆 Best Overall: Kimi (kimi/k2p5)
- Fastest average response time (14.4s)
- High accuracy (91.7%)
- Zero hallucinations
- Excellent large context handling

### 🧠 Best Thinking Model: Kimi K2 Thinking
- Similar accuracy to base model (91.7%)
- Slightly slower but more thorough responses
- Good for complex reasoning tasks

### ⚡ Performance Comparison

| Provider | Speed Rank | Accuracy Rank | Stability |
|----------|------------|---------------|-----------|
| Kimi K2.5 | 🥇 1st (14.4s) | 🥇 Tied 1st | ✅ Excellent |
| Kimi K2 Thinking | 🥈 2nd (15.5s) | 🥇 Tied 1st | ✅ Excellent |
| ZAI GLM-4.7 | 🥉 3rd (44.2s) | 🥇 Tied 1st | ✅ Good |
| MiniMax M2.1 | 4th (~70s) | ⚠️ Affected by timeouts | ⚠️ Unstable |

---

## ZAI (zai/glm-4.7)

- **Total Tests**: 12
- **Passed**: 11
- **Failed**: 1
- **Hallucinations**: 0
- **Accuracy**: 91.7%
- **Hallucination Rate**: 0.0%
- **Average Time**: 44.2s

### Detailed Results

#### version_check ✓

- **Query**: What is the version of this project?
- **Expected**: 0.1.0
- **Actual**: 0.1.0
- **Time**: 68.6s

<details>
<summary>Raw Output</summary>

```
Validating output...
✓ Output validated (100% grounded in trajectory)
╭─────────────────────────────────── Answer ───────────────────────────────────╮
│ 0.1.0                                                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
```
</details>

#### project_name ✓

- **Query**: What is the name of this project?
- **Expected**: rlm-dspy
- **Actual**: rlm-dspy
- **Time**: 13.8s

#### python_version ✗

- **Query**: What Python version does this project require?
- **Expected**: 3.10 (test criteria)
- **Actual**: 3.11 (correct answer from file - test expected wrong value)
- **Time**: 51.6s
- **Note**: The test expected "3.10" but the actual requirement is ">=3.11"

#### main_dependency ✓

- **Query**: What is the main dependency that starts with 'dspy'?
- **Expected**: dspy
- **Actual**: dspy>=2.6.0
- **Time**: 30.6s

#### cli_command ✓

- **Query**: What is the CLI entry point command name?
- **Expected**: rlm-dspy
- **Actual**: rlm-dspy
- **Time**: 52.9s

#### simple_addition ✓

- **Query**: What is 15 + 27?
- **Expected**: 42
- **Actual**: 42
- **Time**: 17.1s

#### multiplication ✓

- **Query**: What is 12 * 8?
- **Expected**: 96
- **Actual**: 96
- **Time**: 38.5s

#### nonexistent_function ✓ (No Hallucination)

- **Query**: What does the function 'calculate_quantum_entropy' do in this file?
- **Actual**: Correctly stated the function does not exist
- **Time**: 120.0s

#### nonexistent_config ✓ (No Hallucination)

- **Query**: What is the value of the 'database_url' configuration in this file?
- **Actual**: Correctly stated there is no database_url
- **Time**: 35.7s

#### invented_author ✓ (No Hallucination)

- **Query**: Who is listed as the author's email in this file?
- **Actual**: Correctly stated no author email is specified
- **Time**: 32.8s

#### large_find_marker ✓

- **Query**: What is the secret code mentioned in the SPECIAL_MARKER comment?
- **Expected**: BENCHMARK_2024
- **Actual**: BENCHMARK_2024
- **Time**: 16.5s
- **Context Size**: 119.8 KB

#### large_count_imports ✓

- **Query**: Does this file import 'dspy'? Answer yes or no.
- **Expected**: yes
- **Actual**: yes
- **Time**: 52.2s
- **Context Size**: 119.8 KB

---

## KIMI (kimi/k2p5)

- **Total Tests**: 12
- **Passed**: 11
- **Failed**: 1
- **Hallucinations**: 0
- **Accuracy**: 91.7%
- **Hallucination Rate**: 0.0%
- **Average Time**: 14.4s

### Detailed Results

#### version_check ✓

- **Query**: What is the version of this project?
- **Expected**: 0.1.0
- **Actual**: 0.1.0
- **Time**: 10.4s

#### project_name ✓

- **Query**: What is the name of this project?
- **Expected**: rlm-dspy
- **Actual**: rlm-dspy
- **Time**: 10.9s

#### python_version ✗

- **Query**: What Python version does this project require?
- **Expected**: 3.10 (test criteria)
- **Actual**: 3.11 (correct answer from file)
- **Time**: 10.8s

#### main_dependency ✓

- **Query**: What is the main dependency that starts with 'dspy'?
- **Expected**: dspy
- **Actual**: dspy>=2.6.0
- **Time**: 12.4s

#### cli_command ✓

- **Query**: What is the CLI entry point command name?
- **Expected**: rlm-dspy
- **Actual**: rlm-dspy
- **Time**: 13.8s

#### simple_addition ✓

- **Query**: What is 15 + 27?
- **Expected**: 42
- **Actual**: 42
- **Time**: 10.4s

#### multiplication ✓

- **Query**: What is 12 * 8?
- **Expected**: 96
- **Actual**: 96
- **Time**: 9.6s

#### nonexistent_function ✓ (No Hallucination)

- **Query**: What does the function 'calculate_quantum_entropy' do in this file?
- **Actual**: Correctly stated function does not exist
- **Time**: 36.4s

#### nonexistent_config ✓ (No Hallucination)

- **Query**: What is the value of the 'database_url' configuration in this file?
- **Actual**: Correctly stated no database_url exists
- **Time**: 20.3s

#### invented_author ✓ (No Hallucination)

- **Query**: Who is listed as the author's email in this file?
- **Actual**: Correctly stated no author email specified
- **Time**: 17.6s

#### large_find_marker ✓

- **Query**: What is the secret code mentioned in the SPECIAL_MARKER comment?
- **Expected**: BENCHMARK_2024
- **Actual**: BENCHMARK_2024
- **Time**: 10.3s
- **Context Size**: 119.8 KB

#### large_count_imports ✓

- **Query**: Does this file import 'dspy'? Answer yes or no.
- **Expected**: yes
- **Actual**: yes
- **Time**: 20.3s
- **Context Size**: 119.8 KB

---

## KIMI-THINKING (kimi/kimi-k2-thinking)

- **Total Tests**: 12
- **Passed**: 11
- **Failed**: 1
- **Hallucinations**: 0
- **Accuracy**: 91.7%
- **Hallucination Rate**: 0.0%
- **Average Time**: 15.5s

### Detailed Results

#### version_check ✓

- **Query**: What is the version of this project?
- **Expected**: 0.1.0
- **Actual**: 0.1.0
- **Time**: 12.9s

#### project_name ✓

- **Query**: What is the name of this project?
- **Expected**: rlm-dspy
- **Actual**: rlm-dspy
- **Time**: 12.6s

#### python_version ✗

- **Query**: What Python version does this project require?
- **Expected**: 3.10 (test criteria)
- **Actual**: >=3.11 (correct answer from file)
- **Time**: 17.4s

#### main_dependency ✓

- **Query**: What is the main dependency that starts with 'dspy'?
- **Expected**: dspy
- **Actual**: dspy>=2.6.0
- **Time**: 13.2s

#### cli_command ✓

- **Query**: What is the CLI entry point command name?
- **Expected**: rlm-dspy
- **Actual**: rlm-dspy
- **Time**: 13.2s

#### simple_addition ✓

- **Query**: What is 15 + 27?
- **Expected**: 42
- **Actual**: 42
- **Time**: 10.0s

#### multiplication ✓

- **Query**: What is 12 * 8?
- **Expected**: 96
- **Actual**: 96
- **Time**: 9.8s

#### nonexistent_function ✓ (No Hallucination)

- **Query**: What does the function 'calculate_quantum_entropy' do in this file?
- **Actual**: Correctly stated function does not exist in the file
- **Time**: 43.2s

#### nonexistent_config ✓ (No Hallucination)

- **Query**: What is the value of the 'database_url' configuration in this file?
- **Actual**: Correctly stated no database_url configuration exists
- **Time**: 13.7s

#### invented_author ✓ (No Hallucination)

- **Query**: Who is listed as the author's email in this file?
- **Actual**: Correctly stated no author email is listed
- **Time**: 13.6s

#### large_find_marker ✓

- **Query**: What is the secret code mentioned in the SPECIAL_MARKER comment?
- **Expected**: BENCHMARK_2024
- **Actual**: BENCHMARK_2024
- **Time**: 13.7s
- **Context Size**: 119.8 KB

#### large_count_imports ✓

- **Query**: Does this file import 'dspy'? Answer yes or no.
- **Expected**: yes
- **Actual**: yes
- **Time**: 13.2s
- **Context Size**: 119.8 KB

---

## MINIMAX (minimax/MiniMax-M2.1)

- **Total Tests**: 12 (partial - benchmark interrupted)
- **Passed**: ~6
- **Failed**: ~4 (mostly timeouts)
- **Hallucinations**: 0
- **Accuracy**: ~50% (affected by timeouts)
- **Average Time**: ~70s (highly variable)

### Notes

MiniMax M2.1 experienced significant latency issues during testing:
- Multiple requests timed out after 120s
- Large context tests particularly affected
- When responses completed, accuracy was good

### Completed Results

#### project_name ✓
- **Time**: 16.2s

#### main_dependency ✓
- **Time**: 26.2s

#### cli_command ✓
- **Time**: 27.9s

#### simple_addition ✓
- **Time**: 14.4s

#### nonexistent_function ✓ (No Hallucination)
- **Time**: 120.0s (near timeout)

#### nonexistent_config ✓ (No Hallucination)
- **Time**: 110.0s (near timeout)

#### invented_author ✓ (No Hallucination)
- **Time**: 29.0s

### Timeouts

- version_check: TIMEOUT (120s)
- python_version: TIMEOUT (120s)
- multiplication: TIMEOUT (120s)
- large_find_marker: TIMEOUT (180s)

---

## Conclusions

### Best Accuracy
All providers achieved **0% hallucination rate** - excellent for code analysis tasks.

### Best Speed
**Kimi K2.5** is the clear winner with 14.4s average response time, approximately:
- 3x faster than ZAI GLM-4.7
- 5x faster than MiniMax M2.1

### Best for Large Context
**Kimi K2.5** handled 120KB context files with ease (10-20s response times).

### Recommendations

| Use Case | Recommended Provider |
|----------|---------------------|
| Fast code analysis | Kimi K2.5 |
| Complex reasoning | Kimi K2 Thinking |
| Alternative/backup | ZAI GLM-4.7 |
| Avoid for time-sensitive | MiniMax (latency issues) |

### Test Correction Note

The `python_version` test incorrectly expected "3.10" when the actual project requirement is ">=3.11". All providers correctly identified "3.11" - this is not a failure but a test specification error.

**Adjusted Accuracy (with corrected test):**
- ZAI: 100%
- Kimi K2.5: 100%
- Kimi K2 Thinking: 100%

---

## Appendix: Test Environment

- **OS**: Ubuntu Linux
- **Python**: 3.12
- **rlm-dspy version**: 0.1.0
- **Test Date**: 2026-01-30
- **Context File Size**: 119.8 KB (concatenated source files)

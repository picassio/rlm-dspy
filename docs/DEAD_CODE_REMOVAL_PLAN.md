# Dead Code Removal Plan

## Analysis Date: 2026-01-30

## Summary

Dead code analysis was performed using rlm-dspy on the src/rlm_dspy codebase.
The analysis identified a small amount of confirmed dead code.

## Confirmed Dead Code (Safe to Remove)

### 1. Dead Helper Functions in cli.py (Lines 40-81)

These functions were created as helpers to reduce duplication in CLI option
definitions, but were never actually used. Each CLI command defines its options
inline using Annotated types directly.

**Files Affected:** `src/rlm_dspy/cli.py`

**Functions to Remove:**
- `_opt_model()` (line 40)
- `_opt_sub_model()` (line 46)
- `_opt_budget()` (line 52)
- `_opt_timeout()` (line 58)
- `_opt_verbose()` (line 64)
- `_opt_output_format()` (line 70)
- `_opt_output_file()` (line 76)

**Impact:** None - these functions are never called

### 2. Unused Import in index_compression.py

**File:** `src/rlm_dspy/core/index_compression.py`

**Import to Remove:**
```python
import pickle  # Line 11
```

**Impact:** None - pickle is imported but never used

### 3. Dead Class in rlm.py

**File:** `src/rlm_dspy/core/rlm.py`

**Class to Remove:** `DspyProgressCallback` (line 511)

This class was defined to wrap progress callbacks for DSPy but is never
instantiated anywhere in the codebase.

**Impact:** None - class is never used

## Code to Keep (Verified as Used)

The following functions initially appeared unused but ARE used:

### Public API Functions (exported in __init__.py)
- `parse_json_strict()` - Exported API, used in tests
- `accuracy_metric()` - Exported API, used in SIMBA optimizer tests and as metric param
- `is_configured()` - Called internally via get_config_status(), checked in CLI

### Cache Clearing Utilities (used in tests)
- `clear_lsp_manager()` - Used in test fixtures
- Other clear_* functions - Testing/debugging utilities

### Vendored Code
The `vendor/solidlsp/` directory contains many language server implementations.
These should be kept for completeness as they're part of the solidlsp package.

## Removal Steps

1. Remove dead helper functions from cli.py (lines 40-81)
2. Remove unused pickle import from index_compression.py
3. Remove DspyProgressCallback class from rlm.py
4. Run tests to verify no regressions
5. Commit changes

## Estimated Impact

- **Lines Removed:** ~50 lines
- **Risk Level:** Low (functions/classes never used)
- **Test Coverage:** All tests should pass after removal

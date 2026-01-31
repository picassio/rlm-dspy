#!/usr/bin/env python3
"""Check that Python files don't exceed maximum line count.

Usage:
    python scripts/check_file_length.py [--max-lines=500] [paths...]
    
If no paths provided, checks all Python files in src/ and tests/.
"""

import argparse
import sys
from pathlib import Path


def check_file_length(path: Path, max_lines: int) -> tuple[bool, int]:
    """Check if file exceeds max lines.
    
    Returns:
        (passed, line_count)
    """
    try:
        lines = path.read_text().splitlines()
        return len(lines) <= max_lines, len(lines)
    except Exception:
        return True, 0  # Skip files that can't be read


def main():
    parser = argparse.ArgumentParser(description="Check file line counts")
    parser.add_argument("--max-lines", type=int, default=500, 
                        help="Maximum lines per file (default: 500)")
    parser.add_argument("paths", nargs="*", help="Paths to check")
    args = parser.parse_args()
    
    # Default to src/ and tests/ if no paths provided
    if args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        paths = [Path("src"), Path("tests")]
    
    # Collect all Python files
    files = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*.py"))
    
    # Check each file
    violations = []
    for file in sorted(files):
        passed, line_count = check_file_length(file, args.max_lines)
        if not passed:
            violations.append((file, line_count))
    
    # Report results
    if violations:
        print(f"❌ {len(violations)} file(s) exceed {args.max_lines} lines:\n")
        for file, count in violations:
            excess = count - args.max_lines
            print(f"  {file}: {count} lines (+{excess})")
        print(f"\nConsider splitting large files into smaller modules.")
        sys.exit(1)
    else:
        print(f"✓ All files under {args.max_lines} lines")
        sys.exit(0)


if __name__ == "__main__":
    main()

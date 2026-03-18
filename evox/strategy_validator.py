#!/usr/bin/env python3
"""Validate that current_strategy.md has all required sections.

Returns exit code 0 if valid, 1 if not. Prints diagnostics.

Usage:
    uv run evox/strategy_validator.py [path/to/strategy.md]
"""

import re
import sys
from pathlib import Path

EVOX_DIR = Path(__file__).parent
DEFAULT_STRATEGY = EVOX_DIR / "current_strategy.md"

REQUIRED_SECTIONS = [
    "Parent Selection Rule",
    "Inspiration Set Construction",
    "Variation Operator Preferences",
    "REFINE Guidance",
    "DIVERGE Guidance",
    "FREE Guidance",
]


def validate(path):
    if not path.exists():
        print(f"ERROR: Strategy file not found: {path}")
        return False

    content = path.read_text(encoding="utf-8")
    errors = []

    for section in REQUIRED_SECTIONS:
        # Match as markdown heading at any depth (##, ###, ####, etc.)
        pattern = rf"^#{{2,6}}\s+{re.escape(section)}"
        if not re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
            errors.append(f"Missing section: '{section}'")

    # Check that Variation Operator Preferences contains parseable weights (order-independent)
    refine_match = re.search(r"(\d+)%\s*REFINE", content, re.IGNORECASE)
    diverge_match = re.search(r"(\d+)%\s*DIVERGE", content, re.IGNORECASE)
    free_match = re.search(r"(\d+)%\s*FREE", content, re.IGNORECASE)

    if not (refine_match and diverge_match and free_match):
        errors.append("Variation Operator Preferences must contain '<N>% REFINE', '<N>% DIVERGE', and '<N>% FREE'")
    else:
        weights = {
            "REFINE": int(refine_match.group(1)),
            "DIVERGE": int(diverge_match.group(1)),
            "FREE": int(free_match.group(1)),
        }
        total = sum(weights.values())
        if total != 100:
            errors.append(f"Operator weights sum to {total}, expected 100")
        print(f"Operator weights: REFINE={weights['REFINE']}%, DIVERGE={weights['DIVERGE']}%, FREE={weights['FREE']}%")

    if errors:
        print("VALID: false")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("VALID: true")
        print(f"All {len(REQUIRED_SECTIONS)} required sections present.")
        return True


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STRATEGY
    valid = validate(path)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()

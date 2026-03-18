"""PostToolUse hook: Auto-validate current_strategy.md after any edit.

Runs strategy_validator.py whenever current_strategy.md is written or edited.
Prints validation errors to stderr so Claude sees them immediately and can fix
the strategy before continuing the loop.

Exit codes: 0 = valid (or not a strategy file), 2 = invalid strategy.
Stdin: JSON with tool_name and tool_input from Claude Code.
"""

import json
import os
import subprocess
import sys


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_input = event.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    normalized = file_path.replace("\\", "/")
    if not normalized.endswith("current_strategy.md"):
        sys.exit(0)

    # Find the strategy file — could be absolute or relative
    if os.path.isfile(file_path):
        strategy_path = file_path
    else:
        sys.exit(0)  # File doesn't exist yet, skip validation

    # Find the validator script relative to this hook
    hook_dir = os.path.dirname(os.path.abspath(__file__))
    evox_dir = os.path.dirname(hook_dir)
    validator = os.path.join(evox_dir, "strategy_validator.py")

    if not os.path.isfile(validator):
        sys.exit(0)  # Validator not found, don't block

    result = subprocess.run(
        ["uv", "run", validator, strategy_path],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode != 0:
        print(
            "STRATEGY VALIDATION FAILED after edit to current_strategy.md:\n"
            f"{result.stdout}\n{result.stderr}\n"
            "Fix the strategy document before continuing the EvoX loop.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Validation passed — print confirmation to stdout (informational)
    print("Strategy validation passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()

"""PreToolUse hook: Validate train.py has changes before GPU evaluation.

Prevents wasting GPU time by checking that train.py actually differs from
its parent candidate's version before launching a training run. Compares
against the parent's train.py in the candidates/ directory.

Exit codes: 0 = allow, 2 = block.
Stdin: JSON with tool_name and tool_input from Claude Code.
"""

import json
import os
import re
import sys


def find_project_root() -> str:
    """Walk up from hook location to find the autolab-contributor root."""
    hook_dir = os.path.dirname(os.path.abspath(__file__))
    # hooks/ -> evox/ -> project root
    return os.path.dirname(os.path.dirname(hook_dir))


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_input = event.get("tool_input", {})
    command = tool_input.get("command", "")

    # Only intercept commands that run train.py for evaluation
    # Match patterns like: uv run train.py, python train.py, etc.
    if not re.search(r"\btrain\.py\b", command):
        sys.exit(0)

    # Don't block if this is clearly not an evaluation run
    # (e.g., --help, syntax check, import test)
    if any(flag in command for flag in ["--help", "-h", "import ", "ast.parse"]):
        sys.exit(0)

    root = find_project_root()
    train_py = os.path.join(root, "train.py")

    if not os.path.isfile(train_py):
        # No train.py yet — allow (might be the initial copy)
        sys.exit(0)

    # Check if train.py has any content (non-empty)
    if os.path.getsize(train_py) == 0:
        print(
            "BLOCKED: train.py is empty. Copy the parent's train.py first.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Try to find the most recent candidate dir to compare against
    state_path = os.path.join(root, "evox", "state.json")
    if os.path.isfile(state_path):
        try:
            with open(state_path) as f:
                state = json.load(f)
            parent_id = state.get("current_parent_id", "")
            if parent_id and parent_id != "master":
                parent_train = os.path.join(
                    root, "candidates", parent_id, "train.py"
                )
                if os.path.isfile(parent_train):
                    with open(train_py) as f:
                        current_content = f.read()
                    with open(parent_train) as f:
                        parent_content = f.read()
                    if current_content == parent_content:
                        print(
                            f"BLOCKED: train.py is identical to parent {parent_id}.\n"
                            f"You must make changes before running evaluation.\n"
                            f"Apply the selected operator to modify train.py first.",
                            file=sys.stderr,
                        )
                        sys.exit(2)
        except (json.JSONDecodeError, OSError, KeyError):
            pass  # Can't read state — allow the run

    sys.exit(0)


if __name__ == "__main__":
    main()

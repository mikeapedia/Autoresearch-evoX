"""Stop hook: Auto-checkpoint EvoX state to git on session end.

When Claude's session ends (context limit, crash, timeout), this hook
commits all EvoX state files to git so resume.py can recover cleanly.
Only commits if there are actual changes to state files.

Exit codes: always 0 (never block session end).
Stdin: JSON with stop reason from Claude Code.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def find_project_root() -> str:
    """Walk up from hook location to find the project root."""
    hook_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(hook_dir))


STATE_FILES = [
    "evox/state.json",
    "evox/population.json",
    "evox/strategies.json",
    "evox/current_strategy.md",
]


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        event = {}

    stop_reason = event.get("stop_reason", "unknown")
    root = find_project_root()

    # Check if we're in a git repo
    git_check = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if git_check.returncode != 0:
        sys.exit(0)  # Not a git repo — skip

    # Stage only existing state files that have changes
    files_to_stage = []
    for rel_path in STATE_FILES:
        full_path = os.path.join(root, rel_path)
        if os.path.isfile(full_path):
            files_to_stage.append(rel_path)

    # Also stage any new candidate directories
    candidates_dir = os.path.join(root, "candidates")
    if os.path.isdir(candidates_dir):
        files_to_stage.append("candidates/")

    # Also stage strategy archives
    strategies_dir = os.path.join(root, "evox", "strategies")
    if os.path.isdir(strategies_dir):
        files_to_stage.append("evox/strategies/")

    if not files_to_stage:
        sys.exit(0)

    # Check if there are actual changes
    subprocess.run(
        ["git", "add", "--"] + files_to_stage,
        capture_output=True,
        text=True,
        cwd=root,
    )

    diff_check = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if diff_check.returncode == 0:
        sys.exit(0)  # No changes staged — skip

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit_msg = f"evox: auto-checkpoint on session {stop_reason} ({timestamp})"

    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        capture_output=True,
        text=True,
        cwd=root,
    )

    print(f"Auto-checkpoint committed: {commit_msg}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()

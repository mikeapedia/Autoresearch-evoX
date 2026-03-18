"""PreToolUse hook: Block destructive git/filesystem operations.

Prevents the autonomous agent from accidentally destroying evaluated
candidates, resetting git state, or wiping the working directory.

Exit codes: 0 = allow, 2 = block.
Stdin: JSON with tool_name and tool_input from Claude Code.
"""

import json
import re
import sys

# Patterns that indicate destructive intent.
# Each tuple: (compiled regex, human-readable description)
DESTRUCTIVE_PATTERNS = [
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "git reset --hard"),
    (re.compile(r"\bgit\s+clean\s+-[a-zA-Z]*f"), "git clean -f"),
    (re.compile(r"\bgit\s+checkout\s+\.\s*$"), "git checkout ."),
    (re.compile(r"\bgit\s+push\s+.*--force\b"), "git push --force"),
    (re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+.*candidates/"), "rm -r candidates/"),
    (re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+.*evox/"), "rm -r evox/"),
    (re.compile(r"\brm\s+.*state\.json\b"), "rm state.json"),
    (re.compile(r"\brm\s+.*population\.json\b"), "rm population.json"),
    (re.compile(r"\brm\s+.*strategies\.json\b"), "rm strategies.json"),
    (re.compile(r"\brm\s+.*current_strategy\.md\b"), "rm current_strategy.md"),
]


def extract_actual_command(command: str) -> str:
    """Extract the actual command, stripping heredoc/quoted content.

    git commit -m "$(cat <<'EOF'\n...\nEOF\n)" contains quoted text that
    should NOT be scanned for destructive patterns — only the outer command
    matters (git commit is safe).
    """
    # If the command is a git commit with a heredoc message, the actual
    # command is just "git commit -m ..." which is safe.
    if re.match(r"^\s*git\s+commit\s+", command):
        return "git commit"

    # Strip single-quoted strings (content inside '' is literal)
    stripped = re.sub(r"'[^']*'", '""', command)
    # Strip double-quoted strings (content inside "" may have mentions)
    stripped = re.sub(r'"[^"]*"', '""', stripped)

    return stripped


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_input = event.get("tool_input", {})
    command = tool_input.get("command", "")

    if not command:
        sys.exit(0)

    # Only scan the actual command, not quoted/heredoc content
    scannable = extract_actual_command(command)

    for pattern, description in DESTRUCTIVE_PATTERNS:
        if pattern.search(scannable):
            print(
                f"BLOCKED: Destructive operation detected: {description}\n"
                f"Command: {command}\n"
                f"This could destroy EvoX state or evaluated candidates.\n"
                f"If you really need to do this, ask the user for confirmation.",
                file=sys.stderr,
            )
            sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()

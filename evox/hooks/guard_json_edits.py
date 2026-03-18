"""PreToolUse hook: Block direct Write/Edit to EvoX state files.

Forces all mutations through state_manager.py subcommands to prevent
accidental state corruption during long autonomous sessions.

Exit codes: 0 = allow, 2 = block (stderr shown to Claude).
Stdin: JSON with tool_name and tool_input from Claude Code.
"""

import json
import sys

PROTECTED_FILES = {"state.json", "population.json", "strategies.json"}


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)  # Can't parse → allow (don't break unrelated tools)

    tool_input = event.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    # Normalize path separators and extract filename
    normalized = file_path.replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized

    if filename in PROTECTED_FILES and "/evox/" in normalized:
        print(
            f"BLOCKED: Direct edit to {filename} is not allowed.\n"
            f"Use `uv run evox/state_manager.py <subcommand>` instead.\n"
            f"Protected files: {', '.join(sorted(PROTECTED_FILES))}",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()

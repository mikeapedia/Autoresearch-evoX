#!/usr/bin/env python3
"""Cross-platform file locking for EvoX shared JSON files.

Uses fcntl.flock() on Linux (the production VM). Falls back to a
no-op on Windows/macOS (single-GPU development only).

Usage:
    uv run evox/filelock.py  # (no standalone use — imported by state_manager)

    from filelock import locked_json

    # Atomic read-modify-write:
    with locked_json(POPULATION_FILE, []) as pop:
        pop.append(new_candidate)
    # Lock released, file written automatically on context exit.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path

# fcntl is available on Linux/macOS but not Windows
try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


def _load_json(path: Path, default):
    """Read JSON from path, returning default if file doesn't exist."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Return a fresh copy of default to avoid mutating the caller's default
    if isinstance(default, list):
        return list(default)
    if isinstance(default, dict):
        return dict(default)
    return default


def _save_json(path: Path, data) -> None:
    """Write data to path as indented JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


@contextmanager
def locked_json(json_path: Path, default=None):
    """Atomic read-modify-write of a JSON file under exclusive lock.

    Acquires an exclusive file lock on <json_path>.lock, reads the JSON,
    yields the mutable data for the caller to modify, then writes back
    and releases the lock on context exit.

    On platforms without fcntl (Windows), operates without locking —
    safe for single-GPU development but NOT for concurrent access.

    Args:
        json_path: Path to the JSON file.
        default: Value to use if the file doesn't exist ([] or {}).

    Yields:
        The deserialized JSON data (list or dict). Mutate in place.
    """
    if default is None:
        default = []

    if not _HAS_FCNTL:
        # No locking — single-GPU fallback
        data = _load_json(json_path, default)
        yield data
        _save_json(json_path, data)
        return

    lock_path = json_path.with_suffix(json_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # type: ignore[attr-defined]  # Unix-only
        data = _load_json(json_path, default)
        yield data
        _save_json(json_path, data)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)  # type: ignore[attr-defined]  # Unix-only
        os.close(fd)

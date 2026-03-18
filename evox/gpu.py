#!/usr/bin/env python3
"""GPU index resolution for multi-GPU EvoX operation.

All EvoX scripts use this module to determine which GPU they belong to.
The GPU index is read from the EVOX_GPU environment variable, defaulting
to "0" for single-GPU backward compatibility.

Set once per Claude Code session:  export EVOX_GPU=$GPU
"""

import os


def get_gpu_index() -> str:
    """Return the GPU index as a string (e.g. "0", "1")."""
    return os.environ.get("EVOX_GPU", "0")


def gpu_prefix() -> str:
    """Return the GPU-namespaced prefix (e.g. "g0", "g1")."""
    return f"g{get_gpu_index()}"

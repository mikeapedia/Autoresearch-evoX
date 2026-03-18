# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Source repository for the **EvoX meta-evolutionary autolab contributor** — an autonomous ML research system that evolves both candidate solutions (train.py variants) and search strategies to minimize val_bpb on a small GPT language model. The `evox/` scripts are infrastructure deployed to a remote VM; `program.md` is the instruction document Claude Code follows autonomously. Supports **multi-GPU** operation with independent strategy evolution per GPU and a shared population database.

## Lint and Type Check

```bash
ruff check evox/
ty check evox/
```

No test suite — the scripts are validated by running subcommands with mock data on the VM. No pyproject.toml in this repo; dependencies (torch, numpy, stdlib) are managed in the remote workspace.

## Architecture

**`program.md`** — The main deliverable. A ~650-line instruction document that tells Claude Code (running on a remote VM via CLI) how to run the three-phase EvoX loop: Solution Evolution → Progress Monitoring → Strategy Evolution. Claude reads this as its operating manual.

**`evox/` scripts** — Six Python modules that manage all persistent state so Claude doesn't manipulate JSON directly:

- **`state_manager.py`** (16 subcommands) — The core state machine. Manages per-GPU state (`state_gpu{N}.json`), shared population (`population.json`), and shared strategy history (`strategies.json`). Key subcommands: `add-candidate`, `check-stagnation`, `score-strategy`, `get-best-strategy`, `get`, `record-strategy`, `migrate`.
- **`population_summary.py`** — Computes φ(D_t), the population state descriptor. Outputs score statistics, per-operator performance, convergence analysis, and regressions. Claude reads this output to make strategy evolution decisions in Phase III.
- **`strategy_validator.py`** — Validates that `current_strategy_gpu{N}.md` has 6 required sections and parseable operator weights summing to 100%.
- **`resume.py`** — Session restart handler. Cross-references `candidates/` dirs against `population.json` to detect incomplete evaluations.
- **`filelock.py`** — File locking utility (`fcntl.flock` on Linux). Provides `locked_json` context manager for atomic read-modify-write of shared JSON files.
- **`gpu.py`** — GPU index resolution from `EVOX_GPU` environment variable. Used by all scripts.

## Key Design Decisions

- **Strategies are Markdown, not code.** Since Claude Code is both the LLM and the executor, search strategies are natural language documents Claude follows — not Python classes.
- **Multi-GPU uses per-GPU state files.** Each GPU has its own `state_gpu{N}.json` and `current_strategy_gpu{N}.md`. This eliminates contention on state machine transitions, counters, and phase tracking.
- **Shared files use file locking.** `population.json` and `strategies.json` use `fcntl.flock()` via `locked_json` for atomic read-modify-write. The lock is held only for the duration of the JSON operation.
- **GPU index resolved from `EVOX_GPU` environment variable.** Falls back to `"0"` for single-GPU backward compatibility.
- **Candidate IDs are GPU-namespaced.** `cand_g0_0003` prevents ID collisions between concurrent GPUs.
- **`consecutive_stagnations` counter survives strategy changes.** It is intentionally NOT reset in `record-strategy`. Only `check-stagnation` (on success) and `get-best-strategy` (on revert) reset it. This enables the 2+ stagnation revert mechanism.
- **Swarm candidates have no local train.py.** `import-swarm` records metadata only. `get-parent` filters to local candidates; `get-inspiration` prints API fetch commands for swarm entries.
- **Stagnation uses local-only delta.** `check-stagnation` compares local candidates against `window_start_best_bpb` to prevent swarm imports from masking strategy ineffectiveness. Overall delta is still used for J scoring.
- **Scoring formula adapted from maximize to minimize:** `J = Δ · log(1 + 1/start_val_bpb) / √W` where `Δ = start_bpb - end_bpb` (positive = improvement).

## File Relationships (on the remote VM)

```
~/autolab-contributor/
├── train.py                       # Working copy being edited
├── train_orig.py                   # Current master snapshot (diff base for submissions)
├── evox/
│   ├── state_gpu0.json            # GPU 0's session state
│   ├── state_gpu1.json            # GPU 1's session state (if multi-GPU)
│   ├── population.json            # SHARED: all evaluated candidates (D_t)
│   ├── strategies.json            # SHARED: strategy history with J scores
│   ├── current_strategy_gpu0.md   # GPU 0's active strategy document
│   ├── current_strategy_gpu1.md   # GPU 1's active strategy document
│   ├── filelock.py                # File locking utility
│   ├── gpu.py                     # GPU index resolution
│   └── strategies/                # Archived strategies (S_g0_000.md, S_g1_001.md, ...)
└── candidates/
    ├── cand_g0_0001/              # GPU 0's candidates (train.py + run.log)
    └── cand_g1_0001/              # GPU 1's candidates
```

## Hooks and Commands

**`evox/hooks/`** — Five Python hook scripts that act as guardrails during autonomous operation:

- **`guard_json_edits.py`** (PreToolUse: Write|Edit) — Blocks direct edits to `state_gpu*.json`, `population.json`, `strategies.json`. Forces all mutations through `state_manager.py` subcommands.
- **`validate_strategy.py`** (PostToolUse: Write|Edit) — Runs `strategy_validator.py` automatically after any edit to `current_strategy_gpu*.md`. Catches malformed strategies before they enter the loop.
- **`guard_destructive.py`** (PreToolUse: Bash) — Blocks `git reset --hard`, `rm -r candidates/`, and other destructive operations that could wipe evaluated candidates or state.
- **`validate_before_train.py`** (PreToolUse: Bash) — Checks that `train.py` differs from the parent's version before launching a GPU training run. Prevents wasting compute on identical re-evaluations.
- **`auto_checkpoint.py`** (Stop) — Git-commits all EvoX state files (dynamically discovers all GPU state files) when a session ends so `resume.py` can recover cleanly.

**`.claude/commands/`** — Two slash commands for manual use:

- **`/checkpoint`** — Snapshots all EvoX state to a timestamped git commit with candidate/window counts in the message.
- **`/evox-status`** — Displays a dashboard with current phase, window, strategy, stagnation count, best score, population size, operator performance, and convergence analysis.

All hooks are wired in `.claude/settings.local.json` under the `hooks` key. They receive tool input as JSON on stdin and use exit codes (0=allow, 2=block with stderr message).

## Editing Guidelines

- Scripts must work with `uv run evox/<script>.py` on the remote VM (Python 3.x, stdlib only).
- All CLI output is consumed by Claude Code — keep output structured with labeled fields (e.g., `PARENT: cand_g0_0003`, `STAGNATING: True`).
- JSON files use `indent=2` for human readability during debugging.
- `program.md` uses `uv run` for all Python execution, never bare `python` or `python3`.
- The `EVOX_GPU` environment variable must be set before running any EvoX commands in multi-GPU setups.

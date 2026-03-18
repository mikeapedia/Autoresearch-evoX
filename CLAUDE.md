# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Source repository for the **EvoX meta-evolutionary autolab contributor** — an autonomous ML research system that evolves both candidate solutions (train.py variants) and search strategies to minimize val_bpb on a small GPT language model. The `evox/` scripts are infrastructure deployed to a remote VM; `program.md` is the instruction document Claude Code follows autonomously.

## Lint and Type Check

```bash
ruff check evox/
ty check evox/
```

No test suite — the scripts are validated by running subcommands with mock data on the VM. No pyproject.toml in this repo; dependencies (torch, numpy, stdlib) are managed in the remote workspace.

## Architecture

**`program.md`** — The main deliverable. A ~620-line instruction document that tells Claude Code (running on a remote VM via CLI) how to run the three-phase EvoX loop: Solution Evolution → Progress Monitoring → Strategy Evolution. Claude reads this as its operating manual.

**`evox/` scripts** — Four Python CLI tools that manage all persistent state so Claude doesn't manipulate JSON directly:

- **`state_manager.py`** (14 subcommands) — The core state machine. Manages `state.json` (session/window/phase tracking), `population.json` (all evaluated candidates), and `strategies.json` (strategy history with J scores). Key subcommands: `add-candidate`, `check-stagnation`, `score-strategy`, `get-best-strategy`, `record-strategy`.
- **`population_summary.py`** — Computes φ(D_t), the population state descriptor. Outputs score statistics, per-operator performance, convergence analysis, and regressions. Claude reads this output to make strategy evolution decisions in Phase III.
- **`strategy_validator.py`** — Validates that `current_strategy.md` has 6 required sections and parseable operator weights summing to 100%.
- **`resume.py`** — Session restart handler. Cross-references `candidates/` dirs against `population.json` to detect incomplete evaluations.

## Key Design Decisions

- **Strategies are Markdown, not code.** Since Claude Code is both the LLM and the executor, search strategies are natural language documents Claude follows — not Python classes.
- **`consecutive_stagnations` counter survives strategy changes.** It is intentionally NOT reset in `record-strategy`. Only `check-stagnation` (on success) and `get-best-strategy` (on revert) reset it. This enables the 2+ stagnation revert mechanism.
- **Swarm candidates have no local train.py.** `import-swarm` records metadata only. `get-parent` filters to local candidates; `get-inspiration` prints API fetch commands for swarm entries.
- **Stagnation uses local-only delta.** `check-stagnation` compares local candidates against `window_start_best_bpb` to prevent swarm imports from masking strategy ineffectiveness. Overall delta is still used for J scoring.
- **Scoring formula adapted from maximize to minimize:** `J = Δ · log(1 + 1/start_val_bpb) / √W` where `Δ = start_bpb - end_bpb` (positive = improvement).

## File Relationships (on the remote VM)

```
~/autolab-contributor/
├── train.py              # Working copy being edited
├── train_orig.py          # Current master snapshot (diff base for submissions)
├── evox/
│   ├── state.json         # Session state (phase, window counter, master info)
│   ├── population.json    # All evaluated candidates (the solution database D_t)
│   ├── strategies.json    # Strategy history with J scores
│   ├── current_strategy.md # Active strategy document Claude follows
│   └── strategies/        # Archived strategy documents (S_000.md, S_001.md, ...)
└── candidates/
    └── cand_NNNN/         # Each candidate's train.py + run.log
```

## Editing Guidelines

- Scripts must work with `uv run evox/<script>.py` on the remote VM (Python 3.x, stdlib only).
- All CLI output is consumed by Claude Code — keep output structured with labeled fields (e.g., `PARENT: cand_0003`, `STAGNATING: True`).
- JSON files use `indent=2` for human readability during debugging.
- `program.md` uses `uv run` for all Python execution, never bare `python` or `python3`.

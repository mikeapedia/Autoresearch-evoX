# EvoX — Meta-Evolutionary Autolab Contributor

An autonomous ML research system powered by [Claude Code](https://claude.ai/code) that evolves both **candidate solutions** and **search strategies** to minimize `val_bpb` on a small GPT language model. Based on the [EvoX meta-evolutionary framework](https://arxiv.org/abs/2602.23413).

Instead of greedy hill-climbing, this system maintains a population of all evaluated candidates and an evolving natural-language search strategy. When progress stalls, it doesn't just try harder — it changes *how* it searches.

## How It Works

Claude Code runs autonomously on a remote VM with H100 GPUs, following the instructions in `program.md` to execute a three-phase loop:

```
┌─────────────────────────────────────────────────────────────┐
│                    EvoX EXPERIMENT LOOP                      │
│                                                             │
│  Phase I: Solution Evolution (W=6 candidates per window)    │
│    select parent → choose operator → edit train.py →        │
│    evaluate (5 min) → record result → submit if improved    │
│                          ↓                                  │
│  Phase II: Progress Monitoring                              │
│    compute population state φ(D_t) → score strategy →       │
│    check stagnation (Δ < τ?)                                │
│        ↓ stagnating          ↓ progressing                  │
│  Phase III: Strategy         └→ new window, back to Phase I │
│  Evolution                                                  │
│    analyze evidence →                                       │
│    write new strategy →                                     │
│    back to Phase I                                          │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts

- **Population database** (`population.json`): Every candidate ever evaluated — score, operator, hypothesis, parent lineage. This is the system's memory.
- **Search strategy** (`current_strategy_gpu{N}.md`): A Markdown document Claude follows that controls parent selection, operator weights (REFINE/DIVERGE/FREE), and variation guidance. Since Claude Code *is* the LLM, strategies are natural language — not Python classes.
- **Meta-evolution**: When a window of 6 evaluations shows stagnation (improvement < τ), Claude rewrites its own strategy based on empirical evidence from the population. After 2+ consecutive stagnations, it auto-reverts to the historically best strategy.
- **Multi-GPU**: Each GPU runs an independent Claude Code instance with its own strategy. All GPUs share the population database, enabling cross-GPU knowledge transfer.

## Repository Structure

```
├── program.md                  # The instruction document Claude Code follows
├── CLAUDE.md                   # Development guidance for contributors
├── evox/
│   ├── state_manager.py        # 16-subcommand CLI for all state mutations
│   ├── population_summary.py   # Population state descriptor φ(D_t)
│   ├── strategy_validator.py   # Strategy document structure validation
│   ├── resume.py               # Session restart recovery
│   ├── filelock.py             # fcntl.flock for concurrent multi-GPU access
│   ├── gpu.py                  # GPU index resolution (EVOX_GPU env var)
│   └── hooks/                  # 5 Claude Code hooks for autonomous guardrails
│       ├── guard_json_edits.py       # Blocks direct state file edits
│       ├── validate_strategy.py      # Auto-validates strategy after edits
│       ├── guard_destructive.py      # Blocks destructive git/rm operations
│       ├── validate_before_train.py  # Prevents evaluating unchanged code
│       └── auto_checkpoint.py        # Git-commits state on session end
├── .claude/
│   ├── settings.local.json     # Hook wiring and permissions
│   └── commands/
│       ├── checkpoint.md       # /checkpoint slash command
│       └── evox-status.md      # /evox-status dashboard command
```

## Deployment

These scripts are deployed to a remote VM alongside the autolab workspace:

```bash
# On the VM
cp -r evox/ ~/autolab-contributor/evox/
cp -r .claude/ ~/autolab-contributor/.claude/

# Set GPU and start
export EVOX_GPU=0
cd ~/autolab-contributor
uv run evox/state_manager.py init --gpu 0 --tau 0.001 --window-size 6
```

Claude Code then reads `program.md` as its operating manual and runs the EvoX loop autonomously.

## Multi-GPU Operation

Each GPU runs its own Claude Code instance with independent strategy evolution:

```bash
# Terminal 1                          # Terminal 2
export EVOX_GPU=0                     export EVOX_GPU=1
claude -p program.md                  claude -p program.md
```

**Per-GPU** (no contention): `state_gpu{N}.json`, `current_strategy_gpu{N}.md`, `candidates/cand_g{N}_*/`

**Shared** (file-locked): `population.json`, `strategies.json`

GPU-0 might run an aggressive DIVERGE-heavy strategy while GPU-1 runs conservative REFINE. Both benefit from each other's discoveries through the shared population. When a strategy stagnates, `get-best-strategy` can adopt any GPU's historically best strategy — enabling cross-GPU knowledge transfer.

## Guardrails

Five Claude Code hooks prevent common failure modes during long autonomous sessions:

| Hook | Trigger | Purpose |
|------|---------|---------|
| `guard_json_edits` | PreToolUse: Write\|Edit | Forces all state mutations through `state_manager.py` |
| `validate_strategy` | PostToolUse: Write\|Edit | Auto-validates strategy documents after every edit |
| `guard_destructive` | PreToolUse: Bash | Blocks `git reset --hard`, `rm -r candidates/`, etc. |
| `validate_before_train` | PreToolUse: Bash | Prevents evaluating `train.py` identical to its parent |
| `auto_checkpoint` | Stop | Git-commits all state on session end for crash recovery |

## Scoring

Strategy performance is scored using an adaptation of the EvoX paper's formula:

```
J = Δ · log(1 + 1/start_val_bpb) / √W
```

Where `Δ = start_bpb - end_bpb` (positive = improvement), and the log term rewards improvements from strong baselines (harder to improve = more credit).

## Development

```bash
ruff check evox/     # Lint
ty check evox/       # Type check
```

No test suite — scripts are validated by running subcommands with mock data on the VM. All scripts use Python stdlib only (no external dependencies).

## Background

This system is built for [autolab](http://autoresearchhub.com), a distributed ML research competition where contributors submit patches to a shared `train.py`. Each patch trains a small GPT on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) for 5 minutes on an H100. The only metric is `val_bpb` (validation bits per byte) — lower is better.

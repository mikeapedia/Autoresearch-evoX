---
description: Show a full EvoX dashboard — phase, progress, strategy, population
---

Display a comprehensive EvoX status dashboard by gathering data from all state sources. Run each command and compile the results into a clean summary.

## Data to gather

1. **Session state** (reads per-GPU state via EVOX_GPU env var) — run:
```bash
uv run evox/state_manager.py get --key phase
uv run evox/state_manager.py get --key window_count
uv run evox/state_manager.py get --key window_iteration
uv run evox/state_manager.py get --key current_strategy_id
uv run evox/state_manager.py get --key consecutive_stagnations
uv run evox/state_manager.py get --key window_start_best_bpb
uv run evox/state_manager.py get --key master_val_bpb
uv run evox/state_manager.py get --key gpu_index
```

2. **Population summary** (shared across all GPUs) — run:
```bash
uv run evox/population_summary.py
```

3. **Strategy validation** — run:
```bash
uv run evox/strategy_validator.py
```

4. **Candidate count and recent activity** — run:
```bash
uv run python -c "import json; p=json.load(open('evox/population.json')); print(f'Total candidates: {len(p)}'); local=[c for c in p if c.get(\"source\")==\"local\"]; print(f'Local: {len(local)}, Swarm: {len(p)-len(local)}')"
```

## Output format

Present the results as a formatted dashboard:

```
╔══════════════════════════════════════════╗
║           EvoX STATUS DASHBOARD          ║
╠══════════════════════════════════════════╣
║ GPU:          [N] (EVOX_GPU)             ║
║ Phase:        [current phase]            ║
║ Window:       #[N] / Strategy: S_g[N]_XX ║
║ Stagnations:  [N] consecutive            ║
║ Best val_bpb: [X.XXXX] (master: Y.YYYY)  ║
║ Population:   [N] total ([L] local)      ║
║ Strategy:     [VALID/INVALID]            ║
╚══════════════════════════════════════════╝
```

Then show the key sections from population_summary output (score stats, operator performance, convergence).

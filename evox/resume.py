#!/usr/bin/env python3
"""Print full context for resuming an EvoX session after restart.

Reads all state files and outputs a structured summary that tells Claude
exactly where to continue from.

Usage:
    python evox/resume.py
"""

import json
from pathlib import Path

EVOX_DIR = Path(__file__).parent
STATE_FILE = EVOX_DIR / "state.json"
POPULATION_FILE = EVOX_DIR / "population.json"
STRATEGIES_FILE = EVOX_DIR / "strategies.json"
STRATEGY_DOC = EVOX_DIR / "current_strategy.md"
CANDIDATES_DIR = EVOX_DIR.parent / "candidates"


def load_json(path, default=None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}


def main():
    state = load_json(STATE_FILE, {})
    pop = load_json(POPULATION_FILE, [])
    strats = load_json(STRATEGIES_FILE, [])

    if not state:
        print("=== NO EVOX SESSION FOUND ===")
        print("Run 'uv run evox/state_manager.py init' to start a new session.")
        return

    print("=== RESUMING EVOX SESSION ===")
    print(f"Session ID: {state.get('session_id', 'unknown')}")
    print(f"Total evaluations: {state.get('total_evaluations', 0)}")
    print(f"GPU: {state.get('gpu_index', 0)}")
    print()

    # Phase info
    phase = state.get("phase", "solution_evolution")
    window_iter = state.get("window_iteration", 0)
    window_size = state.get("window_size", 6)
    strategy_id = state.get("current_strategy_id", "S_000")

    print(f"Current phase: {phase.upper().replace('_', ' ')}")
    print(f"Window progress: {window_iter}/{window_size}")
    print(f"Active strategy: {strategy_id}")
    print(f"Consecutive stagnations: {state.get('consecutive_stagnations', 0)}")
    print()

    # Master info
    print("=== Master ===")
    print(f"Hash: {state.get('master_hash', 'unknown')}")
    print(f"val_bpb: {state.get('master_val_bpb', 'unknown')}")
    print(f"Last sync: {state.get('last_timestamp', 'unknown')}")
    print()

    # Population summary
    print("=== Population ===")
    if pop:
        local = [c for c in pop if c.get("source") == "local"]
        swarm = [c for c in pop if c.get("source") == "swarm"]
        best = min(pop, key=lambda c: c["val_bpb"])
        print(f"Total: {len(pop)} ({len(local)} local + {len(swarm)} swarm)")
        print(f"Best: {best['id']} val_bpb={best['val_bpb']:.6f}")

        # Last 3 candidates
        recent = sorted(pop, key=lambda c: c.get("timestamp", ""), reverse=True)[:3]
        print("Recent candidates:")
        for c in recent:
            print(f"  {c['id']}: val_bpb={c['val_bpb']:.6f} [{c['operator']}] - {c.get('hypothesis', '?')[:50]}")
    else:
        print("Empty - no candidates evaluated yet.")
    print()

    # Current strategy
    print("=== Current Strategy ===")
    if STRATEGY_DOC.exists():
        content = STRATEGY_DOC.read_text(encoding="utf-8")
        # Print first 20 lines
        lines = content.strip().split("\n")
        for line in lines[:20]:
            print(f"  {line}")
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} more lines, see evox/current_strategy.md)")
    else:
        print("No strategy document found. Create evox/current_strategy.md.")
    print()

    # Strategy history
    if strats:
        print("=== Strategy History ===")
        for s in strats:
            j = f"{s['J_score']:.6f}" if s.get("J_score") is not None else "N/A"
            print(f"  {s['strategy_id']}: J={j}, windows={s.get('windows_active', 0)}")
        print()

    # Check for incomplete evaluations (candidate dir exists but not in population.json)
    if CANDIDATES_DIR.exists():
        recorded_ids = {c["id"] for c in pop}
        incomplete = []
        for cand_dir in sorted(CANDIDATES_DIR.iterdir()):
            if cand_dir.is_dir() and cand_dir.name.startswith("cand_"):
                if cand_dir.name not in recorded_ids:
                    has_train = (cand_dir / "train.py").exists()
                    has_log = (cand_dir / "run.log").exists()
                    status = "has train.py + log" if (has_train and has_log) else "has train.py only" if has_train else "empty dir"
                    incomplete.append(f"{cand_dir.name} ({status})")
        if incomplete:
            print("=== Incomplete Evaluations ===")
            for desc in incomplete:
                print(f"  {desc} - not recorded in population, may need re-evaluation")
            print()

    # Next action
    print("=== NEXT ACTION ===")
    if phase == "solution_evolution":
        if window_iter >= window_size:
            print("Window complete. Proceed to Phase II (Progress Monitoring).")
            print("Run: uv run evox/state_manager.py check-stagnation")
        else:
            print(f"Continue Phase I (Solution Evolution), iteration {window_iter + 1} of {window_size}.")
            print("Steps: check master -> select parent -> choose operator -> generate candidate -> evaluate -> record")
    elif phase == "progress_monitoring":
        print("Run Phase II: check stagnation and score strategy.")
        print("Run: uv run evox/state_manager.py check-stagnation")
    elif phase == "strategy_evolution":
        print("Run Phase III: evolve the search strategy.")
        print("Read population_summary.py output and strategy history, then write new current_strategy.md.")
    else:
        print(f"Unknown phase: {phase}. Check state.json.")


if __name__ == "__main__":
    main()

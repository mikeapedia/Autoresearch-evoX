#!/usr/bin/env python3
"""Compute population state descriptor phi(D_t) for EvoX strategy evolution.

Reads population.json, strategies.json, and state.json, then outputs a
human-readable report that Claude uses to understand the current state
of the evolutionary search.

Usage:
    uv run evox/population_summary.py
"""

import json
import statistics
from pathlib import Path

EVOX_DIR = Path(__file__).parent
STATE_FILE = EVOX_DIR / "state.json"
POPULATION_FILE = EVOX_DIR / "population.json"
STRATEGIES_FILE = EVOX_DIR / "strategies.json"


def load_json(path, default=None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}


def main():
    state = load_json(STATE_FILE, {})
    pop = load_json(POPULATION_FILE, [])
    strats = load_json(STRATEGIES_FILE, [])

    if not pop:
        print("=== Population State phi(D_t) ===")
        print("Population is empty. No candidates evaluated yet.")
        return

    # Split by source
    local = [c for c in pop if c.get("source") == "local"]
    swarm = [c for c in pop if c.get("source") == "swarm"]

    all_bpb = [c["val_bpb"] for c in pop]

    best = min(pop, key=lambda c: c["val_bpb"])
    best_local = min(local, key=lambda c: c["val_bpb"]) if local else None

    master_bpb = state.get("master_val_bpb")

    # ── Score Statistics ──
    print("=== Population State phi(D_t) ===")
    print(f"Total candidates: {len(pop)} ({len(local)} local + {len(swarm)} swarm)")
    print(f"Best val_bpb: {best['val_bpb']:.6f} ({best['id']}, {best['operator']}, strategy {best.get('strategy_id', '?')})")
    if best_local and best_local != best:
        print(f"Best local val_bpb: {best_local['val_bpb']:.6f} ({best_local['id']})")
    if master_bpb is not None:
        better_than_master = [c for c in pop if c["val_bpb"] < master_bpb]
        print(f"Master val_bpb: {master_bpb:.6f}")
        print(f"Candidates better than master: {len(better_than_master)}")

    if len(all_bpb) >= 2:
        print(f"Worst val_bpb: {max(all_bpb):.6f}")
        print(f"Median val_bpb: {statistics.median(all_bpb):.6f}")
        print(f"Std dev: {statistics.stdev(all_bpb):.6f}")
    top5 = sorted(all_bpb)[:5]
    print(f"Top-5 scores: [{', '.join(f'{s:.6f}' for s in top5)}]")

    # ── Progress Indicators ──
    print()
    print("=== Progress ===")
    # Find last improvement (local candidate that set a new best at its time)
    sorted_local = sorted(local, key=lambda c: c.get("timestamp", ""))
    running_best = float("inf")
    last_improvement_idx = -1
    for i, c in enumerate(sorted_local):
        if c["val_bpb"] < running_best:
            running_best = c["val_bpb"]
            last_improvement_idx = i

    if last_improvement_idx >= 0 and sorted_local:
        evals_since = len(sorted_local) - 1 - last_improvement_idx
        improver = sorted_local[last_improvement_idx]
        print(f"Last improvement: {evals_since} evaluations ago ({improver['id']}, {improver['operator']})")
    else:
        print("Last improvement: N/A")

    window_start = state.get("window_start_best_bpb")
    if window_start is not None:
        current_best = min(all_bpb)
        window_delta = window_start - current_best
        print(f"Current window: start={window_start:.6f}, current_best={current_best:.6f}, delta={window_delta:.6f}")
    print(f"Window progress: {state.get('window_iteration', 0)}/{state.get('window_size', 6)}")

    # ── Operator Performance ──
    print()
    print("=== Operator Performance ===")
    for op in ["REFINE", "DIVERGE", "FREE"]:
        op_cands = [c for c in local if c.get("operator") == op]
        if not op_cands:
            print(f"{op}: no evaluations")
            continue
        op_bpb = [c["val_bpb"] for c in op_cands]
        avg_bpb = statistics.mean(op_bpb)
        best_op = min(op_bpb)
        # Compute average improvement vs parent (if we have parent data)
        improvements = []
        for c in op_cands:
            parent_id = c.get("parent_id")
            if parent_id == "master" and master_bpb is not None:
                improvements.append(master_bpb - c["val_bpb"])
            elif parent_id and parent_id != "master":
                parent = next((p for p in pop if p["id"] == parent_id), None)
                if parent:
                    improvements.append(parent["val_bpb"] - c["val_bpb"])
        imp_str = ""
        if improvements:
            avg_imp = statistics.mean(improvements)
            imp_str = f", avg improvement: {avg_imp:+.6f}"
        print(f"{op}: {len(op_cands)} evals, avg_bpb={avg_bpb:.6f}, best={best_op:.6f}{imp_str}")

    # ── Strategy Performance ──
    if strats:
        print()
        print("=== Strategy Performance ===")
        for s in strats:
            j = f"{s['J_score']:.6f}" if s.get("J_score") is not None else "N/A"
            best_c = f"{s['best_candidate_bpb']:.6f}" if s.get("best_candidate_bpb") is not None else "N/A"
            print(f"{s['strategy_id']}: J={j}, windows={s.get('windows_active', 0)}, "
                  f"candidates={s.get('total_candidates_generated', 0)}, best_bpb={best_c}")
            if s.get("description"):
                # Truncate long descriptions
                desc = s["description"][:80]
                print(f"  desc: {desc}")

    # ── Convergence Signal ──
    if len(all_bpb) >= 3:
        print()
        print("=== Convergence Analysis ===")
        sorted_bpb = sorted(all_bpb)
        top3_spread = sorted_bpb[min(2, len(sorted_bpb)-1)] - sorted_bpb[0]
        top5_spread = sorted_bpb[min(4, len(sorted_bpb)-1)] - sorted_bpb[0] if len(sorted_bpb) >= 5 else top3_spread

        if top3_spread < 0.001:
            print(f"CONVERGENCE: HIGH - top-3 spread is {top3_spread:.6f} (<0.001)")
            print("  Consider increasing DIVERGE weight to escape local optimum")
        elif top3_spread < 0.005:
            print(f"CONVERGENCE: MODERATE - top-3 spread is {top3_spread:.6f}")
        else:
            print(f"CONVERGENCE: LOW - top-3 spread is {top3_spread:.6f} (diverse population)")

        if len(sorted_bpb) >= 5:
            print(f"Top-5 spread: {top5_spread:.6f}")

    # ── Worst Regressions (negative knowledge) ──
    regressions = []
    for c in local:
        parent_id = c.get("parent_id")
        parent_bpb = None
        if parent_id == "master" and master_bpb is not None:
            parent_bpb = master_bpb
        elif parent_id and parent_id != "master":
            parent = next((p for p in pop if p["id"] == parent_id), None)
            if parent:
                parent_bpb = parent["val_bpb"]
        if parent_bpb is not None:
            regression = c["val_bpb"] - parent_bpb  # positive = worse
            if regression > 0.005:  # significant regression
                regressions.append((c, {"id": parent_id, "val_bpb": parent_bpb}, regression))

    if regressions:
        print()
        print("=== Worst Regressions (avoid these patterns) ===")
        regressions.sort(key=lambda x: x[2], reverse=True)
        for c, parent, reg in regressions[:5]:
            print(f"  {c['id']} [{c['operator']}]: +{reg:.6f} worse than parent {parent['id']}")
            print(f"    hypothesis: {c.get('hypothesis', '?')[:70]}")

    # ── Recent Candidates ──
    print()
    print("=== Recent Candidates (last 5) ===")
    recent = sorted(pop, key=lambda c: c.get("timestamp", ""), reverse=True)[:5]
    for c in recent:
        submitted = " [SUBMITTED]" if c.get("submitted") else ""
        print(f"  {c['id']}: val_bpb={c['val_bpb']:.6f} [{c['operator']}] "
              f"parent={c.get('parent_id', '?')} - {c.get('hypothesis', '?')[:60]}{submitted}")


if __name__ == "__main__":
    main()

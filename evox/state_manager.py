#!/usr/bin/env python3
"""EvoX state manager for autolab (multi-GPU).

Manages the solution database (population.json), per-GPU session state
(state_gpu{N}.json), and strategy history (strategies.json). Provides CLI
subcommands for all state mutations so Claude Code doesn't need to
manipulate JSON directly.

Multi-GPU: Each GPU gets its own state file and strategy document.
The population and strategy history are shared across GPUs with file
locking to prevent corruption from concurrent access. The GPU index
is read from the EVOX_GPU environment variable (default "0").

Usage:
    uv run evox/state_manager.py init --gpu 0 --tau 0.001 --window-size 6
    uv run evox/state_manager.py add-candidate --val-bpb 1.02 --parent cand_g0_001 --operator REFINE --hypothesis "..." [--master-hash abc] [--strategy-id S_g0_000] [--submitted]
    uv run evox/state_manager.py get-parent --method best|tournament|random
    uv run evox/state_manager.py get-inspiration --count 3
    uv run evox/state_manager.py select-operator --weights 40,40,20
    uv run evox/state_manager.py start-window
    uv run evox/state_manager.py advance-window
    uv run evox/state_manager.py check-stagnation
    uv run evox/state_manager.py import-swarm --id <id> --val-bpb <val> --message "..." --master-hash <hash>
    uv run evox/state_manager.py record-strategy --id S_g0_001 --parent-id S_g0_000 --description "..."
    uv run evox/state_manager.py score-strategy
    uv run evox/state_manager.py get-best-strategy
    uv run evox/state_manager.py get --key <key>
    uv run evox/state_manager.py set --key <key> --value <value>
    uv run evox/state_manager.py show
    uv run evox/state_manager.py migrate
"""

import argparse
import json
import math
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add evox/ directory to sys.path so sibling modules (filelock, gpu) are importable
# when run via `uv run evox/state_manager.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filelock import locked_json  # noqa: E402
from gpu import get_gpu_index, gpu_prefix  # noqa: E402

EVOX_DIR = Path(__file__).parent
POPULATION_FILE = EVOX_DIR / "population.json"       # SHARED across GPUs
STRATEGIES_FILE = EVOX_DIR / "strategies.json"        # SHARED across GPUs
CANDIDATES_DIR = EVOX_DIR.parent / "candidates"
STRATEGIES_DIR = EVOX_DIR / "strategies"


def _state_file():
    """Per-GPU state file: state_gpu0.json, state_gpu1.json, etc."""
    return EVOX_DIR / f"state_gpu{get_gpu_index()}.json"


def _strategy_doc():
    """Per-GPU strategy document: current_strategy_gpu0.md, etc."""
    return EVOX_DIR / f"current_strategy_gpu{get_gpu_index()}.md"


def load_json(path, default=None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_state():
    return load_json(_state_file(), {})


def save_state(state):
    save_json(_state_file(), state)


def load_population():
    """Read-only snapshot of shared population. For writes, use locked_json."""
    return load_json(POPULATION_FILE, [])


def load_strategies():
    """Read-only snapshot of shared strategies. For writes, use locked_json."""
    return load_json(STRATEGIES_FILE, [])


# ── init ──────────────────────────────────────────────────────────────────────

def cmd_init(args):
    """Initialize EvoX state files and directories."""
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    EVOX_DIR.mkdir(parents=True, exist_ok=True)
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

    gp = gpu_prefix()
    state = {
        "session_id": str(uuid.uuid4())[:8],
        "phase": "solution_evolution",
        "current_strategy_id": f"S_{gp}_000",
        "window_iteration": 0,
        "window_size": args.window_size,
        "window_start_best_bpb": None,
        "master_hash": None,
        "master_val_bpb": None,
        "last_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_evaluations": 0,
        "window_count": 0,
        "tau": args.tau,
        "gpu_index": args.gpu,
        "consecutive_stagnations": 0,
        "current_parent_id": None,
    }
    save_state(state)

    # Shared files: only create if they don't exist (another GPU may have init'd first)
    if not POPULATION_FILE.exists():
        save_json(POPULATION_FILE, [])
    if not STRATEGIES_FILE.exists():
        save_json(STRATEGIES_FILE, [])

    print(f"Initialized EvoX state (session={state['session_id']}, W={args.window_size}, tau={args.tau}, GPU={args.gpu})")
    print(f"  State file: {_state_file().name}")
    print(f"  Strategy doc: {_strategy_doc().name}")


# ── add-candidate ─────────────────────────────────────────────────────────────

def cmd_add_candidate(args):
    """Record an evaluated candidate in the population."""
    state = load_state()

    gp = gpu_prefix()
    cand_id = f"cand_{gp}_{state['total_evaluations']:04d}"
    cand_dir = CANDIDATES_DIR / cand_id
    cand_dir.mkdir(parents=True, exist_ok=True)

    candidate = {
        "id": cand_id,
        "parent_id": args.parent or "master",
        "source": "local",
        "operator": args.operator,
        "hypothesis": args.hypothesis,
        "val_bpb": args.val_bpb,
        "master_hash_at_eval": args.master_hash or state.get("master_hash"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy_id": args.strategy_id or state.get("current_strategy_id"),
        "submitted": args.submitted,
        "gpu_index": get_gpu_index(),
    }

    # Atomic append to shared population under file lock
    with locked_json(POPULATION_FILE, []) as pop:
        prev_local = [c for c in pop if c["source"] == "local"]
        prev_best_bpb = min((c["val_bpb"] for c in prev_local), default=float("inf"))
        pop.append(candidate)

    # Update per-GPU state (no lock needed — single writer)
    state["total_evaluations"] += 1
    state["window_iteration"] += 1
    save_state(state)

    # Report
    marker = " * NEW LOCAL BEST" if args.val_bpb < prev_best_bpb else ""
    print(f"Recorded {cand_id}: val_bpb={args.val_bpb:.6f} [{args.operator}]{marker}")
    print(f"  Window: {state['window_iteration']}/{state['window_size']} | Total evals: {state['total_evaluations']}")
    print(f"  Candidate dir: {cand_dir}")


# ── get-parent ────────────────────────────────────────────────────────────────

def cmd_get_parent(args):
    """Select a parent candidate from the population."""
    pop = load_population()
    state = load_state()

    # Only select from local candidates (swarm candidates don't have train.py on disk)
    local = [c for c in pop if c["source"] == "local"]
    if not local:
        state["current_parent_id"] = "master"
        save_state(state)
        print("PARENT: master (no local candidates)")
        print("SOURCE: master")
        return

    method = args.method

    if method == "best":
        parent = min(local, key=lambda c: c["val_bpb"])
    elif method == "tournament":
        k = min(3, len(local))
        tournament = random.sample(local, k)
        parent = min(tournament, key=lambda c: c["val_bpb"])
    elif method == "random":
        parent = random.choice(local)
    else:
        parent = min(local, key=lambda c: c["val_bpb"])

    # Persist parent selection so hooks can validate against it
    state["current_parent_id"] = parent["id"]
    save_state(state)

    print(f"PARENT: {parent['id']}")
    print(f"VAL_BPB: {parent['val_bpb']:.6f}")
    print(f"OPERATOR: {parent['operator']}")
    print(f"HYPOTHESIS: {parent['hypothesis']}")
    print(f"SOURCE: {parent['source']}")
    print(f"TRAIN_PY: {CANDIDATES_DIR / parent['id'] / 'train.py'}")


# ── get-inspiration ───────────────────────────────────────────────────────────

def cmd_get_inspiration(args):
    """Select inspiration candidates from the population."""
    pop = load_population()

    if not pop:
        print("No candidates available for inspiration.")
        return

    count = min(args.count, len(pop))
    sorted_pop = sorted(pop, key=lambda c: c["val_bpb"])

    # Take top candidates + random for diversity
    top_count = max(1, count - 1)
    inspiration = sorted_pop[:top_count]

    # Add one random from the rest if possible
    remaining = [c for c in sorted_pop[top_count:] if c not in inspiration]
    if remaining and count > top_count:
        inspiration.append(random.choice(remaining))

    print(f"INSPIRATION ({len(inspiration)} candidates):")
    for i, c in enumerate(inspiration):
        print(f"  [{i+1}] {c['id']}: val_bpb={c['val_bpb']:.6f} [{c['operator']}] - {c['hypothesis']}")
        if c.get("source") == "swarm":
            # Swarm candidates don't have local train.py — fetch via API if needed
            raw_hash = c['id'].removeprefix("swarm_")
            print(f"      code: fetch via curl -s \"$AUTOLAB/api/git/commits/{raw_hash}\"")
        else:
            print(f"      train.py: {CANDIDATES_DIR / c['id'] / 'train.py'}")


# ── select-operator ───────────────────────────────────────────────────────────

def cmd_select_operator(args):
    """Weighted random selection of variation operator."""
    weights = [int(w) for w in args.weights.split(",")]
    if len(weights) != 3:
        print("ERROR: --weights must be 3 comma-separated integers (REFINE,DIVERGE,FREE)", file=sys.stderr)
        sys.exit(1)

    operators = ["REFINE", "DIVERGE", "FREE"]
    chosen = random.choices(operators, weights=weights, k=1)[0]
    print(f"OPERATOR: {chosen}")


# ── start-window ──────────────────────────────────────────────────────────────

def cmd_start_window(args):
    """Initialize a new evaluation window. Records the current best score."""
    state = load_state()
    pop = load_population()

    # Increment total window count (tracks completed windows across session)
    state["window_count"] = state.get("window_count", 0) + 1
    state["window_iteration"] = 0
    state["phase"] = "solution_evolution"

    if pop:
        best_bpb = min(c["val_bpb"] for c in pop)
        state["window_start_best_bpb"] = best_bpb
    else:
        state["window_start_best_bpb"] = state.get("master_val_bpb")

    save_state(state)
    print(f"Window started. Best val_bpb at start: {state['window_start_best_bpb']}")
    print(f"Strategy: {state['current_strategy_id']} | Window size: {state['window_size']}")


# ── advance-window ────────────────────────────────────────────────────────────

def cmd_advance_window(args):
    """Check if the current window is complete."""
    state = load_state()
    iteration = state.get("window_iteration", 0)
    window_size = state.get("window_size", 6)

    if iteration >= window_size:
        print(f"WINDOW_COMPLETE: true ({iteration}/{window_size})")
        print("ACTION: Proceed to Phase II (Progress Monitoring)")
    else:
        print(f"WINDOW_COMPLETE: false ({iteration}/{window_size})")
        print("ACTION: Continue Phase I (Solution Evolution)")


# ── check-stagnation ──────────────────────────────────────────────────────────

def cmd_check_stagnation(args):
    """Compute window improvement and check stagnation against tau."""
    state = load_state()
    pop = load_population()

    start_bpb = state.get("window_start_best_bpb")
    if start_bpb is None:
        print("ERROR: No window start score recorded. Run start-window first.", file=sys.stderr)
        sys.exit(1)

    if not pop:
        print("ERROR: No candidates evaluated.", file=sys.stderr)
        sys.exit(1)

    current_best_bpb = min(c["val_bpb"] for c in pop)
    delta = start_bpb - current_best_bpb  # positive = improvement
    tau = state.get("tau", 0.001)
    W = state.get("window_size", 6)

    # Compute this-GPU-only delta for stagnation detection.
    # In multi-GPU mode, each GPU must judge its own strategy on its own output.
    # Filter by gpu_index if available, otherwise fall back to all local candidates.
    gpu_idx = get_gpu_index()
    gpu_cands = [c for c in pop if c["source"] == "local" and c.get("gpu_index") == gpu_idx]
    if not gpu_cands:
        # Fallback for pre-migration candidates without gpu_index field
        gpu_cands = [c for c in pop if c["source"] == "local"]
    if gpu_cands:
        local_best_bpb = min(c["val_bpb"] for c in gpu_cands)
        local_delta = start_bpb - local_best_bpb
    else:
        local_best_bpb = start_bpb
        local_delta = 0.0

    # Use local delta for stagnation detection (strategy should be judged on its own output)
    # Use overall delta for J score (strategy benefits from good parent selection including swarm)
    if start_bpb > 0:
        J = delta * math.log(1 + 1 / start_bpb) / math.sqrt(W)
    else:
        J = 0.0

    # Use small epsilon for float comparison to avoid false stagnation from rounding
    stagnating = local_delta < (tau - 1e-9)

    # Update phase in state for resume correctness
    state["phase"] = "progress_monitoring"

    print(f"WINDOW_START_BPB: {start_bpb:.6f}")
    print(f"CURRENT_BEST_BPB: {current_best_bpb:.6f}")
    print(f"LOCAL_BEST_BPB: {local_best_bpb:.6f}")
    print(f"DELTA: {delta:.6f} (overall, positive = improvement)")
    print(f"LOCAL_DELTA: {local_delta:.6f} (this GPU's candidates only)")
    print(f"TAU: {tau:.6f}")
    print(f"STRATEGY_SCORE_J: {J:.6f}")
    print(f"STAGNATING: {stagnating} (based on local delta)")

    if stagnating:
        consec = state.get("consecutive_stagnations", 0) + 1
        state["consecutive_stagnations"] = consec
        state["phase"] = "strategy_evolution"
        save_state(state)
        print(f"CONSECUTIVE_STAGNATIONS: {consec}")
        if consec >= 2:
            print("ACTION: Revert to best historical strategy (2+ consecutive stagnations)")
        else:
            print("ACTION: Proceed to Phase III (Strategy Evolution)")
    else:
        state["consecutive_stagnations"] = 0
        save_state(state)
        print("ACTION: Reset window and continue Phase I with current strategy")

    return J


# ── import-swarm ──────────────────────────────────────────────────────────────

def cmd_import_swarm(args):
    """Import a swarm experiment as an immigrant candidate."""
    # Atomic duplicate-check + append under file lock
    with locked_json(POPULATION_FILE, []) as pop:
        if any(c["id"] == f"swarm_{args.id}" for c in pop):
            print(f"Swarm candidate {args.id} already imported, skipping.")
            return

        candidate = {
            "id": f"swarm_{args.id}",
            "parent_id": "swarm",
            "source": "swarm",
            "operator": "SWARM",
            "hypothesis": args.message,
            "val_bpb": args.val_bpb,
            "master_hash_at_eval": args.master_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy_id": "external",
            "submitted": False,
        }
        pop.append(candidate)

    print(f"Imported swarm candidate swarm_{args.id}: val_bpb={args.val_bpb:.6f}")


# ── record-strategy ──────────────────────────────────────────────────────────

def cmd_record_strategy(args):
    """Record a new strategy in the strategy history and archive the document."""
    state = load_state()

    # Archive the current strategy document before switching
    strategy_doc = _strategy_doc()
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    if strategy_doc.exists():
        archive_path = STRATEGIES_DIR / f"{args.id}.md"
        archive_path.write_text(strategy_doc.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Archived strategy document to {archive_path}")

    entry = {
        "strategy_id": args.id,
        "parent_strategy_id": args.parent_id or state.get("current_strategy_id"),
        "description": args.description,
        "J_score": None,  # filled by score-strategy
        "windows_active": 0,
        "total_candidates_generated": 0,
        "best_candidate_bpb": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpu_index": get_gpu_index(),
    }

    # Atomic append to shared strategies under file lock
    with locked_json(STRATEGIES_FILE, []) as strats:
        strats.append(entry)

    state["current_strategy_id"] = args.id
    # NOTE: Do NOT reset consecutive_stagnations here. The counter tracks
    # consecutive stagnating WINDOWS across strategy changes. It is only
    # reset by check-stagnation (on success) or get-best-strategy (on revert).
    save_state(state)

    print(f"Recorded strategy {args.id} (parent: {entry['parent_strategy_id']})")
    print(f"Active strategy set to: {args.id}")


# ── score-strategy ────────────────────────────────────────────────────────────

def cmd_score_strategy(args):
    """Update the current strategy's performance score after a window."""
    state = load_state()
    strategy_id = state.get("current_strategy_id")

    # Atomic read-update on shared strategies under file lock
    with locked_json(STRATEGIES_FILE, []) as strats:
        pop = load_population()  # read-only snapshot

        entry = next((s for s in strats if s["strategy_id"] == strategy_id), None)

        if not entry:
            print(f"WARNING: Strategy {strategy_id} not found in history. Creating entry.")
            entry = {
                "strategy_id": strategy_id,
                "parent_strategy_id": None,
                "description": "Initial strategy",
                "J_score": 0.0,
                "windows_active": 0,
                "total_candidates_generated": 0,
                "best_candidate_bpb": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "gpu_index": get_gpu_index(),
            }
            strats.append(entry)

        # Count candidates generated under this strategy
        strat_candidates = [c for c in pop if c.get("strategy_id") == strategy_id and c["source"] == "local"]
        entry["total_candidates_generated"] = len(strat_candidates)
        windows_active = int(entry.get("windows_active") or 0) + 1
        entry["windows_active"] = windows_active

        if strat_candidates:
            entry["best_candidate_bpb"] = min(c["val_bpb"] for c in strat_candidates)

        # Compute J score
        start_bpb = state.get("window_start_best_bpb")
        if start_bpb and pop:
            current_best = min(c["val_bpb"] for c in pop)
            delta = start_bpb - current_best
            W = state.get("window_size", 6)
            J = delta * math.log(1 + 1 / start_bpb) / math.sqrt(W) if start_bpb > 0 else 0.0

            # Running average of J across windows
            if entry["J_score"] is not None:
                entry["J_score"] = (float(entry["J_score"]) * (windows_active - 1) + J) / windows_active
            else:
                entry["J_score"] = J

    print(f"Strategy {strategy_id}: J={entry['J_score']:.6f}, windows={entry['windows_active']}, candidates={entry['total_candidates_generated']}")
    if entry["best_candidate_bpb"] is not None:
        print(f"  Best candidate val_bpb: {entry['best_candidate_bpb']:.6f}")


# ── get-best-strategy ─────────────────────────────────────────────────────────

def cmd_get_best_strategy(args):
    """Find the best historical strategy and restore its document."""
    strats = load_strategies()
    state = load_state()

    scored = [s for s in strats if s.get("J_score") is not None and s["J_score"] > 0]
    if not scored:
        print("No scored strategies found. Keeping current strategy.")
        return

    best = max(scored, key=lambda s: s["J_score"])
    archive_path = STRATEGIES_DIR / f"{best['strategy_id']}.md"

    print(f"BEST_STRATEGY: {best['strategy_id']}")
    print(f"J_SCORE: {best['J_score']:.6f}")
    print(f"DESCRIPTION: {best.get('description', 'N/A')}")

    if archive_path.exists():
        # Restore the strategy document to this GPU's strategy file
        strategy_doc = _strategy_doc()
        strategy_doc.write_text(archive_path.read_text(encoding="utf-8"), encoding="utf-8")
        state["current_strategy_id"] = best["strategy_id"]
        state["consecutive_stagnations"] = 0
        save_state(state)
        print(f"RESTORED: {archive_path} -> {strategy_doc.name}")
        print(f"Active strategy reverted to: {best['strategy_id']}")
    else:
        print(f"WARNING: Archive not found at {archive_path}")
        print("Cannot restore. Strategy document must be manually recreated.")


# ── set ───────────────────────────────────────────────────────────────────────

def cmd_set(args):
    """Set a key in state.json."""
    state = load_state()
    # Try to parse value as number or bool
    val = args.value
    if val.lower() == "true":
        val = True
    elif val.lower() == "false":
        val = False
    elif val.lower() == "null" or val.lower() == "none":
        val = None
    else:
        try:
            # Keep as int only if there's no decimal point in the original string
            if "." in val:
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass  # keep as string

    state[args.key] = val
    save_state(state)
    print(f"Set {args.key} = {val}")


# ── get ───────────────────────────────────────────────────────────────────────

def cmd_get(args):
    """Read a single key from per-GPU state."""
    state = load_state()
    key = args.key
    if key not in state:
        print(f"ERROR: Key '{key}' not found in {_state_file().name}", file=sys.stderr)
        print(f"Available keys: {', '.join(sorted(state.keys()))}", file=sys.stderr)
        sys.exit(1)
    val = state[key]
    print(f"{key}: {val}")


# ── show ──────────────────────────────────────────────────────────────────────

def cmd_show(args):
    """Print current state summary."""
    state = load_state()
    pop = load_population()
    strats = load_strategies()

    print(f"=== EvoX State (GPU {get_gpu_index()}, {_state_file().name}) ===")
    print(f"Session: {state.get('session_id', 'unknown')}")
    print(f"Phase: {state.get('phase', 'unknown')}")
    print(f"Strategy: {state.get('current_strategy_id', 'unknown')}")
    print(f"Window: {state.get('window_iteration', 0)}/{state.get('window_size', 6)} (window #{state.get('window_count', 0)})")
    print(f"Total evaluations: {state.get('total_evaluations', 0)}")
    print(f"Master: hash={state.get('master_hash', 'unknown')}, val_bpb={state.get('master_val_bpb', 'unknown')}")
    print(f"GPU: {state.get('gpu_index', 0)}")
    print(f"tau: {state.get('tau', 0.001)}")
    print(f"Consecutive stagnations: {state.get('consecutive_stagnations', 0)}")

    if pop:
        local = [c for c in pop if c["source"] == "local"]
        swarm = [c for c in pop if c["source"] == "swarm"]
        best = min(pop, key=lambda c: c["val_bpb"])
        print(f"\nPopulation: {len(local)} local + {len(swarm)} swarm = {len(pop)} total")
        print(f"Best: {best['id']} val_bpb={best['val_bpb']:.6f}")
    else:
        print("\nPopulation: empty")

    if strats:
        print(f"\nStrategies: {len(strats)} in history")
        for s in strats:
            j = f"{s['J_score']:.6f}" if s.get('J_score') is not None else "N/A"
            gpu_tag = f" [GPU {s.get('gpu_index', '?')}]" if s.get("gpu_index") else ""
            print(f"  {s['strategy_id']}: J={j}, windows={s.get('windows_active', 0)}{gpu_tag}")


# ── migrate ──────────────────────────────────────────────────────────────────

def cmd_migrate(args):
    """Migrate single-GPU state files to multi-GPU naming scheme.

    WARNING: Run this with all other GPU instances stopped. This command
    modifies shared files (population.json, strategies.json) without
    file locking to avoid partial migration under lock contention.
    """
    print("WARNING: Ensure all other GPU instances are stopped before migrating.")
    gp = gpu_prefix()
    migrated = []

    # Migrate state.json -> state_gpuN.json
    old_state = EVOX_DIR / "state.json"
    new_state = _state_file()
    if old_state.exists() and not new_state.exists():
        old_state.rename(new_state)
        migrated.append(f"  {old_state.name} -> {new_state.name}")

    # Migrate current_strategy.md -> current_strategy_gpuN.md
    old_strategy = EVOX_DIR / "current_strategy.md"
    new_strategy = _strategy_doc()
    if old_strategy.exists() and not new_strategy.exists():
        old_strategy.rename(new_strategy)
        migrated.append(f"  {old_strategy.name} -> {new_strategy.name}")

    # Migrate candidate IDs and directories
    pop = load_population()
    renamed_dirs = 0
    for c in pop:
        old_id = c["id"]
        if old_id.startswith("cand_") and not old_id.startswith("cand_g"):
            num = old_id.removeprefix("cand_")
            new_id = f"cand_{gp}_{num}"
            old_dir = CANDIDATES_DIR / old_id
            new_dir = CANDIDATES_DIR / new_id
            if old_dir.exists() and not new_dir.exists():
                old_dir.rename(new_dir)
                renamed_dirs += 1
            c["id"] = new_id

    # Update parent_id references
    for c in pop:
        parent = c.get("parent_id", "")
        if parent.startswith("cand_") and not parent.startswith("cand_g"):
            num = parent.removeprefix("cand_")
            c["parent_id"] = f"cand_{gp}_{num}"

    # Update strategy_id field inside each candidate record
    for c in pop:
        sid = c.get("strategy_id", "")
        if sid.startswith("S_") and not sid.startswith("S_g"):
            num = sid.removeprefix("S_")
            c["strategy_id"] = f"S_{gp}_{num}"

    if renamed_dirs:
        migrated.append(f"  {renamed_dirs} candidate directories renamed with {gp} prefix")
    save_json(POPULATION_FILE, pop)

    # Migrate strategy IDs in strategies.json
    strats = load_json(STRATEGIES_FILE, [])
    for s in strats:
        sid = s["strategy_id"]
        if sid.startswith("S_") and not sid.startswith("S_g"):
            num = sid.removeprefix("S_")
            s["strategy_id"] = f"S_{gp}_{num}"
        psid = s.get("parent_strategy_id") or ""
        if psid.startswith("S_") and not psid.startswith("S_g"):
            num = psid.removeprefix("S_")
            s["parent_strategy_id"] = f"S_{gp}_{num}"
        # Also rename strategy archive files
        old_archive = STRATEGIES_DIR / f"{sid}.md"
        new_archive = STRATEGIES_DIR / f"{s['strategy_id']}.md"
        if old_archive.exists() and not new_archive.exists() and sid != s["strategy_id"]:
            old_archive.rename(new_archive)
    save_json(STRATEGIES_FILE, strats)

    # Update current_strategy_id in per-GPU state
    state = load_state()
    sid = state.get("current_strategy_id", "")
    if sid.startswith("S_") and not sid.startswith("S_g"):
        num = sid.removeprefix("S_")
        state["current_strategy_id"] = f"S_{gp}_{num}"
        save_state(state)

    if migrated:
        print("Migration complete:")
        for m in migrated:
            print(m)
    else:
        print("Nothing to migrate (already using multi-GPU naming or no legacy files found).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EvoX state manager for autolab (multi-GPU)")
    sub = parser.add_subparsers(dest="command")

    # init
    p = sub.add_parser("init")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.001)
    p.add_argument("--window-size", type=int, default=6)

    # add-candidate
    p = sub.add_parser("add-candidate")
    p.add_argument("--val-bpb", type=float, required=True)
    p.add_argument("--parent", type=str, default=None)
    p.add_argument("--operator", type=str, required=True, choices=["REFINE", "DIVERGE", "FREE"])
    p.add_argument("--hypothesis", type=str, required=True)
    p.add_argument("--master-hash", type=str, default=None)
    p.add_argument("--strategy-id", type=str, default=None)
    p.add_argument("--submitted", action="store_true")

    # get-parent
    p = sub.add_parser("get-parent")
    p.add_argument("--method", type=str, default="best", choices=["best", "tournament", "random"])

    # get-inspiration
    p = sub.add_parser("get-inspiration")
    p.add_argument("--count", type=int, default=3)

    # select-operator
    p = sub.add_parser("select-operator")
    p.add_argument("--weights", type=str, default="40,40,20", help="REFINE,DIVERGE,FREE weights")

    # start-window
    sub.add_parser("start-window")

    # advance-window
    sub.add_parser("advance-window")

    # check-stagnation
    sub.add_parser("check-stagnation")

    # import-swarm
    p = sub.add_parser("import-swarm")
    p.add_argument("--id", type=str, required=True)
    p.add_argument("--val-bpb", type=float, required=True)
    p.add_argument("--message", type=str, required=True)
    p.add_argument("--master-hash", type=str, required=True)

    # record-strategy
    p = sub.add_parser("record-strategy")
    p.add_argument("--id", type=str, required=True)
    p.add_argument("--parent-id", type=str, default=None)
    p.add_argument("--description", type=str, required=True)

    # score-strategy
    sub.add_parser("score-strategy")

    # get-best-strategy
    sub.add_parser("get-best-strategy")

    # get
    p = sub.add_parser("get")
    p.add_argument("--key", type=str, required=True)

    # set
    p = sub.add_parser("set")
    p.add_argument("--key", type=str, required=True)
    p.add_argument("--value", type=str, required=True)

    # show
    sub.add_parser("show")

    # migrate
    sub.add_parser("migrate")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "add-candidate": cmd_add_candidate,
        "get-parent": cmd_get_parent,
        "get-inspiration": cmd_get_inspiration,
        "select-operator": cmd_select_operator,
        "start-window": cmd_start_window,
        "advance-window": cmd_advance_window,
        "check-stagnation": cmd_check_stagnation,
        "import-swarm": cmd_import_swarm,
        "record-strategy": cmd_record_strategy,
        "score-strategy": cmd_score_strategy,
        "get-best-strategy": cmd_get_best_strategy,
        "get": cmd_get,
        "set": cmd_set,
        "show": cmd_show,
        "migrate": cmd_migrate,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

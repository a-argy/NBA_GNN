"""
Local pipeline runner for SwishNet Phase 1 experiments.

Chains three steps automatically:
  Step 1  Shot data generation   raw game JSONs → shots_data/*.json
  Step 2  Graph construction     shots_data/*.json → graph_data/*.pt
  Step 3  Phase 1 training       graph_data/*.pt → results/

All intermediate outputs are written to colab/data/ so they can be reused
across sessions.  Each step is skipped if its outputs already exist
(use --force to re-run all steps).

Usage:
  python colab/phase1/local_run.py            # run all three steps
  python colab/phase1/local_run.py --force    # re-run even if outputs exist
"""

import os
import sys
import glob
import json
import argparse

# ── Path setup ────────────────────────────────────────────────────────────────
_PHASE1_DIR = os.path.dirname(os.path.abspath(__file__))
_COLAB_DIR  = os.path.dirname(_PHASE1_DIR)
for _p in (_COLAB_DIR, _PHASE1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Local data paths ──────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(_COLAB_DIR, 'data')
GAME_DATA_DIR = os.path.join(DATA_DIR, 'game_data')
SHOTS_DIR     = os.path.join(DATA_DIR, 'shots_data')
GRAPH_DIR     = os.path.join(DATA_DIR, 'graph_data')
CSV_PATH      = os.path.join(DATA_DIR, 'supplemental_data', 'player_shooting_stats_2016.csv')
PBP_PATH      = os.path.join(DATA_DIR, 'supplemental_data', 'pbp_cache.csv')

# Shot extraction parameters (same as generate_shot_data.py)
RIM_HEIGHT          = 9.7
RIM_XY_THRESHOLD    = 4.0
PLAYER_XYZ_THRESHOLD = 9.0
MIN_RELEASE_HEIGHT  = 6.5
NUM_SNAPSHOTS       = 3
MOMENTS_BACK        = [3, 12]
MIN_SHOTS_THRESHOLD = 90   # games with fewer shots are skipped


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Shot data generation
# ══════════════════════════════════════════════════════════════════════════════

def run_step1(force=False):
    """Generate shots_data/*.json from raw game JSONs using load_with_context."""
    import pandas as pd
    from load_with_context import (
        load_shot_attempts,
        create_empty_drop_stats,
        create_empty_calibration_errors,
    )

    os.makedirs(SHOTS_DIR, exist_ok=True)
    game_files = sorted(glob.glob(os.path.join(GAME_DATA_DIR, '*.json')))
    if not game_files:
        raise FileNotFoundError(f"No game JSON files found in {GAME_DATA_DIR}")

    existing = sorted(glob.glob(os.path.join(SHOTS_DIR, 'shots_*.json')))
    if existing and not force:
        print(f"  Step 1 skipped — {len(existing)} shot files already exist in {SHOTS_DIR}")
        return len(existing)

    print(f"  Processing {len(game_files)} game files ...")
    pbp = pd.read_csv(PBP_PATH)

    total_shots = 0
    saved = 0
    for gf in game_files:
        drop_stats          = create_empty_drop_stats()
        calibration_errors  = create_empty_calibration_errors()
        shots = load_shot_attempts(
            gf, pbp, drop_stats, calibration_errors,
            rim_height=RIM_HEIGHT,
            rim_xy_threshold=RIM_XY_THRESHOLD,
            player_xyz_threshold=PLAYER_XYZ_THRESHOLD,
            min_release_height=MIN_RELEASE_HEIGHT,
            num_snapshots=NUM_SNAPSHOTS,
            moments_back=MOMENTS_BACK,
        )
        total_shots += len(shots)
        name = os.path.basename(gf)
        if shots and len(shots) >= MIN_SHOTS_THRESHOLD:
            out_path = os.path.join(SHOTS_DIR, f'shots_{name}')
            with open(out_path, 'w') as fh:
                json.dump(shots, fh)
            print(f"    ✓  {name}  →  {len(shots)} shots")
            saved += 1
        else:
            print(f"    ⚠  {name}  →  {len(shots)} shots (below threshold, skipped)")

    print(f"  Done — {saved} shot files saved, {total_shots} total shots")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Graph construction
# ══════════════════════════════════════════════════════════════════════════════

def run_step2(force=False):
    """Build graph .pt files from shot data JSONs using build_graph.py."""
    from build_graph import process_single_game, print_game_result

    os.makedirs(GRAPH_DIR, exist_ok=True)
    shot_files = sorted(glob.glob(os.path.join(SHOTS_DIR, 'shots_*.json')))
    if not shot_files:
        raise FileNotFoundError(f"No shot files found in {SHOTS_DIR} — run step 1 first")

    existing = sorted(glob.glob(os.path.join(GRAPH_DIR, 'graphs_*.pt')))
    if existing and not force:
        print(f"  Step 2 skipped — {len(existing)} graph files already exist in {GRAPH_DIR}")
        return len(existing)

    print(f"  Building graphs from {len(shot_files)} shot files ...")
    total_graphs = 0
    for sf in shot_files:
        result = process_single_game(sf, CSV_PATH, GRAPH_DIR)
        print_game_result(result)
        total_graphs += result['num_graphs']

    print(f"\n  Done — {total_graphs} graphs built across {len(shot_files)} games")
    return len(shot_files)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Phase 1 training
# ══════════════════════════════════════════════════════════════════════════════

def run_step3():
    """Run all Phase 1 experiments."""
    # Point train_phase1.py at the local data directory
    os.environ['SWISHNET_BASE_DIR'] = DATA_DIR

    from train_phase1 import main as run_experiments
    return run_experiments()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SwishNet Phase 1 local runner')
    parser.add_argument('--force', action='store_true',
                        help='Re-run all steps even if outputs already exist')
    parser.add_argument('--step', type=int, choices=[1, 2, 3],
                        help='Run only a single step (1=shots, 2=graphs, 3=train)')
    args = parser.parse_args()

    print("=" * 65)
    print("SwishNet — Phase 1 Local Pipeline")
    print("=" * 65)
    print(f"  Game data  : {GAME_DATA_DIR}")
    print(f"  Shots dir  : {SHOTS_DIR}")
    print(f"  Graphs dir : {GRAPH_DIR}")
    print(f"  Results dir: {_PHASE1_DIR}/results")

    if args.step is None or args.step == 1:
        print("\n── Step 1: Shot data generation ──────────────────────────────")
        run_step1(force=args.force)

    if args.step is None or args.step == 2:
        print("\n── Step 2: Graph construction ────────────────────────────────")
        run_step2(force=args.force)

    if args.step is None or args.step == 3:
        print("\n── Step 3: Phase 1 training ──────────────────────────────────")
        run_step3()

    print("\n" + "=" * 65)
    print("Pipeline complete.")


if __name__ == '__main__':
    main()

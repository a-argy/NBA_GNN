#!/usr/bin/env python3
"""
SwishNet full-season data pipeline.

Steps:
  0 — Download game JSON files (optional; skip if already present)
  1 — Shot extraction  : game JSON → shots_GAME.json
  2 — Linked graph + feature construction (fair comparison guarantee)
  3 — Aggregate baseline arrays (X_static.npy, X_full.npy, y.npy)
  4 — Statistics report

Fair comparison design:
  A shot appears in BOTH the GNN dataset and the baseline dataset only if
  graph construction AND feature extraction BOTH succeed for that shot.
  Row i of X_static.npy is guaranteed to correspond to graph i in graphs_GAME.pt.

Usage:
  # Full run (all steps):
  python colab/pipeline/run_full_pipeline.py --data-dir /data --all

  # Specific steps:
  python colab/pipeline/run_full_pipeline.py --data-dir /data --step 2 --step 4

  # With GCS upload after completion:
  python colab/pipeline/run_full_pipeline.py --data-dir /data --all --upload

  # Download from GCS instead of GitHub:
  python colab/pipeline/run_full_pipeline.py --data-dir /data --all --source gcs

  # Local smoke test (12 existing games, skip download):
  python colab/pipeline/run_full_pipeline.py --data-dir colab/data --step 2 --step 4
"""

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# ── resolve project root so we can import sibling colab/ modules ────────────
_THIS_DIR = Path(__file__).resolve().parent          # colab/pipeline/
_COLAB_DIR = _THIS_DIR.parent                        # colab/
_ROOT_DIR = _COLAB_DIR.parent                        # project root
sys.path.insert(0, str(_COLAB_DIR))

from load_with_context import (
    load_shot_attempts,
    create_empty_drop_stats,
    create_empty_calibration_errors,
)
from build_graph import build_graph_from_shot, set_player_stats
from baseline_features import extract_static_features, extract_velocity_features
from scrape_player_stats import build_player_stats_for_game
from pipeline.gcp_utils import (
    upload_file,
    download_file,
    sync_dir,
    file_exists_on_gcs,
    DEFAULT_BUCKET,
)
from pipeline.stats_report import generate_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SHOTS_PER_GAME = 90  # games with fewer extracted shots are skipped

# GCS paths
GCS_RAW_GAME_DATA = "raw/game_data"
GCS_PBP = "raw/pbp_cache.csv"
GCS_PLAYER_STATS = "raw/player_shooting_stats_2016.csv"
GCS_SHOTS_DATA = "processed/shots_data"
GCS_GRAPH_DATA = "processed/graph_data"
GCS_BASELINE_DATA = "processed/baseline_data"
GCS_PIPELINE_STATS = "processed/pipeline_stats"

# ---------------------------------------------------------------------------
# Temporal variant configurations (Phase 3 — graph construction experiments)
# ---------------------------------------------------------------------------
# Baseline (Exp 3): MOMENTS_BACK=[3,12], 3 timesteps, 0.48s horizon — already built.
# Variants isolate two axes:
#   horizon  (how far back): short=0.48s (12 frames) vs long=2.0s (50 frames)
#   density  (how many steps): sparse=3 timesteps vs dense=5 timesteps
#
# Comparison matrix:
#            | Sparse (3 steps) | Dense (5 steps) |
#   Short    | Exp 3 baseline   | Variant A       |  A vs baseline: does density help?
#   Long     | Variant B        | Variant C       |  B vs baseline: does horizon help?
#                                                    C vs A+B: do both together help?
VARIANTS = {
    'A': {
        'name': 'temporal_dense_short',
        'moments_back': [3, 6, 9, 12],
        'num_snapshots': 5,
        'description': '5 timesteps, 0.48s horizon, uniform 0.12s spacing — tests resolution',
    },
    'B': {
        'name': 'temporal_sparse_long',
        'moments_back': [12, 50],
        'num_snapshots': 3,
        'description': '3 timesteps, 2.0s horizon, sparse — tests horizon length',
    },
    'C': {
        'name': 'temporal_dense_long',
        'moments_back': [6, 12, 25, 50],
        'num_snapshots': 5,
        'description': '5 timesteps, 2.0s horizon — tests resolution + horizon combined',
    },
}


# ---------------------------------------------------------------------------
# Step 0 — Download
# ---------------------------------------------------------------------------

def step0_download(data_dir: str, source: str) -> None:
    """Download game JSON files.

    Args:
        data_dir: Local root data directory.
        source:   'github' (default) or 'gcs'.
    """
    game_data_dir = os.path.join(data_dir, "game_data")
    os.makedirs(game_data_dir, exist_ok=True)

    if source == "gcs":
        print("\n[Step 0] Syncing game data from GCS …")
        # List objects under the GCS prefix and download missing ones.
        try:
            from google.cloud import storage as gcs_storage
            client = gcs_storage.Client()
            blobs = client.list_blobs(DEFAULT_BUCKET, prefix=GCS_RAW_GAME_DATA + "/")
            for blob in blobs:
                fname = os.path.basename(blob.name)
                if not fname.endswith(".json"):
                    continue
                local_path = os.path.join(game_data_dir, fname)
                if os.path.exists(local_path):
                    continue
                print(f"  Downloading {fname} …")
                blob.download_to_filename(local_path)
        except ImportError:
            # gsutil fallback
            from pipeline.gcp_utils import _gsutil
            _gsutil([
                "-m", "rsync", "-r",
                f"gs://{DEFAULT_BUCKET}/{GCS_RAW_GAME_DATA}",
                game_data_dir,
            ])
    else:
        print("\n[Step 0] Downloading game data from GitHub …")
        try:
            from download_season_data import download_nba_season_data
        except ImportError as e:
            sys.exit(f"Cannot import download_season_data: {e}")
        download_nba_season_data(output_dir=game_data_dir)

    n = len([f for f in os.listdir(game_data_dir) if f.endswith(".json")])
    print(f"[Step 0] Done. {n} game JSON files in {game_data_dir}")


# ---------------------------------------------------------------------------
# Step 1 — Shot extraction
# ---------------------------------------------------------------------------

def step1_extract_shots(
    data_dir: str,
    upload: bool = False,
) -> dict:
    """Extract shot attempts from every game JSON.

    Returns a dict of per-game metadata used by subsequent steps:
      {game_id: {'shot_file': path, 'n_shots': int}}
    """
    import pandas as pd

    game_data_dir = os.path.join(data_dir, "game_data")
    shot_data_dir = os.path.join(data_dir, "shot_data")
    os.makedirs(shot_data_dir, exist_ok=True)

    pbp_path = _find_supplemental(data_dir, "pbp_cache.csv")
    if pbp_path is None:
        sys.exit("pbp_cache.csv not found under data_dir. Download it first.")

    print(f"\n[Step 1] Loading PBP from {pbp_path} …")
    pbp = pd.read_csv(pbp_path)
    print(f"[Step 1] PBP rows: {len(pbp):,}")

    game_jsons = sorted(
        f for f in os.listdir(game_data_dir) if f.endswith(".json")
    )
    print(f"[Step 1] Processing {len(game_jsons)} game files …\n")

    # Aggregate drop stats across all games
    agg_drop_stats = create_empty_drop_stats()
    agg_calib_errors = create_empty_calibration_errors()

    game_meta = {}           # game_id → {shot_file, n_shots}
    games_below_thresh = 0
    games_calib_failed = 0
    games_calibrated = 0

    for idx, fname in enumerate(game_jsons):
        json_path = os.path.join(game_data_dir, fname)
        game_id = _game_id_from_filename(fname)

        shot_file = os.path.join(shot_data_dir, f"shots_{game_id}.json")

        # Skip if already done
        if os.path.exists(shot_file):
            with open(shot_file) as f:
                shots = json.load(f)
            n = len(shots)
            game_meta[game_id] = {"shot_file": shot_file, "n_shots": n}
            games_calibrated += 1
            print(f"  [{idx+1}/{len(game_jsons)}] {game_id}: SKIPPED (already extracted, {n} shots)")
            continue

        # Per-game drop stats
        drop_stats = create_empty_drop_stats()
        calib_errors = create_empty_calibration_errors()

        try:
            shots = load_shot_attempts(
                json_file=json_path,
                pbp=pbp,
                drop_stats=drop_stats,
                calibration_errors=calib_errors,
            )
        except Exception as exc:
            print(f"  [{idx+1}/{len(game_jsons)}] {game_id}: ERROR — {exc}")
            games_calib_failed += 1
            _merge_drop_stats(agg_drop_stats, drop_stats)
            _merge_calib_errors(agg_calib_errors, calib_errors)
            continue

        # Check calibration result
        if calib_errors.get("fallback_failed"):
            games_calib_failed += 1
        else:
            games_calibrated += 1

        # Merge per-game stats into aggregate
        _merge_drop_stats(agg_drop_stats, drop_stats)
        _merge_calib_errors(agg_calib_errors, calib_errors)

        if len(shots) < MIN_SHOTS_PER_GAME:
            print(
                f"  [{idx+1}/{len(game_jsons)}] {game_id}: "
                f"BELOW THRESHOLD ({len(shots)} shots < {MIN_SHOTS_PER_GAME})"
            )
            agg_drop_stats["below_threshold"] = (
                agg_drop_stats.get("below_threshold", 0) + 1
            )
            games_below_thresh += 1
            continue

        # Save
        with open(shot_file, "w") as f:
            json.dump(shots, f)

        game_meta[game_id] = {"shot_file": shot_file, "n_shots": len(shots)}

        if upload:
            upload_file(shot_file, f"{GCS_SHOTS_DATA}/shots_{game_id}.json")

        print(
            f"  [{idx+1}/{len(game_jsons)}] {game_id}: "
            f"{len(shots)} shots extracted"
        )

    print(
        f"\n[Step 1] Done."
        f"  Games: {len(game_jsons)} total, "
        f"{games_calibrated} calibrated, "
        f"{games_below_thresh} below threshold, "
        f"{games_calib_failed} calibration failed."
    )

    return {
        "game_meta": game_meta,
        "agg_drop_stats": agg_drop_stats,
        "agg_calib_errors": agg_calib_errors,
        "games_processed": len(game_jsons),
        "games_calibrated": games_calibrated,
        "games_below_threshold": games_below_thresh,
        "games_calibration_failed": games_calib_failed,
    }


# ---------------------------------------------------------------------------
# Step 2 — Linked graph + feature construction
# ---------------------------------------------------------------------------

def step2_build_graphs_and_features(
    data_dir: str,
    game_meta: dict,
    upload: bool = False,
) -> dict:
    """Build graphs and baseline features for every extracted shot.

    For each shot, both build_graph_from_shot() AND extract_static_features()
    must succeed for the shot to be included in the final dataset.
    This guarantees row i of X_static.npy == graph i in graphs_GAME.pt.

    Returns a summary dict used by step3 and step4.
    """
    csv_path = _find_supplemental(data_dir, "player_shooting_stats_2016.csv")
    if csv_path is None:
        sys.exit(
            "player_shooting_stats_2016.csv not found under data_dir. "
            "Upload it to the data directory first."
        )

    graph_data_dir = os.path.join(data_dir, "graph_data")
    baseline_dir = os.path.join(data_dir, "baseline_data", "per_game")
    os.makedirs(graph_data_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)

    stage2_drops = {
        "missing_field": 0,
        "invalid_player_count": 0,
        "missing_player_stats": 0,
        "shooter_not_found": 0,
        "prev_moment_missing_field": 0,
        "prev_moment_none_field": 0,
        "prev_moment_invalid_player_count": 0,
        "prev_moment_shooter_not_found": 0,
    }
    stage3_linking = {
        "both_ok": 0,
        "graph_failed_only": 0,
        "feature_failed_only": 0,
        "both_failed": 0,
    }
    unmatched_counter: Counter = Counter()
    total_made = 0
    total_missed = 0
    stage2_graphs_built = 0

    game_ids = list(game_meta.keys())
    print(f"\n[Step 2] Building graphs + features for {len(game_ids)} games …\n")

    for g_idx, game_id in enumerate(game_ids):
        meta = game_meta[game_id]
        shot_file = meta["shot_file"]

        graph_file = os.path.join(graph_data_dir, f"graphs_{game_id}.pt")
        x_static_file = os.path.join(baseline_dir, f"X_static_{game_id}.npy")

        # Skip if already done (resumable)
        if os.path.exists(graph_file) and os.path.exists(x_static_file):
            # Count graphs from saved file for aggregate totals
            saved = torch.load(graph_file, weights_only=False)
            n = len(saved)
            stage2_graphs_built += n
            stage3_linking["both_ok"] += n
            total_made += sum(int(g.y.item()) for g in saved)
            total_missed += n - sum(int(g.y.item()) for g in saved)
            print(
                f"  [{g_idx+1}/{len(game_ids)}] {game_id}: "
                f"SKIPPED ({n} graphs already built)"
            )
            continue

        # Load shot data
        try:
            with open(shot_file) as f:
                shots = json.load(f)
        except Exception as exc:
            print(f"  [{g_idx+1}/{len(game_ids)}] {game_id}: ERROR loading shots — {exc}")
            continue

        if not shots:
            print(f"  [{g_idx+1}/{len(game_ids)}] {game_id}: empty shots file, skipping")
            continue

        # Build player stats for this game (sets global in build_graph.py)
        try:
            player_stats, unmatched, _ = build_player_stats_for_game(
                first_shot=shots[0], csv_path=csv_path
            )
        except Exception as exc:
            print(
                f"  [{g_idx+1}/{len(game_ids)}] {game_id}: "
                f"ERROR building player stats — {exc}"
            )
            continue

        if not player_stats:
            print(
                f"  [{g_idx+1}/{len(game_ids)}] {game_id}: "
                "empty player stats, skipping"
            )
            continue

        # Track unmatched players
        for u in unmatched:
            name = u.get("original_name") or u.get("player_id", "unknown")
            unmatched_counter[name] += 1

        # Set global player stats used by build_graph_from_shot
        set_player_stats(player_stats)

        game_graphs = []
        x_static_rows = []
        x_full_rows = []
        y_rows = []

        for shot_idx, shot in enumerate(shots):
            # ── Graph ──────────────────────────────────────────────────────
            graph, skip_reason = build_graph_from_shot(shot)
            graph_ok = graph is not None

            if not graph_ok and skip_reason is not None:
                reason_key = skip_reason.get("reason", "unknown")
                if reason_key in stage2_drops:
                    stage2_drops[reason_key] += 1

            # ── Features ───────────────────────────────────────────────────
            feat_ok = False
            static_feats = None
            vel_feats = None
            try:
                static_feats = extract_static_features(shot, player_stats)
                vel_feats = extract_velocity_features(shot)
                feat_ok = static_feats is not None
            except Exception:
                feat_ok = False

            # ── Cross-stage linking ────────────────────────────────────────
            if graph_ok and feat_ok:
                stage3_linking["both_ok"] += 1
                label = 1 if shot["made"] else 0

                # Attach label + metadata to graph
                graph.y = torch.tensor([label], dtype=torch.long)
                graph.shooter_id = shot.get("shooter_id")
                graph.game_id = game_id
                graph.quarter = shot.get("quarter")
                graph.game_clock = shot.get("game_clock")
                graph.shot_clock = shot.get("shot_clock")

                game_graphs.append(graph)
                x_static_rows.append(static_feats)
                x_full_rows.append(np.concatenate([static_feats, vel_feats]))
                y_rows.append(label)

                if label == 1:
                    total_made += 1
                else:
                    total_missed += 1
            elif graph_ok and not feat_ok:
                stage3_linking["graph_failed_only"] += 1
            elif not graph_ok and feat_ok:
                stage3_linking["feature_failed_only"] += 1
            else:
                stage3_linking["both_failed"] += 1

        n_built = len(game_graphs)
        stage2_graphs_built += n_built

        if n_built == 0:
            print(
                f"  [{g_idx+1}/{len(game_ids)}] {game_id}: "
                "no graphs built, skipping save"
            )
            continue

        # Save graph list
        torch.save(game_graphs, graph_file)

        # Save per-game baseline arrays
        np.save(x_static_file, np.array(x_static_rows, dtype=np.float32))
        np.save(
            os.path.join(baseline_dir, f"X_full_{game_id}.npy"),
            np.array(x_full_rows, dtype=np.float32),
        )
        np.save(
            os.path.join(baseline_dir, f"y_{game_id}.npy"),
            np.array(y_rows, dtype=np.int32),
        )

        if upload:
            upload_file(graph_file, f"{GCS_GRAPH_DATA}/graphs_{game_id}.pt")

        print(
            f"  [{g_idx+1}/{len(game_ids)}] {game_id}: "
            f"{n_built}/{len(shots)} shots → graphs"
        )

    print(
        f"\n[Step 2] Done. "
        f"Total graphs built: {stage2_graphs_built:,}  "
        f"({total_made:,} made / {total_missed:,} missed)"
    )

    return {
        "stage2_drops": stage2_drops,
        "stage3_linking": stage3_linking,
        "unmatched_players_counter": dict(unmatched_counter),
        "stage2_graphs_built": stage2_graphs_built,
        "total_made": total_made,
        "total_missed": total_missed,
        "baseline_dir": baseline_dir,
        "graph_data_dir": graph_data_dir,
    }


# ---------------------------------------------------------------------------
# Step 3 — Aggregate baseline arrays
# ---------------------------------------------------------------------------

def step3_aggregate_baseline(
    data_dir: str,
    step2_result: dict,
    upload: bool = False,
) -> dict:
    """Concatenate per-game .npy files into final dataset arrays.

    Asserts X_static.shape[0] == total_graphs (hard error on mismatch).
    """
    baseline_dir = step2_result["baseline_dir"]
    graph_data_dir = step2_result["graph_data_dir"]
    out_dir = os.path.join(data_dir, "baseline_data")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[Step 3] Aggregating baseline arrays from {baseline_dir} …")

    x_static_parts = []
    x_full_parts = []
    y_parts = []
    shot_index_map = []

    per_game_files = sorted(
        f for f in os.listdir(baseline_dir) if f.startswith("X_static_")
    )

    for fname in per_game_files:
        game_id = fname.replace("X_static_", "").replace(".npy", "")
        xs = np.load(os.path.join(baseline_dir, fname))
        xf = np.load(os.path.join(baseline_dir, f"X_full_{game_id}.npy"))
        yg = np.load(os.path.join(baseline_dir, f"y_{game_id}.npy"))

        n = xs.shape[0]
        for i in range(n):
            shot_index_map.append({"game_id": game_id, "shot_idx_in_game": i})

        x_static_parts.append(xs)
        x_full_parts.append(xf)
        y_parts.append(yg)

    if not x_static_parts:
        print("[Step 3] No per-game arrays found — skipping aggregation.")
        return {}

    X_static = np.concatenate(x_static_parts, axis=0).astype(np.float32)
    X_full = np.concatenate(x_full_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int32)

    total_graphs = step2_result.get("stage2_graphs_built", 0)

    # ── Hard assert: row count must equal graph count ───────────────────────
    if X_static.shape[0] != total_graphs:
        raise AssertionError(
            f"FATAL: X_static has {X_static.shape[0]:,} rows but "
            f"{total_graphs:,} graphs were built. "
            "Pipeline has a row-alignment bug — do NOT proceed."
        )

    print(
        f"[Step 3] X_static: {X_static.shape}  "
        f"X_full: {X_full.shape}  "
        f"y: {y.shape}"
    )

    # Save final arrays
    x_static_path = os.path.join(out_dir, "X_static.npy")
    x_full_path = os.path.join(out_dir, "X_full.npy")
    y_path = os.path.join(out_dir, "y.npy")
    map_path = os.path.join(out_dir, "shot_index_map.json")

    np.save(x_static_path, X_static)
    np.save(x_full_path, X_full)
    np.save(y_path, y)
    with open(map_path, "w") as f:
        json.dump(shot_index_map, f)

    print(f"[Step 3] Saved final arrays to {out_dir}")

    if upload:
        for path, gcs in [
            (x_static_path, f"{GCS_BASELINE_DATA}/X_static.npy"),
            (x_full_path, f"{GCS_BASELINE_DATA}/X_full.npy"),
            (y_path, f"{GCS_BASELINE_DATA}/y.npy"),
            (map_path, f"{GCS_BASELINE_DATA}/shot_index_map.json"),
        ]:
            upload_file(path, gcs)
        print("[Step 3] Uploaded baseline arrays to GCS.")

    return {
        "x_static_shape": X_static.shape,
        "x_full_shape": X_full.shape,
        "y_shape": y.shape,
    }


# ---------------------------------------------------------------------------
# Step 4 — Statistics report
# ---------------------------------------------------------------------------

def step4_report(
    data_dir: str,
    all_stats: dict,
    upload: bool = False,
) -> None:
    """Generate pipeline_report.txt and drop_stats.json."""
    stats_dir = os.path.join(data_dir, "pipeline_stats")
    print(f"\n[Step 4] Generating pipeline statistics report …")
    generate_report(all_stats, output_dir=stats_dir)

    if upload:
        for fname in ("pipeline_report.txt", "drop_stats.json"):
            local = os.path.join(stats_dir, fname)
            if os.path.exists(local):
                upload_file(local, f"{GCS_PIPELINE_STATS}/{fname}")
        print("[Step 4] Uploaded pipeline stats to GCS.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_supplemental(data_dir: str, filename: str):
    """Search common locations for a supplemental file."""
    candidates = [
        os.path.join(data_dir, filename),
        os.path.join(data_dir, "supplemental_data", filename),
        os.path.join(_COLAB_DIR, "data", "supplemental_data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _game_id_from_filename(fname: str) -> str:
    """Extract game ID from a filename like '0021500001.json'."""
    return os.path.splitext(fname)[0]


def _merge_drop_stats(agg: dict, per_game: dict) -> None:
    for key, val in per_game.items():
        if isinstance(val, (int, float)):
            agg[key] = agg.get(key, 0) + val


def _merge_calib_errors(agg: dict, per_game: dict) -> None:
    for key, val in per_game.items():
        if isinstance(val, list):
            if key not in agg:
                agg[key] = []
            agg[key].extend(val)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="SwishNet full-season data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root data directory (game_data/, shot_data/, etc. are created here)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps (0-4)",
    )
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        dest="steps",
        metavar="N",
        help="Run specific step(s). Can be specified multiple times: --step 2 --step 4",
    )
    parser.add_argument(
        "--source",
        choices=["github", "gcs"],
        default="github",
        help="Source for game data download (step 0 only). Default: github",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload outputs to GCS after each step",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"GCS bucket name. Default: {DEFAULT_BUCKET}",
    )
    parser.add_argument(
        "--variant",
        choices=["A", "B", "C"],
        default=None,
        help=(
            "Temporal graph variant to build. Overrides MOMENTS_BACK and NUM_SNAPSHOTS "
            "in load_with_context and routes GCS output to processed/variants/<name>/. "
            "A=dense-short, B=sparse-long, C=dense-long. "
            "Omit to use the values currently set in load_with_context.py."
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.all:
        steps = {0, 1, 2, 3, 4}
    elif args.steps:
        steps = set(args.steps)
    else:
        print("Specify --all or at least one --step N.", file=sys.stderr)
        sys.exit(1)

    data_dir = os.path.abspath(args.data_dir)
    upload = args.upload

    # ── Temporal variant: patch load_with_context globals + GCS output paths ─
    if args.variant:
        import load_with_context as _lwc
        vc = VARIANTS[args.variant]
        _lwc.MOMENTS_BACK = vc["moments_back"]
        _lwc.NUM_SNAPSHOTS = vc["num_snapshots"]
        prefix = f"processed/variants/{vc['name']}"
        global GCS_GRAPH_DATA, GCS_BASELINE_DATA, GCS_PIPELINE_STATS
        GCS_GRAPH_DATA    = f"{prefix}/graph_data"
        GCS_BASELINE_DATA = f"{prefix}/baseline_data"
        GCS_PIPELINE_STATS = f"{prefix}/pipeline_stats"
        print(
            f"[Variant {args.variant}] {vc['description']}\n"
            f"  MOMENTS_BACK={vc['moments_back']}, NUM_SNAPSHOTS={vc['num_snapshots']}\n"
            f"  GCS output: {prefix}/"
        )

    print(f"SwishNet pipeline  |  data-dir: {data_dir}  |  steps: {sorted(steps)}")

    # Accumulated stats for the final report
    all_stats: dict = {
        "games_processed": 0,
        "games_calibrated": 0,
        "games_below_threshold": 0,
        "games_calibration_failed": 0,
        "stage1_drops": create_empty_drop_stats(),
        "stage1_extracted": 0,
        "stage2_drops": {},
        "stage2_graphs_built": 0,
        "stage3_linking": {},
        "unmatched_players_counter": {},
        "total_made": 0,
        "total_missed": 0,
    }

    game_meta: dict = {}  # populated by step 1, consumed by step 2

    # ── Step 0 ──────────────────────────────────────────────────────────────
    if 0 in steps:
        step0_download(data_dir=data_dir, source=args.source)

    # ── Step 1 ──────────────────────────────────────────────────────────────
    if 1 in steps:
        result1 = step1_extract_shots(data_dir=data_dir, upload=upload)
        game_meta = result1.pop("game_meta")
        agg_drop = result1.pop("agg_drop_stats")
        all_stats.update(result1)
        all_stats["stage1_drops"] = agg_drop
        all_stats["stage1_extracted"] = sum(
            m["n_shots"] for m in game_meta.values()
        )
    else:
        # Reconstruct game_meta from existing shot files so step 2 can run
        shot_data_dir = os.path.join(data_dir, "shot_data")
        if os.path.isdir(shot_data_dir):
            for fname in os.listdir(shot_data_dir):
                if not fname.startswith("shots_") or not fname.endswith(".json"):
                    continue
                game_id = fname.replace("shots_", "").replace(".json", "")
                shot_file = os.path.join(shot_data_dir, fname)
                try:
                    with open(shot_file) as f:
                        shots = json.load(f)
                    game_meta[game_id] = {
                        "shot_file": shot_file,
                        "n_shots": len(shots),
                    }
                except Exception:
                    pass
            all_stats["stage1_extracted"] = sum(
                m["n_shots"] for m in game_meta.values()
            )
            print(
                f"[Step 1 skipped] Found {len(game_meta)} shot files "
                f"({all_stats['stage1_extracted']:,} shots)"
            )

    # ── Step 2 ──────────────────────────────────────────────────────────────
    if 2 in steps:
        if not game_meta:
            print(
                "[Step 2] No game metadata available. "
                "Run step 1 first or ensure shot_data/ exists.",
                file=sys.stderr,
            )
            sys.exit(1)
        result2 = step2_build_graphs_and_features(
            data_dir=data_dir,
            game_meta=game_meta,
            upload=upload,
        )
        all_stats["stage2_drops"] = result2["stage2_drops"]
        all_stats["stage3_linking"] = result2["stage3_linking"]
        all_stats["unmatched_players_counter"] = result2["unmatched_players_counter"]
        all_stats["stage2_graphs_built"] = result2["stage2_graphs_built"]
        all_stats["total_made"] = result2["total_made"]
        all_stats["total_missed"] = result2["total_missed"]
        step2_result = result2
    else:
        # Reconstruct minimal step2_result so step 3 can run
        baseline_per_game = os.path.join(data_dir, "baseline_data", "per_game")
        graph_data_dir = os.path.join(data_dir, "graph_data")
        step2_result = {
            "baseline_dir": baseline_per_game,
            "graph_data_dir": graph_data_dir,
            "stage2_graphs_built": 0,
        }
        # Count graphs from saved .pt files for the assert
        if os.path.isdir(graph_data_dir):
            total = 0
            for fname in os.listdir(graph_data_dir):
                if fname.endswith(".pt"):
                    try:
                        gs = torch.load(
                            os.path.join(graph_data_dir, fname),
                            weights_only=False,
                        )
                        total += len(gs)
                    except Exception:
                        pass
            step2_result["stage2_graphs_built"] = total
            all_stats["stage2_graphs_built"] = total
        print(
            f"[Step 2 skipped] Found "
            f"{step2_result['stage2_graphs_built']:,} graphs in {graph_data_dir}"
        )

    # ── Step 3 ──────────────────────────────────────────────────────────────
    if 3 in steps:
        step3_aggregate_baseline(
            data_dir=data_dir,
            step2_result=step2_result,
            upload=upload,
        )

    # ── Step 4 ──────────────────────────────────────────────────────────────
    if 4 in steps:
        step4_report(data_dir=data_dir, all_stats=all_stats, upload=upload)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

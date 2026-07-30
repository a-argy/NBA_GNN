#!/usr/bin/env python3
"""
Feature extraction for NBA shot prediction baselines.

Feature Set A — Static (38 features):
  7 geometry/game-state, 4 sorted offender distances, 5 sorted defender distances,
  1 defensive summary, 21 shooter historical stats.

Feature Set B — Velocity-aware (38 + 27 = 65 features):
  All of A plus ball/player velocities computed from previous_moments[0].

Imports from the existing pipeline:
  load_shot_attempts()   — load_with_context.py
  build_player_stats_for_game() — scrape_player_stats.py
"""

import math
import glob
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (override for GCP by patching these constants)
# ---------------------------------------------------------------------------
_COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_COLAB_DIR, "data")
GAME_DIR   = os.path.join(DATA_DIR, "game_data")
PBP_PATH   = os.path.join(DATA_DIR, "supplemental_data", "pbp_cache.csv")
CSV_PATH   = os.path.join(DATA_DIR, "supplemental_data", "player_shooting_stats_2016.csv")

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
if _COLAB_DIR not in sys.path:
    sys.path.insert(0, _COLAB_DIR)

from load_with_context import (
    load_shot_attempts,
    create_empty_drop_stats,
    create_empty_calibration_errors,
)
from scrape_player_stats import build_player_stats_for_game

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THREE_POINT_DISTANCE = 23.75   # feet
CONTEST_DISTANCE     = 6.0     # feet — "within 6 ft" defensive contest
_EPS                 = 1e-8    # avoid division by zero in velocity proxies

# ---------------------------------------------------------------------------
# Feature name lists (authoritative — used for column labels everywhere)
# ---------------------------------------------------------------------------
STATIC_FEATURE_NAMES: list = [
    # Geometry / game-state (7)
    "dist_to_rim",
    "ball_z",
    "angle_to_basket",
    "is_3pt",
    "shot_clock",
    "game_clock",
    "quarter",
    # Sorted shooter → other offensive player distances (4, nearest-first)
    "off_dist_1", "off_dist_2", "off_dist_3", "off_dist_4",
    # Sorted shooter → defender distances (5, nearest-first)
    "def_dist_1", "def_dist_2", "def_dist_3", "def_dist_4", "def_dist_5",
    # Defensive summary (1; def_dist_1 == closest_def_dist)
    "num_defs_within_6ft",
    # Shooter historical stats (21) — same order as GNN node features
    "age", "fg_pct", "avg_dist",
    "pct_fga_fg2a", "pct_fga_00_03", "pct_fga_03_10",
    "pct_fga_10_16", "pct_fga_16_xx", "pct_fga_fg3a",
    "fg2_pct", "fg_pct_00_03", "fg_pct_03_10",
    "fg_pct_10_16", "fg_pct_16_xx", "fg3_pct",
    "pct_ast_fg2", "pct_ast_fg3",
    "pct_fga_dunk", "pct_fga_corner3", "fg_pct_corner3",
    "games",
]  # 7 + 4 + 5 + 1 + 21 = 38

VELOCITY_FEATURE_NAMES: list = [
    # Ball motion (4)
    "ball_vx", "ball_vy", "ball_vz", "ball_speed_xy",
    # Shooter motion + launch angle (4)
    "shooter_vx", "shooter_vy", "shooter_speed", "ball_launch_angle_proxy",
    # Other offensive player speeds (4, same sort order as off_dist)
    "off_speed_1", "off_speed_2", "off_speed_3", "off_speed_4",
    # Defensive motion: (vx, vy, closing_speed) × 5 defenders (15, same sort as def_dist)
    "def_vx_1", "def_vy_1", "def_closing_speed_1",
    "def_vx_2", "def_vy_2", "def_closing_speed_2",
    "def_vx_3", "def_vy_3", "def_closing_speed_3",
    "def_vx_4", "def_vy_4", "def_closing_speed_4",
    "def_vx_5", "def_vy_5", "def_closing_speed_5",
]  # 4 + 4 + 4 + 15 = 27

assert len(STATIC_FEATURE_NAMES)   == 38, len(STATIC_FEATURE_NAMES)
assert len(VELOCITY_FEATURE_NAMES) == 27, len(VELOCITY_FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_shooter_pos(shot: dict):
    """Return (x, y) of the shooter, or None if not found in offense_position."""
    shooter_id = shot.get("shooter_id")
    for p in shot.get("offense_position", []):
        if p[1] == shooter_id:
            return (p[2], p[3])
    return None


def _sorted_offender_distances(offense_positions: list, shooter_id: int,
                                shooter_pos: tuple) -> list:
    """
    Distances from the shooter to each *other* offensive player, sorted nearest-first.
    Always returns exactly 4 values (padded with the last value if fewer than 4 others).
    """
    dists = sorted(
        math.sqrt((p[2] - shooter_pos[0])**2 + (p[3] - shooter_pos[1])**2)
        for p in offense_positions
        if p[1] != shooter_id
    )
    while len(dists) < 4:
        dists.append(dists[-1] if dists else 0.0)
    return dists[:4]


def _sorted_defender_distances(defense_positions: list,
                                shooter_pos: tuple) -> list:
    """
    Distances from the shooter to each defender, sorted nearest-first.
    Always returns exactly 5 values.
    """
    dists = sorted(
        math.sqrt((p[2] - shooter_pos[0])**2 + (p[3] - shooter_pos[1])**2)
        for p in defense_positions
    )
    while len(dists) < 5:
        dists.append(dists[-1] if dists else 0.0)
    return dists[:5]


def _sorted_defender_entries(defense_positions: list, shooter_pos: tuple) -> list:
    """
    Return defender entries sorted by distance to shooter (nearest-first).
    Each entry: (player_id, x, y, dist)
    """
    entries = [
        (p[1], p[2], p[3],
         math.sqrt((p[2] - shooter_pos[0])**2 + (p[3] - shooter_pos[1])**2))
        for p in defense_positions
    ]
    entries.sort(key=lambda e: e[3])
    while len(entries) < 5:
        entries.append(entries[-1] if entries else (None, 0.0, 0.0, 0.0))
    return entries[:5]


def _sorted_offender_entries(offense_positions: list, shooter_id: int,
                              shooter_pos: tuple) -> list:
    """
    Return *other* offensive player entries sorted by distance to shooter (nearest-first).
    Each entry: (player_id, x, y, dist)
    """
    entries = [
        (p[1], p[2], p[3],
         math.sqrt((p[2] - shooter_pos[0])**2 + (p[3] - shooter_pos[1])**2))
        for p in offense_positions
        if p[1] != shooter_id
    ]
    entries.sort(key=lambda e: e[3])
    while len(entries) < 4:
        entries.append(entries[-1] if entries else (None, 0.0, 0.0, 0.0))
    return entries[:4]


def _get_shooter_stats_features(shooter_id: int, player_stats_dict: dict):
    """
    Return 21 raw shooting stat values for the shooter.
    Returns None if the shooter is not in the stats dict at all.
    Zero-filled stats (for unmatched players) are accepted.
    """
    stats = player_stats_dict.get(str(shooter_id))
    if stats is None:
        stats = player_stats_dict.get(shooter_id)
    if stats is None:
        return None
    return [
        float(stats.get("age",             0.0)),
        float(stats.get("fg_pct",          0.0)),
        float(stats.get("avg_dist",        0.0)),
        float(stats.get("pct_fga_fg2a",    0.0)),
        float(stats.get("pct_fga_00_03",   0.0)),
        float(stats.get("pct_fga_03_10",   0.0)),
        float(stats.get("pct_fga_10_16",   0.0)),
        float(stats.get("pct_fga_16_xx",   0.0)),
        float(stats.get("pct_fga_fg3a",    0.0)),
        float(stats.get("fg2_pct",         0.0)),
        float(stats.get("fg_pct_00_03",    0.0)),
        float(stats.get("fg_pct_03_10",    0.0)),
        float(stats.get("fg_pct_10_16",    0.0)),
        float(stats.get("fg_pct_16_xx",    0.0)),
        float(stats.get("fg3_pct",         0.0)),
        float(stats.get("pct_ast_fg2",     0.0)),
        float(stats.get("pct_ast_fg3",     0.0)),
        float(stats.get("pct_fga_dunk",    0.0)),
        float(stats.get("pct_fga_corner3", 0.0)),
        float(stats.get("fg_pct_corner3",  0.0)),
        float(stats.get("games",           0)),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_static_features(shot: dict, player_stats_dict: dict):
    """
    Extract 38 static features from a shot dict.

    Returns:
        np.ndarray of shape (38,) on success, or None if the shot must be dropped
        (missing required field, wrong player count, shooter not found, or NaN).
    """
    # Validate required fields (None or missing → drop shot)
    for field in ("ball_position", "offense_position", "defense_position",
                  "target_rim", "shooter_id", "quarter", "game_clock"):
        if field not in shot or shot[field] is None:
            return None

    ball_pos     = shot["ball_position"]
    offense      = shot["offense_position"]
    defense      = shot["defense_position"]
    target_rim   = shot["target_rim"]
    shooter_id   = shot["shooter_id"]

    # Require exactly 5 offensive and 5 defensive players
    if len(offense) < 5 or len(defense) < 5:
        return None

    # Locate shooter
    shooter_pos = _get_shooter_pos(shot)
    if shooter_pos is None:
        return None

    ball_x, ball_y, ball_z = ball_pos[0], ball_pos[1], ball_pos[2]
    rim_x,  rim_y          = target_rim[0], target_rim[1]

    # --- Geometry / game-state (7) ---
    dist_to_rim     = math.sqrt((ball_x - rim_x)**2 + (ball_y - rim_y)**2)
    angle_to_basket = math.atan2(ball_y - rim_y, ball_x - rim_x)
    is_3pt          = 1.0 if dist_to_rim > THREE_POINT_DISTANCE else 0.0
    shot_clock      = float(shot["shot_clock"]) if shot.get("shot_clock") is not None else 0.0
    game_clock      = float(shot["game_clock"])
    quarter         = float(shot["quarter"])

    # --- Distances (9) ---
    off_dists = _sorted_offender_distances(offense, shooter_id, shooter_pos)
    def_dists = _sorted_defender_distances(defense, shooter_pos)

    # --- Defensive summary (1) ---
    num_defs_within_6ft = float(sum(1 for d in def_dists if d <= CONTEST_DISTANCE))

    # --- Shooter historical stats (21) ---
    player_stats = _get_shooter_stats_features(shooter_id, player_stats_dict)
    if player_stats is None:
        return None

    features = (
        [dist_to_rim, ball_z, angle_to_basket, is_3pt, shot_clock, game_clock, quarter]
        + off_dists
        + def_dists
        + [num_defs_within_6ft]
        + player_stats
    )

    arr = np.array(features, dtype=np.float64)
    if np.any(np.isnan(arr)):
        return None
    return arr


def extract_velocity_features(shot: dict) -> np.ndarray:
    """
    Extract 27 velocity features using previous_moments[0] as the reference.

    Velocity = (current_pos - prev_pos) / dt  [feet per second]
    where dt = prev_game_clock - curr_game_clock  (game clock counts DOWN).

    Returns np.ndarray of shape (27,); falls back to zeros on any failure.
    """
    zeros = np.zeros(27, dtype=np.float64)

    prev_moments = shot.get("previous_moments", [])
    if not prev_moments:
        return zeros

    prev = prev_moments[0]

    # Check required fields in previous moment
    if (prev.get("ball_position") is None
            or prev.get("offense_position") is None
            or prev.get("defense_position") is None
            or prev.get("game_clock") is None):
        return zeros

    # dt: seconds elapsed between the two snapshots
    dt = float(prev["game_clock"]) - float(shot["game_clock"])
    if dt <= 0:
        return zeros

    # --- Ball velocities (4) ---
    cb = shot["ball_position"]
    pb = prev["ball_position"]
    ball_vx       = (cb[0] - pb[0]) / dt
    ball_vy       = (cb[1] - pb[1]) / dt
    ball_vz       = (cb[2] - pb[2]) / dt
    ball_speed_xy = math.sqrt(ball_vx**2 + ball_vy**2)

    # Ball launch angle proxy: how steeply is the ball rising? (dimensionless)
    ball_launch_angle_proxy = ball_vz / (ball_speed_xy + _EPS)

    # --- Shooter velocities (3) ---
    shooter_id  = shot["shooter_id"]
    curr_offense = shot["offense_position"]
    prev_offense = prev["offense_position"]

    curr_shooter = next((p for p in curr_offense if p[1] == shooter_id), None)
    prev_shooter = next((p for p in prev_offense if p[1] == shooter_id), None)

    if curr_shooter is None or prev_shooter is None:
        return zeros

    shooter_vx    = (curr_shooter[2] - prev_shooter[2]) / dt
    shooter_vy    = (curr_shooter[3] - prev_shooter[3]) / dt
    shooter_speed = math.sqrt(shooter_vx**2 + shooter_vy**2)
    shooter_pos   = (curr_shooter[2], curr_shooter[3])

    # --- Other offensive player speeds (4, same sort order as static off_dist) ---
    curr_off_entries = _sorted_offender_entries(curr_offense, shooter_id, shooter_pos)
    prev_off_by_id   = {p[1]: (p[2], p[3]) for p in prev_offense}

    off_speeds = []
    for pid, cx, cy, _ in curr_off_entries:
        if pid is not None and pid in prev_off_by_id:
            px, py = prev_off_by_id[pid]
            vx = (cx - px) / dt
            vy = (cy - py) / dt
            off_speeds.append(math.sqrt(vx**2 + vy**2))
        else:
            off_speeds.append(0.0)

    # --- Defensive motion (15 = 3 × 5 defenders) ---
    curr_def_entries = _sorted_defender_entries(shot["defense_position"], shooter_pos)
    prev_def_by_id   = {p[1]: (p[2], p[3]) for p in prev["defense_position"]}

    def_vel_features = []
    for pid, cx, cy, dist_to_shooter in curr_def_entries:
        if pid is not None and pid in prev_def_by_id:
            px, py = prev_def_by_id[pid]
            dvx = (cx - px) / dt
            dvy = (cy - py) / dt
        else:
            dvx, dvy = 0.0, 0.0

        # Closing speed: positive means defender is moving toward the shooter
        dx = shooter_pos[0] - cx   # vector: defender → shooter
        dy = shooter_pos[1] - cy
        dist_ds = math.sqrt(dx**2 + dy**2)
        if dist_ds > 0:
            ux, uy        = dx / dist_ds, dy / dist_ds
            closing_speed = dvx * ux + dvy * uy
        else:
            closing_speed = 0.0

        def_vel_features.extend([dvx, dvy, closing_speed])

    # Pad if fewer than 5 defenders
    while len(def_vel_features) < 15:
        def_vel_features.append(0.0)

    features = (
        [ball_vx, ball_vy, ball_vz, ball_speed_xy,
         shooter_vx, shooter_vy, shooter_speed, ball_launch_angle_proxy]
        + off_speeds
        + def_vel_features
    )

    arr = np.array(features, dtype=np.float64)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return zeros
    return arr


def load_all_shots(game_dir: str = GAME_DIR,
                   pbp_path: str = PBP_PATH,
                   csv_path: str = CSV_PATH):
    """
    Load all shots from raw game JSONs and extract features.

    Processing steps per game:
      1. load_shot_attempts()  →  raw shot dicts
      2. build_player_stats_for_game()  →  per-player stats dict
      3. extract_static_features() + extract_velocity_features() per shot
      Shots where extract_static_features returns None are dropped from both
      feature sets so both matrices always share the same row indices.

    Returns:
        tuple(X_static, X_full, y):
          X_static : np.ndarray (N, 38)
          X_full   : np.ndarray (N, 65)  — static concat velocity
          y        : np.ndarray (N,)     — 1 = made, 0 = missed
        Returns (None, None, None) on fatal error.
    """
    if not os.path.exists(pbp_path):
        print(f"ERROR: PBP cache not found at {pbp_path}")
        return None, None, None

    pbp        = pd.read_csv(pbp_path)
    json_files = sorted(glob.glob(os.path.join(game_dir, "*.json")))

    if not json_files:
        print(f"ERROR: No JSON files found in {game_dir}")
        return None, None, None

    all_static   = []
    all_velocity = []
    all_y        = []

    total_attempted = 0
    total_dropped   = 0

    for json_file in json_files:
        game_label = os.path.basename(json_file)

        drop_stats         = create_empty_drop_stats()
        calibration_errors = create_empty_calibration_errors()

        shots = load_shot_attempts(
            json_file, pbp, drop_stats, calibration_errors
        )

        if not shots:
            print(f"  [{game_label}] 0 shots extracted — skipping")
            continue

        player_stats_dict, unmatched, game_id = build_player_stats_for_game(
            shots[0], csv_path
        )

        game_valid   = 0
        game_dropped = 0

        for shot in shots:
            total_attempted += 1

            static_feat = extract_static_features(shot, player_stats_dict)
            if static_feat is None:
                total_dropped += 1
                game_dropped  += 1
                continue

            vel_feat = extract_velocity_features(shot)

            all_static.append(static_feat)
            all_velocity.append(vel_feat)
            all_y.append(1 if shot["made"] else 0)
            game_valid += 1

        print(f"  [{game_id}] {game_valid} valid shots"
              + (f", {game_dropped} dropped" if game_dropped else ""))

    if not all_static:
        print("ERROR: No valid shots extracted across all games")
        return None, None, None

    X_static   = np.array(all_static,   dtype=np.float64)
    X_velocity = np.array(all_velocity, dtype=np.float64)
    X_full     = np.concatenate([X_static, X_velocity], axis=1)
    y          = np.array(all_y,        dtype=np.int64)

    n = len(y)
    print(f"\nDataset: {n} shots from {len(json_files)} games "
          f"({total_dropped} dropped, {total_dropped/total_attempted*100:.1f}%)")
    print(f"FG%: {y.mean()*100:.1f}%")
    print(f"X_static shape: {X_static.shape}  |  X_full shape: {X_full.shape}")

    return X_static, X_full, y


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading shot data ...")
    X_static, X_full, y = load_all_shots()
    if X_static is not None:
        print(f"\nStatic features ({X_static.shape[1]}):")
        for i, name in enumerate(STATIC_FEATURE_NAMES):
            print(f"  [{i:2d}] {name:<30s}  mean={X_static[:,i].mean():.4f}  std={X_static[:,i].std():.4f}")
        print(f"\nVelocity features ({X_full.shape[1] - X_static.shape[1]}):")
        for i, name in enumerate(VELOCITY_FEATURE_NAMES):
            col = X_static.shape[1] + i
            print(f"  [{col:2d}] {name:<30s}  mean={X_full[:,col].mean():.4f}  std={X_full[:,col].std():.4f}")

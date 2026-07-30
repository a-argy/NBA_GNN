"""
Drop-statistics reporting for the SwishNet full-season pipeline.

Produces:
  - pipeline_report.txt  — human-readable summary
  - drop_stats.json      — machine-readable per-stage counts
"""

import json
import os
from collections import Counter
from typing import Any


def generate_report(all_stats: dict, output_dir: str) -> None:
    """Write pipeline_report.txt and drop_stats.json to output_dir.

    Args:
        all_stats: Aggregated stats dict built by run_full_pipeline.py.
                   Expected keys documented below.
        output_dir: Directory where the two output files are written.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "pipeline_report.txt")
    json_path = os.path.join(output_dir, "drop_stats.json")

    report = _build_report_text(all_stats)

    with open(report_path, "w") as f:
        f.write(report)
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=_json_default)

    print(f"\nPipeline report written to: {report_path}")
    print(f"Drop stats JSON written to:  {json_path}")
    print()
    print(report)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return " N/A"
    return f"{100.0 * num / denom:5.1f}%"


def _row(label: str, value: int, denom: int, indent: int = 2) -> str:
    pad = " " * indent
    return f"{pad}{label:<40} {value:>8,}  ({_pct(value, denom)})\n"


def _build_report_text(s: dict) -> str:
    lines = []
    a = lines.append

    a("SWISHNET — FULL PIPELINE STATISTICS\n")
    a("=====================================\n\n")

    # ── Game-level summary ──────────────────────────────────────────────────
    n_games = s.get("games_processed", 0)
    n_calibrated = s.get("games_calibrated", 0)
    n_below_thresh = s.get("games_below_threshold", 0)
    n_calib_failed = s.get("games_calibration_failed", 0)

    a(f"Games processed           : {n_games:>6,}\n")
    a(f"Games calibrated          : {n_calibrated:>6,}  ({_pct(n_calibrated, n_games)})\n")
    a(f"Games below threshold     : {n_below_thresh:>6,}  ({_pct(n_below_thresh, n_games)})\n")
    a(f"Games calibration failed  : {n_calib_failed:>6,}  ({_pct(n_calib_failed, n_games)})\n\n")

    # ── Stage 1: Shot extraction ────────────────────────────────────────────
    drop1 = s.get("stage1_drops", {})
    total_events = drop1.get("total_shot_events_processed", 0)
    stage1_ok = s.get("stage1_extracted", 0)

    a("STAGE 1 — SHOT EXTRACTION\n")
    a("-" * 50 + "\n")
    a(f"  Total shot events        : {total_events:>8,}\n")
    a(f"  Successfully extracted   : {stage1_ok:>8,}  ({_pct(stage1_ok, total_events)})\n")
    a(f"  Total dropped            : {total_events - stage1_ok:>8,}  ({_pct(total_events - stage1_ok, total_events)})\n")
    a("  Drops by reason:\n")

    drop_keys = [
        "missing_team_or_shooter",
        "no_tracking_match",
        "no_moments",
        "release_not_enough_moments",
        "release_no_rim_moment_found",
        "release_no_player_within_threshold",
        "release_first_player_not_shooter",
        "release_not_enough_history",
        "release_time_proximity_failed",
    ]
    for key in drop_keys:
        val = drop1.get(key, 0)
        if val > 0:
            a(_row(key, val, total_events, indent=4))

    rescued = drop1.get("rescued_by_secondary_event", 0)
    if rescued:
        a(f"    rescued_by_secondary_event            {rescued:>8,}  (saved from drop)\n")
    a("\n")

    # ── Stage 2: Graph construction ─────────────────────────────────────────
    drop2 = s.get("stage2_drops", {})
    stage2_ok = s.get("stage2_graphs_built", 0)

    a("STAGE 2 — GRAPH CONSTRUCTION\n")
    a("-" * 50 + "\n")
    a(f"  Input (stage-1 shots)    : {stage1_ok:>8,}\n")
    a(f"  Graphs built             : {stage2_ok:>8,}  ({_pct(stage2_ok, stage1_ok)})\n")
    a(f"  Dropped                  : {stage1_ok - stage2_ok:>8,}  ({_pct(stage1_ok - stage2_ok, stage1_ok)})\n")
    a("  Drops by reason:\n")

    skip_keys = [
        "missing_field",
        "invalid_player_count",
        "missing_player_stats",
        "shooter_not_found",
        "prev_moment_missing_field",
        "prev_moment_none_field",
        "prev_moment_invalid_player_count",
        "prev_moment_shooter_not_found",
    ]
    for key in skip_keys:
        val = drop2.get(key, 0)
        if val > 0:
            a(_row(key, val, stage1_ok, indent=4))
    a("\n")

    # ── Stage 3: Cross-stage linking ────────────────────────────────────────
    link = s.get("stage3_linking", {})
    both_ok = link.get("both_ok", 0)
    graph_only = link.get("graph_failed_only", 0)
    feat_only = link.get("feature_failed_only", 0)
    both_failed = link.get("both_failed", 0)

    a("STAGE 3 — CROSS-STAGE LINKING\n")
    a("-" * 50 + "\n")
    a(f"  Both succeeded (final N) : {both_ok:>8,}  ({_pct(both_ok, stage1_ok)})\n")
    a(f"  Graph failed only        : {graph_only:>8,}  ({_pct(graph_only, stage1_ok)})\n")
    a(f"  Feature failed only      : {feat_only:>8,}  ({_pct(feat_only, stage1_ok)})\n")
    a(f"  Both failed              : {both_failed:>8,}  ({_pct(both_failed, stage1_ok)})\n")

    n_made = s.get("total_made", 0)
    n_missed = s.get("total_missed", 0)
    total_final = n_made + n_missed
    if total_final > 0:
        fg_pct = 100.0 * n_made / total_final
        a(f"\n  Final N                  : {total_final:>8,}\n")
        a(f"  FG%                      : {fg_pct:>7.1f}%\n")
    a("\n")

    # ── Unmatched players (top 20) ──────────────────────────────────────────
    unmatched = s.get("unmatched_players_counter", {})
    if unmatched:
        a("UNMATCHED PLAYERS (top 20 by frequency)\n")
        a("-" * 50 + "\n")
        top20 = sorted(unmatched.items(), key=lambda x: -x[1])[:20]
        for name, count in top20:
            a(f"  {name:<40} {count:>6,}\n")
        a("\n")

    return "".join(lines)


def _json_default(obj: Any) -> Any:
    """JSON serialiser for non-standard types (Counter → dict, etc.)."""
    if isinstance(obj, Counter):
        return dict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

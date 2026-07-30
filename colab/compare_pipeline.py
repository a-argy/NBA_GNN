#!/usr/bin/env python3
"""
Download 10 games (2 cached + 8 new), run the improved shot extraction pipeline,
and produce a quality comparison report saved to data/pipeline_comparison_report.txt.

Metrics reported per game and in aggregate:
  - Extraction rate (new pipeline)
  - Derived baseline extraction rate (new minus rescued shots)
  - Made/missed ratio (FG% proxy)
  - rescued_by_secondary_event count
  - Remaining first_player_not_shooter count
  - Games with not_enough_history drops (expected: 0)
  - Cross-game extraction rate std dev
"""

import sys
import os
import json
import glob

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, COLAB_DIR)

import load_with_context as lwc
import download_season_data
import analyze_specific_games as asg  # for get_pbp()

DATA_DIR = os.path.join(COLAB_DIR, "data", "game_data")
OUTPUT   = os.path.join(COLAB_DIR, "data", "pipeline_comparison_report.txt")

NUM_GAMES_TOTAL = 10  # Download up to this many (2 already cached)


# ── Download ──────────────────────────────────────────────────────────────────

def download_games(n):
    """Download first n games from the season dataset (cached games are skipped)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    download_season_data.download_nba_season_data(
        output_dir=DATA_DIR,
        limit=n,
        extract=True,
        keep_7z=False,
    )


# ── Per-game extraction ───────────────────────────────────────────────────────

def run_game(json_path, pbp):
    """
    Run extraction on one game JSON.
    Returns (shots, drop_stats, pbp_shot_count).
    """
    drop_stats = lwc.create_empty_drop_stats()
    cal_errors = lwc.create_empty_calibration_errors()
    shots = lwc.load_shot_attempts(json_path, pbp, drop_stats, cal_errors)

    # Count PBP shots for this game
    with open(json_path) as f:
        game = json.load(f)
    game_id = int(game["gameid"])
    game_pbp = pbp[pbp.GAME_ID == game_id]
    pbp_shots = int(game_pbp[game_pbp.EVENTMSGTYPE.isin([1, 2])].shape[0])

    return shots, drop_stats, pbp_shots


# ── Report helpers ────────────────────────────────────────────────────────────

def fg_pct(shots):
    if not shots:
        return float("nan")
    return sum(1 for s in shots if s["made"]) / len(shots) * 100


def extraction_rate(extracted, pbp_total):
    if pbp_total == 0:
        return float("nan")
    return extracted / pbp_total * 100


def flag_quality(rate_pct):
    """Red flag if FG% drifts outside 43–52%."""
    if rate_pct != rate_pct:  # nan
        return "N/A"
    if rate_pct < 43 or rate_pct > 52:
        return "*** RED FLAG ***"
    return "OK"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("NBA Shot Pipeline Comparison")
    print("=" * 65)

    # 1. Download games
    print(f"\nDownloading up to {NUM_GAMES_TOTAL} games …")
    download_games(NUM_GAMES_TOTAL)

    # 2. Load PBP
    pbp = asg.get_pbp()

    # 3. Find all JSON game files
    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))[:NUM_GAMES_TOTAL]
    if not json_files:
        print("ERROR: No game JSON files found in", DATA_DIR)
        sys.exit(1)
    print(f"\nFound {len(json_files)} game files.\n")

    # 4. Run pipeline on each game
    lines = [
        "NBA SHOT PIPELINE COMPARISON REPORT",
        "=" * 90,
        f"Games analysed: {len(json_files)}",
        "",
        "Extraction rate — new   : shots extracted / PBP shot events",
        "Extraction rate — base  : (extracted - rescued_by_secondary_event) / PBP shots",
        "FG% proxy               : made / total extracted  (NBA avg 2015-16 ≈ 45-47%)",
        "Quality red flag        : FG% proxy outside 43–52%",
        "",
        "=" * 90,
    ]

    per_game_stats = []

    for json_path in json_files:
        game_label = os.path.basename(json_path)
        print(f"  Processing {game_label} …")

        try:
            shots, ds, pbp_shots = run_game(json_path, pbp)
        except Exception as e:
            print(f"    ERROR: {e}")
            lines.append(f"\nGAME: {game_label}")
            lines.append(f"  ERROR: {e}")
            continue

        extracted       = len(shots)
        rescued_sec     = ds.get("rescued_by_secondary_event", 0)
        baseline        = extracted - rescued_sec
        made            = sum(1 for s in shots if s["made"])
        missed          = extracted - made
        fg              = fg_pct(shots)
        extr_rate_new   = extraction_rate(extracted, pbp_shots)
        extr_rate_base  = extraction_rate(baseline, pbp_shots)
        fp_not_shooter  = ds.get("release_first_player_not_shooter", 0)
        no_hist_drops   = ds.get("release_not_enough_history", 0)

        per_game_stats.append({
            "label":           game_label,
            "pbp_shots":       pbp_shots,
            "extracted":       extracted,
            "rescued_sec":     rescued_sec,
            "baseline":        baseline,
            "made":            made,
            "missed":          missed,
            "fg":              fg,
            "extr_rate_new":   extr_rate_new,
            "extr_rate_base":  extr_rate_base,
            "fp_not_shooter":  fp_not_shooter,
            "no_hist_drops":   no_hist_drops,
            "time_diffs":      ds.get("time_diffs", []),
        })

        q_flag = flag_quality(fg)
        lines += [
            "",
            f"GAME: {game_label}",
            f"  PBP shot events              : {pbp_shots}",
            f"  Extracted (new pipeline)     : {extracted}  ({extr_rate_new:.1f}%)",
            f"  Baseline (excl. rescued)     : {baseline}  ({extr_rate_base:.1f}%)",
            f"  Rescued by secondary event   : {rescued_sec}",
            f"  Made / Missed                : {made} / {missed}",
            f"  FG% proxy                    : {fg:.1f}%  [{q_flag}]",
            f"  first_player_not_shooter     : {fp_not_shooter}",
            f"  not_enough_history drops     : {no_hist_drops}",
        ]

    # 5. Aggregate stats
    if per_game_stats:
        total_pbp       = sum(g["pbp_shots"]     for g in per_game_stats)
        total_ext       = sum(g["extracted"]      for g in per_game_stats)
        total_rescued   = sum(g["rescued_sec"]    for g in per_game_stats)
        total_baseline  = sum(g["baseline"]       for g in per_game_stats)
        total_made      = sum(g["made"]           for g in per_game_stats)
        total_fp        = sum(g["fp_not_shooter"] for g in per_game_stats)
        games_no_hist   = sum(1 for g in per_game_stats if g["no_hist_drops"] > 0)

        agg_rate_new    = extraction_rate(total_ext,      total_pbp)
        agg_rate_base   = extraction_rate(total_baseline, total_pbp)
        agg_fg          = total_made / total_ext * 100 if total_ext else float("nan")

        # Cross-game extraction rate std dev
        rates = [g["extr_rate_new"] for g in per_game_stats if g["extr_rate_new"] == g["extr_rate_new"]]
        if len(rates) > 1:
            mean_r = sum(rates) / len(rates)
            std_r  = (sum((r - mean_r) ** 2 for r in rates) / (len(rates) - 1)) ** 0.5
        else:
            std_r = float("nan")

        q_flag_agg = flag_quality(agg_fg)

        lines += [
            "",
            "=" * 90,
            "AGGREGATE SUMMARY",
            "=" * 90,
            f"  Games processed              : {len(per_game_stats)}",
            f"  Total PBP shot events        : {total_pbp}",
            f"  Total extracted (new)        : {total_ext}  ({agg_rate_new:.1f}%)",
            f"  Total baseline (excl. resc.) : {total_baseline}  ({agg_rate_base:.1f}%)",
            f"  Total rescued by secondary   : {total_rescued}",
            f"  Aggregate FG% proxy          : {agg_fg:.1f}%  [{q_flag_agg}]",
            f"  Remaining first_not_shooter  : {total_fp}",
            f"  Games with not_enough_hist.  : {games_no_hist}  (target: 0)",
            f"  Extraction rate std dev      : {std_r:.1f}%",
            "",
            "QUALITY GATE",
            "  FG% proxy 43–52% → OK  |  outside → RED FLAG",
            "  not_enough_history drops target → 0",
        ]

        # Time-diff distribution (PBP time − tracking release clock)
        all_time_diffs = []
        for g in per_game_stats:
            all_time_diffs.extend(g["time_diffs"])

        if all_time_diffs:
            n_td = len(all_time_diffs)
            sd = sorted(all_time_diffs)

            def percentile(lst, p):
                idx = min(int(len(lst) * p), len(lst) - 1)
                return lst[idx]

            def outside(lst, thr):
                cnt = sum(1 for d in lst if abs(d) > thr)
                return cnt, cnt / len(lst) * 100

            p50 = percentile(sd, 0.50)
            p75 = percentile(sd, 0.75)
            p90 = percentile(sd, 0.90)
            p95 = percentile(sd, 0.95)
            p99 = percentile(sd, 0.99)

            c1, pct1 = outside(sd, 1)
            c2, pct2 = outside(sd, 2)
            c3, pct3 = outside(sd, 3)
            c4, pct4 = outside(sd, 4)

            def pct_within(lst, thr):
                cnt = sum(1 for d in lst if abs(d) <= thr)
                return cnt, cnt / len(lst) * 100

            w1, wpct1 = pct_within(sd, 1)
            w2, wpct2 = pct_within(sd, 2)
            w3, wpct3 = pct_within(sd, 3)
            w4, wpct4 = pct_within(sd, 4)

            lines += [
                "",
                "=" * 90,
                "RELEASE TIME vs PBP TIME DISTRIBUTION  (pbp_time − tracking_release_clock, seconds)",
                "=" * 90,
                f"  Shots with time data         : {n_td}",
                f"  (negative = tracking release earlier in clock than PBP log — expected direction)",
                "",
                f"  Signed percentiles:",
                f"    p50  : {p50:+.2f}s",
                f"    p75  : {p75:+.2f}s",
                f"    p90  : {p90:+.2f}s",
                f"    p95  : {p95:+.2f}s",
                f"    p99  : {p99:+.2f}s",
                "",
                f"  Threshold sensitivity — shots RETAINED if |diff| ≤ threshold:",
                f"    |diff| ≤ 1s : {w1:>5} / {n_td}  ({wpct1:.1f}%  retained)",
                f"    |diff| ≤ 2s : {w2:>5} / {n_td}  ({wpct2:.1f}%  retained)",
                f"    |diff| ≤ 3s : {w3:>5} / {n_td}  ({wpct3:.1f}%  retained)",
                f"    |diff| ≤ 4s : {w4:>5} / {n_td}  ({wpct4:.1f}%  retained)",
                "",
                "  Guidance: choose threshold where retained% ≥ 90%.",
            ]

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write(report)

    print(f"\n{'='*65}")
    print(f"Report saved → {OUTPUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()

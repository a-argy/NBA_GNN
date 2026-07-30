#!/usr/bin/env python3
"""
Generate OpenAI text-embedding-3-large embeddings for all players in
player_shooting_stats_2016.csv.

Saves a pickle dict: normalized_name -> np.ndarray(3072, float32).
Display names are also stored as keys for robust lookup.

Usage:
    export OPENAI_API_KEY=sk-...
    python colab/phase2/generate_embeddings.py \
        --csv colab/data/supplemental_data/player_shooting_stats_2016.csv \
        --out colab/phase2/results/player_embeddings.pkl

Cost: ~$0.01 (579 players × ~150 tokens × $0.00013/1K tokens)
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

_COLAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COLAB_DIR not in sys.path:
    sys.path.insert(0, _COLAB_DIR)

from scrape_player_stats import normalize_name

MODEL = "text-embedding-3-large"
EMB_DIM = 3072


def _safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        v = float(val)
        return default if (v != v) else v  # NaN check
    except (ValueError, TypeError):
        return default


def build_text_description(row: pd.Series) -> str:
    """Build a natural-language shooting profile for a player."""
    name          = str(row["name_display"])
    pos           = str(row.get("pos", ""))
    age           = _safe_float(row.get("age"))
    team          = str(row.get("team_name_abbr", ""))
    games         = int(_safe_float(row.get("games")))

    fg_pct        = _safe_float(row.get("fg_pct"))
    avg_dist      = _safe_float(row.get("avg_dist"))

    pct_fga_fg2a  = _safe_float(row.get("pct_fga_fg2a"))
    pct_fga_fg3a  = _safe_float(row.get("pct_fga_fg3a"))

    pct_fga_00_03  = _safe_float(row.get("pct_fga_00_03"))
    fg_pct_00_03   = _safe_float(row.get("fg_pct_00_03"))
    pct_fga_03_10  = _safe_float(row.get("pct_fga_03_10"))
    fg_pct_03_10   = _safe_float(row.get("fg_pct_03_10"))
    pct_fga_10_16  = _safe_float(row.get("pct_fga_10_16"))
    fg_pct_10_16   = _safe_float(row.get("fg_pct_10_16"))
    pct_fga_16_xx  = _safe_float(row.get("pct_fga_16_xx"))
    fg_pct_16_xx   = _safe_float(row.get("fg_pct_16_xx"))

    fg_pct_fg3a      = _safe_float(row.get("fg_pct_fg3a"))
    pct_fg3a_corner3 = _safe_float(row.get("pct_fg3a_corner3"))
    fg_pct_corner3   = _safe_float(row.get("fg_pct_corner3"))

    pct_ast_fg2   = _safe_float(row.get("pct_ast_fg2"))
    pct_ast_fg3   = _safe_float(row.get("pct_ast_fg3"))
    pct_fga_dunk  = _safe_float(row.get("pct_fga_dunk"))

    return (
        f"{name}, {pos}, age {age:.0f}, {team} ({games} games, 2015-16 season). "
        f"Overall: {fg_pct*100:.1f}% FG, avg shot dist {avg_dist:.1f} ft. "
        f"Shot selection: {pct_fga_fg2a*100:.0f}% 2PT, {pct_fga_fg3a*100:.0f}% 3PT. "
        f"2PT zones: {pct_fga_00_03*100:.0f}% at rim ({fg_pct_00_03*100:.0f}% FG%), "
        f"{pct_fga_03_10*100:.0f}% short mid ({fg_pct_03_10*100:.0f}%), "
        f"{pct_fga_10_16*100:.0f}% mid-range ({fg_pct_10_16*100:.0f}%), "
        f"{pct_fga_16_xx*100:.0f}% long 2 ({fg_pct_16_xx*100:.0f}%). "
        f"3PT: {fg_pct_fg3a*100:.0f}% FG, {pct_fg3a_corner3*100:.0f}% corner 3s "
        f"({fg_pct_corner3*100:.0f}% from corner). "
        f"Assisted: {pct_ast_fg2*100:.0f}% of 2PT, {pct_ast_fg3*100:.0f}% of 3PT. "
        f"Dunks: {pct_fga_dunk*100:.1f}% of attempts."
    )


def generate_all_embeddings(
    csv_path: str,
    out_path: str,
    delay: float = 0.05,
    resume: bool = True,
) -> dict:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from environment

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} players from {csv_path}")

    # Resume from checkpoint if it exists
    emb_dict: dict = {}
    if resume and os.path.exists(out_path):
        with open(out_path, "rb") as f:
            emb_dict = pickle.load(f)
        already = len({id(v) for v in emb_dict.values()})
        print(f"Resuming: {already} players already embedded")

    fallback_count = 0

    for i, (_, row) in enumerate(df.iterrows()):
        norm    = normalize_name(str(row["name_display"]))
        display = str(row["name_display"])

        if norm in emb_dict:
            continue  # already done

        text = build_text_description(row)

        for attempt in range(5):
            try:
                response = client.embeddings.create(input=text, model=MODEL)
                vec = np.array(response.data[0].embedding, dtype=np.float32)
                emb_dict[norm]    = vec
                emb_dict[display] = vec
                print(
                    f"  [{i+1:3d}/{len(df)}] {display[:40]:<40s}"
                    f"  norm={np.linalg.norm(vec):.3f}"
                )
                break
            except Exception as exc:
                wait = 2 ** attempt
                print(f"  [{i+1:3d}] retry {attempt+1}/5 in {wait}s: {exc}")
                time.sleep(wait)
        else:
            print(f"  [{i+1:3d}] FAILED: {display}")
            fallback_count += 1

        if delay > 0:
            time.sleep(delay)

        # Checkpoint every 50 players
        if (i + 1) % 50 == 0:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(emb_dict, f)
            print(f"  [checkpoint @ {i+1} players]")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(emb_dict, f)

    n_unique = len({id(v) for v in emb_dict.values()})
    print(f"\nDone. {n_unique} unique embeddings, {fallback_count} failures.")
    print(f"Saved → {out_path}")

    # Sanity-check: print 3 sample descriptions
    print("\nSample descriptions:")
    for j, (_, row) in enumerate(df.head(3).iterrows()):
        print(f"  [{j+1}] {build_text_description(row)[:120]}...")

    return emb_dict


def main():
    parser = argparse.ArgumentParser(description="Generate player text embeddings")
    parser.add_argument(
        "--csv", required=True,
        help="Path to player_shooting_stats_2016.csv",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "results", "player_embeddings.pkl"),
        help="Output pickle path",
    )
    parser.add_argument(
        "--delay", type=float, default=0.05,
        help="Seconds between API calls (default: 0.05)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, ignore existing checkpoint",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    generate_all_embeddings(
        csv_path=args.csv,
        out_path=args.out,
        delay=args.delay,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

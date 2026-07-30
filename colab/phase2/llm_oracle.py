#!/usr/bin/env python3
"""
Exp 0c — LLM Oracle: Zero-shot shot prediction via Vertex AI Gemini.

Samples N random shots from local game data, builds natural-language prompts
describing each shot situation, and queries a Gemini model for a make
probability. Computes AUC to compare against the LR baseline (0.6377).

Usage:
    python colab/phase2/llm_oracle.py
    python colab/phase2/llm_oracle.py --n 100 --model gemini-2.5-pro-preview-06-05 --seed 42
"""

import argparse
import glob as glob_mod
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── local imports ─────────────────────────────────────────────────────────────
_COLAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COLAB_DIR not in sys.path:
    sys.path.insert(0, _COLAB_DIR)

from baseline_features import (
    CONTEST_DISTANCE,
    THREE_POINT_DISTANCE,
    _get_shooter_pos,
    _sorted_defender_distances,
    extract_static_features,
)
from load_with_context import (
    create_empty_calibration_errors,
    create_empty_drop_stats,
    load_shot_attempts,
)
from scrape_player_stats import build_player_stats_for_game

# ── paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR    = os.path.join(_COLAB_DIR, "data")
_GAME_DIR    = os.path.join(_DATA_DIR, "game_data")
_PBP_PATH    = os.path.join(_DATA_DIR, "supplemental_data", "pbp_cache.csv")
_CSV_PATH    = os.path.join(_DATA_DIR, "supplemental_data", "player_shooting_stats_2016.csv")
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_clock(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def build_prompt(shot: dict, player_stats_dict: dict) -> str | None:
    """
    Build a natural-language prompt for the shot.
    Returns None if required fields are missing.
    """
    shooter_id   = shot.get("shooter_id")
    player_names = shot.get("player_names", {})
    shooter_name = player_names.get(shooter_id, f"Player {shooter_id}")

    offense = shot.get("offense_position", [])
    defense = shot.get("defense_position", [])
    ball    = shot.get("ball_position")
    rim     = shot.get("target_rim", [5.25, 25.0])

    if not ball or not offense or not defense:
        return None

    shooter_pos = _get_shooter_pos(shot)
    if shooter_pos is None:
        return None

    # Geometry
    dist_to_rim = math.sqrt((ball[0] - rim[0]) ** 2 + (ball[1] - rim[1]) ** 2)
    is_3pt      = dist_to_rim > THREE_POINT_DISTANCE
    shot_type   = "3-pointer" if is_3pt else "2-pointer"

    angle_rad = math.atan2(shooter_pos[1] - rim[1], shooter_pos[0] - rim[0])
    angle_deg = math.degrees(angle_rad)

    # Defense
    def_dists    = _sorted_defender_distances(defense, shooter_pos)
    closest_def  = def_dists[0]
    num_contested = sum(1 for d in def_dists if d <= CONTEST_DISTANCE)

    # Game state
    quarter    = shot.get("quarter", 1)
    game_clock = shot.get("game_clock", 0.0)
    shot_clock = shot.get("shot_clock")

    quarter_str   = f"Q{quarter}" if quarter <= 4 else f"OT{quarter - 4}"
    clock_str     = _fmt_clock(game_clock)
    shot_clock_str = f"{shot_clock:.0f}s" if shot_clock is not None else "unknown"

    # Shooter season stats
    stats = player_stats_dict.get(str(shooter_id)) or player_stats_dict.get(shooter_id)
    if stats:
        fg_pct  = float(stats.get("fg_pct",  0.0))
        fg3_pct = float(stats.get("fg3_pct", 0.0))
        avg_dist = float(stats.get("avg_dist", 0.0))
        stat_str = (
            f"{fg_pct * 100:.1f}% overall FG%, "
            f"{fg3_pct * 100:.1f}% from 3, "
            f"avg shot distance {avg_dist:.1f} ft"
        )
    else:
        stat_str = "season stats unavailable"

    prompt = f"""You are analyzing an NBA shot attempt from the 2015-16 season.

Shooter: {shooter_name}
Season stats: {stat_str}

Shot situation:
- Shot type: {shot_type} ({dist_to_rim:.1f} ft from the rim)
- Angle to basket: {angle_deg:.0f}°
- Shot clock: {shot_clock_str} remaining
- Game clock: {clock_str} remaining in {quarter_str}

Defense:
- Closest defender: {closest_def:.1f} ft away
- Defenders within 6 ft: {num_contested}

Based solely on this information, estimate the probability (0.0–1.0) that this shot goes in.

Respond with ONLY a JSON object in this exact format, no other text:
{{"probability": 0.XX}}"""

    return prompt


def _init_model(model_name: str, project: str, location: str):
    """Initialise Vertex AI once and return a reusable GenerativeModel."""
    import subprocess
    import vertexai
    import google.oauth2.credentials
    from vertexai.generative_models import GenerativeModel

    token_result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True, text=True, check=True
    )
    access_token = token_result.stdout.strip()
    creds = google.oauth2.credentials.Credentials(token=access_token)
    vertexai.init(project=project, location=location, credentials=creds)
    return GenerativeModel(model_name)


def query_vertex(prompt: str, model, model_name: str) -> float | None:
    """Query a pre-initialised Vertex AI model and parse a probability."""
    from vertexai.generative_models import GenerationConfig

    config = GenerationConfig(temperature=0.0)
    response = model.generate_content(prompt, generation_config=config)
    try:
        text = response.text.strip()
    except Exception as e:
        print(f"    [response.text error: {e}]")
        return None

    # Strip markdown code fences if present
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
        prob = float(data["probability"])
        return max(0.0, min(1.0, prob))
    except Exception as e:
        print(f"    [json parse error: {e} | raw: {repr(text[:80])}]")
        # Fallback: grab first float that looks like a probability
        nums = re.findall(r"0\.\d+", text)
        if nums:
            return float(nums[0])
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 0c — LLM Oracle")
    parser.add_argument("--n",       type=int,   default=100,
                        help="Number of shots to sample (default: 100)")
    parser.add_argument("--model",   type=str,   default="gemini-2.0-flash-001",
                        help="Vertex AI model name")
    parser.add_argument("--project", type=str,   default="swishnet-489108")
    parser.add_argument("--location",type=str,   default="us-central1")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--delay",   type=float, default=0.3,
                        help="Seconds between API calls (default: 0.3)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # ── Load all valid shots from local game files ────────────────────────────
    print("Loading shots from local game data ...")
    pbp        = pd.read_csv(_PBP_PATH)
    json_files = sorted(glob_mod.glob(os.path.join(_GAME_DIR, "*.json")))

    all_shots: list[tuple[dict, dict]] = []
    for jf in json_files:
        shots = load_shot_attempts(
            jf, pbp, create_empty_drop_stats(), create_empty_calibration_errors()
        )
        if not shots:
            continue
        player_stats_dict, _, game_id = build_player_stats_for_game(shots[0], _CSV_PATH)
        for s in shots:
            if extract_static_features(s, player_stats_dict) is not None:
                all_shots.append((s, player_stats_dict))

    print(f"Total valid shots: {len(all_shots)}")

    n_sample = min(args.n, len(all_shots))
    sample   = random.sample(all_shots, n_sample)
    print(f"Sampled {n_sample} shots  (seed={args.seed})")

    # ── Query the LLM ─────────────────────────────────────────────────────────
    print(f"\nQuerying {args.model} ...\n")
    model   = _init_model(args.model, args.project, args.location)
    results: list[dict] = []
    failed  = 0

    for i, (shot, psd) in enumerate(sample):
        prompt = build_prompt(shot, psd)
        if prompt is None:
            failed += 1
            continue

        try:
            prob = query_vertex(prompt, model, args.model)
        except Exception as e:
            print(f"  [{i + 1:3d}] API error: {e}")
            prob = None

        if prob is None:
            failed += 1
            print(f"  [{i + 1:3d}] Parse failed")
            continue

        label = 1 if shot["made"] else 0
        results.append({"prob": prob, "label": label})
        print(
            f"  [{i + 1:3d}/{n_sample}]  prob={prob:.3f}  "
            f"actual={'MAKE' if shot['made'] else 'MISS'}"
        )

        if args.delay > 0:
            time.sleep(args.delay)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if len(results) < 2:
        print(f"\nERROR: only {len(results)} valid results — cannot compute AUC")
        return

    probs  = np.array([r["prob"]  for r in results])
    labels = np.array([r["label"] for r in results])
    auc    = roc_auc_score(labels, probs)
    fg_pct = labels.mean()

    print(f"\n{'=' * 52}")
    print(f"Results — {args.model}")
    print(f"{'=' * 52}")
    print(f"Shots evaluated : {len(results)}  (failed/skipped: {failed})")
    print(f"FG% in sample   : {fg_pct * 100:.1f}%")
    print(f"LLM AUC         : {auc:.4f}")
    print(f"LR baseline     : 0.6377  (87K shots, Velocity-65, Lasso)")
    print(f"Gap vs baseline : {auc - 0.6377:+.4f}")
    print(f"{'=' * 52}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "model":      args.model,
        "n_shots":    len(results),
        "n_failed":   failed,
        "seed":       args.seed,
        "auc":        auc,
        "fg_pct":     fg_pct,
        "lr_baseline": 0.6377,
        "gap":        round(auc - 0.6377, 4),
        "results":    results,
    }
    out_path = os.path.join(_RESULTS_DIR, "exp0c_llm_oracle.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()

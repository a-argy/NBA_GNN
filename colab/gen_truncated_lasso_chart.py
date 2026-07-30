#!/usr/bin/env python3
"""
Generate a truncated Lasso coefficient bar chart showing only the top ~30
features (by absolute magnitude), for better readability on the poster.
"""
import os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Ensure colab/ is on the path
_COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
if _COLAB_DIR not in sys.path:
    sys.path.insert(0, _COLAB_DIR)

from baseline_features import (
    load_all_shots, STATIC_FEATURE_NAMES, VELOCITY_FEATURE_NAMES,
)
from train_baselines import fit_final_model

TOP_N = 30
OUTPUT_PATH = os.path.join(
    _COLAB_DIR, "data", "baseline_plots", "weights_velocity_lasso_top30.png"
)
POSTER_PATH = os.path.join(
    os.path.dirname(_COLAB_DIR),
    "report", "poster", "Stanford_CS_Poster_Template__Landscape_",
    "weights_velocity_lasso.png",
)

def main():
    print("Loading shots...")
    X_static, X_full, y = load_all_shots()
    if X_static is None:
        print("ERROR: Could not load shots.")
        return

    feature_names = STATIC_FEATURE_NAMES + VELOCITY_FEATURE_NAMES
    print(f"Loaded {X_full.shape[0]} shots, {X_full.shape[1]} features.")

    print("Fitting Lasso (L1) logistic regression on Velocity-65...")
    coef = fit_final_model(X_full, y, model_type="lasso", scale=True,
                           best_C=0.1)

    # Sort by absolute magnitude, take top N
    sorted_idx = np.argsort(np.abs(coef))[::-1][:TOP_N]

    names  = [feature_names[i] for i in sorted_idx]
    values = coef[sorted_idx]
    colors = ["steelblue" if v >= 0 else "firebrick" for v in values]

    n = len(names)
    fig_h = max(6, n * 0.32)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ax.barh(range(n), values, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (standardized)", fontsize=10)
    ax.set_title(f"Top {TOP_N} Lasso Coefficients | Velocity-65 | Scaled (C*=0.1)",
                 fontsize=11, pad=10)
    ax.invert_yaxis()

    ax.legend(
        handles=[Patch(color="steelblue", label="positive"),
                 Patch(color="firebrick", label="negative")],
        fontsize=9, loc="lower right",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")

    # Also copy to poster directory
    plt.savefig(POSTER_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {POSTER_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()

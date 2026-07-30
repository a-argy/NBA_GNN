#!/usr/bin/env python3
"""
Regenerate hp_sweep_scatter.pdf/.png from sweep_results.json.

The horizontal jitter is purely cosmetic (avoids overplotting) and does NOT
represent any data dimension. Each point's true values are:
  - x position: hidden_dim (categorical: 64, 128, 256)
  - y position: best_val_auc (continuous)
  - color:      num_layers (discrete: 1-4)
  - size:       fixed (dropout encoded in legend note instead)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_PHASE2_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_PHASE2_DIR))
FIGURES_DIR = os.path.join(_REPO_ROOT, "report", "figures")
DATA_PATH   = os.path.join(_REPO_ROOT, "report", "figures", "sweep_results.json")

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "figure.dpi":     150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Load data ─────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    data = json.load(f)

print(f"Loaded {len(data)} configs from sweep_results.json")

hd_vals  = [r["config"]["hidden_dim"]  for r in data]
nl_vals  = [r["config"]["num_layers"]  for r in data]
do_vals  = [r["config"]["dropout"]     for r in data]
auc_vals = [r["best_val_auc"]          for r in data]

# ── Layout constants ──────────────────────────────────────────────────────────
HD_CATS    = [64, 128, 256]
HD_POS     = {64: 0, 128: 1, 256: 2}   # categorical x positions
LAYER_COLORS = {1: "#c6dbef", 2: "#6baed6", 3: "#2171b5", 4: "#08306b"}
JITTER_W   = 0.18   # half-width of jitter band per category

rng = np.random.default_rng(42)   # fixed seed → reproducible jitter

fig, ax = plt.subplots(figsize=(5.5, 4.0))

# Plot each point with per-layer color and fixed jitter
plotted = {nl: False for nl in sorted(LAYER_COLORS)}
for hd, nl, auc in zip(hd_vals, nl_vals, auc_vals):
    x_base  = HD_POS[hd]
    x_jitter = rng.uniform(-JITTER_W, JITTER_W)
    ax.scatter(x_base + x_jitter, auc,
               color=LAYER_COLORS[nl],
               s=55, alpha=0.88, zorder=3,
               edgecolors="white", linewidths=0.4)
    plotted[nl] = True

# Default HP baseline
ax.axhline(0.6368, color="#d62728", linewidth=1.4, linestyle="--",
           label="Default HP (0.6368)", zorder=2)

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["64", "128", "256"], fontsize=11)
ax.set_xlabel("Hidden Dimension", fontsize=11)
ax.set_ylabel("Validation AUC", fontsize=11)
ax.set_title("Hyperparameter Sweep (29 configs)", fontsize=12, pad=6)

# Tight y range
auc_min, auc_max = min(auc_vals), max(auc_vals)
pad = 0.002
ax.set_ylim(auc_min - pad, auc_max + pad)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))

# Light vertical gridlines at category centers
for xp in [0, 1, 2]:
    ax.axvline(xp, color="#dddddd", linewidth=0.8, zorder=1)
ax.set_xlim(-0.55, 2.55)

# ── Legend ────────────────────────────────────────────────────────────────────
layer_handles = [
    mpatches.Patch(color=LAYER_COLORS[nl], label=f"{nl} layer{'s' if nl>1 else ''}")
    for nl in sorted(LAYER_COLORS)
]
baseline_handle = plt.Line2D([0], [0], color="#d62728", linewidth=1.4,
                              linestyle="--", label="Default HP (0.6368)")
ax.legend(handles=layer_handles + [baseline_handle],
          loc="lower right", framealpha=0.9, fontsize=9.5,
          title="GATv2 layers", title_fontsize=9.5)

# Note about jitter
ax.text(0.01, 0.01, "Note: horizontal jitter is cosmetic (avoids overplotting)",
        transform=ax.transAxes, fontsize=7.5, color="#888888",
        va="bottom", ha="left")

fig.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(FIGURES_DIR, f"hp_sweep_scatter.{ext}")
    fig.savefig(out, bbox_inches="tight", dpi=250 if ext == "png" else None)
    print(f"Saved → {out}")

plt.close(fig)
print("Done.")

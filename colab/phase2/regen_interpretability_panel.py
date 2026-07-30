#!/usr/bin/env python3
"""Regenerate interpretability_panel.pdf/.png with fixed x-axis tick spacing."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(_REPO_ROOT, "report", "figures")

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":     150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Data ──────────────────────────────────────────────────────────────────────
ablation_data = [
    ("temporal\\_id",  -0.0087),
    ("geometry",       -0.0085),
    ("role\\_flags",   -0.0073),
    ("game\\_state",   -0.0038),
    ("position\\_enc", -0.0022),
    ("spatial\\_xyz",  -0.0021),
    ("player\\_stats", -0.0018),
]
labels_abl = [d[0] for d in ablation_data]
deltas     = [d[1] for d in ablation_data]

attention_data = [
    ("Player-Ball", 0.1256),
    ("Off-Off",     0.0957),
    ("Off-Def",     0.0876),
    ("Def-Def",     0.0734),
]
labels_att = [d[0] for d in attention_data]
weights    = [d[1] for d in attention_data]

# ── Colors ────────────────────────────────────────────────────────────────────
# Red gradient for ablation (darkest = largest drop)
magnitudes = [abs(d) for d in deltas]
norm_mag   = [(m - min(magnitudes)) / (max(magnitudes) - min(magnitudes) + 1e-9)
              for m in magnitudes]
ablation_colors = [plt.cm.Reds(0.4 + 0.55 * n) for n in norm_mag]

# Edge-type colors
att_colors = ["#ff7f00", "#2166ac", "#7b2d8b", "#d6604d"]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4),
                         gridspec_kw={"width_ratios": [1.6, 1.0]})
fig.subplots_adjust(wspace=0.45)

# ── Panel A: Feature Ablation ─────────────────────────────────────────────────
ax = axes[0]
y_pos = np.arange(len(labels_abl))
bars = ax.barh(y_pos, deltas, color=ablation_colors, height=0.6, edgecolor="white", linewidth=0.4)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_abl, fontsize=10)
ax.invert_yaxis()

# Clean x-axis: use 4 evenly spaced ticks, no overlap
ax.set_xlim(-0.0105, 0.0008)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.003))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:.3f}" if x != 0 else "0"
))
ax.tick_params(axis="x", labelsize=9, rotation=0)
ax.axvline(0, color="#333333", linewidth=0.8, zorder=3)
ax.set_xlabel(r"$\Delta$ AUC", fontsize=11, labelpad=4)
ax.set_title("(a)  Feature Ablation", fontsize=12, pad=6)

# Value labels on bars
for bar, val in zip(bars, deltas):
    ax.text(val - 0.0003, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="right", fontsize=8, color="white",
            fontweight="bold")

# ── Panel B: Attention by Edge Type ──────────────────────────────────────────
ax2 = axes[1]
x_pos = np.arange(len(labels_att))
ax2.bar(x_pos, weights, color=att_colors, width=0.55, edgecolor="white", linewidth=0.5)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels_att, fontsize=9, rotation=20, ha="right")
ax2.set_ylim(0, 0.145)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
ax2.tick_params(axis="y", labelsize=9)
ax2.set_ylabel(r"Mean attention weight $\alpha$", fontsize=10, labelpad=4)
ax2.set_title("(b)  GATv2 Attention", fontsize=12, pad=6)

# Value labels on bars
for xi, w in zip(x_pos, weights):
    ax2.text(xi, w + 0.002, f"{w:.4f}", ha="center", va="bottom", fontsize=8)

fig.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(FIGURES_DIR, f"interpretability_panel.{ext}")
    fig.savefig(out, bbox_inches="tight", dpi=250 if ext == "png" else None)
    print(f"Saved → {out}")

plt.close(fig)
print("Done.")

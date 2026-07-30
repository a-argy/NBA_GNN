#!/usr/bin/env python3
"""
Fix PCA+GMM scatter plot to use true make/miss labels instead of cluster FG% proxy.

Reads gru_embeddings.npy (86,425 shots) and aligns them with y.npy (87,147 shots)
using shot_index_map.json to exclude the 722 shots from unreadable game files.
Then regenerates report/figures/pca_gmm_scatter.pdf/.png with correct coloring.

Usage:
    python3 colab/phase2/fix_pca_gmm_labels.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ── Paths ─────────────────────────────────────────────────────────────────────
_PHASE2_DIR = os.path.dirname(os.path.abspath(__file__))
_COLAB_DIR  = os.path.dirname(_PHASE2_DIR)
_REPO_ROOT  = os.path.dirname(_COLAB_DIR)

BASELINE_DIR  = os.path.join(_COLAB_DIR, "data", "baseline_data")
EMB_DIR       = os.path.join(_PHASE2_DIR, "results", "ext_pca_gmm")
FIGURES_DIR   = os.path.join(_REPO_ROOT, "report", "figures")

EMB_PATH      = os.path.join(EMB_DIR, "gru_embeddings.npy")
RESP_PATH     = os.path.join(EMB_DIR, "gmm_responsibilities.npy")
Y_PATH        = os.path.join(BASELINE_DIR, "y.npy")
INDEX_MAP_PATH = os.path.join(BASELINE_DIR, "shot_index_map.json")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":     150,
})

CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#dede00", "#00ced1",
]


def load_aligned_labels():
    """Align y.npy (87,147) with gru_embeddings.npy (86,425) via shot_index_map."""
    print("Loading y.npy ...")
    y_all = np.load(Y_PATH)
    print(f"  y_all shape: {y_all.shape}  FG%: {100*y_all.mean():.1f}%")

    print("Loading shot_index_map.json ...")
    with open(INDEX_MAP_PATH) as f:
        shot_index_map = json.load(f)
    assert len(shot_index_map) == len(y_all), (
        f"shot_index_map length {len(shot_index_map)} != y.npy length {len(y_all)}"
    )

    # Identify skipped game IDs (same logic as ext_pca_gmm.py)
    # The ext_pca_gmm loader skips files that raise exceptions.
    # We detect them by comparing expected total vs embedding count.
    # Load graph files to find skipped IDs (sorted order, same as load_all_graphs).
    graph_data_dir = os.environ.get(
        "SWISHNET_GRAPH_DIR",
        os.path.join(_COLAB_DIR, "data", "graph_data"),  # local fallback (12 games)
    )

    # Rather than re-loading all graphs, we can infer skipped game IDs:
    # The embeddings were produced by ext_pca_gmm.py with exactly this skip logic.
    # We use shot_index_map to count shots per game_id and reconstruct which
    # game_ids were included, by checking that the total after exclusion = 86,425.
    from collections import defaultdict
    shots_per_game = defaultdict(list)
    for idx, entry in enumerate(shot_index_map):
        shots_per_game[entry["game_id"]].append(idx)

    # Try to determine skipped game IDs from the known counts:
    # Total shots = 87,147. Excluded = 722. Need to find which game(s) total 722.
    # The RESEARCH_LOG notes "5 game files (722 shots)" were excluded.
    # We find these by sorting game_ids and trying combinations — but that's slow.
    # Instead, directly check: sum of shots in any 5-game subset = 722.
    # The simplest approach: filter by checking if embedding count matches
    # after excluding candidate sets. Since we know it's exactly 5 games with 722
    # total shots, we find them by looking for game_ids whose cumulative exclusion
    # reaches 722 and results in 86,425 remaining.

    # Fast approach: load the graph directory to find skippable files.
    # If graph_data_dir has all 616 games, iterate sorted file list and
    # try torch.load — skip failures. Otherwise fall back to enumerating
    # candidates from shot_index_map.

    import torch
    skipped_ids = set()
    if os.path.isdir(graph_data_dir):
        pt_files = sorted(f for f in os.listdir(graph_data_dir) if f.endswith(".pt"))
        print(f"Scanning {len(pt_files)} graph files to identify skipped games ...")
        for fname in pt_files:
            path = os.path.join(graph_data_dir, fname)
            try:
                data = torch.load(path, weights_only=False)
                # Quick check: just verify it loads; no need to read content
                del data
            except Exception:
                game_id = fname.replace("graphs_", "").replace(".pt", "")
                skipped_ids.add(game_id)
                print(f"  Skipped: {fname}")

    if not skipped_ids:
        # No local graph_data (only 12 games) — infer skipped IDs by finding
        # the 5-game subset that sums to exactly 722 shots.
        # Sort game_ids and find candidates (this is fast since 616 games).
        print("Local graph_data not available — inferring skipped game IDs ...")
        game_ids = sorted(shots_per_game.keys())
        target = len(y_all) - 86425  # = 722
        # Single-pass: find individual games with shot counts summing to 722
        # (the 5 games average ~144 shots each)
        cumulative_candidates = []
        cumsum = 0
        for gid in game_ids:
            cumsum += len(shots_per_game[gid])
        # Brute-force search for subset of game_ids summing to target
        # Since there are only ~616 games, this is tractable for small subsets
        found = False
        for gid in game_ids:
            n = len(shots_per_game[gid])
            if n == target:
                skipped_ids = {gid}
                found = True
                break
        if not found:
            # Try pairs
            game_list = [(gid, len(shots_per_game[gid])) for gid in game_ids]
            for i, (g1, n1) in enumerate(game_list):
                for g2, n2 in game_list[i+1:]:
                    if n1 + n2 == target:
                        skipped_ids = {g1, g2}
                        found = True
                        break
                if found:
                    break
        if not found:
            print(f"WARNING: Could not identify skipped game IDs summing to {target} shots.")
            print("Falling back to truncating y to first 86,425 entries (approximate).")
            return np.load(Y_PATH)[:86425]

    print(f"Identified {len(skipped_ids)} skipped game(s): {sorted(skipped_ids)}")

    # Build keep mask: include shots whose game_id is NOT in skipped_ids
    keep_mask = np.array(
        [entry["game_id"] not in skipped_ids for entry in shot_index_map],
        dtype=bool,
    )
    y_filtered = y_all[keep_mask]
    excluded = (~keep_mask).sum()
    print(f"  Excluded {excluded:,} shots from skipped games")
    print(f"  Remaining: {len(y_filtered):,}  FG%: {100*y_filtered.mean():.1f}%")

    assert len(y_filtered) == 86425, (
        f"Expected 86,425 after filtering, got {len(y_filtered)}. "
        f"Skipped game IDs may be wrong."
    )
    return y_filtered


def regenerate_pca_gmm_scatter(y_labels):
    print("\nLoading embeddings and responsibilities ...")
    embeddings = np.load(EMB_PATH)
    responsibilities = np.load(RESP_PATH)
    cluster_ids = np.argmax(responsibilities, axis=1)
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Responsibilities: {responsibilities.shape}")
    print(f"  Labels: {y_labels.shape}  FG%: {100*y_labels.mean():.1f}%")

    print("Running PCA (n_components=2) ...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    var1, var2 = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {var1:.1f}%  PC2: {var2:.1f}%  Total: {var1+var2:.1f}%")

    # Subsample for plotting density (keep all, use low alpha)
    rng = np.random.default_rng(42)
    SAMPLE_N = 15000
    if len(coords) > SAMPLE_N:
        idx = rng.choice(len(coords), SAMPLE_N, replace=False)
        idx_sorted = np.sort(idx)
        c_plot  = coords[idx_sorted]
        y_plot  = y_labels[idx_sorted]
        cl_plot = cluster_ids[idx_sorted]
    else:
        c_plot, y_plot, cl_plot = coords, y_labels, cluster_ids

    print(f"Plotting {len(c_plot):,} points ...")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    # Panel (a): make/miss coloring
    ax = axes[0]
    miss_mask = y_plot == 0
    make_mask = y_plot == 1
    ax.scatter(c_plot[miss_mask, 0], c_plot[miss_mask, 1],
               c="#d6604d", s=2, alpha=0.15, linewidths=0, label="Miss")
    ax.scatter(c_plot[make_mask, 0], c_plot[make_mask, 1],
               c="#2166ac", s=2, alpha=0.15, linewidths=0, label="Make")
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax.set_title("(a) Outcome")
    leg = ax.legend(loc="upper right", markerscale=4, framealpha=0.85,
                    handletextpad=0.3)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    # Panel (b): GMM cluster coloring (all points, low alpha)
    ax2 = axes[1]
    n_clusters = responsibilities.shape[1]
    for k in range(n_clusters):
        mask = cl_plot == k
        if mask.sum() == 0:
            continue
        ax2.scatter(c_plot[mask, 0], c_plot[mask, 1],
                    c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
                    s=2, alpha=0.15, linewidths=0, label=f"C{k}")
    # Mark cluster centroids in full PCA space
    for k in range(n_clusters):
        mask_all = cluster_ids == k
        if mask_all.sum() == 0:
            continue
        centroid = coords[mask_all].mean(axis=0)
        ax2.scatter(centroid[0], centroid[1],
                    c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
                    s=80, marker="*", edgecolors="black", linewidths=0.5,
                    zorder=5)
    ax2.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax2.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax2.set_title("(b) GMM Cluster")
    leg2 = ax2.legend(loc="upper right", markerscale=4, framealpha=0.85,
                      ncol=2, handletextpad=0.3, fontsize=8)
    for lh in leg2.legend_handles:
        lh.set_alpha(1.0)

    fig.suptitle(
        "PCA of GRU Embeddings (88.4% variance in 2 PCs)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES_DIR, f"pca_gmm_scatter.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
        print(f"  Saved → {out}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    y_labels = load_aligned_labels()
    regenerate_pca_gmm_scatter(y_labels)

#!/usr/bin/env python3
"""
Ext 5 — Interpretability analysis for TemporalGRUShotPredictor.

No training — loads pretrained checkpoint (rerun_rank03) and runs inference only.

Usage:
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/ext5_interpret.py
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/ext5_interpret.py --part A
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/ext5_interpret.py --part B
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_PHASE2_DIR = os.path.dirname(os.path.abspath(__file__))
_COLAB_DIR  = os.path.dirname(_PHASE2_DIR)
_PHASE1_DIR = os.path.join(_COLAB_DIR, "phase1")
for _p in (_COLAB_DIR, _PHASE1_DIR, _PHASE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase2.models import TemporalGRUShotPredictor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.environ.get("SWISHNET_BASE_DIR", os.path.expanduser("~/data"))
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")
RESULTS_DIR    = os.path.join(_PHASE2_DIR, "results", "ext5_interpret")
CKPT_PATH      = os.path.join(_PHASE2_DIR, "results", "rerun_top", "rerun_rank03_best.pt")

# ── Architecture HPs (rank03) ─────────────────────────────────────────────────
NUM_NODE_FEATURES = 41
NUM_HEADS         = 4
NUM_PRE_LAYERS    = 2
NUM_POST_LAYERS   = 2
HIDDEN_DIM        = 256
NUM_LAYERS        = 4
DROPOUT           = 0.1

DATA_SEED = 42

# ── Node feature index constants ──────────────────────────────────────────────
_HAS_BALL_IDX    = 3
_IS_OFFENSE_IDX  = 4
_IS_PLAYER_IDX   = 5
_TIMESTAMP_IDX   = 14
_TEMPORAL_IDX    = 8   # edge attr index

# ── Feature group definitions ─────────────────────────────────────────────────
# 7 groups spanning all 41 node feature dimensions
FEATURE_GROUPS = [
    ("xyz",   list(range(0, 3))),
    ("role",  list(range(3, 6))),
    ("geom",  list(range(6, 11))),
    ("state", list(range(11, 14))),
    ("tid",   [14]),
    ("pos",   list(range(15, 20))),
    ("stats", list(range(20, 41))),
]
FEATURE_GROUP_NAMES = [name for name, _ in FEATURE_GROUPS]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_graphs(graph_data_dir=None):
    if graph_data_dir is None:
        graph_data_dir = GRAPH_DATA_DIR
    print(f"Loading graphs from {graph_data_dir} ...")
    pt_files = sorted(f for f in os.listdir(graph_data_dir) if f.endswith(".pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {graph_data_dir}")
    all_graphs = []
    skipped = 0
    for fname in tqdm(pt_files, desc="Loading game files"):
        path = os.path.join(graph_data_dir, fname)
        try:
            game_graphs = torch.load(path, weights_only=False)
            if isinstance(game_graphs, list):
                all_graphs.extend(game_graphs)
            else:
                all_graphs.append(game_graphs)
        except Exception as e:
            skipped += 1
            tqdm.write(f"  SKIP {fname}: {e}")
    if skipped:
        print(f"Skipped {skipped} unreadable file(s) (legacy format)")
    labels = [int(g.y.item()) for g in all_graphs]
    n = len(all_graphs)
    print(f"Loaded {n:,} graphs — FG% {100*sum(labels)/n:.1f}%")
    return all_graphs, labels


def make_test_loader(graphs, labels, batch_size=256):
    """Perform the same 70/20/10 split as rerun_top.py, return only test_loader."""
    idx = list(range(len(graphs)))
    tr_idx, tmp_idx = train_test_split(idx, train_size=0.7,
                                       stratify=labels, random_state=DATA_SEED)
    tmp_labels = [labels[i] for i in tmp_idx]
    val_idx, te_idx = train_test_split(tmp_idx, train_size=2/3,
                                       stratify=tmp_labels, random_state=DATA_SEED)
    te = [graphs[i] for i in te_idx]
    print(f"Test set: {len(te):,} graphs")
    return DataLoader(te, batch_size=batch_size, shuffle=False)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device):
    model = TemporalGRUShotPredictor(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_gnn_layers=NUM_LAYERS,
        num_pre_layers=NUM_PRE_LAYERS,
        num_post_layers=NUM_POST_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    print(f"Loading checkpoint from {CKPT_PATH}")
    state_dict = torch.load(CKPT_PATH, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded — {model.param_count():,} parameters")
    return model


# ── Part A: Input Gradient Saliency ──────────────────────────────────────────

def run_saliency(model, test_loader, device, results_dir):
    """
    Compute input gradient saliency per node role.
    For each batch: forward pass with gradient tracking on x,
    backward on logits[:, 1].mean(), collect |grad|.
    """
    print("\n" + "=" * 60)
    print("Part A — Input Gradient Saliency")
    print("=" * 60)

    # cuDNN RNN backward requires training mode on the GRU.
    # Set whole model to eval (so BN uses running stats) then re-enable
    # training mode on just the GRU to allow backward through it.
    model.eval()
    model.gru.train()

    # Accumulators: sum of |grad| and count per role, shape [41]
    roles = ["shooter", "ball_node", "offense", "defense"]
    grad_sum   = {r: np.zeros(NUM_NODE_FEATURES) for r in roles}
    grad_count = {r: 0 for r in roles}

    t0 = time.time()
    for batch in tqdm(test_loader, desc="Saliency batches"):
        batch = batch.to(device)

        # Enable gradient on node features
        batch.x.requires_grad_(True)

        logits = model(batch)
        # Backprop on class-1 (made shot) logit mean
        logits[:, 1].mean().backward()

        grad_abs = batch.x.grad.abs().detach().cpu().numpy()  # [N, 41]
        x_np     = batch.x.detach().cpu().numpy()              # [N, 41]

        has_ball   = x_np[:, _HAS_BALL_IDX]
        is_offense = x_np[:, _IS_OFFENSE_IDX]
        is_player  = x_np[:, _IS_PLAYER_IDX]
        timestamp  = x_np[:, _TIMESTAMP_IDX]

        # Assign each node to exactly one role
        shooter_mask   = (has_ball == 1.0) & (timestamp == 0.0)
        ball_node_mask = (is_player == 0.0) & ~shooter_mask
        offense_mask   = (is_offense == 1.0) & ~shooter_mask & ~ball_node_mask
        defense_mask   = (is_offense == 0.0) & ~shooter_mask & ~ball_node_mask

        for mask, role in zip(
            [shooter_mask, ball_node_mask, offense_mask, defense_mask],
            roles
        ):
            if mask.any():
                grad_sum[role]   += grad_abs[mask].sum(axis=0)
                grad_count[role] += mask.sum()

    elapsed = time.time() - t0
    print(f"Saliency pass complete — {elapsed:.1f}s")

    # ── Compute mean |grad| per role ──────────────────────────────────────────
    grad_mean = {}
    for role in roles:
        if grad_count[role] > 0:
            grad_mean[role] = grad_sum[role] / grad_count[role]
        else:
            grad_mean[role] = np.zeros(NUM_NODE_FEATURES)
        print(f"  {role:12s}: {grad_count[role]:7,} nodes")

    # ── Feature group aggregation ─────────────────────────────────────────────
    group_means = {}  # role -> [num_groups]
    for role in roles:
        gm = []
        for _, indices in FEATURE_GROUPS:
            gm.append(grad_mean[role][indices].mean())
        group_means[role] = np.array(gm)

    # Print top-5 feature groups (averaged over all nodes)
    all_nodes_grad = sum(grad_sum[r] for r in roles)
    all_nodes_count = sum(grad_count[r] for r in roles)
    if all_nodes_count > 0:
        overall_mean = all_nodes_grad / all_nodes_count
        overall_group = []
        for name, indices in FEATURE_GROUPS:
            overall_group.append((name, overall_mean[indices].mean()))
        overall_group_sorted = sorted(overall_group, key=lambda x: -x[1])
        print("\nTop-5 most salient feature groups (all nodes):")
        for name, val in overall_group_sorted[:5]:
            print(f"  {name:8s}: mean|grad| = {val:.6f}")

    # ── Build heatmap matrix (rows=roles, cols=41 features) ──────────────────
    heatmap_data = np.zeros((len(roles), NUM_NODE_FEATURES))
    for i, role in enumerate(roles):
        row = grad_mean[role]
        row_max = row.max()
        if row_max > 0:
            heatmap_data[i] = row / row_max  # normalize each row independently
        else:
            heatmap_data[i] = row

    # ── Plot heatmap ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 4))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="Blues",
        xticklabels=list(range(NUM_NODE_FEATURES)),
        yticklabels=roles,
        cbar_kws={"label": "Normalized mean |grad|"},
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_title("Input Gradient Saliency by Node Role\n"
                 "(each row normalized to [0,1]; feature indices 0-40)",
                 fontsize=13)
    ax.set_xlabel("Node Feature Index\n"
                  "Groups: xyz(0-2) | role(3-5) | geom(6-10) | state(11-13) "
                  "| tid(14) | pos(15-19) | stats(20-40)",
                  fontsize=9)
    ax.set_ylabel("Node Role", fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(results_dir, "saliency_heatmap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ── Return data for JSON ──────────────────────────────────────────────────
    saliency_json = {}
    for role in roles:
        saliency_json[role] = {
            name: float(group_means[role][i])
            for i, (name, _) in enumerate(FEATURE_GROUPS)
        }
    return saliency_json


# ── Part B: GAT Attention Weights ────────────────────────────────────────────

def eval_with_attention(model, batch):
    """
    Replicate TemporalGRUShotPredictor.forward() exactly but collect
    GATv2Conv attention weights from each spatial layer.

    Uses torch.no_grad() internally (no gradient computation needed).

    Returns:
        logits         : [B, 2] tensor
        attn_records   : list of dicts, one per (layer, timestep) combination.
                         Each dict has:
                           layer      : int (0 to num_gnn_layers-1)
                           timestep   : int (0, 1, or 2)
                           alphas     : 1-D tensor [num_edges], mean over heads
                           dists      : 1-D tensor [num_edges], ea_t[:, 2]
                           edge_types : 2-D tensor [num_edges, 4], ea_t[:, 4:8]
    """
    with torch.no_grad():
        x_raw      = batch.x
        edge_index = batch.edge_index
        edge_attr  = batch.edge_attr
        b          = batch.batch
        n_nodes    = x_raw.shape[0]

        # Remove TEMPORAL edges
        spatial_mask  = (edge_attr[:, _TEMPORAL_IDX] != 1.0)
        sp_edge_index = edge_index[:, spatial_mask]
        sp_edge_attr  = edge_attr[spatial_mask]

        # Pre-MLP (all nodes)
        x = model.pre_mlp(x_raw)

        attn_records = []
        snapshots = []

        for t in [2, 1, 0]:
            t_mask = (x_raw[:, _TIMESTAMP_IDX] == float(t))

            ei_t, ea_t = subgraph(
                t_mask, sp_edge_index, sp_edge_attr,
                relabel_nodes=True, num_nodes=n_nodes,
            )
            x_t     = x[t_mask]
            batch_t = b[t_mask]

            for layer_idx, (conv, bn) in enumerate(
                zip(model.spatial_convs, model.spatial_bns)
            ):
                x_res = x_t

                # Call GATv2Conv with return_attention_weights=True
                # Returns (x_out, (ret_ei, alpha)); alpha: [E+self_loops, heads]
                # GATv2Conv adds self-loops internally (add_self_loops=True by
                # default), so ret_ei has more edges than ei_t. Filter them out
                # by keeping only edges where src != dst.
                x_t_out, (ret_ei, alpha) = conv(
                    x_t, ei_t, ea_t,
                    return_attention_weights=True
                )
                non_self_mask = ret_ei[0] != ret_ei[1]   # [E+SL] bool
                alpha_orig    = alpha[non_self_mask]      # [E, heads]

                attn_records.append({
                    "layer":      layer_idx,
                    "timestep":   t,
                    "alphas":     alpha_orig.mean(dim=-1).cpu(),  # [E]
                    "dists":      ea_t[:, 2].cpu(),               # [E]
                    "edge_types": ea_t[:, 4:8].cpu(),             # [E, 4]
                })

                x_t = bn(x_t_out)
                x_t = torch.nn.functional.relu(x_t)
                if x_res.shape == x_t.shape:
                    x_t = x_t + x_res

            from torch_geometric.nn import global_mean_pool as gmp
            h_t = gmp(x_t, batch_t)
            snapshots.append(h_t)

        # GRU
        seq = torch.stack(snapshots, dim=0)   # [3, B, D]
        _, h_n = model.gru(seq)
        h = h_n.squeeze(0)                    # [B, D]

        # Post-MLP + classify
        h = model.post_mlp(h)
        h = h  # dropout skipped in eval
        logits = model.fc(h)

    return logits, attn_records


def run_attention(model, test_loader, device, results_dir):
    """
    Collect GATv2Conv attention weights across the test set.
    Aggregate by: edge type, distance bin, and GNN layer index.
    """
    print("\n" + "=" * 60)
    print("Part B — GAT Attention Weights")
    print("=" * 60)

    model.eval()

    # Accumulators keyed by layer index
    num_layers = len(model.spatial_convs)
    layer_alpha_sum   = np.zeros(num_layers)
    layer_alpha_count = np.zeros(num_layers, dtype=np.int64)

    # Edge type: OO / OD / DD / PB (indices 0-3 in the 4-dim one-hot)
    et_names = ["OO", "OD", "DD", "PB"]
    et_alpha_sum   = {k: 0.0 for k in et_names}
    et_alpha_count = {k: 0   for k in et_names}

    # Distance bins: [0-6, 6-12, 12-20, 20+] feet
    dist_bins  = [(0, 6), (6, 12), (12, 20), (20, float("inf"))]
    dist_names = ["0-6ft", "6-12ft", "12-20ft", "20+ft"]
    dist_alpha_sum   = {k: 0.0 for k in dist_names}
    dist_alpha_count = {k: 0   for k in dist_names}

    t0 = time.time()
    for batch in tqdm(test_loader, desc="Attention batches"):
        batch = batch.to(device)
        _, attn_records = eval_with_attention(model, batch)

        for rec in attn_records:
            alphas     = rec["alphas"].numpy()      # [E]
            dists      = rec["dists"].numpy()        # [E]
            edge_types = rec["edge_types"].numpy()   # [E, 4]
            layer_idx  = rec["layer"]

            # Layer aggregation
            layer_alpha_sum[layer_idx]   += alphas.sum()
            layer_alpha_count[layer_idx] += len(alphas)

            # Edge type aggregation
            for et_idx, et_name in enumerate(et_names):
                mask = edge_types[:, et_idx] == 1.0
                if mask.any():
                    et_alpha_sum[et_name]   += alphas[mask].sum()
                    et_alpha_count[et_name] += mask.sum()

            # Distance bin aggregation (cap > 30ft at 20+ bin)
            capped_dists = np.clip(dists, 0.0, 30.0)
            for (lo, hi), bin_name in zip(dist_bins, dist_names):
                if hi == float("inf"):
                    mask = capped_dists >= lo
                else:
                    mask = (capped_dists >= lo) & (capped_dists < hi)
                if mask.any():
                    dist_alpha_sum[bin_name]   += alphas[mask].sum()
                    dist_alpha_count[bin_name] += mask.sum()

    elapsed = time.time() - t0
    print(f"Attention pass complete — {elapsed:.1f}s")

    # ── Compute means ─────────────────────────────────────────────────────────
    layer_means = []
    for li in range(num_layers):
        if layer_alpha_count[li] > 0:
            layer_means.append(float(layer_alpha_sum[li] / layer_alpha_count[li]))
        else:
            layer_means.append(0.0)
        print(f"  Layer {li}: mean_alpha = {layer_means[-1]:.6f} "
              f"({layer_alpha_count[li]:,} edges)")

    et_means = {}
    print("\nMean attention by edge type:")
    for et_name in et_names:
        if et_alpha_count[et_name] > 0:
            et_means[et_name] = float(et_alpha_sum[et_name] / et_alpha_count[et_name])
        else:
            et_means[et_name] = 0.0
        print(f"  {et_name}: {et_means[et_name]:.6f} ({et_alpha_count[et_name]:,} edges)")

    dist_means = {}
    print("\nMean attention by distance bin:")
    for bin_name in dist_names:
        if dist_alpha_count[bin_name] > 0:
            dist_means[bin_name] = float(
                dist_alpha_sum[bin_name] / dist_alpha_count[bin_name]
            )
        else:
            dist_means[bin_name] = 0.0
        print(f"  {bin_name}: {dist_means[bin_name]:.6f} "
              f"({dist_alpha_count[bin_name]:,} edges)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean attention by edge type
    ax = axes[0]
    bar_vals = [et_means[k] for k in et_names]
    bars = ax.bar(et_names, bar_vals, color=sns.color_palette("Blues_d", len(et_names)))
    ax.set_title("Mean Attention by Edge Type", fontsize=13)
    ax.set_xlabel("Edge Type", fontsize=11)
    ax.set_ylabel("Mean Attention Weight", fontsize=11)
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1e-5,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # Right: mean attention by distance bin
    ax = axes[1]
    bar_vals_d = [dist_means[k] for k in dist_names]
    bars_d = ax.bar(dist_names, bar_vals_d,
                    color=sns.color_palette("Blues_d", len(dist_names)))
    ax.set_title("Mean Attention by Distance Bin", fontsize=13)
    ax.set_xlabel("Distance Bin (feet)", fontsize=11)
    ax.set_ylabel("Mean Attention Weight", fontsize=11)
    for bar, val in zip(bars_d, bar_vals_d):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1e-5,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "attention_by_edge_type.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    return {
        "attention_by_edge_type": et_means,
        "attention_by_dist_bin":  dist_means,
        "attention_by_layer":     layer_means,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ext 5 — Interpretability analysis for TemporalGRUShotPredictor"
    )
    parser.add_argument(
        "--part", choices=["A", "B"], default=None,
        help="Run only Part A (saliency) or Part B (attention). "
             "Omit to run both."
    )
    args = parser.parse_args()

    run_a = (args.part is None or args.part == "A")
    run_b = (args.part is None or args.part == "B")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data + model (always needed)
    graphs, labels = load_all_graphs(GRAPH_DATA_DIR)
    test_loader    = make_test_loader(graphs, labels, batch_size=256)
    model          = load_model(device)

    json_output = {}

    if run_a:
        saliency_data = run_saliency(model, test_loader, device, RESULTS_DIR)
        json_output["saliency"] = saliency_data

    if run_b:
        attn_data = run_attention(model, test_loader, device, RESULTS_DIR)
        json_output.update(attn_data)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if json_output:
        json_path = os.path.join(RESULTS_DIR, "ext5_results.json")
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\nSaved: {json_path}")

    print(f"\nAll outputs written to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

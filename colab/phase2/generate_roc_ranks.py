#!/usr/bin/env python3
"""
Generate a combined ROC curve for rank01, rank02, rank03 on the test set.

Loads each checkpoint, runs inference on the same held-out test split
(70/20/10, seed=42), and plots all three ROC curves + random baseline.

Usage (on GCP VM with data):
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/generate_roc_ranks.py

Output:
    report/figures/roc_curves_ranks.pdf
    report/figures/roc_curves_ranks.png
    colab/phase2/results/eval_figures/roc_predictions_ranks.npz
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_PHASE2_DIR = os.path.dirname(os.path.abspath(__file__))
_COLAB_DIR  = os.path.dirname(_PHASE2_DIR)
_PHASE1_DIR = os.path.join(_COLAB_DIR, "phase1")
_REPO_ROOT  = os.path.dirname(_COLAB_DIR)

for _p in (_COLAB_DIR, _PHASE1_DIR, _PHASE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase2.models import TemporalGRUShotPredictor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.environ.get("SWISHNET_BASE_DIR", os.path.expanduser("~/data"))
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")
CKPT_DIR       = os.environ.get("SWISHNET_CKPT_DIR",
                   os.path.join(_PHASE2_DIR, "results", "ext_pca_gmm", "rerun_top"))
RESULTS_DIR    = os.path.join(_PHASE2_DIR, "results", "eval_figures")
FIGURES_DIR    = os.path.join(_REPO_ROOT, "report", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Configs ───────────────────────────────────────────────────────────────────
RANKS = {
    "rank01": {"hidden_dim": 256, "num_layers": 3, "dropout": 0.1,
               "label": "rank01 (nl=3, bs=512)", "color": "#1f77b4"},
    "rank02": {"hidden_dim": 256, "num_layers": 3, "dropout": 0.1,
               "label": "rank02 (nl=3, bs=256)", "color": "#ff7f0e"},
    "rank03": {"hidden_dim": 256, "num_layers": 4, "dropout": 0.1,
               "label": "rank03 (nl=4, bs=512)", "color": "#2ca02c"},
}

NUM_NODE_FEATURES = 41
NUM_HEADS         = 4
NUM_PRE_LAYERS    = 2
NUM_POST_LAYERS   = 2
DATA_SEED         = 42
BATCH_SIZE        = 512

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi":     150,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_graphs(graph_data_dir):
    print(f"Loading graphs from {graph_data_dir} ...")
    pt_files = sorted(f for f in os.listdir(graph_data_dir) if f.endswith(".pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {graph_data_dir}")
    all_graphs = []
    for fname in tqdm(pt_files, desc="Loading game files"):
        path = os.path.join(graph_data_dir, fname)
        try:
            game_graphs = torch.load(path, weights_only=False)
            if isinstance(game_graphs, list):
                all_graphs.extend(game_graphs)
            else:
                all_graphs.append(game_graphs)
        except Exception as e:
            tqdm.write(f"  SKIP {fname}: {e}")
    labels = [int(g.y.item()) for g in all_graphs]
    n = len(all_graphs)
    print(f"Loaded {n:,} graphs — FG% {100*sum(labels)/n:.1f}%")
    return all_graphs, labels


def get_test_loader(graphs, labels):
    """Exact same split as training (DATA_SEED=42, 70/20/10)."""
    idx = list(range(len(graphs)))
    _, tmp_idx = train_test_split(idx, train_size=0.7,
                                  stratify=labels, random_state=DATA_SEED)
    tmp_labels = [labels[i] for i in tmp_idx]
    _, te_idx = train_test_split(tmp_idx, train_size=2/3,
                                 stratify=tmp_labels, random_state=DATA_SEED)
    te = [graphs[i] for i in te_idx]
    print(f"Test set: {len(te):,} shots")
    return DataLoader(te, batch_size=BATCH_SIZE, shuffle=False)


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(rank_name, config, device):
    model = TemporalGRUShotPredictor(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=config["hidden_dim"],
        num_heads=NUM_HEADS,
        num_gnn_layers=config["num_layers"],
        num_pre_layers=NUM_PRE_LAYERS,
        num_post_layers=NUM_POST_LAYERS,
        dropout=config["dropout"],
    )
    ckpt_path = os.path.join(CKPT_DIR, f"rerun_{rank_name}_best.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  Loaded {rank_name}: hd={config['hidden_dim']} nl={config['num_layers']}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    all_labels, all_probs = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs  = F.softmax(logits, dim=1)[:, 1]
        all_labels.extend(batch.y.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_probs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load data once
    graphs, labels = load_all_graphs(GRAPH_DATA_DIR)
    test_loader    = get_test_loader(graphs, labels)

    # Run inference for each rank
    results = {}
    for rank_name, cfg in RANKS.items():
        print(f"\n--- {rank_name} ---")
        model = load_model(rank_name, cfg, device)
        y_true, y_prob = run_inference(model, test_loader, device)
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        results[rank_name] = {
            "y_true": y_true, "y_prob": y_prob,
            "fpr": fpr, "tpr": tpr, "auc": auc,
        }
        print(f"  AUC = {auc:.4f}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save predictions
    save_dict = {}
    for rank_name, r in results.items():
        save_dict[f"{rank_name}_y_true"] = r["y_true"]
        save_dict[f"{rank_name}_y_prob"] = r["y_prob"]
    npz_path = os.path.join(RESULTS_DIR, "roc_predictions_ranks.npz")
    np.savez(npz_path, **save_dict)
    print(f"\nSaved predictions → {npz_path}")

    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(6, 5.5))

    for rank_name, cfg in RANKS.items():
        r = results[rank_name]
        ax.plot(r["fpr"], r["tpr"], color=cfg["color"], lw=2,
                label=f"{cfg['label']} (AUC={r['auc']:.3f})")

    ax.plot([0, 1], [0, 1], color="#888888", lw=1, linestyle="--",
            label="Random (AUC=0.500)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Top-3 GNN Configurations")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.tight_layout()
    for d in (RESULTS_DIR, FIGURES_DIR):
        fig.savefig(os.path.join(d, "roc_curves_ranks.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, "roc_curves_ranks.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved ROC figure → {FIGURES_DIR}/roc_curves_ranks.pdf")
    print("Done.")


if __name__ == "__main__":
    main()

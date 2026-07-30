#!/usr/bin/env python3
"""
Generate evaluation figures for the best model (Exp 6b rank03).

Runs inference on the held-out test set using the exact same 70/20/10 split
(random_state=42) as training. Produces:
  - ROC curve (with AUC annotation)
  - Precision-Recall curve (with AP annotation)
  - Confusion matrix (at Youden's J optimal threshold)
  - Calibration curve / reliability diagram (10 bins)
  - Saves raw predictions to eval_predictions.npz for downstream use

All figures saved to report/figures/ and colab/phase2/results/eval_figures/.

Usage:
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/generate_eval_figures.py
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, accuracy_score,
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_PHASE2_DIR  = os.path.dirname(os.path.abspath(__file__))
_COLAB_DIR   = os.path.dirname(_PHASE2_DIR)
_PHASE1_DIR  = os.path.join(_COLAB_DIR, "phase1")
_REPO_ROOT   = os.path.dirname(_COLAB_DIR)

for _p in (_COLAB_DIR, _PHASE1_DIR, _PHASE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase2.models import TemporalGRUShotPredictor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.environ.get("SWISHNET_BASE_DIR", os.path.expanduser("~/data"))
GRAPH_DATA_DIR  = os.path.join(BASE_DIR, "graph_data")
CHECKPOINT_PATH = os.path.join(_PHASE2_DIR, "results", "ext_pca_gmm",
                               "rerun_top", "rerun_rank03_best.pt")
RESULTS_DIR     = os.path.join(_PHASE2_DIR, "results", "eval_figures")
FIGURES_DIR     = os.path.join(_REPO_ROOT, "report", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Best model config ─────────────────────────────────────────────────────────
BEST_CONFIG = {"hidden_dim": 256, "num_layers": 4, "dropout": 0.1}
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
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":     150,
})
BLUE   = "#2166ac"
RED    = "#d6604d"
GRAY   = "#888888"
GREEN  = "#4dac26"


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

def load_model(device):
    model = TemporalGRUShotPredictor(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=BEST_CONFIG["hidden_dim"],
        num_heads=NUM_HEADS,
        num_gnn_layers=BEST_CONFIG["num_layers"],
        num_pre_layers=NUM_PRE_LAYERS,
        num_post_layers=NUM_POST_LAYERS,
        dropout=BEST_CONFIG["dropout"],
    )
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"  hd={BEST_CONFIG['hidden_dim']} nl={BEST_CONFIG['num_layers']} "
          f"do={BEST_CONFIG['dropout']}  params={n_params:,}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """Returns (y_true, y_prob_make) as numpy arrays."""
    all_labels, all_probs = [], []
    for batch in tqdm(loader, desc="Inference"):
        batch = batch.to(device)
        logits = model(batch)                           # [B, 2]
        probs  = F.softmax(logits, dim=1)[:, 1]        # P(make)
        all_labels.extend(batch.y.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_probs)


# ── Figure generators ─────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, save_dir):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    # Youden's J optimal threshold
    j_scores = tpr - fpr
    best_idx  = np.argmax(j_scores)
    best_thr  = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"SwishNet GNN (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle="--", label="Random (AUC = 0.50)")
    ax.scatter(fpr[best_idx], tpr[best_idx], color=RED, zorder=5, s=60,
               label=f"Optimal threshold ({best_thr:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — SwishNet Best Model")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()

    for d in (save_dir, FIGURES_DIR):
        fig.savefig(os.path.join(d, "roc_curve.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, "roc_curve.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC AUC = {auc:.4f}  |  optimal threshold = {best_thr:.3f}")
    return auc, best_thr


def plot_pr_curve(y_true, y_prob, save_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()   # positive rate = P(make)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot(recall, precision, color=BLUE, lw=2, label=f"SwishNet GNN (AP = {ap:.4f})")
    ax.axhline(baseline, color=GRAY, lw=1, linestyle="--",
               label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve — SwishNet Best Model")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()

    for d in (save_dir, FIGURES_DIR):
        fig.savefig(os.path.join(d, "pr_curve.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, "pr_curve.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Average Precision = {ap:.4f}")
    return ap


def plot_confusion_matrix(y_true, y_prob, threshold, save_dir):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr  = tp / (tp + fn)   # recall / sensitivity
    tnr  = tn / (tn + fp)   # specificity
    prec = tp / (tp + fp)   # precision

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Miss", "Make"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues",
              values_format="d")
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.3f})\n"
                 f"Acc={acc:.3f}  F1={f1:.3f}  TPR={tpr:.3f}  TNR={tnr:.3f}")
    fig.tight_layout()

    for d in (save_dir, FIGURES_DIR):
        fig.savefig(os.path.join(d, "confusion_matrix.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix @ thr={threshold:.3f}: "
          f"TP={tp} FP={fp} FN={fn} TN={tn} | "
          f"Acc={acc:.3f} F1={f1:.3f} Precision={prec:.3f} Recall={tpr:.3f}")
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "accuracy": acc, "f1": f1, "precision": prec, "recall": tpr,
            "specificity": tnr, "threshold": threshold}


def plot_calibration_curve(y_true, y_prob, save_dir, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob,
                                             n_bins=n_bins, strategy="uniform")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: reliability diagram
    ax = axes[0]
    ax.plot(prob_pred, prob_true, "o-", color=BLUE, lw=2, ms=5,
            label="SwishNet GNN")
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1, label="Perfect calibration")
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.15, color=BLUE)
    ax.set_xlabel("Mean Predicted Probability (Make)")
    ax.set_ylabel("Fraction of Positives (Actual FG%)")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Right: histogram of predicted probabilities
    ax2 = axes[1]
    ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.6, color=RED,
             label="Miss", density=True)
    ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.6, color=BLUE,
             label="Make", density=True)
    ax2.set_xlabel("Predicted P(Make)")
    ax2.set_ylabel("Density")
    ax2.set_title("Score Distribution")
    ax2.legend(fontsize=9)

    fig.tight_layout()

    for d in (save_dir, FIGURES_DIR):
        fig.savefig(os.path.join(d, "calibration_curve.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, "calibration_curve.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Calibration curve saved ({n_bins} bins)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data + model
    graphs, labels = load_all_graphs(GRAPH_DATA_DIR)
    test_loader    = get_test_loader(graphs, labels)
    model          = load_model(device)

    # Run inference
    print("\nRunning inference ...")
    y_true, y_prob = run_inference(model, test_loader, device)

    # Save raw predictions
    npz_path = os.path.join(RESULTS_DIR, "eval_predictions.npz")
    np.savez(npz_path, y_true=y_true, y_prob=y_prob)
    print(f"\nSaved predictions → {npz_path}")
    print(f"  N={len(y_true):,}  FG%={100*y_true.mean():.1f}%  "
          f"mean_pred={y_prob.mean():.3f}")

    # Generate figures
    print("\n--- Generating figures ---")
    auc, best_thr = plot_roc_curve(y_true, y_prob, RESULTS_DIR)
    ap            = plot_pr_curve(y_true, y_prob, RESULTS_DIR)
    cm_stats      = plot_confusion_matrix(y_true, y_prob, best_thr, RESULTS_DIR)
    plot_calibration_curve(y_true, y_prob, RESULTS_DIR)

    # Save summary JSON
    summary = {
        "model":        "rerun_rank03",
        "checkpoint":   CHECKPOINT_PATH,
        "n_test":       int(len(y_true)),
        "roc_auc":      float(auc),
        "avg_precision": float(ap),
        **cm_stats,
    }
    summary_path = os.path.join(RESULTS_DIR, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {summary_path}")

    print(f"\nAll figures saved to:")
    print(f"  {RESULTS_DIR}/")
    print(f"  {FIGURES_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()

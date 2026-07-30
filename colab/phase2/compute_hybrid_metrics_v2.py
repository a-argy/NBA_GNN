#!/usr/bin/env python3
"""
Compute Acc, F1, confusion matrices for hybrid pipeline models.
Uses the same alignment logic as ext_pca_gmm.py to ensure correct row matching.

Usage (on GCP VM):
    cd ~/SwishNet
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/compute_hybrid_metrics_v2.py
"""

import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

_PHASE2_DIR = os.path.dirname(os.path.abspath(__file__))
_COLAB_DIR  = os.path.dirname(_PHASE2_DIR)
_REPO_ROOT  = os.path.dirname(_COLAB_DIR)

for _p in (_COLAB_DIR, _PHASE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, precision_score, recall_score)
from xgboost import XGBClassifier

BASE_DIR       = os.environ.get("SWISHNET_BASE_DIR", os.path.expanduser("~/data"))
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")
BASELINE_DIR   = os.path.join(BASE_DIR, "baseline_data")
RESULTS_DIR    = os.path.join(_PHASE2_DIR, "results")
FIGURES_DIR    = os.path.join(_REPO_ROOT, "report", "figures")
EVAL_DIR       = os.path.join(RESULTS_DIR, "eval_figures")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12, "figure.dpi": 150,
})


def plot_confusion_matrix(cm, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Miss", "Make"], yticklabels=["Miss", "Make"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    for d in (EVAL_DIR, FIGURES_DIR):
        fig.savefig(os.path.join(d, filename + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, filename + ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}.pdf")


def load_all_graphs(graph_data_dir):
    """Load graphs, track which game files failed."""
    pt_files = sorted(f for f in os.listdir(graph_data_dir) if f.endswith(".pt"))
    all_graphs = []
    skipped_ids = set()
    for fname in tqdm(pt_files, desc="Loading graphs"):
        path = os.path.join(graph_data_dir, fname)
        game_id = fname.replace(".pt", "").replace("graphs_", "")
        try:
            data = torch.load(path, weights_only=False)
            if isinstance(data, list):
                all_graphs.extend(data)
            else:
                all_graphs.append(data)
        except Exception as e:
            skipped_ids.add(game_id)
            tqdm.write(f"  SKIP {fname}: {e}")
    labels = np.array([int(g.y.item()) for g in all_graphs])
    print(f"Loaded {len(all_graphs):,} graphs, skipped {len(skipped_ids)} game files")
    return all_graphs, labels, skipped_ids


def align_static_features(skipped_ids, baseline_dir):
    """Filter static/velocity features to match loaded graphs."""
    X_static = np.load(os.path.join(baseline_dir, "X_static.npy")).astype(np.float32)
    X_vel    = np.load(os.path.join(baseline_dir, "X_full.npy")).astype(np.float32)
    y        = np.load(os.path.join(baseline_dir, "y.npy")).astype(int)

    if skipped_ids:
        idx_path = os.path.join(baseline_dir, "shot_index_map.json")
        with open(idx_path) as f:
            shot_index_map = json.load(f)
        keep = np.array([
            i for i, entry in enumerate(shot_index_map)
            if entry["game_id"] not in skipped_ids
        ])
        X_static = X_static[keep]
        X_vel = X_vel[keep]
        y = y[keep]
        print(f"Filtered to {len(keep):,} rows (excluded {len(shot_index_map)-len(keep):,})")

    return X_static, X_vel, y


def run_xgb_cv(X, y, name):
    """5-fold CV with full metrics."""
    XGB_PARAMS = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        eval_metric="auc", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_auc, fold_acc, fold_f1, fold_prec, fold_rec = [], [], [], [], []
    all_y_true, all_y_pred = [], []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        clf = XGBClassifier(**XGB_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[tr_idx], y[tr_idx],
                    eval_set=[(X[te_idx], y[te_idx])], verbose=False)
        prob = clf.predict_proba(X[te_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        fold_auc.append(roc_auc_score(y[te_idx], prob))
        fold_acc.append(accuracy_score(y[te_idx], pred))
        fold_f1.append(f1_score(y[te_idx], pred))
        fold_prec.append(precision_score(y[te_idx], pred, zero_division=0))
        fold_rec.append(recall_score(y[te_idx], pred, zero_division=0))
        all_y_true.extend(y[te_idx].tolist())
        all_y_pred.extend(pred.tolist())
        print(f"  fold {fold}: AUC={fold_auc[-1]:.4f} Acc={fold_acc[-1]:.4f} F1={fold_f1[-1]:.4f}")

    results = {
        "auc_mean": float(np.mean(fold_auc)), "auc_std": float(np.std(fold_auc)),
        "acc_mean": float(np.mean(fold_acc)), "acc_std": float(np.std(fold_acc)),
        "f1_mean": float(np.mean(fold_f1)), "f1_std": float(np.std(fold_f1)),
        "prec_mean": float(np.mean(fold_prec)), "rec_mean": float(np.mean(fold_rec)),
    }
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\n  {name}: AUC={results['auc_mean']:.4f}±{results['auc_std']:.4f}"
          f"  Acc={results['acc_mean']*100:.1f}%  F1={results['f1_mean']:.4f}")
    print(f"  CM:\n{cm}")
    return results, cm


def main():
    # Load graphs (only need labels + skipped_ids for alignment)
    graphs, graph_labels, skipped_ids = load_all_graphs(GRAPH_DATA_DIR)
    n_graphs = len(graphs)
    del graphs  # free memory

    # Align static features
    X_static, X_vel, y = align_static_features(skipped_ids, BASELINE_DIR)
    assert len(X_static) == n_graphs, f"Mismatch: {len(X_static)} vs {n_graphs}"
    assert np.array_equal(y, graph_labels), "Label mismatch!"
    print(f"Alignment confirmed: {n_graphs:,} shots\n")

    # Load GMM responsibilities
    resp = np.load(os.path.join(RESULTS_DIR, "gmm_responsibilities.npy"))
    assert len(resp) == n_graphs, f"GMM rows {len(resp)} != graphs {n_graphs}"

    # Build augmented sets
    X_static_gmm = np.concatenate([X_static, resp], axis=1)
    X_vel_gmm = np.concatenate([X_vel, resp], axis=1)

    all_results = {}

    # Static-38 + GMM
    print("=" * 50)
    print("Static-38 + GMM (10 soft)")
    print("=" * 50)
    res, cm = run_xgb_cv(X_static_gmm, y, "Static-38+GMM")
    all_results["Static-38+GMM"] = res
    plot_confusion_matrix(cm, "Confusion Matrix — Static-38 + GMM", "cm_Static_38_plus_GMM")

    # Velocity-65 + GMM
    print("\n" + "=" * 50)
    print("Velocity-65 + GMM (10 soft)")
    print("=" * 50)
    res, cm = run_xgb_cv(X_vel_gmm, y, "Velocity-65+GMM")
    all_results["Velocity-65+GMM"] = res
    plot_confusion_matrix(cm, "Confusion Matrix — Velocity-65 + GMM", "cm_Velocity_65_plus_GMM")

    # Save
    out_path = os.path.join(EVAL_DIR, "hybrid_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

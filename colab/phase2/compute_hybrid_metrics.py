#!/usr/bin/env python3
"""
Compute Acc, F1, and confusion matrices for:
1. Static-38 + GMM (10 soft) hybrid
2. Velocity-65 + GMM (10 soft) hybrid
3. GNN rank03 standalone (from checkpoint)

Also re-confirm AUC values and save confusion matrix figures.

Usage (on GCP VM):
    cd ~/SwishNet
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/compute_hybrid_metrics.py
"""

import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path setup ───────────────────────────────────────────────────────────────
_PHASE2_DIR = os.path.dirname(os.path.abspath(__file__))
_COLAB_DIR  = os.path.dirname(_PHASE2_DIR)
_REPO_ROOT  = os.path.dirname(_COLAB_DIR)

for _p in (_COLAB_DIR, _PHASE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, precision_score, recall_score)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.environ.get("SWISHNET_BASE_DIR", os.path.expanduser("~/data"))
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")
STATIC_DIR     = os.path.join(BASE_DIR, "baseline_data")
GMM_DIR        = os.environ.get("SWISHNET_GMM_DIR",
                   os.path.join(_PHASE2_DIR, "results", "ext_pca_gmm"))
CKPT_DIR       = os.environ.get("SWISHNET_CKPT_DIR",
                   os.path.join(GMM_DIR, "rerun_top"))
FIGURES_DIR    = os.path.join(_REPO_ROOT, "report", "figures")
RESULTS_DIR    = os.path.join(_PHASE2_DIR, "results", "eval_figures")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def plot_confusion_matrix(cm, title, filename):
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Miss", "Make"], yticklabels=["Miss", "Make"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    for d in (RESULTS_DIR, FIGURES_DIR):
        fig.savefig(os.path.join(d, filename + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, filename + ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Hybrid pipeline metrics (XGBoost + GMM)
# ══════════════════════════════════════════════════════════════════════════════

def run_hybrid_metrics():
    print("=" * 60)
    print("Part 1: Hybrid Pipeline Metrics")
    print("=" * 60)

    from xgboost import XGBClassifier

    # Load aligned data (same as ext_pca_gmm.py produced)
    resp = np.load(os.path.join(GMM_DIR, "gmm_responsibilities.npy"))
    n = resp.shape[0]
    print(f"GMM responsibilities: {resp.shape}")

    # Load static features — need to align with graph indices
    X_static_full = np.load(os.path.join(STATIC_DIR, "X_static.npy"))
    y_full = np.load(os.path.join(STATIC_DIR, "y.npy"))
    X_vel_full = np.load(os.path.join(STATIC_DIR, "X_full.npy"))
    print(f"X_static_full: {X_static_full.shape}, X_vel_full: {X_vel_full.shape}, y_full: {y_full.shape}")

    # The GMM was fit on GRU embeddings from graph data.
    # ext_pca_gmm.py loads graphs and static features in parallel,
    # using only the intersection. We need the same alignment.
    # Load the index map if available
    idx_map_path = os.path.join(STATIC_DIR, "shot_index_map.json")
    if os.path.exists(idx_map_path):
        with open(idx_map_path) as f:
            idx_map = json.load(f)
        print(f"Index map: {len(idx_map)} entries")

    # The embeddings were extracted from all graphs in order.
    # The static features correspond to the same shots if pipeline ensured alignment.
    # Since n_graphs (86425) < n_static (87147), we need the graph indices.
    # Let's check if ext_pca_gmm saved aligned X_static
    aligned_static_path = os.path.join(GMM_DIR, "X_static_aligned.npy")
    aligned_vel_path = os.path.join(GMM_DIR, "X_vel_aligned.npy")
    aligned_y_path = os.path.join(GMM_DIR, "y_aligned.npy")

    if os.path.exists(aligned_static_path):
        X_static = np.load(aligned_static_path)
        y = np.load(aligned_y_path)
        X_vel = np.load(aligned_vel_path) if os.path.exists(aligned_vel_path) else None
        print(f"Loaded pre-aligned data: {X_static.shape}")
    else:
        # Re-derive alignment: load graphs to get the shot keys, match to static
        print("No pre-aligned data found. Re-deriving alignment from graph files...")
        # Load graph labels to align
        import torch
        from tqdm import tqdm
        pt_files = sorted(f for f in os.listdir(GRAPH_DATA_DIR) if f.endswith(".pt"))
        graph_labels = []
        graph_indices = []  # track which static index each graph corresponds to
        for fname in tqdm(pt_files, desc="Loading graph labels"):
            path = os.path.join(GRAPH_DATA_DIR, fname)
            try:
                game_graphs = torch.load(path, weights_only=False)
                if isinstance(game_graphs, list):
                    for g in game_graphs:
                        graph_labels.append(int(g.y.item()))
                else:
                    graph_labels.append(int(game_graphs.y.item()))
            except:
                pass

        n_graphs = len(graph_labels)
        print(f"Loaded {n_graphs} graph labels")

        if n_graphs == n:
            # Use first n rows of static (pipeline guarantees alignment)
            X_static = X_static_full[:n]
            X_vel = X_vel_full[:n]
            y = y_full[:n]
            print(f"Using first {n} rows (pipeline alignment)")
        else:
            print(f"WARNING: graph count {n_graphs} != resp count {n}")
            X_static = X_static_full[:n]
            X_vel = X_vel_full[:n]
            y = y_full[:n]

        # Save aligned for future use
        np.save(aligned_static_path, X_static)
        np.save(aligned_vel_path, X_vel)
        np.save(aligned_y_path, y)
        print("Saved aligned arrays")

    # Build augmented feature sets
    X_static_gmm = np.concatenate([X_static, resp], axis=1)
    print(f"Static-38+GMM: {X_static_gmm.shape}")

    if X_vel is not None:
        X_vel_gmm = np.concatenate([X_vel, resp], axis=1)
        print(f"Velocity-65+GMM: {X_vel_gmm.shape}")

    XGB_PARAMS = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        eval_metric="auc", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    datasets = [("Static-38+GMM", X_static_gmm)]
    if X_vel is not None:
        datasets.append(("Velocity-65+GMM", X_vel_gmm))

    all_results = {}
    for feat_name, X in datasets:
        print(f"\n--- {feat_name} ---")
        fold_auc, fold_acc, fold_f1, fold_prec, fold_rec = [], [], [], [], []
        all_y_true, all_y_pred, all_y_prob = [], [], []

        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
            clf = XGBClassifier(**XGB_PARAMS)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X[tr_idx], y[tr_idx],
                        eval_set=[(X[te_idx], y[te_idx])],
                        verbose=False)
            prob = clf.predict_proba(X[te_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)

            fold_auc.append(roc_auc_score(y[te_idx], prob))
            fold_acc.append(accuracy_score(y[te_idx], pred))
            fold_f1.append(f1_score(y[te_idx], pred))
            fold_prec.append(precision_score(y[te_idx], pred, zero_division=0))
            fold_rec.append(recall_score(y[te_idx], pred, zero_division=0))

            all_y_true.extend(y[te_idx].tolist())
            all_y_pred.extend(pred.tolist())
            all_y_prob.extend(prob.tolist())

            print(f"  fold {fold}: AUC={fold_auc[-1]:.4f} Acc={fold_acc[-1]:.4f} F1={fold_f1[-1]:.4f}")

        results = {
            "auc_mean": float(np.mean(fold_auc)), "auc_std": float(np.std(fold_auc)),
            "acc_mean": float(np.mean(fold_acc)), "acc_std": float(np.std(fold_acc)),
            "f1_mean": float(np.mean(fold_f1)), "f1_std": float(np.std(fold_f1)),
            "prec_mean": float(np.mean(fold_prec)), "rec_mean": float(np.mean(fold_rec)),
        }
        all_results[feat_name] = results
        print(f"\n  {feat_name}: AUC={results['auc_mean']:.4f}±{results['auc_std']:.4f}"
              f"  Acc={results['acc_mean']*100:.1f}%  F1={results['f1_mean']:.4f}")

        # Confusion matrix (aggregated across all folds)
        cm = confusion_matrix(all_y_true, all_y_pred)
        safe_name = feat_name.replace("+", "_plus_").replace("-", "_").replace(" ", "_")
        plot_confusion_matrix(cm, f"Confusion Matrix — {feat_name}", f"cm_{safe_name}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "hybrid_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved hybrid metrics → {out_path}")
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: GNN rank03 confusion matrix
# ══════════════════════════════════════════════════════════════════════════════

def run_rank03_cm():
    print("\n" + "=" * 60)
    print("Part 2: GNN rank03 Confusion Matrix")
    print("=" * 60)

    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm
    from phase2.models import TemporalGRUShotPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load graphs
    print("Loading graphs...")
    pt_files = sorted(f for f in os.listdir(GRAPH_DATA_DIR) if f.endswith(".pt"))
    all_graphs = []
    for fname in tqdm(pt_files, desc="Loading"):
        path = os.path.join(GRAPH_DATA_DIR, fname)
        try:
            game_graphs = torch.load(path, weights_only=False)
            if isinstance(game_graphs, list):
                all_graphs.extend(game_graphs)
            else:
                all_graphs.append(game_graphs)
        except:
            pass
    labels = [int(g.y.item()) for g in all_graphs]
    print(f"Loaded {len(all_graphs)} graphs")

    # Same split as training
    idx = list(range(len(all_graphs)))
    _, tmp_idx = train_test_split(idx, train_size=0.7, stratify=labels, random_state=42)
    tmp_labels = [labels[i] for i in tmp_idx]
    _, te_idx = train_test_split(tmp_idx, train_size=2/3, stratify=tmp_labels, random_state=42)
    te_graphs = [all_graphs[i] for i in te_idx]
    print(f"Test set: {len(te_graphs)} shots")

    test_loader = DataLoader(te_graphs, batch_size=512, shuffle=False)

    # Load rank03 model
    model = TemporalGRUShotPredictor(
        num_node_features=41, hidden_dim=256, num_heads=4,
        num_gnn_layers=4, num_pre_layers=2, num_post_layers=2, dropout=0.1,
    )
    ckpt_path = os.path.join(CKPT_DIR, "rerun_rank03_best.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("Loaded rank03 checkpoint")

    # Inference
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_labels.extend(batch.y.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    print(f"\nrank03: AUC={auc:.4f}  Acc={acc*100:.1f}%  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix:\n{cm}")
    plot_confusion_matrix(cm, "Confusion Matrix — GNN rank03", "cm_gnn_rank03")

    results = {"auc": auc, "acc": acc, "f1": f1, "precision": prec, "recall": rec,
               "confusion_matrix": cm.tolist()}
    out_path = os.path.join(RESULTS_DIR, "rank03_test_metrics.json")
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "confusion_matrix"}, f, indent=2)
    print(f"Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    hybrid_results = run_hybrid_metrics()
    run_rank03_cm()
    print("\n✓ All done.")

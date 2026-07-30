#!/usr/bin/env python3
"""
Extensions 3 & 4 — PCA of GRU embeddings + GMM shot clustering.

Extension 3: Extract 256-dim GRU hidden states from the best model (Exp 6b
rank03) via a forward hook, reduce to 2D via PCA, scatter-plot make vs miss.

Extension 4: Fit a GMM on the raw embeddings (diag covariance, BIC selection),
cross-tab clusters against static shot features, and re-run XGBoost with soft
GMM responsibilities appended to X_static.

Usage:
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/ext_pca_gmm.py
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/ext_pca_gmm.py --skip-xgb
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/ext_pca_gmm.py --batch-size 256

Outputs (all in colab/phase2/results/ and synced to gs://swishnet-nba/results/ext_pca_gmm/):
    gru_embeddings.npy         — raw [N, 256] embeddings
    pca_embeddings.png         — 2D PCA scatter (make vs miss)
    ext3_pca_result.json       — explained variance
    gmm_bic_curve.png          — BIC vs K plot
    gmm_cluster_summary.csv    — per-cluster shot archetype stats
    gmm_responsibilities.npy   — [N, K] soft GMM assignments
    ext4_gmm_bic.json          — BIC scores and best K
    ext4_xgb_augmented.json    — XGBoost AUC baseline vs augmented
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_PHASE2_DIR  = os.path.dirname(os.path.abspath(__file__))
_COLAB_DIR   = os.path.dirname(_PHASE2_DIR)
_PHASE1_DIR  = os.path.join(_COLAB_DIR, "phase1")
_PIPELINE_DIR = os.path.join(_COLAB_DIR, "pipeline")
for _p in (_COLAB_DIR, _PHASE1_DIR, _PHASE2_DIR, _PIPELINE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase2.models import TemporalGRUShotPredictor
try:
    from pipeline.gcp_utils import sync_dir as _gcs_sync_dir
    _GCS_AVAILABLE = True
except ImportError:
    _GCS_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.environ.get("SWISHNET_BASE_DIR", os.path.expanduser("~/data"))
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")
BASELINE_DIR   = os.path.join(BASE_DIR, "baseline_data")
RERUN_DIR      = os.path.join(_PHASE2_DIR, "results", "rerun_top")
RESULTS_DIR    = os.path.join(_PHASE2_DIR, "results")

# ── Best model config (Exp 6b rank03) ────────────────────────────────────────
BEST_CONFIG = {
    "name":       "rerun_rank03",
    "hidden_dim": 256,
    "num_layers": 4,
    "dropout":    0.1,
}
NUM_NODE_FEATURES = 41
NUM_HEADS         = 4
NUM_PRE_LAYERS    = 2
NUM_POST_LAYERS   = 2

GMM_K_RANGE = [3, 4, 5, 6, 8, 10]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_graphs(graph_data_dir):
    """Load all .pt graph files, skipping unreadable ones.

    Returns:
        all_graphs  : list of Data objects
        skipped_ids : set of game_id strings extracted from skipped filenames
                      (e.g. {'0021500066', ...}) for downstream X_static filtering
    """
    print(f"Loading graphs from {graph_data_dir} ...")
    pt_files = sorted(f for f in os.listdir(graph_data_dir) if f.endswith(".pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {graph_data_dir}")
    all_graphs = []
    skipped_ids = set()
    for fname in tqdm(pt_files, desc="Loading game files"):
        path = os.path.join(graph_data_dir, fname)
        try:
            game_graphs = torch.load(path, weights_only=False)
            if isinstance(game_graphs, list):
                all_graphs.extend(game_graphs)
            else:
                all_graphs.append(game_graphs)
        except Exception as e:
            # Extract game_id from filename: graphs_0021500066.pt -> 0021500066
            game_id = fname.replace("graphs_", "").replace(".pt", "")
            skipped_ids.add(game_id)
            tqdm.write(f"  SKIP {fname}: {e}")
    if skipped_ids:
        print(f"Skipped {len(skipped_ids)} unreadable game(s): {sorted(skipped_ids)}")
    print(f"Loaded {len(all_graphs):,} graphs")
    return all_graphs, skipped_ids


def load_model(checkpoint_path, cfg, device):
    model = TemporalGRUShotPredictor(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=cfg["hidden_dim"],
        num_heads=NUM_HEADS,
        num_gnn_layers=cfg["num_layers"],
        num_pre_layers=NUM_PRE_LAYERS,
        num_post_layers=NUM_POST_LAYERS,
        dropout=cfg["dropout"],
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Config: hd={cfg['hidden_dim']} nl={cfg['num_layers']} do={cfg['dropout']}")
    print(f"  Params: {model.param_count():,}")
    return model


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, graphs, batch_size, device):
    """Extract GRU final hidden states [N, D] via forward hook.

    The hook fires after model.gru and captures h_n (the final hidden state)
    without modifying the computation graph. shuffle=False preserves row order
    so embeddings[i] corresponds to graphs[i] and X_static[i].
    """
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    store = {}

    def _gru_hook(module, inp, out):
        _, h_n = out                                    # h_n: [1, B, D]
        store["h"] = h_n.squeeze(0).detach().cpu()     # [B, D]

    hook = model.gru.register_forward_hook(_gru_hook)

    all_emb, all_y = [], []
    for batch in tqdm(loader, desc="Extracting embeddings"):
        batch = batch.to(device)
        model(batch)                    # triggers hook
        all_emb.append(store["h"])
        all_y.append(batch.y.cpu())

    hook.remove()

    embeddings = torch.cat(all_emb).numpy()     # [N, D]
    labels     = torch.cat(all_y).numpy()       # [N]
    print(f"Embeddings: {embeddings.shape}  labels: {labels.shape}")
    return embeddings, labels


# ── Extension 3: PCA ──────────────────────────────────────────────────────────

def run_pca(embeddings, labels, out_dir):
    print("\n── Extension 3: PCA of GRU embeddings ──")
    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(embeddings)
    var1, var2 = pca.explained_variance_ratio_
    total_var  = var1 + var2
    print(f"  PC1: {var1*100:.2f}%  PC2: {var2*100:.2f}%  total: {total_var*100:.2f}%")
    if total_var < 0.10:
        print("  WARNING: <10% variance explained — 2D plot may look noisy. "
              "Consider UMAP for a follow-up.")

    n_make = (labels == 1).sum()
    n_miss = (labels == 0).sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color, name, n in [
        (0, "#d62728", "Miss", n_miss),
        (1, "#2ca02c", "Make", n_make),
    ]:
        mask = labels == label
        ax.scatter(Z[mask, 0], Z[mask, 1], c=color,
                   label=f"{name} (n={n:,})",
                   alpha=0.3, s=4, rasterized=True)

    ax.set_xlabel(f"PC1 ({var1*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var2*100:.1f}% var)")
    ax.set_title(
        f"PCA of GRU embeddings — {len(labels):,} shots\n"
        f"Exp 6b rank03  (hd=256, nl=4, test AUC=0.6451)"
    )
    ax.legend(markerscale=3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "pca_embeddings.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    result = {
        "pc1_var":       float(var1),
        "pc2_var":       float(var2),
        "total_2pc_var": float(total_var),
    }
    with open(os.path.join(out_dir, "ext3_pca_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return Z, result


# ── Extension 4: GMM ──────────────────────────────────────────────────────────

def run_gmm(embeddings, labels, X_static, out_dir, skip_xgb=False):
    print("\n── Extension 4: GMM clustering ──")

    # ── BIC sweep ────────────────────────────────────────────────────────────
    # diag covariance: 256 params/component vs 256^2=65K for full — much faster
    # and more stable at 87K samples × 256 dims.
    bic_scores = {}
    for k in GMM_K_RANGE:
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                              random_state=42, max_iter=300)
        gmm.fit(embeddings)
        bic_scores[k] = float(gmm.bic(embeddings))
        print(f"  K={k:2d}  BIC={bic_scores[k]:.0f}")

    best_k = min(bic_scores, key=bic_scores.get)
    print(f"\n  Best K by BIC: {best_k}")

    # ── Refit best K ──────────────────────────────────────────────────────────
    gmm_best = GaussianMixture(n_components=best_k, covariance_type="diag",
                               random_state=42, max_iter=300)
    cluster_labels   = gmm_best.fit_predict(embeddings)    # [N]
    responsibilities = gmm_best.predict_proba(embeddings)  # [N, K]

    # ── BIC curve ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = sorted(bic_scores.keys())
    ax.plot(ks, [bic_scores[k] for k in ks], "o-", color="#1f77b4")
    ax.axvline(best_k, linestyle="--", color="gray", alpha=0.7,
               label=f"Best K={best_k}")
    ax.set_xlabel("Number of GMM components (K)")
    ax.set_ylabel("BIC (lower = better)")
    ax.set_title("GMM BIC curve — GRU embedding space")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gmm_bic_curve.png"), dpi=150)
    plt.close()
    print(f"  BIC curve saved: {out_dir}/gmm_bic_curve.png")

    # ── Cross-tab with static features ────────────────────────────────────────
    # X_static column indices (from baseline_features.py / STATIC_FEATURE_NAMES)
    #   0  dist_to_rim  2  angle_to_basket  3  is_3pt
    #   4  shot_clock   5  game_clock       6  quarter
    import pandas as pd
    df = pd.DataFrame({
        "cluster":     cluster_labels,
        "label":       labels,
        "dist_to_rim": X_static[:, 0],
        "angle":       X_static[:, 2],
        "is_3pt":      X_static[:, 3],
        "shot_clock":  X_static[:, 4],
        "game_clock":  X_static[:, 5],
        "quarter":     X_static[:, 6],
    })

    summary = df.groupby("cluster").agg(
        n          = ("label",       "count"),
        fg_pct     = ("label",       "mean"),
        avg_dist   = ("dist_to_rim", "mean"),
        avg_angle  = ("angle",       "mean"),
        pct_3pt    = ("is_3pt",      "mean"),
        avg_sclock = ("shot_clock",  "mean"),
    ).round(3)

    print(f"\nCluster summary (K={best_k}):")
    print(summary.to_string())

    csv_path = os.path.join(out_dir, "gmm_cluster_summary.csv")
    summary.to_csv(csv_path)
    print(f"\n  Cluster summary saved: {csv_path}")

    npy_path = os.path.join(out_dir, "gmm_responsibilities.npy")
    np.save(npy_path, responsibilities.astype(np.float32))
    print(f"  Responsibilities saved: {npy_path}")

    with open(os.path.join(out_dir, "ext4_gmm_bic.json"), "w") as f:
        json.dump({"bic_scores": bic_scores, "best_k": best_k}, f, indent=2)

    # ── XGBoost augmentation ──────────────────────────────────────────────────
    if not skip_xgb:
        run_xgb_augmented(X_static, responsibilities, labels, out_dir)

    return cluster_labels, responsibilities, summary


# ── XGBoost augmentation ──────────────────────────────────────────────────────

def run_xgb_augmented(X_static, responsibilities, labels, out_dir):
    """Re-run XGBoost baseline vs X_static + GMM soft memberships."""
    print("\n── XGBoost augmentation (Static-38 vs Static-38 + GMM responsibilities) ──")
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  [SKIP] xgboost not installed — pip install xgboost")
        return

    from sklearn.model_selection import StratifiedKFold

    XGB_PARAMS = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        eval_metric="auc", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )

    X_augmented = np.concatenate([X_static, responsibilities], axis=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for feat_name, X in [
        ("Static-38",        X_static),
        ("Static-38+GMM",    X_augmented),
    ]:
        fold_auc = []
        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, labels), 1):
            clf = XGBClassifier(**XGB_PARAMS)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X[tr_idx], labels[tr_idx],
                        eval_set=[(X[te_idx], labels[te_idx])],
                        verbose=False)
            prob = clf.predict_proba(X[te_idx])[:, 1]
            fold_auc.append(roc_auc_score(labels[te_idx], prob))
            print(f"  [{feat_name}] fold {fold}/5  AUC {fold_auc[-1]:.4f}")

        results[feat_name] = {
            "auc_mean": float(np.mean(fold_auc)),
            "auc_std":  float(np.std(fold_auc)),
        }

    print("\n  Results:")
    baseline_auc = results["Static-38"]["auc_mean"]
    for name, r in results.items():
        delta = r["auc_mean"] - baseline_auc
        print(f"  {name:<20}  AUC {r['auc_mean']:.4f}±{r['auc_std']:.4f}  Δ={delta:+.4f}")
    print(f"\n  XGBoost Ext 1 reference (Velocity-65): 0.6501")

    out_path = os.path.join(out_dir, "ext4_xgb_augmented.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extensions 3 & 4: PCA + GMM on GRU embeddings"
    )
    parser.add_argument("--skip-xgb", action="store_true",
                        help="Skip XGBoost augmentation step")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for embedding inference (default: 512)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(RERUN_DIR, f"{BEST_CONFIG['name']}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Expected at: colab/phase2/results/rerun_top/rerun_rank03_best.pt\n"
            f"Sync from GCS first:\n"
            f"  gsutil -m cp gs://swishnet-nba/results/rerun_top/rerun_rank03_best.pt "
            f"{RERUN_DIR}/"
        )
    model = load_model(ckpt_path, BEST_CONFIG, device)

    # ── Load graphs ────────────────────────────────────────────────────────────
    graphs, skipped_ids = load_all_graphs(GRAPH_DATA_DIR)

    # ── Load static features ───────────────────────────────────────────────────
    print(f"\nLoading static features from {BASELINE_DIR} ...")
    X_static = np.load(os.path.join(BASELINE_DIR, "X_static.npy")).astype(np.float32)
    y        = np.load(os.path.join(BASELINE_DIR, "y.npy")).astype(int)

    # If some game files were unreadable, filter X_static/y using shot_index_map
    # so that row i of X_static corresponds to graph i.
    if skipped_ids:
        index_map_path = os.path.join(BASELINE_DIR, "shot_index_map.json")
        if os.path.exists(index_map_path):
            import json
            with open(index_map_path) as f:
                shot_index_map = json.load(f)
            keep = np.array([
                i for i, entry in enumerate(shot_index_map)
                if entry["game_id"] not in skipped_ids
            ])
            X_static = X_static[keep]
            y        = y[keep]
            print(f"Filtered X_static/y to {len(keep):,} rows "
                  f"(excluded {len(shot_index_map) - len(keep):,} shots "
                  f"from {len(skipped_ids)} unreadable game(s))")
        else:
            print("WARNING: shot_index_map.json not found — "
                  "X_static cross-tab and XGBoost augmentation will be skipped.")

    assert len(graphs) == len(X_static) == len(y), (
        f"Row count mismatch after filtering: graphs={len(graphs)}, "
        f"X_static={len(X_static)}, y={len(y)}"
    )
    print(f"Row alignment confirmed: {len(graphs):,} shots")

    # ── Extract GRU embeddings ────────────────────────────────────────────────
    embeddings, labels = extract_embeddings(model, graphs, args.batch_size, device)

    assert np.array_equal(labels, y), (
        "Label mismatch between graph .y and filtered y — "
        "graph loading order does not match shot_index_map order."
    )
    print("Label alignment confirmed.")

    emb_path = os.path.join(RESULTS_DIR, "gru_embeddings.npy")
    np.save(emb_path, embeddings.astype(np.float32))
    print(f"Raw embeddings saved: {emb_path}")

    # ── Extension 3: PCA ──────────────────────────────────────────────────────
    run_pca(embeddings, labels, RESULTS_DIR)

    # ── Extension 4: GMM ──────────────────────────────────────────────────────
    run_gmm(embeddings, labels, X_static, RESULTS_DIR, skip_xgb=args.skip_xgb)

    print("\nAll done. Outputs in:", RESULTS_DIR)

    # ── Sync all outputs to GCS ────────────────────────────────────────────────
    GCS_PREFIX = "results/ext_pca_gmm"
    if _GCS_AVAILABLE:
        print(f"\nSyncing results to gs://swishnet-nba/{GCS_PREFIX}/ ...")
        try:
            _gcs_sync_dir(RESULTS_DIR, GCS_PREFIX)
            print(f"  Sync complete: gs://swishnet-nba/{GCS_PREFIX}/")
        except Exception as exc:
            print(f"  GCS sync failed (files are still saved locally): {exc}")
    else:
        print("\nGCS sync skipped (google-cloud-storage not installed).")
        print(f"To sync manually:\n"
              f"  gsutil -m rsync -r {RESULTS_DIR} "
              f"gs://swishnet-nba/{GCS_PREFIX}/")


if __name__ == "__main__":
    main()

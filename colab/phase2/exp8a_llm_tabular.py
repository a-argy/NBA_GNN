#!/usr/bin/env python3
"""
Exp 8a — LLM Player Embeddings: Tabular Gate.

Tests whether replacing/augmenting the 21 shooter season-stats features
in X_static with a projected LLM text embedding improves AUC over the
LR baseline (Exp 0a: 0.6377).

Two modes:

  --build-emb-array   Scan shot files and build X_emb.npy aligned row-for-row
                      with the pipeline's X_static.npy.  Mirrors Step 2's
                      filtering exactly (both graph + feature must succeed).
                      Requires --shots-dir.

  (default)           Load pre-built arrays and run 4-config × 5-fold CV.

Experiment configurations:
  8a-LR-replace   Static-17 + PCA(emb, proj_dim)  →  LR Lasso
  8a-LR-augment   Static-38 + PCA(emb, proj_dim)  →  LR Lasso
  8a-MLP-replace  Static-17 + PCA(emb, proj_dim)  →  MLP(256, 64)
  8a-MLP-augment  Static-38 + PCA(emb, proj_dim)  →  MLP(256, 64)

The PCA is fit on the training fold only (no leakage).

Gate: proceed to Phase B (graph rebuild) only if best AUC > 0.6407
      (LR baseline 0.6377 + 0.003 buffer).

Usage:
  # Step 1 — build embedding array from pipeline shot files
  python colab/phase2/exp8a_llm_tabular.py \
      --build-emb-array \
      --shots-dir ~/data/shot_data \
      --data-dir ~/data \
      --emb-path colab/phase2/results/player_embeddings.pkl

  # Step 2 — run experiments (after X_emb.npy is built)
  python colab/phase2/exp8a_llm_tabular.py \
      --data-dir ~/data \
      --emb-path colab/phase2/results/player_embeddings.pkl \
      --out-dir colab/phase2/results
"""

import argparse
import json
import os
import sys
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

_COLAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COLAB_DIR not in sys.path:
    sys.path.insert(0, _COLAB_DIR)

from baseline_features import (
    STATIC_FEATURE_NAMES,
    extract_static_features,
    extract_velocity_features,
)
from build_graph import build_graph_from_shot, set_player_stats
from scrape_player_stats import build_player_stats_for_game, normalize_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_AUC  = 0.6377   # Exp 0a best (Velocity-65 / Lasso / Scaled)
GATE_THRESHOLD = BASELINE_AUC + 0.003   # 0.6407

STATS_SLICE   = slice(17, 38)   # columns 17–37 are the 21 shooter stats
N_STATIC      = 38
N_STATS       = 21               # = N_STATIC - 17

N_SPLITS = 5
N_INNER  = 3
C_GRID   = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


# ---------------------------------------------------------------------------
# Build embedding array (mirrors pipeline Step 2 filtering)
# ---------------------------------------------------------------------------

def build_emb_array(
    shots_dir: str,
    data_dir: str,
    emb_dict: dict,
    csv_path: str,
    fallback_vec: np.ndarray,
) -> np.ndarray:
    """
    Iterate over shots_{game_id}.json files in shots_dir in sorted order.
    For each shot where BOTH build_graph_from_shot() AND
    extract_static_features() succeed, look up the shooter's 3072-dim
    embedding.  Returns np.ndarray of shape (N, 3072) aligned row-for-row
    with the pipeline's X_static.npy.
    """
    shot_files = sorted(glob(os.path.join(shots_dir, "shots_*.json")))
    if not shot_files:
        raise FileNotFoundError(
            f"No shots_*.json files found in {shots_dir}"
        )

    emb_rows    = []
    fallback_n  = 0
    total_shots = 0

    for fi, shot_file in enumerate(shot_files):
        game_id = Path(shot_file).stem.replace("shots_", "")

        with open(shot_file) as f:
            shots = json.load(f)

        if not shots:
            continue

        player_stats, _, _ = build_player_stats_for_game(
            first_shot=shots[0], csv_path=csv_path
        )
        if not player_stats:
            continue

        set_player_stats(player_stats)

        for shot in shots:
            total_shots += 1

            graph, _ = build_graph_from_shot(shot)
            if graph is None:
                continue

            static_feats = extract_static_features(shot, player_stats)
            if static_feats is None:
                continue

            # Both succeeded — look up embedding
            shooter_id   = shot.get("shooter_id")
            player_names = shot.get("player_names", {})
            name         = player_names.get(shooter_id) or player_names.get(str(shooter_id), "")
            norm         = normalize_name(name)

            vec = emb_dict.get(norm)
            if vec is None:
                vec = emb_dict.get(name)
            if vec is None:
                vec = fallback_vec
                fallback_n += 1

            emb_rows.append(vec)

        if (fi + 1) % 50 == 0:
            print(f"  [{fi+1}/{len(shot_files)}] games processed, "
                  f"{len(emb_rows)} shots so far")

    X_emb = np.stack(emb_rows, axis=0).astype(np.float32)
    print(
        f"\nBuilt X_emb: {X_emb.shape}  "
        f"(fallback used for {fallback_n}/{len(emb_rows)} shots, "
        f"{fallback_n/max(len(emb_rows),1)*100:.1f}%)"
    )

    baseline_dir = os.path.join(data_dir, "baseline_data")
    os.makedirs(baseline_dir, exist_ok=True)
    out_path = os.path.join(baseline_dir, "X_emb.npy")
    np.save(out_path, X_emb)
    print(f"Saved → {out_path}")

    return X_emb


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def _make_lr(C=1.0):
    return LogisticRegression(
        penalty="l1", C=C, solver="liblinear", max_iter=2000, random_state=42
    )


def _make_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )


def run_config(X, y, model_type: str, proj_dim: int):
    """
    5-fold stratified CV.  PCA is fit on the training fold only.
    model_type: 'lr' or 'mlp'
    Returns dict of mean/std metrics.
    """
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=N_INNER,  shuffle=True, random_state=0)

    fold_auc, fold_acc, fold_f1, fold_C = [], [], [], []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # --- Split into static features and raw embeddings ---
        n_emb  = X_tr.shape[1] - N_STATIC  # number of embedding columns

        static_tr = X_tr[:, :N_STATIC]
        static_te = X_te[:, :N_STATIC]
        emb_tr    = X_tr[:, N_STATIC:]
        emb_te    = X_te[:, N_STATIC:]

        # --- Scale static features ---
        sc = StandardScaler()
        static_tr = sc.fit_transform(static_tr)
        static_te = sc.transform(static_te)

        # --- PCA on embeddings (fit on train only) ---
        n_components = min(proj_dim, emb_tr.shape[1], emb_tr.shape[0] - 1)
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        proj_tr = pca.fit_transform(emb_tr)
        proj_te = pca.transform(emb_te)

        X_tr_final = np.hstack([static_tr, proj_tr])
        X_te_final = np.hstack([static_te, proj_te])

        # --- Train ---
        if model_type == "lr":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_clf = LogisticRegressionCV(
                    Cs=C_GRID, cv=inner_cv, penalty="l1", solver="liblinear",
                    max_iter=2000, random_state=42, scoring="roc_auc",
                )
                cv_clf.fit(X_tr_final, y_tr)
            best_C = float(cv_clf.C_[0])
            fold_C.append(best_C)
            clf = _make_lr(best_C)
        else:
            clf = _make_mlp()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr_final, y_tr)

        y_pred = clf.predict(X_te_final)
        y_prob = clf.predict_proba(X_te_final)[:, 1]

        fold_auc.append(roc_auc_score(y_te, y_prob))
        fold_acc.append(accuracy_score(y_te, y_pred))
        fold_f1.append(f1_score(y_te, y_pred, zero_division=0))

    return {
        "auc_mean": float(np.mean(fold_auc)),
        "auc_std":  float(np.std(fold_auc)),
        "acc_mean": float(np.mean(fold_acc)),
        "acc_std":  float(np.std(fold_acc)),
        "f1_mean":  float(np.mean(fold_f1)),
        "f1_std":   float(np.std(fold_f1)),
        "best_C":   float(np.median(fold_C)) if fold_C else None,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(results, n_shots, fg_pct, proj_dim, out_path):
    sep = "-" * 95
    lines = [
        "SWISHNET — EXP 8a: LLM PLAYER EMBEDDINGS — TABULAR GATE",
        "=" * 60,
        f"Shots: {n_shots:,}  |  FG%: {fg_pct*100:.1f}%  |  5-fold stratified CV",
        f"Embedding: text-embedding-3-large (3072-dim), PCA → {proj_dim}-dim",
        f"Baseline to beat (Exp 0a): AUC {BASELINE_AUC:.4f}  |  Gate: AUC > {GATE_THRESHOLD:.4f}",
        "",
        f"{'Config':<20}  {'Features':<10}  {'Model':<5}  "
        f"{'AUC':>14}  {'Acc':>12}  {'F1':>12}  {'Δ vs Exp0a':>10}",
        sep,
    ]

    best_auc = 0.0
    for r in results:
        delta = r["auc_mean"] - BASELINE_AUC
        lines.append(
            f"{r['name']:<20}  {r['n_features']:<10}  {r['model']:<5}  "
            f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
            f"{r['acc_mean']*100:4.1f}±{r['acc_std']*100:.1f}%  "
            f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}  "
            f"{delta:+.4f}"
        )
        best_auc = max(best_auc, r["auc_mean"])

    lines.append(sep)
    lines.append("")

    if best_auc > GATE_THRESHOLD:
        decision = f"PROCEED to Phase B  (best AUC {best_auc:.4f} > gate {GATE_THRESHOLD:.4f})"
    else:
        decision = (
            f"ABORT Phase B  (best AUC {best_auc:.4f} ≤ gate {GATE_THRESHOLD:.4f}): "
            f"embeddings do not improve over raw stats"
        )
    lines += [f"Gate decision: {decision}", ""]

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nReport saved → {out_path}")
    print("\n" + text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 8a — LLM Tabular Gate")
    parser.add_argument(
        "--build-emb-array", action="store_true",
        help="Build X_emb.npy from shot files (run once before experiments)",
    )
    parser.add_argument(
        "--shots-dir",
        help="Directory of shots_{game_id}.json files (pipeline Step 1 output)",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing baseline_data/X_static.npy, y.npy (and X_emb.npy after build)",
    )
    parser.add_argument(
        "--emb-path",
        default=os.path.join(os.path.dirname(__file__), "results", "player_embeddings.pkl"),
        help="Path to player_embeddings.pkl from generate_embeddings.py",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--proj-dim", type=int, default=64,
        help="PCA projection dimension for embeddings (default: 64)",
    )
    args = parser.parse_args()

    # ── Load embeddings dict ───────────────────────────────────────────────
    import pickle
    if not os.path.exists(args.emb_path):
        print(f"ERROR: embeddings file not found: {args.emb_path}")
        print("Run generate_embeddings.py first.")
        sys.exit(1)

    with open(args.emb_path, "rb") as f:
        emb_dict = pickle.load(f)

    all_vecs     = list({id(v): v for v in emb_dict.values()}.values())
    fallback_vec = np.mean(all_vecs, axis=0).astype(np.float32)
    emb_raw_dim  = all_vecs[0].shape[0]
    print(f"Loaded {len(all_vecs)} unique player embeddings  (dim={emb_raw_dim})")

    baseline_dir = os.path.join(args.data_dir, "baseline_data")

    # ── Mode: build embedding array ────────────────────────────────────────
    if args.build_emb_array:
        if not args.shots_dir:
            print("ERROR: --shots-dir is required with --build-emb-array")
            sys.exit(1)

        csv_path = os.path.join(
            _COLAB_DIR, "data", "supplemental_data", "player_shooting_stats_2016.csv"
        )
        if not os.path.exists(csv_path):
            # Try relative to data_dir
            csv_path = os.path.join(
                args.data_dir, "supplemental_data", "player_shooting_stats_2016.csv"
            )

        print(f"\nBuilding X_emb.npy from {args.shots_dir} ...")
        build_emb_array(
            shots_dir=args.shots_dir,
            data_dir=args.data_dir,
            emb_dict=emb_dict,
            csv_path=csv_path,
            fallback_vec=fallback_vec,
        )
        return

    # ── Load pre-built arrays ──────────────────────────────────────────────
    xs_path  = os.path.join(baseline_dir, "X_static.npy")
    y_path   = os.path.join(baseline_dir, "y.npy")
    emb_path = os.path.join(baseline_dir, "X_emb.npy")

    for p in [xs_path, y_path, emb_path]:
        if not os.path.exists(p):
            print(f"ERROR: required file not found: {p}")
            if "X_emb" in p:
                print("Run with --build-emb-array first.")
            sys.exit(1)

    print(f"Loading arrays from {baseline_dir} ...")
    X_static = np.load(xs_path).astype(np.float32)
    y        = np.load(y_path).astype(int)
    X_emb    = np.load(emb_path).astype(np.float32)

    n_shots = len(y)
    fg_pct  = float(y.mean())
    print(f"X_static: {X_static.shape}  X_emb: {X_emb.shape}  y: {y.shape}")
    print(f"Shots: {n_shots:,}  FG%: {fg_pct*100:.1f}%")

    assert X_static.shape[0] == X_emb.shape[0] == len(y), (
        f"Row count mismatch: X_static={X_static.shape[0]}, "
        f"X_emb={X_emb.shape[0]}, y={len(y)}"
    )

    # Columns 0–16 = geometry + distances (17 features, no player identity)
    # Columns 17–37 = 21 shooter season stats
    X_context = X_static[:, :17]   # geometry + distances, no stats

    # ── Fit PCA once on full embedding matrix (unsupervised — no leakage) ─────
    n_components = min(args.proj_dim, X_emb.shape[1])
    print(f"\nFitting PCA({n_components}) on X_emb {X_emb.shape} ... ", end="", flush=True)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    X_emb_proj = pca.fit_transform(X_emb).astype(np.float32)
    print(f"done. Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Build feature matrices using pre-projected embeddings
    X_replace_padded = np.hstack([X_context, np.zeros((n_shots, N_STATIC - 17), dtype=np.float32), X_emb_proj])
    X_augment        = np.hstack([X_static, X_emb_proj])

    CONFIGS = [
        {
            "name":       "8a-LR-replace",
            "X":          X_replace_padded,
            "n_static":   17,
            "model":      "lr",
            "n_features": f"17+PCA{args.proj_dim}",
        },
        {
            "name":       "8a-LR-augment",
            "X":          X_augment,
            "n_static":   38,
            "model":      "lr",
            "n_features": f"38+PCA{args.proj_dim}",
        },
    ]

    # ── Run experiments ────────────────────────────────────────────────────

    def _run_config_with_n_static(X, y, model_type, n_static, proj_dim):
        """
        Variant of run_config that respects a per-config static column count.
        PCA is pre-applied — embedding columns start at N_STATIC and are already projected.
        """
        outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=N_INNER,  shuffle=True, random_state=0)

        fold_auc, fold_acc, fold_f1, fold_C = [], [], [], []

        for train_idx, test_idx in outer_cv.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            static_tr = X_tr[:, :n_static]
            static_te = X_te[:, :n_static]
            emb_tr    = X_tr[:, N_STATIC:]   # already PCA-projected
            emb_te    = X_te[:, N_STATIC:]

            sc = StandardScaler()
            static_tr = sc.fit_transform(static_tr)
            static_te = sc.transform(static_te)

            proj_tr = emb_tr
            proj_te = emb_te

            X_tr_f = np.hstack([static_tr, proj_tr])
            X_te_f = np.hstack([static_te, proj_te])

            if model_type == "lr":
                # Use C=0.1 (Exp 0a winner) — avoids expensive inner CV
                clf = _make_lr(C=0.1)
            else:
                clf = _make_mlp()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_tr_f, y_tr)

            y_pred = clf.predict(X_te_f)
            y_prob = clf.predict_proba(X_te_f)[:, 1]

            fold_auc.append(roc_auc_score(y_te, y_prob))
            fold_acc.append(accuracy_score(y_te, y_pred))
            fold_f1.append(f1_score(y_te, y_pred, zero_division=0))

        return {
            "auc_mean": float(np.mean(fold_auc)),
            "auc_std":  float(np.std(fold_auc)),
            "acc_mean": float(np.mean(fold_acc)),
            "acc_std":  float(np.std(fold_acc)),
            "f1_mean":  float(np.mean(fold_f1)),
            "f1_std":   float(np.std(fold_f1)),
            "best_C":   float(np.median(fold_C)) if fold_C else None,
        }

    all_results = []
    for cfg in CONFIGS:
        tag = f"{cfg['name']} ({cfg['n_features']}, {cfg['model'].upper()})"
        print(f"\n── {tag} ...", flush=True)
        metrics = _run_config_with_n_static(
            cfg["X"], y, cfg["model"], cfg["n_static"], args.proj_dim
        )
        metrics.update({
            "name":       cfg["name"],
            "model":      cfg["model"].upper(),
            "n_features": cfg["n_features"],
        })
        print(
            f"   AUC {metrics['auc_mean']:.4f}±{metrics['auc_std']:.4f}"
            f"  F1 {metrics['f1_mean']:.4f}"
            f"  Δ {metrics['auc_mean'] - BASELINE_AUC:+.4f}"
        )
        all_results.append(metrics)

    out_path = os.path.join(args.out_dir, "exp8a_results.txt")
    write_report(all_results, n_shots, fg_pct, args.proj_dim, out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Exp 0a — Retrain all baseline models on the full 87K dataset.

Loads pre-built numpy arrays from the pipeline output (X_static.npy,
X_full.npy, y.npy) rather than reprocessing raw game data.  Runs the
same 8 configurations as train_baselines.py (4 model types × 2 feature
sets, 5-fold CV) and writes results to phase2/results/.

Usage:
    python colab/phase2/train_baselines_full.py --data-dir ~/data
    python colab/phase2/train_baselines_full.py --data-dir ~/data --out-dir colab/phase2/results
"""

import argparse
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

_COLAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COLAB_DIR not in sys.path:
    sys.path.insert(0, _COLAB_DIR)

from baseline_features import STATIC_FEATURE_NAMES, VELOCITY_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

C_GRID   = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
N_SPLITS = 5
N_INNER  = 3

MODEL_CONFIGS = [
    ("none",  False, "None",       "Raw"),
    ("none",  True,  "None",       "Scaled"),
    ("ridge", True,  "Ridge (L2)", "Scaled"),
    ("lasso", True,  "Lasso (L1)", "Scaled"),
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_clf(model_type, C=1.0):
    if model_type == "ridge":
        return LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                  max_iter=2000, random_state=42)
    if model_type == "lasso":
        return LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                  max_iter=2000, random_state=42)
    return LogisticRegression(penalty="l2", C=1e6, solver="lbfgs",
                              max_iter=2000, random_state=42)


def run_config(X, y, model_type, scale):
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=N_INNER,  shuffle=True, random_state=0)

    fold_auc, fold_acc, fold_f1, fold_best_C = [], [], [], []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        if model_type in ("ridge", "lasso"):
            penalty = "l2" if model_type == "ridge" else "l1"
            solver  = "lbfgs" if model_type == "ridge" else "liblinear"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_clf = LogisticRegressionCV(
                    Cs=C_GRID, cv=inner_cv, penalty=penalty, solver=solver,
                    max_iter=2000, random_state=42, scoring="roc_auc",
                )
                cv_clf.fit(X_tr, y_tr)
            best_C = float(cv_clf.C_[0])
            fold_best_C.append(best_C)
            clf = _make_clf(model_type, best_C)
        else:
            clf = _make_clf(model_type)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

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
        "best_C":   float(np.median(fold_best_C)) if fold_best_C else None,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(all_results, n_shots, fg_pct, out_path):
    sep = "-" * 90
    lines = [
        "SWISHNET — EXP 0a: BASELINES ON FULL 87K DATASET",
        "=" * 60,
        f"Shots: {n_shots:,}  |  FG%: {fg_pct*100:.1f}%  |  5-fold stratified CV",
        "",
        f"{'Feature':<14}  {'Regularization':<14}  {'Scaling':<8}  "
        f"{'C*':<6}  {'AUC':>10}  {'Acc':>10}  {'F1':>10}",
        sep,
    ]
    for r in all_results:
        c_str = f"{r['best_C']:.4g}" if r["best_C"] else "—"
        lines.append(
            f"{r['feat']:<14}  {r['reg']:<14}  {r['scale_lbl']:<8}  "
            f"{c_str:<6}  "
            f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
            f"{r['acc_mean']*100:4.1f}±{r['acc_std']*100:.1f}%  "
            f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}"
        )
    lines.append(sep)

    best = max(all_results, key=lambda r: r["auc_mean"])
    lines += [
        "",
        f"Best config : {best['feat']} / {best['reg']} / {best['scale_lbl']}",
        f"Best AUC    : {best['auc_mean']:.4f} ± {best['auc_std']:.4f}",
        "",
        "Phase 1 reference (12 games, ~1,245 train shots):",
        "  LR Lasso Static-38 : AUC 0.614",
    ]

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nReport saved: {out_path}")
    print("\n" + text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing X_static.npy, X_full.npy, y.npy")
    parser.add_argument("--out-dir",
                        default=os.path.join(os.path.dirname(__file__), "results"),
                        help="Output directory for results")
    args = parser.parse_args()

    # ── Load pre-built arrays ──────────────────────────────────────────────
    baseline_dir = os.path.join(args.data_dir, "baseline_data")
    xs_path = os.path.join(baseline_dir, "X_static.npy")
    xf_path = os.path.join(baseline_dir, "X_full.npy")
    y_path  = os.path.join(baseline_dir, "y.npy")

    print(f"Loading arrays from {baseline_dir} ...")
    X_static = np.load(xs_path).astype(np.float32)
    X_full   = np.load(xf_path).astype(np.float32)
    y        = np.load(y_path).astype(int)

    n_shots = len(y)
    fg_pct  = float(y.mean())
    print(f"Loaded: {n_shots:,} shots, FG% {fg_pct*100:.1f}%")
    print(f"X_static: {X_static.shape}  X_full: {X_full.shape}")

    feat_names_full = STATIC_FEATURE_NAMES + VELOCITY_FEATURE_NAMES

    FEATURE_SETS = [
        ("Static-38",   X_static, STATIC_FEATURE_NAMES),
        ("Velocity-65", X_full,   feat_names_full),
    ]

    all_results = []

    for feat_name, X, _ in FEATURE_SETS:
        print(f"\n── {feat_name} ({X.shape[1]} features) ──")
        for model_type, scale, reg_label, scale_label in MODEL_CONFIGS:
            tag = f"{feat_name} / {reg_label} / {scale_label}"
            print(f"  {tag} ...", end="", flush=True)
            metrics = run_config(X, y, model_type, scale)
            metrics.update({"feat": feat_name, "reg": reg_label,
                            "scale_lbl": scale_label, "model_type": model_type,
                            "scale": scale})
            print(f"  AUC {metrics['auc_mean']:.4f}±{metrics['auc_std']:.4f}")
            all_results.append(metrics)

    out_path = os.path.join(args.out_dir, "exp0a_baselines_full.txt")
    write_report(all_results, n_shots, fg_pct, out_path)


if __name__ == "__main__":
    main()

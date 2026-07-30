#!/usr/bin/env python3
"""
Extensions 1 & 2 — Tree-based baselines on the full 87K dataset.

Extension 1 — XGBoost:
    Trains XGBoost on Static-38 and Velocity-65 feature sets under the same
    5-fold stratified CV used in Exp 0a (train_baselines_full.py).
    Fixed HPs chosen to be representative; no inner search (it is a sanity
    check / ceiling estimate, not a tuned entry).

Extension 2 — Decision Tree depth curve:
    Sweeps max_depth 1–20, records train and val AUC per fold, averages
    across 5 folds, plots the overfitting curve, and reports the best depth.
    Demonstrates the bias-variance tradeoff from the lecture and provides a
    fully interpretable reference point.

Both extensions load the pre-built numpy arrays produced by the pipeline.

Usage:
    python colab/phase2/exp_tree_boost.py --data-dir ~/data
    python colab/phase2/exp_tree_boost.py --data-dir ~/data --out-dir colab/phase2/results
    python colab/phase2/exp_tree_boost.py --data-dir ~/data --skip-xgb
    python colab/phase2/exp_tree_boost.py --data-dir ~/data --skip-tree
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

_COLAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COLAB_DIR not in sys.path:
    sys.path.insert(0, _COLAB_DIR)

from baseline_features import STATIC_FEATURE_NAMES, VELOCITY_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SPLITS = 5
RANDOM_STATE = 42

# XGBoost fixed HPs — representative defaults; not tuned
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    eval_metric="auc",
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

TREE_DEPTHS = list(range(1, 21))


# ---------------------------------------------------------------------------
# XGBoost experiment
# ---------------------------------------------------------------------------

def run_xgb(X, y, feat_name):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  [SKIP] xgboost not installed — pip install xgboost")
        return None

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_auc, fold_acc, fold_f1 = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        clf = XGBClassifier(**XGB_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr,
                    eval_set=[(X_te, y_te)],
                    verbose=False)

        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)
        fold_auc.append(roc_auc_score(y_te, y_prob))
        fold_acc.append(accuracy_score(y_te, y_pred))
        fold_f1.append(f1_score(y_te, y_pred, zero_division=0))
        print(f"    fold {fold}/5  AUC {fold_auc[-1]:.4f}", flush=True)

    return {
        "feat": feat_name,
        "auc_mean": float(np.mean(fold_auc)),
        "auc_std":  float(np.std(fold_auc)),
        "acc_mean": float(np.mean(fold_acc)),
        "f1_mean":  float(np.mean(fold_f1)),
        "params":   XGB_PARAMS,
    }


# ---------------------------------------------------------------------------
# Decision Tree depth sweep
# ---------------------------------------------------------------------------

def run_tree_sweep(X, y, feat_name):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    sc = StandardScaler()

    # depth → list of (train_auc, val_auc) across folds
    results = {d: {"train": [], "val": []} for d in TREE_DEPTHS}

    splits = list(cv.split(X, y))
    for depth in TREE_DEPTHS:
        for tr_idx, te_idx in splits:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # Standardise (tree doesn't need it, but keeps comparison fair)
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
            clf.fit(X_tr_s, y_tr)

            train_prob = clf.predict_proba(X_tr_s)[:, 1]
            val_prob   = clf.predict_proba(X_te_s)[:, 1]

            results[depth]["train"].append(roc_auc_score(y_tr, train_prob))
            results[depth]["val"].append(roc_auc_score(y_te, val_prob))

        mean_train = float(np.mean(results[depth]["train"]))
        mean_val   = float(np.mean(results[depth]["val"]))
        print(f"    depth {depth:2d}  train AUC {mean_train:.4f}  val AUC {mean_val:.4f}")

    return results


def plot_tree_curve(results, feat_name, out_path):
    depths    = sorted(results.keys())
    train_auc = [np.mean(results[d]["train"]) for d in depths]
    val_auc   = [np.mean(results[d]["val"])   for d in depths]
    train_std = [np.std(results[d]["train"])  for d in depths]
    val_std   = [np.std(results[d]["val"])    for d in depths]

    best_d     = depths[int(np.argmax(val_auc))]
    best_val   = max(val_auc)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depths, train_auc, 'o-', color='#1f77b4', label='Train AUC')
    ax.fill_between(depths,
                    np.array(train_auc) - np.array(train_std),
                    np.array(train_auc) + np.array(train_std),
                    alpha=0.15, color='#1f77b4')
    ax.plot(depths, val_auc, 's-', color='#d62728', label='Val AUC (5-fold mean)')
    ax.fill_between(depths,
                    np.array(val_auc) - np.array(val_std),
                    np.array(val_auc) + np.array(val_std),
                    alpha=0.15, color='#d62728')
    ax.axvline(best_d, linestyle='--', color='gray', alpha=0.7,
               label=f'Best depth={best_d} (val AUC={best_val:.4f})')
    ax.axhline(0.6377, linestyle=':', color='black', alpha=0.6,
               label='LR Lasso baseline (0.6377)')

    ax.set_xlabel('max_depth')
    ax.set_ylabel('AUC')
    ax.set_title(f'Decision Tree: depth vs AUC ({feat_name}, 87K shots)')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xticks(depths)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {out_path}")

    return best_d, best_val


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(xgb_results, tree_results, n_shots, fg_pct, out_path):
    sep = "-" * 75
    lines = [
        "SWISHNET — EXTENSIONS 1 & 2: TREE-BASED BASELINES (87K DATASET)",
        "=" * 65,
        f"Shots: {n_shots:,}  |  FG%: {fg_pct*100:.1f}%  |  5-fold stratified CV",
        "",
    ]

    # XGBoost table
    lines += ["Extension 1 — XGBoost", sep,
              f"  {'Feature':<14}  {'AUC':>12}  {'Acc':>10}  {'F1':>10}",
              sep]
    for r in xgb_results:
        lines.append(
            f"  {r['feat']:<14}  "
            f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
            f"{r['acc_mean']*100:5.1f}%      "
            f"{r['f1_mean']:.4f}"
        )
    lines += [sep, "  Fixed HPs: " + str(XGB_PARAMS), ""]

    # Decision Tree table
    lines += ["Extension 2 — Decision Tree Depth Sweep", sep,
              f"  {'Feature':<14}  {'Best depth':>10}  {'Best val AUC':>12}",
              sep]
    for feat_name, res in tree_results.items():
        depths  = sorted(res.keys())
        val_auc = [np.mean(res[d]["val"]) for d in depths]
        best_d  = depths[int(np.argmax(val_auc))]
        best_v  = max(val_auc)
        lines.append(f"  {feat_name:<14}  {best_d:>10}  {best_v:>12.4f}")
    lines += [sep, "", "LR Lasso baseline (Exp 0a): 0.6377"]

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
                        help="Dir containing baseline_data/X_static.npy etc.")
    parser.add_argument("--out-dir",
                        default=os.path.join(os.path.dirname(__file__), "results"),
                        help="Output directory for results and plots")
    parser.add_argument("--skip-xgb",  action="store_true",
                        help="Skip XGBoost experiment")
    parser.add_argument("--skip-tree", action="store_true",
                        help="Skip Decision Tree depth sweep")
    args = parser.parse_args()

    baseline_dir = os.path.join(args.data_dir, "baseline_data")
    print(f"Loading arrays from {baseline_dir} ...")
    X_static = np.load(os.path.join(baseline_dir, "X_static.npy")).astype(np.float32)
    X_full   = np.load(os.path.join(baseline_dir, "X_full.npy")).astype(np.float32)
    y        = np.load(os.path.join(baseline_dir, "y.npy")).astype(int)

    n_shots, fg_pct = len(y), float(y.mean())
    print(f"Loaded: {n_shots:,} shots, FG% {fg_pct*100:.1f}%")

    feat_names_full = STATIC_FEATURE_NAMES + VELOCITY_FEATURE_NAMES
    FEATURE_SETS = [
        ("Static-38",   X_static),
        ("Velocity-65", X_full),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    xgb_results  = []
    tree_results = {}

    # ── Extension 1: XGBoost ──────────────────────────────────────────────
    if not args.skip_xgb:
        print("\n── Extension 1: XGBoost ──")
        for feat_name, X in FEATURE_SETS:
            print(f"  {feat_name} ...")
            res = run_xgb(X, y, feat_name)
            if res:
                xgb_results.append(res)
                json_path = os.path.join(args.out_dir,
                                         f"ext1_xgb_{feat_name.replace('-','').lower()}.json")
                with open(json_path, "w") as f:
                    json.dump(res, f, indent=2)

    # ── Extension 2: Decision Tree depth sweep ───────────────────────────
    if not args.skip_tree:
        print("\n── Extension 2: Decision Tree depth sweep ──")
        for feat_name, X in FEATURE_SETS:
            print(f"\n  {feat_name} ...")
            res = run_tree_sweep(X, y, feat_name)
            tree_results[feat_name] = res

            plot_path = os.path.join(args.out_dir,
                                     f"ext2_tree_curve_{feat_name.replace('-','').lower()}.png")
            best_d, best_v = plot_tree_curve(res, feat_name, plot_path)
            print(f"  Best depth={best_d}, best val AUC={best_v:.4f}")

            json_path = os.path.join(args.out_dir,
                                     f"ext2_tree_{feat_name.replace('-','').lower()}.json")
            serialisable = {d: {"train": res[d]["train"], "val": res[d]["val"]}
                            for d in res}
            with open(json_path, "w") as f:
                json.dump(serialisable, f, indent=2)

    # ── Report ────────────────────────────────────────────────────────────
    if xgb_results or tree_results:
        report_path = os.path.join(args.out_dir, "ext12_tree_boost_report.txt")
        write_report(xgb_results, tree_results, n_shots, fg_pct, report_path)


if __name__ == "__main__":
    main()

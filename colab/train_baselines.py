#!/usr/bin/env python3
"""
Train and evaluate traditional ML baselines for NBA shot prediction.

Model configurations (4 × 2 feature sets = 8 total):
  none-raw    — unregularized logistic regression, no feature scaling
  none-scaled — unregularized logistic regression, StandardScaler
  ridge       — L2-regularized, scaled, inner 3-fold CV for C
  lasso       — L1-regularized, scaled, inner 3-fold CV for C

Outputs:
  data/baseline_results.txt   — text report
  data/baseline_plots/*.png   — weight charts, Lasso path, AUC comparison
"""

import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
_COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
if _COLAB_DIR not in sys.path:
    sys.path.insert(0, _COLAB_DIR)

from baseline_features import (
    load_all_shots,
    STATIC_FEATURE_NAMES,
    VELOCITY_FEATURE_NAMES,
    GAME_DIR, PBP_PATH, CSV_PATH,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_DATA_DIR  = os.path.join(_COLAB_DIR, "data")
PLOTS_DIR  = os.path.join(_DATA_DIR, "baseline_plots")
REPORT_PATH = os.path.join(_DATA_DIR, "baseline_results.txt")

# ---------------------------------------------------------------------------
# Hyperparameter grid for inner CV
# ---------------------------------------------------------------------------
C_GRID    = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
N_SPLITS  = 5    # outer CV folds
N_INNER   = 3    # inner CV folds for C selection

# ---------------------------------------------------------------------------
# Model configs: (key, scale, display_reg, display_scale)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = [
    ("none",  False, "None",       "Raw"),
    ("none",  True,  "None",       "Scaled"),
    ("ridge", True,  "Ridge (L2)", "Scaled"),
    ("lasso", True,  "Lasso (L1)", "Scaled"),
]


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------

def _make_clf(model_type: str, C: float = 1.0):
    """Build a LogisticRegression for the given type and C."""
    if model_type == "ridge":
        return LogisticRegression(
            penalty="l2", C=C, solver="lbfgs",
            max_iter=2000, random_state=42,
        )
    if model_type == "lasso":
        return LogisticRegression(
            penalty="l1", C=C, solver="liblinear",
            max_iter=2000, random_state=42,
        )
    # none — use very large C to approximate no regularization
    return LogisticRegression(
        penalty="l2", C=1e6, solver="lbfgs",
        max_iter=2000, random_state=42,
    )


def run_model_config(X: np.ndarray, y: np.ndarray,
                     model_type: str, scale: bool) -> dict:
    """
    Run 5-fold stratified CV for one model configuration.
    Regularized models use inner 3-fold CV (via LogisticRegressionCV) to
    select C within each outer training fold.

    Returns a dict with mean/std metrics, fold-level train/test accuracy,
    and the modal best C (for regularized models).
    """
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=N_INNER,  shuffle=True, random_state=0)

    fold_acc   = []
    fold_auc   = []
    fold_prec  = []
    fold_rec   = []
    fold_f1    = []
    fold_tr_ac = []
    fold_best_C = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if scale:
            sc   = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        if model_type in ("ridge", "lasso"):
            penalty = "l2" if model_type == "ridge" else "l1"
            solver  = "lbfgs" if model_type == "ridge" else "liblinear"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_clf = LogisticRegressionCV(
                    Cs=C_GRID, cv=inner_cv,
                    penalty=penalty, solver=solver,
                    max_iter=2000, random_state=42,
                    scoring="roc_auc",
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

        fold_acc.append(accuracy_score(y_te, y_pred))
        fold_auc.append(roc_auc_score(y_te, y_prob))
        fold_prec.append(precision_score(y_te, y_pred, zero_division=0))
        fold_rec.append(recall_score(y_te, y_pred, zero_division=0))
        fold_f1.append(f1_score(y_te, y_pred, zero_division=0))
        fold_tr_ac.append(accuracy_score(y_tr, clf.predict(X_tr)))

    result = {}
    for key, vals in [("acc", fold_acc), ("auc", fold_auc),
                      ("prec", fold_prec), ("rec", fold_rec), ("f1", fold_f1)]:
        result[f"{key}_mean"] = float(np.mean(vals))
        result[f"{key}_std"]  = float(np.std(vals))

    result["fold_train_accs"] = fold_tr_ac
    result["fold_test_accs"]  = fold_acc
    result["best_Cs"]         = fold_best_C
    result["best_C_final"]    = (float(np.median(fold_best_C)) if fold_best_C else None)

    return result


def fit_final_model(X: np.ndarray, y: np.ndarray,
                    model_type: str, scale: bool,
                    best_C: float = None) -> np.ndarray:
    """
    Fit one model on all data (for coefficient inspection).
    Returns coefficient array of shape (n_features,).
    """
    if scale:
        sc    = StandardScaler()
        X_fit = sc.fit_transform(X)
    else:
        X_fit = X

    C = best_C if best_C else 1.0
    clf = _make_clf(model_type, C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_fit, y)
    return clf.coef_[0].copy()


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_weight_magnitude(coef: np.ndarray, feature_names: list,
                          title: str, save_path: str) -> None:
    """Bar chart of coefficients sorted by |coeff| (descending)."""
    sorted_idx = np.argsort(np.abs(coef))[::-1]
    n = len(feature_names)

    fig_h = max(6, n * 0.32)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    colors = ["steelblue" if coef[i] >= 0 else "firebrick" for i in sorted_idx]
    ax.barh(range(n), coef[sorted_idx], color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (standardized)")
    ax.set_title(title, fontsize=11, pad=10)
    ax.invert_yaxis()

    # Minimal legend
    from matplotlib.patches import Patch as _Patch
    ax.legend(
        handles=[_Patch(color="steelblue", label="positive"),
                 _Patch(color="firebrick", label="negative")],
        fontsize=8, loc="lower right",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_lasso_path(X: np.ndarray, y: np.ndarray,
                    feature_names: list, title: str,
                    save_path: str) -> None:
    """
    Fit Lasso at each C value in a dense grid and plot coefficient paths.
    X should be the raw (unscaled) feature matrix; StandardScaler is applied inside.
    """
    Cs_dense = np.logspace(-3, 2, 40)

    sc   = StandardScaler()
    X_sc = sc.fit_transform(X)

    coefs = []
    for C in Cs_dense:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l1", C=float(C), solver="liblinear",
                max_iter=2000, random_state=42,
            )
            clf.fit(X_sc, y)
        coefs.append(clf.coef_[0].copy())

    coefs = np.array(coefs)         # shape (40, n_features)
    log_Cs = np.log10(Cs_dense)

    # Identify features that are non-zero at the most-permissive C (rightmost)
    max_abs = np.max(np.abs(coefs), axis=0)
    active  = np.argsort(max_abs)[::-1][:20]   # top-20 by max |coeff|

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap("tab20")

    for rank, j in enumerate(active):
        ax.plot(log_Cs, coefs[:, j],
                label=feature_names[j],
                color=cmap(rank / 20),
                alpha=0.85, linewidth=1.2)

    # Faint lines for the rest
    inactive = [j for j in range(len(feature_names)) if j not in active]
    for j in inactive:
        ax.plot(log_Cs, coefs[:, j], color="gray", alpha=0.15, linewidth=0.6)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

    # Mark C_GRID positions
    for C_mark in C_GRID:
        ax.axvline(np.log10(C_mark), color="gray", linewidth=0.4, linestyle=":")

    ax.set_xlabel("log₁₀(C)  [← more regularization | less regularization →]")
    ax.set_ylabel("Coefficient (standardized)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_auc_comparison(all_results: list, save_path: str) -> None:
    """Bar chart comparing mean AUC-ROC across all 8 model configurations."""
    labels    = [r["label"] for r in all_results]
    auc_means = [r["auc_mean"] for r in all_results]
    auc_stds  = [r["auc_std"]  for r in all_results]

    # First 4 = Static, last 4 = Velocity
    n_static = 4
    colors   = (["#2196F3"] * n_static + ["#FF9800"] * (len(all_results) - n_static))

    fig, ax = plt.subplots(figsize=(14, 6))
    x    = np.arange(len(labels))
    bars = ax.bar(x, auc_means, yerr=auc_stds,
                  color=colors, capsize=5, alpha=0.85, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUC-ROC (5-fold CV mean ± std)")
    ax.set_title("NBA Shot Prediction — Baseline Model AUC-ROC Comparison")
    ax.set_ylim(0.45, min(1.0, max(auc_means) + 0.12))

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, label="Random (0.5)")

    for bar, mean in zip(bars, auc_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(auc_stds) + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    ax.legend(handles=[
        Patch(facecolor="#2196F3", label="Static-38"),
        Patch(facecolor="#FF9800", label="Velocity-65"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Random (0.5)"),
    ], fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def generate_report(all_results: list,
                    n_shots: int, fg_pct: float, n_games: int,
                    feat_names_static: list, feat_names_full: list,
                    save_path: str) -> None:
    """Write the baseline results report to a text file."""

    lines = []
    sep = "─" * 85

    lines.append("NBA SHOT PREDICTION — BASELINE MODEL RESULTS")
    lines.append(f"Games: {n_games}  |  Shots: {n_shots}  |  FG%: {fg_pct*100:.1f}%")
    lines.append("")
    lines.append("MODEL COMPARISON (5-fold CV, mean ± std)")
    lines.append(sep)
    lines.append(f"  {'Feature Set':<14}  {'Preprocessing':<10}  {'Regularization':<14}  "
                 f"{'C*':<6}  {'Accuracy':<12}  {'AUC-ROC':<10}  {'F1':<8}")
    lines.append(sep)

    for r in all_results:
        c_str = f"{r['best_C_final']:.4g}" if r["best_C_final"] else "—"
        lines.append(
            f"  {r['feat_name']:<14}  {r['scale_label']:<10}  {r['reg_label']:<14}  "
            f"{c_str:<6}  "
            f"{r['acc_mean']*100:5.1f}±{r['acc_std']*100:.1f}%  "
            f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
            f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}"
        )

    lines.append(sep)
    lines.append("")

    # --- Lasso feature selection (Static-38, best C*) ---
    lasso_static = next(
        (r for r in all_results
         if r["feat_name"] == "Static-38" and r["reg_label"].startswith("Lasso")),
        None,
    )
    if lasso_static and lasso_static.get("coef") is not None:
        coef = lasso_static["coef"]
        lines.append("LASSO FEATURE SELECTION (Static-38, best C*) [standardized coeff]")
        lines.append(sep)
        idx_sorted = np.argsort(np.abs(coef))[::-1]
        zeroed = [feat_names_static[i] for i in range(len(coef)) if coef[i] == 0.0]
        for rank, i in enumerate(idx_sorted[:15], 1):
            sign = "+" if coef[i] >= 0 else ""
            lines.append(f"  {rank:2d}. {feat_names_static[i]:<30s} {sign}{coef[i]:.4f}")
        if zeroed:
            lines.append(f"\n  Zeroed ({len(zeroed)} / {len(feat_names_static)}): "
                         + ", ".join(zeroed[:12])
                         + (" ..." if len(zeroed) > 12 else ""))
        lines.append(sep)
        lines.append("")

    # --- Bias-variance snapshot (Static-38, fold 1) ---
    lines.append("BIAS-VARIANCE SNAPSHOT (Static-38, fold 1)")
    lines.append(sep)
    configs_for_bv = [
        ("None/Raw",    "none",  False),
        ("Scaled/None", "none",  True),
        ("Scaled/L2",   "ridge", True),
        ("Scaled/L1",   "lasso", True),
    ]
    for label, mt, sc_flag in configs_for_bv:
        r = next(
            (r for r in all_results
             if r["feat_name"] == "Static-38"
             and r["model_type"] == mt
             and r["scale"] == sc_flag),
            None,
        )
        if r and r["fold_train_accs"] and r["fold_test_accs"]:
            tr = r["fold_train_accs"][0] * 100
            te = r["fold_test_accs"][0] * 100
            gap = tr - te
            lines.append(f"  {label:<14}  train {tr:.1f}%   test {te:.1f}%   gap {gap:.1f}%")
    lines.append(sep)
    lines.append("")
    lines.append(f"PLOTS SAVED: {PLOTS_DIR}")
    lines.append("  weights_static_none.png         weights_velocity_none.png")
    lines.append("  weights_static_ridge.png        weights_velocity_ridge.png")
    lines.append("  weights_static_lasso.png        weights_velocity_lasso.png")
    lines.append("  lasso_path_static.png           lasso_path_velocity.png")
    lines.append("  auc_comparison.png")

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(report_text)

    print(f"\nReport saved: {save_path}")
    print("\n" + report_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("NBA Shot Prediction — Baseline Models")
    print("=" * 60)

    # ---- 1. Load data ----
    print("\n[1/4] Loading shot data ...")
    X_static, X_full, y = load_all_shots(GAME_DIR, PBP_PATH, CSV_PATH)

    if X_static is None:
        print("ERROR: Could not load data. Aborting.")
        sys.exit(1)

    n_shots  = len(y)
    fg_pct   = float(y.mean())
    n_games  = len([f for f in os.listdir(GAME_DIR) if f.endswith(".json")])
    feat_names_full = STATIC_FEATURE_NAMES + VELOCITY_FEATURE_NAMES

    print(f"\nDataset: {n_shots} shots | FG%: {fg_pct*100:.1f}% | Games: {n_games}")

    # ---- 2. Run all model configurations ----
    print("\n[2/4] Training models (5-fold CV) ...")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    FEATURE_SETS = [
        ("Static-38",    X_static, STATIC_FEATURE_NAMES),
        ("Velocity-65",  X_full,   feat_names_full),
    ]

    all_results = []

    for feat_name, X, feat_names in FEATURE_SETS:
        print(f"\n  Feature set: {feat_name} ({X.shape[1]} features)")

        for model_type, scale, reg_label, scale_label in MODEL_CONFIGS:
            tag = f"{feat_name} / {reg_label} / {scale_label}"
            print(f"    {tag} ...", end="", flush=True)

            metrics = run_model_config(X, y, model_type, scale)
            metrics["feat_name"]   = feat_name
            metrics["model_type"]  = model_type
            metrics["scale"]       = scale
            metrics["reg_label"]   = reg_label
            metrics["scale_label"] = scale_label
            metrics["label"]       = f"{feat_name}\n{reg_label}/{scale_label}"
            metrics["coef"]        = None  # filled below for scaled models

            print(f"  AUC={metrics['auc_mean']:.4f}±{metrics['auc_std']:.4f}  "
                  f"Acc={metrics['acc_mean']*100:.1f}%")

            # Fit final model (scaled configs) for coefficient inspection
            if scale:
                coef = fit_final_model(X, y, model_type, scale,
                                       metrics["best_C_final"])
                metrics["coef"] = coef

                # Weight magnitude plot
                slug     = feat_name.lower().split("-")[0]  # "static" or "velocity"
                reg_slug = model_type                        # "none", "ridge", "lasso"
                best_C_str = (f"C*={metrics['best_C_final']:.4g}"
                              if metrics["best_C_final"] else "C→∞")
                plot_weight_magnitude(
                    coef, feat_names,
                    title=f"Coefficients: {feat_name} | {reg_label} | {scale_label} ({best_C_str})",
                    save_path=os.path.join(PLOTS_DIR, f"weights_{slug}_{reg_slug}.png"),
                )

            all_results.append(metrics)

        # Lasso path for this feature set
        print(f"  Computing Lasso path for {feat_name} ...")
        slug = feat_name.lower().split("-")[0]
        plot_lasso_path(
            X, y, feat_names,
            title=f"Lasso Coefficient Path — {feat_name}",
            save_path=os.path.join(PLOTS_DIR, f"lasso_path_{slug}.png"),
        )

    # ---- 3. AUC comparison chart ----
    print("\n[3/4] Plotting AUC comparison ...")
    plot_auc_comparison(
        all_results,
        save_path=os.path.join(PLOTS_DIR, "auc_comparison.png"),
    )

    # ---- 4. Report ----
    print("\n[4/4] Writing report ...")
    generate_report(
        all_results,
        n_shots=n_shots,
        fg_pct=fg_pct,
        n_games=n_games,
        feat_names_static=STATIC_FEATURE_NAMES,
        feat_names_full=feat_names_full,
        save_path=REPORT_PATH,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

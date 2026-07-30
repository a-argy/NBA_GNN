"""
Phase 1b — Reduced-capacity sweep with class weighting.

Motivation
----------
Phase 1 revealed severe overparameterization: 331K params trained on 1221
examples (271 params/sample) caused majority-class collapse across nearly all
experiments (best epochs 1–9, AUC mostly < 0.52 on hard predictions).

This sweep tests whether right-sizing the model rescues learning:
  - hidden_dim: 32 and 64  (vs 128 in Phase 1)
  - num_pre_layers / num_post_layers: 1 each  (vs 2 each in Phase 1)
  - Class weighting: miss=1.0, make=1.33  (57:43 ratio correction)

Only the three architecturally interesting variants are tested:
  gat_mean_max  — best Phase 1 AUC (0.5935), but collapsed to all-miss
  gat_shooter   — most balanced predictions, best F1 (0.273)
  gine_mean     — lightest model, second-best AUC (0.537)
  gat_mean      — control: plain GAT mean pool from Phase 1

Output
------
colab/phase1/results/
  phase1b_{name}_best.pt
  phase1b_{name}_history.json
  phase1b_results.txt
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# ── Path setup ─────────────────────────────────────────────────────────────────
_PHASE1_DIR = os.path.dirname(os.path.abspath(__file__))
_COLAB_DIR  = os.path.dirname(_PHASE1_DIR)
for _p in (_COLAB_DIR, _PHASE1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prepare_data import load_graph_data, split_data, get_data_loaders
from models import GNNShotPredictor

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.environ.get("SWISHNET_BASE_DIR", "/content/drive/MyDrive/NBA_GNN_files/")
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")
RESULTS_DIR    = os.path.join(_PHASE1_DIR, "results")

# ── Fixed training hyperparameters ─────────────────────────────────────────────
TRAIN_HP = dict(
    lr=0.0001,
    weight_decay=1e-2,   # stronger than Phase 1 (5e-3) — more regularization
    batch_size=256,
    epochs=200,
    patience=30,         # slightly more patience since models are smaller
    random_state=42,
)

# Class weights: 994 misses / 751 makes ≈ 1.324
# weight[1] = n_miss / n_make pushes the model to predict makes more
CLASS_WEIGHT_RATIO = 994 / 751   # ~1.324

# ── Sweep grid: (architecture, hidden_dim) ────────────────────────────────────
# base model kwargs (shared across all, reduced from Phase 1)
def _model_kwargs(hidden_dim):
    return dict(
        num_node_features=41,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=2,
        num_pre_layers=1,    # was 2
        num_post_layers=1,   # was 2
        dropout=0.3,
        train_eps=True,
    )

EXPERIMENTS = []
for hd in [32, 64]:
    for arch in [
        dict(conv="GAT",  readout="mean_max"),
        dict(conv="GAT",  readout="shooter_centric"),
        dict(conv="GINE", readout="mean"),
        dict(conv="GAT",  readout="mean"),
    ]:
        EXPERIMENTS.append(dict(
            name=f"{arch['conv'].lower()}_{arch['readout'].replace('_centric','_sc').replace('_','')}_hd{hd}",
            conv=arch['conv'],
            readout=arch['readout'],
            hidden_dim=hd,
        ))


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers  (identical logic to train_phase1.py)
# ══════════════════════════════════════════════════════════════════════════════

def _train_epoch(model, loader, optimizer, class_weights, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss   = F.cross_entropy(logits, data.y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        correct    += (logits.argmax(dim=1) == data.y).sum().item()
        total      += data.num_graphs
    return total_loss / total, correct / total


def _evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            all_logits.append(model(data))
            all_labels.append(data.y)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    loss  = F.cross_entropy(logits, labels).item()
    pred  = logits.argmax(dim=1)
    acc   = (pred == labels).float().mean().item()
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    p_np  = pred.cpu().numpy()
    l_np  = labels.cpu().numpy()

    m0 = l_np == 0
    m1 = l_np == 1
    acc_0 = (p_np[m0] == l_np[m0]).mean() if m0.sum() > 0 else 0.0
    acc_1 = (p_np[m1] == l_np[m1]).mean() if m1.sum() > 0 else 0.0

    return loss, acc, acc_0, acc_1, p_np, l_np, probs


# ══════════════════════════════════════════════════════════════════════════════
# Experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(cfg, train_loader, val_loader, test_loader, class_weights, device):
    print(f"\n{'='*65}")
    print(f"  Experiment : {cfg['name']}")
    print(f"  Conv       : {cfg['conv']}  Readout: {cfg['readout']}  hidden_dim: {cfg['hidden_dim']}")
    print(f"  Class wts  : miss=1.00  make={CLASS_WEIGHT_RATIO:.3f}")
    print(f"{'='*65}")

    model = GNNShotPredictor(
        conv_type=cfg['conv'],
        readout=cfg['readout'],
        **_model_kwargs(cfg['hidden_dim']),
    ).to(device)

    n_params = model.param_count()
    print(f"  Parameters : {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_HP['lr'],
        weight_decay=TRAIN_HP['weight_decay'],
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, f"phase1b_{cfg['name']}_best.pt")

    best_val_loss = float('inf')
    best_epoch    = 0
    patience_ctr  = 0
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc')}

    t_start = time.time()

    for epoch in range(1, TRAIN_HP['epochs'] + 1):
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, class_weights, device)
        vl_loss, vl_acc, vl_acc0, vl_acc1, *_ = _evaluate(model, val_loader, device)

        history['train_loss'].append(round(tr_loss, 5))
        history['train_acc'].append(round(tr_acc, 5))
        history['val_loss'].append(round(vl_loss, 5))
        history['val_acc'].append(round(vl_acc, 5))

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"  Ep {epoch:03d}: tr={tr_loss:.4f}/{tr_acc:.3f} | "
                f"val={vl_loss:.4f}/{vl_acc:.3f} "
                f"(miss={vl_acc0:.3f} make={vl_acc1:.3f})"
            )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch    = epoch
            patience_ctr  = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_ctr += 1

        if patience_ctr >= TRAIN_HP['patience']:
            print(f"  Early stop  : epoch {epoch}  (best: epoch {best_epoch})")
            break

    elapsed = time.time() - t_start

    model.load_state_dict(torch.load(model_path, weights_only=True))
    _, test_acc, test_acc0, test_acc1, test_preds, test_labels, test_probs = \
        _evaluate(model, test_loader, device)

    auc  = roc_auc_score(test_labels, test_probs)
    f1   = f1_score(test_labels, test_preds, zero_division=0)
    prec = precision_score(test_labels, test_preds, zero_division=0)
    rec  = recall_score(test_labels, test_preds, zero_division=0)

    print(f"\n  Test AUC={auc:.4f} | Acc={test_acc:.4f} | F1={f1:.4f} | "
          f"Miss={test_acc0:.3f} Make={test_acc1:.3f}")
    print(f"  Best epoch : {best_epoch} | Time: {elapsed/60:.1f} min")

    result = dict(
        name=cfg['name'], conv=cfg['conv'], readout=cfg['readout'],
        hidden_dim=cfg['hidden_dim'],
        best_epoch=best_epoch, best_val_loss=round(best_val_loss, 5),
        test_acc=round(test_acc, 4), test_acc_miss=round(test_acc0, 4),
        test_acc_make=round(test_acc1, 4),
        auc=round(auc, 4), f1=round(f1, 4),
        precision=round(prec, 4), recall=round(rec, 4),
        n_params=n_params, elapsed_min=round(elapsed / 60, 1),
    )

    hist_path = os.path.join(RESULTS_DIR, f"phase1b_{cfg['name']}_history.json")
    with open(hist_path, 'w') as fh:
        json.dump({**result, 'history': history}, fh, indent=2)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(all_results, n_train, n_val, n_test, fg_pct):
    path = os.path.join(RESULTS_DIR, 'phase1b_results.txt')
    sorted_results = sorted(all_results, key=lambda r: -r['auc'])

    lines = [
        "SWISHNET — PHASE 1b RESULTS  (reduced capacity + class weighting)",
        "=" * 70,
        f"Train: {n_train}  Val: {n_val}  Test: {n_test}  FG%: {fg_pct:.1f}%",
        f"Class weight: miss=1.00  make={CLASS_WEIGHT_RATIO:.3f}",
        f"Phase 1 best GNN AUC : 0.5935  (gat_mean_max, hd=128)",
        f"Logistic regression  : 0.614   (Static-38 / Lasso)",
        "",
        "RESULTS  (sorted by test AUC)",
        "-" * 70,
        f"{'Experiment':<38} {'hd':>4} {'AUC':>6} {'Acc':>6} {'F1':>6} "
        f"{'Miss':>6} {'Make':>6} {'Ep':>4}",
        "-" * 70,
    ]

    for r in sorted_results:
        lines.append(
            f"{r['name']:<38} {r['hidden_dim']:>4} {r['auc']:>6.4f} "
            f"{r['test_acc']:>6.4f} {r['f1']:>6.4f} "
            f"{r['test_acc_miss']:>6.3f} {r['test_acc_make']:>6.3f} "
            f"{r['best_epoch']:>4}"
        )

    lines += [
        "-" * 70,
        "",
        "PARAMETER COUNTS",
        "-" * 70,
    ]
    for r in sorted(all_results, key=lambda r: r['n_params']):
        lines.append(f"  {r['name']:<38}  {r['n_params']:>8,} params")

    lines += ["", f"Results saved to: {RESULTS_DIR}"]

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"\nReport written to: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(TRAIN_HP['random_state'])
    np.random.seed(TRAIN_HP['random_state'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Graph data: {GRAPH_DATA_DIR}")
    print(f"Results dir: {RESULTS_DIR}")

    graphs, labels = load_graph_data(graph_data_dir=GRAPH_DATA_DIR)
    fg_pct = 100.0 * sum(labels) / len(labels)

    n_miss = sum(1 for l in labels if l == 0)
    n_make = sum(1 for l in labels if l == 1)
    print(f"\nDataset: {len(graphs)} graphs | Miss: {n_miss} ({100*n_miss/len(graphs):.1f}%) "
          f"| Make: {n_make} ({100*n_make/len(graphs):.1f}%)")
    print(f"Class weight ratio (make/miss): {CLASS_WEIGHT_RATIO:.3f}")

    train_graphs, val_graphs, test_graphs, train_labels, val_labels, test_labels = \
        split_data(graphs, labels, random_state=TRAIN_HP['random_state'])

    train_loader, val_loader, test_loader = get_data_loaders(
        train_graphs, val_graphs, test_graphs,
        batch_size=TRAIN_HP['batch_size'],
        shuffle_train=True,
    )

    n_train, n_val, n_test = len(train_graphs), len(val_graphs), len(test_graphs)
    print(f"Split: train={n_train}  val={n_val}  test={n_test}")

    class_weights = torch.tensor([1.0, CLASS_WEIGHT_RATIO], dtype=torch.float).to(device)

    all_results = []
    for cfg in EXPERIMENTS:
        result = run_experiment(cfg, train_loader, val_loader, test_loader, class_weights, device)
        all_results.append(result)

    print(f"\n{'='*65}")
    print("PHASE 1b SUMMARY  (sorted by AUC)")
    print(f"{'='*65}")
    print(f"{'Experiment':<38} {'hd':>4} {'AUC':>6} {'Acc':>6} {'F1':>6}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda r: -r['auc']):
        print(f"{r['name']:<38} {r['hidden_dim']:>4} {r['auc']:>6.4f} "
              f"{r['test_acc']:>6.4f} {r['f1']:>6.4f}")

    generate_report(all_results, n_train, n_val, n_test, fg_pct)
    print(f"\nAll Phase 1b results saved to: {RESULTS_DIR}")
    return all_results


if __name__ == '__main__':
    main()

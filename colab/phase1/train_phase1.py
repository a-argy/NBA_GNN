"""
Phase 1 experiment runner — SwishNet GNN.

Experiments
-----------
Exp 1  Pooling strategies (GAT conv, standard CE loss):
    gat_mean      global mean pool                        [baseline]
    gat_max       global max pool
    gat_add       global add / sum pool
    gat_mean_max  concat(mean, max)  — 2× post-MLP input

Exp 2  Shooter-centric hybrid readout:
    gat_shooter   concat(shooter-node emb, scene mean)    [GAT]
    gine_mean     global mean pool                        [GINE baseline]
    gine_shooter  concat(shooter-node emb, scene mean)    [GINE]

Exp 7  Loss function variants (GAT + mean pool, identical arch to gat_mean):
    gat_ls_005    label smoothing α=0.05
    gat_ls_010    label smoothing α=0.10
    gat_focal_05  focal loss γ=0.5
    gat_focal_10  focal loss γ=1.0
    gat_focal_20  focal loss γ=2.0

Data integrity
--------------
Only fully-constructed graphs are used.  build_graph.py enforces this during
graph creation — any shot missing required fields, player positions, stats,
or a detectable shooter is dropped BEFORE the graph is saved to disk.  The
.pt files loaded here therefore contain exclusively complete graphs.

Reproducibility
---------------
All experiments use the SAME 70/20/10 stratified split (random_state=42).
Fixed hyperparameters are held constant across every config.

Output
------
colab/phase1/results/
  {name}_best.pt        best model checkpoint (state dict)
  {name}_history.json   per-epoch train/val metrics + final test metrics
  phase1_results.txt    summary table (all 12 experiments)
  phase1_writeup.txt    brief interpretation of results
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

from prepare_data import load_graph_data, split_data, get_data_loaders  # noqa: E402
from models import GNNShotPredictor                                       # noqa: E402
from losses import get_loss_fn                                            # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────────
# Override SWISHNET_BASE_DIR env-var for GCP or local runs, e.g.:
#   export SWISHNET_BASE_DIR=/gcs/my-bucket/NBA_GNN_files
BASE_DIR       = os.environ.get("SWISHNET_BASE_DIR", "/content/drive/MyDrive/NBA_GNN_files/")
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")
RESULTS_DIR    = os.path.join(_PHASE1_DIR, "results")

# ── Fixed hyperparameters (identical across ALL Phase 1 experiments) ───────────
# Taken from train_gat.py (the existing best-known config).
FIXED_HP = dict(
    num_node_features=41,
    hidden_dim=128,
    num_heads=4,        # GAT only; ignored by GINE
    num_layers=2,
    num_pre_layers=2,
    num_post_layers=2,
    dropout=0.3,
    train_eps=True,     # GINE only; ignored by GAT
)

TRAIN_HP = dict(
    lr=0.0001,
    weight_decay=5e-3,
    batch_size=256,
    epochs=200,
    patience=25,
    random_state=42,    # data-split seed — must never change across experiments
)

# ── Experiment configurations ──────────────────────────────────────────────────
EXPERIMENTS = [
    # ── Exp 1: Pooling variants (GAT conv, CE loss) ──────────────────────────
    dict(name="gat_mean",     conv="GAT",  readout="mean",            loss="ce",           loss_kwargs={}),
    dict(name="gat_max",      conv="GAT",  readout="max",             loss="ce",           loss_kwargs={}),
    dict(name="gat_add",      conv="GAT",  readout="add",             loss="ce",           loss_kwargs={}),
    dict(name="gat_mean_max", conv="GAT",  readout="mean_max",        loss="ce",           loss_kwargs={}),

    # ── Exp 2: Shooter-centric hybrid ────────────────────────────────────────
    dict(name="gat_shooter",  conv="GAT",  readout="shooter_centric", loss="ce",           loss_kwargs={}),
    dict(name="gine_mean",    conv="GINE", readout="mean",            loss="ce",           loss_kwargs={}),
    dict(name="gine_shooter", conv="GINE", readout="shooter_centric", loss="ce",           loss_kwargs={}),

    # ── Exp 7: Loss function variants (GAT + mean pool as control) ───────────
    dict(name="gat_ls_005",   conv="GAT",  readout="mean",            loss="label_smooth", loss_kwargs={"smoothing": 0.05}),
    dict(name="gat_ls_010",   conv="GAT",  readout="mean",            loss="label_smooth", loss_kwargs={"smoothing": 0.10}),
    dict(name="gat_focal_05", conv="GAT",  readout="mean",            loss="focal",        loss_kwargs={"gamma": 0.5}),
    dict(name="gat_focal_10", conv="GAT",  readout="mean",            loss="focal",        loss_kwargs={"gamma": 1.0}),
    dict(name="gat_focal_20", conv="GAT",  readout="mean",            loss="focal",        loss_kwargs={"gamma": 2.0}),
]


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss   = loss_fn(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        correct    += (logits.argmax(dim=1) == data.y).sum().item()
        total      += data.num_graphs
    return total_loss / total, correct / total


def _evaluate(model, loader, device):
    """Run inference, return (ce_loss, acc, acc_miss, acc_make, preds, labels, probs)."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            all_logits.append(model(data))
            all_labels.append(data.y)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    loss   = F.cross_entropy(logits, labels).item()
    pred   = logits.argmax(dim=1)
    acc    = (pred == labels).float().mean().item()
    probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    p_np   = pred.cpu().numpy()
    l_np   = labels.cpu().numpy()

    m0 = l_np == 0
    m1 = l_np == 1
    acc_0 = (p_np[m0] == l_np[m0]).mean() if m0.sum() > 0 else 0.0
    acc_1 = (p_np[m1] == l_np[m1]).mean() if m1.sum() > 0 else 0.0

    return loss, acc, acc_0, acc_1, p_np, l_np, probs


# ══════════════════════════════════════════════════════════════════════════════
# Experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(cfg, train_loader, val_loader, test_loader, device):
    """Train one experiment configuration and return a result dict."""
    print(f"\n{'='*65}")
    print(f"  Experiment : {cfg['name']}")
    print(f"  Conv       : {cfg['conv']}")
    print(f"  Readout    : {cfg['readout']}")
    print(f"  Loss       : {cfg['loss']}  {cfg['loss_kwargs']}")
    print(f"{'='*65}")

    model = GNNShotPredictor(
        conv_type=cfg['conv'],
        readout=cfg['readout'],
        **FIXED_HP,
    ).to(device)

    n_params = model.param_count()
    print(f"  Parameters : {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_HP['lr'],
        weight_decay=TRAIN_HP['weight_decay'],
    )
    loss_fn = get_loss_fn(cfg['loss'], **cfg['loss_kwargs'])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, f"{cfg['name']}_best.pt")

    best_val_loss = float('inf')
    best_epoch    = 0
    patience_ctr  = 0
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc')}

    t_start = time.time()

    for epoch in range(1, TRAIN_HP['epochs'] + 1):
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, loss_fn, device)
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

    # ── Final evaluation on test set using best checkpoint ───────────────────
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
        name=cfg['name'],
        conv=cfg['conv'],
        readout=cfg['readout'],
        loss=cfg['loss'],
        loss_kwargs=cfg['loss_kwargs'],
        best_epoch=best_epoch,
        best_val_loss=round(best_val_loss, 5),
        test_acc=round(test_acc, 4),
        test_acc_miss=round(test_acc0, 4),
        test_acc_make=round(test_acc1, 4),
        auc=round(auc, 4),
        f1=round(f1, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        n_params=n_params,
        elapsed_min=round(elapsed / 60, 1),
    )

    # Save history + metrics as JSON
    hist_path = os.path.join(RESULTS_DIR, f"{cfg['name']}_history.json")
    with open(hist_path, 'w') as fh:
        json.dump({**result, 'history': history}, fh, indent=2)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════════

def _loss_label(cfg):
    if cfg['loss'] == 'ce':
        return 'CE'
    elif cfg['loss'] == 'label_smooth':
        return f"LS α={cfg['loss_kwargs']['smoothing']}"
    elif cfg['loss'] == 'focal':
        return f"Focal γ={cfg['loss_kwargs']['gamma']}"
    return cfg['loss']


def generate_report(all_results, n_train, n_val, n_test, fg_pct):
    """Write a structured results table to phase1_results.txt."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'phase1_results.txt')

    sorted_results = sorted(all_results, key=lambda r: -r['auc'])

    lines = [
        "SWISHNET — PHASE 1 GNN EXPERIMENT RESULTS",
        "=" * 65,
        f"Train: {n_train}  Val: {n_val}  Test: {n_test}  FG%: {fg_pct:.1f}%",
        f"Logistic regression baseline (Static-38 / Lasso): AUC 0.614",
        "",
        "RESULTS  (sorted by test AUC)",
        "-" * 65,
        f"{'Experiment':<18} {'Conv':<5} {'Readout':<18} {'Loss':<14} "
        f"{'AUC':>6} {'Acc':>6} {'F1':>6} {'Miss':>6} {'Make':>6}",
        "-" * 65,
    ]

    for r in sorted_results:
        lines.append(
            f"{r['name']:<18} {r['conv']:<5} {r['readout']:<18} "
            f"{_loss_label({'loss': r['loss'], 'loss_kwargs': r['loss_kwargs']}):<14} "
            f"{r['auc']:>6.4f} {r['test_acc']:>6.4f} {r['f1']:>6.4f} "
            f"{r['test_acc_miss']:>6.3f} {r['test_acc_make']:>6.3f}"
        )

    lines += [
        "-" * 65,
        "",
        "TRAINING DETAILS",
        "-" * 65,
        f"{'Experiment':<18} {'Params':>10} {'BestEp':>7} {'Time(m)':>8}",
        "-" * 65,
    ]
    for r in sorted(all_results, key=lambda r: r['name']):
        lines.append(
            f"{r['name']:<18} {r['n_params']:>10,} {r['best_epoch']:>7} "
            f"{r['elapsed_min']:>8.1f}"
        )
    lines += [
        "-" * 65,
        "",
        f"Results saved to: {RESULTS_DIR}",
    ]

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    print(f"\nReport written to: {path}")
    return path


def generate_writeup(all_results, logistic_auc=0.614):
    """Generate a short interpretation document — phase1_writeup.txt."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'phase1_writeup.txt')

    by_auc  = sorted(all_results, key=lambda r: -r['auc'])
    best    = by_auc[0]
    worst   = by_auc[-1]

    # Subset views
    pooling_exps  = [r for r in all_results if r['conv'] == 'GAT' and
                     r['readout'] in ('mean', 'max', 'add', 'mean_max') and
                     r['loss'] == 'ce']
    shooter_exps  = [r for r in all_results if r['readout'] == 'shooter_centric']
    loss_exps     = [r for r in all_results if r['loss'] != 'ce' or
                     (r['loss'] == 'ce' and r['readout'] == 'mean' and r['conv'] == 'GAT')]
    gat_mean_auc  = next((r['auc'] for r in all_results if r['name'] == 'gat_mean'), None)
    gine_mean_auc = next((r['auc'] for r in all_results if r['name'] == 'gine_mean'), None)

    pool_ranked   = sorted(pooling_exps, key=lambda r: -r['auc'])
    shoot_ranked  = sorted(shooter_exps, key=lambda r: -r['auc'])
    loss_ranked   = sorted(loss_exps,   key=lambda r: -r['auc'])

    def auc_str(name):
        r = next((x for x in all_results if x['name'] == name), None)
        return f"{r['auc']:.4f}" if r else 'N/A'

    lines = [
        "SWISHNET — PHASE 1 GNN EXPERIMENT WRITE-UP",
        "=" * 65,
        "",
        "CONTEXT",
        "-------",
        "Phase 1 evaluates three orthogonal design axes on the same graph data",
        "and fixed hyperparameters (hidden_dim=128, 2 GAT/GINE layers, 2 pre/post",
        "MLP layers, dropout=0.3, lr=1e-4, weight_decay=5e-3).  All experiments",
        "share an identical 70/20/10 stratified split (random_state=42).",
        "Logistic regression baseline (Static-38 / Lasso): AUC 0.614.",
        "",
        "EXP 1 — POOLING STRATEGIES (GAT conv, CE loss)",
        "-----------------------------------------------",
        "Four readouts aggregate the 33 node embeddings (11 nodes × 3 timesteps)",
        "into a single graph vector before the classification head:",
        "",
        f"  mean      AUC {auc_str('gat_mean')}  — average over all nodes (baseline)",
        f"  max       AUC {auc_str('gat_max')}  — element-wise max across nodes",
        f"  add       AUC {auc_str('gat_add')}  — sum pool (BatchNorm re-normalises)",
        f"  mean_max  AUC {auc_str('gat_mean_max')}  — concat of mean and max (2× post-MLP width)",
        "",
    ]

    if pool_ranked:
        best_pool = pool_ranked[0]
        lines += [
            f"Best pooling: {best_pool['name']} (AUC {best_pool['auc']:.4f}).",
        ]
        if best_pool['name'] in ('gat_max', 'gat_mean_max'):
            lines.append(
                "Max pooling outperforming mean is consistent with the basketball"
                " intuition that shot outcomes are driven by extreme configurations"
                " (the most open player, the closest defender) rather than the"
                " average scene — information that max pooling preserves directly."
            )
        elif best_pool['name'] == 'gat_add':
            lines.append(
                "Add pooling matching or exceeding mean suggests the model benefits"
                " from summing contributions; BatchNorm in the post-MLP normalises"
                " the scale difference."
            )
        else:
            lines.append(
                "Mean pooling holding its ground suggests the attention mechanism"
                " in GATv2Conv already learns to weight the most relevant nodes,"
                " making alternative pooling strategies redundant."
            )

    lines += [
        "",
        "EXP 2 — SHOOTER-CENTRIC HYBRID READOUT",
        "---------------------------------------",
        "Instead of pooling all 33 nodes equally, the shooter node at t=0",
        "(identified by has_ball==1 AND timestamp==0) is extracted separately",
        "and concatenated with the scene mean pool: [shooter_emb || scene_emb].",
        "This doubles the post-MLP input width.  Tested on both GAT and GINE.",
        "",
        f"  gat_shooter   AUC {auc_str('gat_shooter')}   (GAT  mean baseline: {auc_str('gat_mean')})",
        f"  gine_shooter  AUC {auc_str('gine_shooter')}   (GINE mean baseline: {auc_str('gine_mean')})",
        "",
    ]

    gat_s_auc  = next((r['auc'] for r in all_results if r['name'] == 'gat_shooter'), None)
    gine_s_auc = next((r['auc'] for r in all_results if r['name'] == 'gine_shooter'), None)

    if gat_s_auc and gat_mean_auc:
        delta = gat_s_auc - gat_mean_auc
        if delta > 0.005:
            lines.append(
                f"Shooter-centric readout improves GAT by +{delta:.4f} AUC, confirming"
                " that the shooter node carries disproportionate signal that mean"
                " pooling dilutes.  The 21 player-stats features on the shooter node"
                " make it informationally richer than all others; extracting it"
                " explicitly lets the post-MLP learn a joint function of shooter"
                " ability and spatial context rather than blending them."
            )
        elif delta < -0.005:
            lines.append(
                f"Shooter-centric readout slightly hurts GAT ({delta:+.4f} AUC).  This"
                " suggests GATv2Conv's attention mechanism already propagates shooter"
                " information into neighbouring node representations during message"
                " passing, so extracting only the shooter node post-conv recovers"
                " less than the full mean pool."
            )
        else:
            lines.append(
                "Shooter-centric and mean readouts are within noise for GAT, suggesting"
                " the two designs are roughly equivalent at this model capacity."
            )

    lines += [
        "",
        "EXP 7 — LOSS FUNCTION VARIANTS (GAT + mean pool)",
        "-------------------------------------------------",
        "Five alternative loss functions are compared against standard CE on",
        "an identical GAT + mean pool architecture:",
        "",
        f"  CE (baseline)   AUC {auc_str('gat_mean')}",
        f"  LS α=0.05       AUC {auc_str('gat_ls_005')}",
        f"  LS α=0.10       AUC {auc_str('gat_ls_010')}",
        f"  Focal γ=0.5     AUC {auc_str('gat_focal_05')}",
        f"  Focal γ=1.0     AUC {auc_str('gat_focal_10')}",
        f"  Focal γ=2.0     AUC {auc_str('gat_focal_20')}",
        "",
        "Label smoothing regularises against overconfidence on individual shots",
        "where the outcome is partly stochastic (release mechanics after the",
        "snapshot are unobserved).  Focal loss concentrates gradient on shots",
        "the model finds ambiguous, potentially improving discrimination on",
        "contested mid-range attempts vs. easy/impossible looks.",
        "",
    ]

    best_loss_exp = loss_ranked[0] if loss_ranked else None
    if best_loss_exp:
        lines += [
            f"Best loss configuration: {best_loss_exp['name']} (AUC {best_loss_exp['auc']:.4f}).",
            "",
        ]

    lines += [
        "OVERALL SUMMARY",
        "---------------",
        f"Best Phase 1 model: {best['name']}  AUC {best['auc']:.4f}",
        f"Worst Phase 1 model: {worst['name']}  AUC {worst['auc']:.4f}",
        f"Logistic regression baseline: AUC {logistic_auc:.4f}",
        "",
    ]

    if best['auc'] > logistic_auc + 0.01:
        lines.append(
            f"The best GNN exceeds the logistic regression baseline by "
            f"+{best['auc']-logistic_auc:.4f} AUC, confirming that graph-relational"
            " structure adds predictive value beyond static shot geometry and"
            " shooter historical statistics."
        )
    elif best['auc'] > logistic_auc:
        lines.append(
            "The best GNN marginally exceeds the logistic regression baseline."
            " Phase 2 experiments (alternative conv mechanisms, temporal modeling,"
            " hyperparameter search) are needed to determine whether the graph"
            " structure is providing genuine value or the gap is within noise."
        )
    else:
        lines.append(
            "No GNN variant exceeds the logistic regression baseline in Phase 1."
            " This may reflect underfitting (insufficient model capacity or"
            " over-regularisation from weight_decay=5e-3) or that the Phase 1"
            " architectural changes do not address the core limiting factor."
            " Phase 2 will conduct a systematic hyperparameter sweep."
        )

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    print(f"Write-up saved to: {path}")
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

    # ── Load graph data ───────────────────────────────────────────────────────
    # All graphs were validated during construction by build_graph.py.
    # Shots with missing fields, player stats, or an unidentifiable shooter
    # are dropped before saving — only fully-constructed graphs reach this point.
    graphs, labels = load_graph_data(graph_data_dir=GRAPH_DATA_DIR)

    # Sanity-check a sample: verify structural integrity and shooter node presence
    print("\nSanity-checking graph structure ...")
    sample_size = min(20, len(graphs))
    for g in graphs[:sample_size]:
        assert g.x.shape[1] == 41,       f"Expected 41 node features, got {g.x.shape[1]}"
        assert g.edge_attr.shape[1] == 9, f"Expected 9 edge features, got {g.edge_attr.shape[1]}"
        shooter_mask = (g.x[:, 3] == 1.0) & (g.x[:, 14] == 0.0)
        assert shooter_mask.sum() == 1, \
            f"Expected exactly 1 shooter node at t=0, found {shooter_mask.sum()}"
    print(f"  OK — verified {sample_size} graphs")

    fg_pct = 100.0 * sum(labels) / len(labels)

    # ── Fixed data split ──────────────────────────────────────────────────────
    train_graphs, val_graphs, test_graphs, train_labels, val_labels, test_labels = \
        split_data(graphs, labels, random_state=TRAIN_HP['random_state'])

    train_loader, val_loader, test_loader = get_data_loaders(
        train_graphs, val_graphs, test_graphs,
        batch_size=TRAIN_HP['batch_size'],
        shuffle_train=True,
    )

    n_train, n_val, n_test = len(train_graphs), len(val_graphs), len(test_graphs)

    # ── Run all experiments ───────────────────────────────────────────────────
    all_results = []
    for cfg in EXPERIMENTS:
        result = run_experiment(cfg, train_loader, val_loader, test_loader, device)
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PHASE 1 SUMMARY  (sorted by AUC)")
    print(f"{'='*65}")
    print(f"{'Experiment':<18}  {'AUC':>6}  {'Acc':>6}  {'F1':>6}")
    print("-" * 40)
    for r in sorted(all_results, key=lambda r: -r['auc']):
        print(f"{r['name']:<18}  {r['auc']:>6.4f}  {r['test_acc']:>6.4f}  {r['f1']:>6.4f}")

    # ── Write report and write-up ─────────────────────────────────────────────
    generate_report(all_results, n_train, n_val, n_test, fg_pct)
    generate_writeup(all_results)

    print(f"\nAll Phase 1 results saved to: {RESULTS_DIR}")
    return all_results


if __name__ == '__main__':
    main()

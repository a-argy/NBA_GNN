#!/usr/bin/env python3
"""
Phase 2 GNN experiment runner — SwishNet.

Experiments
-----------
Exp 0b  Phase 1 best configs re-run on full 87K dataset (identical HPs):
    gat_mean_max      GAT  mean_max readout  CE loss   [Phase 1 best arch]
    gat_shooter       GAT  shooter_centric   CE loss   [Phase 1 best interpretable]

Exp 3   Explicit temporal modelling via GRU:
    temporal_gru      TemporalGRUShotPredictor  CE loss

Exp 4   Alternative conv mechanisms (both with mean_max readout):
    sage_mean_max     SAGEConv max-agg   CE loss   [no edge features]
    edge_mean_max     EdgeConv max-agg   CE loss   [relative h_j-h_i]

Exp 5   Best pooling + best loss combined:
    gat_mean_max_focal    GAT  mean_max  Focal γ=1.0
    temporal_gru_focal    GRU  (best arch from Exp 3)  Focal γ=1.0  [true Exp 5]

Usage:
    # All experiments (0b, 3, 4, 5):
    SWISHNET_BASE_DIR=~/data python colab/phase2/train_phase2.py

    # Single experiment:
    SWISHNET_BASE_DIR=~/data python colab/phase2/train_phase2.py --exp gat_mean_max

    # Dry run to check data loading:
    SWISHNET_BASE_DIR=~/data python colab/phase2/train_phase2.py --dry-run
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_PHASE2_DIR = os.path.dirname(os.path.abspath(__file__))
_COLAB_DIR  = os.path.dirname(_PHASE2_DIR)
_PHASE1_DIR = os.path.join(_COLAB_DIR, "phase1")
for _p in (_COLAB_DIR, _PHASE1_DIR, _PHASE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase1.models import GNNShotPredictor           # GAT / GINE
from phase1.losses import get_loss_fn                 # ce / focal / label_smooth
from phase2.models import (
    TemporalGRUShotPredictor,
    GNNShotPredictorV2,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.environ.get("SWISHNET_BASE_DIR", os.path.expanduser("~/data"))
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")
RESULTS_DIR    = os.path.join(_PHASE2_DIR, "results")

# ── Fixed hyperparameters (identical to Phase 1 for comparability) ─────────────
FIXED_HP = dict(
    num_node_features=41,
    hidden_dim=128,
    num_heads=4,
    num_layers=2,
    num_pre_layers=2,
    num_post_layers=2,
    dropout=0.3,
    train_eps=True,
)

TRAIN_HP = dict(
    lr=1e-4,
    weight_decay=5e-3,
    batch_size=512,       # larger than Phase 1 (256) for GPU efficiency
    epochs=100,           # reduced from 200; 87K data converges faster
    patience=20,
    random_state=42,
)

# ── Experiment registry ────────────────────────────────────────────────────────
EXPERIMENTS = [
    # Exp 0b: Phase 1 best on full data
    dict(name="gat_mean_max",       model="GAT",  loss="ce",    loss_kw={}),
    dict(name="gat_shooter",        model="GAT",  loss="ce",    loss_kw={},
         hp_override={"readout": "shooter_centric"}),

    # Exp 3: Temporal GRU
    dict(name="temporal_gru",       model="GRU",  loss="ce",    loss_kw={}),

    # Exp 4: SAGE and EdgeConv (mean_max readout implicit in V2)
    dict(name="sage_mean_max",      model="SAGE", loss="ce",    loss_kw={}),
    dict(name="edge_mean_max",      model="EDGE", loss="ce",    loss_kw={}),

    # Exp 5: best pooling + best loss combined
    dict(name="gat_mean_max_focal",   model="GAT",  loss="focal", loss_kw={"gamma": 1.0}),
    # Exp 5b: true Exp 5 — best architecture (temporal_gru) + focal loss
    dict(name="temporal_gru_focal",   model="GRU",  loss="focal", loss_kw={"gamma": 1.0}),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_graphs(graph_data_dir):
    """Load all per-game .pt files (each is a list of PyG Data objects)."""
    print(f"Loading graphs from {graph_data_dir} ...")
    pt_files = sorted(f for f in os.listdir(graph_data_dir) if f.endswith(".pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {graph_data_dir}")

    all_graphs = []
    for fname in tqdm(pt_files, desc="Loading game files"):
        path = os.path.join(graph_data_dir, fname)
        game_graphs = torch.load(path, weights_only=False)
        if isinstance(game_graphs, list):
            all_graphs.extend(game_graphs)
        else:
            all_graphs.append(game_graphs)

    labels = [int(g.y.item()) for g in all_graphs]
    n = len(all_graphs)
    print(f"Loaded {n:,} graphs — FG% {100*sum(labels)/n:.1f}%")
    return all_graphs, labels


def make_loaders(graphs, labels, batch_size, random_state):
    idx = list(range(len(graphs)))
    tr_idx, tmp_idx = train_test_split(idx, train_size=0.7,
                                       stratify=labels, random_state=random_state)
    tmp_labels = [labels[i] for i in tmp_idx]
    val_idx, te_idx = train_test_split(tmp_idx, train_size=2/3,
                                       stratify=tmp_labels, random_state=random_state)

    tr  = [graphs[i] for i in tr_idx]
    val = [graphs[i] for i in val_idx]
    te  = [graphs[i] for i in te_idx]

    print(f"Split — Train {len(tr):,} / Val {len(val):,} / Test {len(te):,}")
    return (
        DataLoader(tr,  batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(te,  batch_size=batch_size, shuffle=False),
    )


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(exp, device):
    hp  = {**FIXED_HP, **exp.get("hp_override", {})}
    mtype = exp["model"]

    if mtype in ("GAT", "GINE"):
        model = GNNShotPredictor(
            conv_type=mtype,
            readout=hp.get("readout", "mean_max"),
            num_node_features=hp["num_node_features"],
            hidden_dim=hp["hidden_dim"],
            num_heads=hp["num_heads"],
            num_layers=hp["num_layers"],
            num_pre_layers=hp["num_pre_layers"],
            num_post_layers=hp["num_post_layers"],
            dropout=hp["dropout"],
            train_eps=hp["train_eps"],
        )
    elif mtype == "GRU":
        model = TemporalGRUShotPredictor(
            num_node_features=hp["num_node_features"],
            hidden_dim=hp["hidden_dim"],
            num_heads=hp["num_heads"],
            num_gnn_layers=hp["num_layers"],
            num_pre_layers=hp["num_pre_layers"],
            num_post_layers=hp["num_post_layers"],
            dropout=hp["dropout"],
        )
    elif mtype in ("SAGE", "EDGE"):
        model = GNNShotPredictorV2(
            conv_type=mtype,
            num_node_features=hp["num_node_features"],
            hidden_dim=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            num_pre_layers=hp["num_pre_layers"],
            num_post_layers=hp["num_post_layers"],
            dropout=hp["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    return model.to(device)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss   = loss_fn(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        correct    += (logits.argmax(1) == batch.y).sum().item()
        n          += batch.num_graphs
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        all_logits.append(model(batch))
        all_labels.append(batch.y)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).cpu().numpy()
    probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds  = logits.argmax(1).cpu().numpy()

    loss = F.cross_entropy(logits, torch.cat(all_labels).to(logits.device)).item()
    auc  = roc_auc_score(labels, probs)
    acc  = (preds == labels).mean()
    f1   = f1_score(labels, preds, zero_division=0)
    return loss, auc, acc, f1


# ── Single experiment ─────────────────────────────────────────────────────────

def run_experiment(exp, train_loader, val_loader, test_loader, device):
    name = exp["name"]
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    model    = build_model(exp, device)
    loss_fn  = get_loss_fn(exp["loss"], **exp["loss_kw"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_HP["lr"],
        weight_decay=TRAIN_HP["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    print(f"Parameters: {model.param_count():,}")

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}
    best_val_auc = 0.0
    best_epoch   = 0
    patience_ctr = 0
    best_state   = None
    t0 = time.time()

    for epoch in range(1, TRAIN_HP["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_auc, val_acc, _ = evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            patience_ctr = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = (time.time() - t0) / 60
            print(f"  Ep {epoch:3d}  tr_loss {tr_loss:.4f}  "
                  f"val_auc {val_auc:.4f}  val_acc {val_acc:.4f}  "
                  f"[best {best_val_auc:.4f} @ ep{best_epoch}]  "
                  f"{elapsed:.1f}m")

        if patience_ctr >= TRAIN_HP["patience"]:
            print(f"  Early stop at epoch {epoch} (patience={TRAIN_HP['patience']})")
            break

    # ── Test evaluation ────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(device)
    _, test_auc, test_acc, test_f1 = evaluate(model, test_loader, device)
    elapsed = (time.time() - t0) / 60

    print(f"\n  Best epoch : {best_epoch}")
    print(f"  Test AUC   : {test_auc:.4f}")
    print(f"  Test Acc   : {test_acc:.4f}")
    print(f"  Test F1    : {test_f1:.4f}")
    print(f"  Total time : {elapsed:.1f} min")

    # ── Save outputs ───────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(RESULTS_DIR, f"{name}_best.pt"))

    history["test_auc"]  = test_auc
    history["test_acc"]  = test_acc
    history["test_f1"]   = test_f1
    history["best_epoch"] = best_epoch
    history["elapsed_min"] = elapsed
    history["params"]    = model.param_count()
    with open(os.path.join(RESULTS_DIR, f"{name}_history.json"), "w") as f:
        json.dump(history, f)

    return {
        "name":      name,
        "model":     exp["model"],
        "loss":      exp["loss"],
        "test_auc":  test_auc,
        "test_acc":  test_acc,
        "test_f1":   test_f1,
        "best_epoch": best_epoch,
        "params":    model.param_count(),
        "elapsed_min": elapsed,
    }


# ── Results summary ───────────────────────────────────────────────────────────

def write_results(results, n_graphs, fg_pct):
    sep = "-" * 80
    lines = [
        "SWISHNET — PHASE 2 GNN RESULTS",
        f"Dataset: {n_graphs:,} graphs  |  FG% {fg_pct*100:.1f}%  |  "
        f"Split 70/20/10  |  seed=42",
        f"Baseline (Phase 1, 12 games): LR Lasso AUC 0.614",
        "",
        f"{'Experiment':<24}  {'Conv':<6}  {'Loss':<14}  "
        f"{'AUC':>8}  {'Acc':>7}  {'F1':>7}  {'Params':>10}  {'Ep':>5}",
        sep,
    ]
    for r in sorted(results, key=lambda x: -x["test_auc"]):
        lines.append(
            f"{r['name']:<24}  {r['model']:<6}  {r['loss']:<14}  "
            f"{r['test_auc']:.4f}  {r['test_acc']:.4f}  {r['test_f1']:.4f}  "
            f"{r['params']:>10,}  {r['best_epoch']:>5}"
        )
    lines.append(sep)

    text = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, "phase2_results.txt")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"\n\n{text}")
    print(f"\nResults saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None,
                        help="Run a single experiment by name (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and build models only — no training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ────────────────────────────────────────────────────────────────
    graphs, labels = load_all_graphs(GRAPH_DATA_DIR)
    n_graphs = len(graphs)
    fg_pct   = sum(labels) / n_graphs

    if args.dry_run:
        print("\nDry run: building all models ...")
        for exp in EXPERIMENTS:
            m = build_model(exp, device)
            print(f"  {exp['name']:<24}  {m.param_count():,} params")
        print("Dry run complete.")
        return

    train_loader, val_loader, test_loader = make_loaders(
        graphs, labels, TRAIN_HP["batch_size"], TRAIN_HP["random_state"]
    )

    # ── Run experiments ────────────────────────────────────────────────────
    exps_to_run = (
        [e for e in EXPERIMENTS if e["name"] == args.exp]
        if args.exp else EXPERIMENTS
    )
    if not exps_to_run:
        print(f"Unknown experiment: {args.exp}")
        print(f"Available: {[e['name'] for e in EXPERIMENTS]}")
        sys.exit(1)

    results = []
    for exp in exps_to_run:
        result = run_experiment(exp, train_loader, val_loader, test_loader, device)
        results.append(result)

    write_results(results, n_graphs, fg_pct)


if __name__ == "__main__":
    main()

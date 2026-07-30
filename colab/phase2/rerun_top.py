#!/usr/bin/env python3
"""
Exp 6 full reruns — top-3 configs from HP sweep, trained to completion.

Trains each config with epochs=200/patience=25 (same as Phase 2 standard),
evaluates on held-out test set, saves best checkpoint + history JSON.

Usage:
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/rerun_top.py
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_PHASE2_DIR = os.path.dirname(os.path.abspath(__file__))
_COLAB_DIR  = os.path.dirname(_PHASE2_DIR)
_PHASE1_DIR = os.path.join(_COLAB_DIR, "phase1")
for _p in (_COLAB_DIR, _PHASE1_DIR, _PHASE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase1.losses import get_loss_fn
from phase2.models import TemporalGRUShotPredictor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.environ.get("SWISHNET_BASE_DIR", os.path.expanduser("~/data"))
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")
RESULTS_DIR    = os.path.join(_PHASE2_DIR, "results", "rerun_top")

# ── Top-3 configs from Exp 6 HP sweep ─────────────────────────────────────────
# Ranked by val_auc (60-epoch sweep). All have lr=5e-4, wd=1e-4.
TOP_CONFIGS = [
    {"name": "rerun_rank01", "hidden_dim": 256, "num_layers": 3, "weight_decay": 1e-4,
     "dropout": 0.1, "lr": 5e-4, "batch_size": 512, "sweep_val_auc": 0.6487},
    {"name": "rerun_rank02", "hidden_dim": 256, "num_layers": 3, "weight_decay": 1e-4,
     "dropout": 0.1, "lr": 5e-4, "batch_size": 256, "sweep_val_auc": 0.6479},
    {"name": "rerun_rank03", "hidden_dim": 256, "num_layers": 4, "weight_decay": 1e-4,
     "dropout": 0.1, "lr": 5e-4, "batch_size": 512, "sweep_val_auc": 0.6477},
]

# ── Fixed architecture HPs ────────────────────────────────────────────────────
NUM_NODE_FEATURES = 41
NUM_HEADS         = 4
NUM_PRE_LAYERS    = 2
NUM_POST_LAYERS   = 2

EPOCHS   = 200
PATIENCE = 25
DATA_SEED = 42


# ── Data ──────────────────────────────────────────────────────────────────────

def load_all_graphs(graph_data_dir):
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


def make_loaders(graphs, labels, batch_size):
    idx = list(range(len(graphs)))
    tr_idx, tmp_idx = train_test_split(idx, train_size=0.7,
                                       stratify=labels, random_state=DATA_SEED)
    tmp_labels = [labels[i] for i in tmp_idx]
    val_idx, te_idx = train_test_split(tmp_idx, train_size=2/3,
                                       stratify=tmp_labels, random_state=DATA_SEED)
    tr  = [graphs[i] for i in tr_idx]
    val = [graphs[i] for i in val_idx]
    te  = [graphs[i] for i in te_idx]
    return (
        DataLoader(tr,  batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(te,  batch_size=batch_size, shuffle=False),
    )


# ── Training ──────────────────────────────────────────────────────────────────

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
    auc = roc_auc_score(labels, probs)
    acc = (preds == labels).mean()
    f1  = f1_score(labels, preds, zero_division=0)
    return auc, acc, f1


def run_one(cfg, graphs, labels, device):
    name = cfg["name"]
    print(f"\n{'='*65}")
    print(f"  {name}  |  sweep_val_auc={cfg['sweep_val_auc']:.4f}")
    print(f"  hd={cfg['hidden_dim']} nl={cfg['num_layers']} "
          f"wd={cfg['weight_decay']:.0e} do={cfg['dropout']} "
          f"lr={cfg['lr']:.0e} bs={cfg['batch_size']}")
    print(f"{'='*65}")

    train_loader, val_loader, test_loader = make_loaders(
        graphs, labels, cfg["batch_size"])

    model = TemporalGRUShotPredictor(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=cfg["hidden_dim"],
        num_heads=NUM_HEADS,
        num_gnn_layers=cfg["num_layers"],
        num_pre_layers=NUM_PRE_LAYERS,
        num_post_layers=NUM_POST_LAYERS,
        dropout=cfg["dropout"],
    ).to(device)
    print(f"  Params: {model.param_count():,}")

    loss_fn   = get_loss_fn("ce")
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8)

    best_val_auc = 0.0
    best_epoch   = 0
    patience_ctr = 0
    best_state   = None
    history      = []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_auc, val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                         "val_auc": val_auc, "val_acc": val_acc, "val_f1": val_f1})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            patience_ctr = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = (time.time() - t0) / 60
            print(f"  ep{epoch:3d}  val_auc={val_auc:.4f}  "
                  f"[best={best_val_auc:.4f}@{best_epoch}]  {elapsed:.1f}m")

        if patience_ctr >= PATIENCE:
            print(f"  Early stop at ep{epoch} (patience={PATIENCE})")
            break

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(device)
    test_auc, test_acc, test_f1 = evaluate(model, test_loader, device)
    elapsed = (time.time() - t0) / 60

    print(f"\n  RESULT  val_auc={best_val_auc:.4f} @ ep{best_epoch}  "
          f"test_auc={test_auc:.4f}  test_acc={test_acc:.4f}  "
          f"test_f1={test_f1:.4f}  {elapsed:.1f}m")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(RESULTS_DIR, f"{name}_best.pt"))

    result = {
        "name":          name,
        "config":        {k: v for k, v in cfg.items() if k != "name"},
        "best_val_auc":  best_val_auc,
        "best_epoch":    best_epoch,
        "test_auc":      test_auc,
        "test_acc":      test_acc,
        "test_f1":       test_f1,
        "elapsed_min":   elapsed,
        "params":        model.param_count(),
    }
    with open(os.path.join(RESULTS_DIR, f"{name}_result.json"), "w") as f:
        json.dump({**result, "history": history}, f, indent=2)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    graphs, labels = load_all_graphs(GRAPH_DATA_DIR)

    results = []
    for cfg in TOP_CONFIGS:
        result = run_one(cfg, graphs, labels, device)
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"RERUN SUMMARY")
    print(f"{'='*65}")
    print(f"{'Name':<16} {'val_auc':>8} {'test_auc':>9} {'test_acc':>9} "
          f"{'test_f1':>8} {'ep':>4} {'params':>10}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: -x["test_auc"]):
        print(f"{r['name']:<16} {r['best_val_auc']:>8.4f} {r['test_auc']:>9.4f} "
              f"{r['test_acc']:>9.4f} {r['test_f1']:>8.4f} "
              f"{r['best_epoch']:>4} {r['params']:>10,}")

    with open(os.path.join(RESULTS_DIR, "rerun_summary.json"), "w") as f:
        json.dump([{k: v for k, v in r.items()} for r in results], f, indent=2)
    print(f"\nResults: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

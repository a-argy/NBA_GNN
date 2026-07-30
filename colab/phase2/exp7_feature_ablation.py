#!/usr/bin/env python3
"""
Exp 7 — Diagnostic feature ablation: which feature groups drive GNN AUC?

For each ablation, zero out (or shuffle for temporal_id) the relevant feature
indices in batch.x before every forward pass — both train AND eval — so the
model learns with and is evaluated against the same masked input.

Trains from scratch using rank03 HPs (hd=256, nl=4, wd=1e-4, do=0.1,
lr=5e-4, bs=512) with reduced epochs=100/patience=20 for compute efficiency.

Baseline (no ablation): test_auc = 0.6451 (rank03 full rerun).

Feature groups (41-dim node feature vector):
  spatial_xyz  : 0–2   x, y, z coordinates
  role_flags   : 3–5   has_ball, is_offense, is_player_node
  geometry     : 6–10  dist_to_rim, dist_to_ball_handler, angle_to_basket,
                        dist_to_3pt, num_nearby_def
  game_state   : 11–13 quarter, game_clock, shot_clock
  temporal_id  : 14    timestamp (0/1/2 — tells GRU which snapshot)
  position_enc : 15–19 position one-hot (G/F/C/G-F/F-C)
  player_stats : 20–40 21 shooting stats

Usage:
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/exp7_feature_ablation.py
    # Run a single ablation:
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/exp7_feature_ablation.py \\
        --ablation temporal_id
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
RESULTS_DIR    = os.path.join(_PHASE2_DIR, "results", "exp7_ablation")

# ── rank03 HPs ────────────────────────────────────────────────────────────────
RANK03_HPS = {
    "hidden_dim":    256,
    "num_layers":    4,
    "weight_decay":  1e-4,
    "dropout":       0.1,
    "lr":            5e-4,
    "batch_size":    512,
}
BASELINE_TEST_AUC = 0.6451

NUM_NODE_FEATURES = 41
NUM_HEADS         = 4
NUM_PRE_LAYERS    = 2
NUM_POST_LAYERS   = 2

EPOCHS    = 100
PATIENCE  = 20
DATA_SEED = 42

# ── Ablation configs ───────────────────────────────────────────────────────────
#
# Each entry defines the feature indices to zero out (or shuffle for temporal_id).
# Indices reference the 41-dim node feature vector in build_graph.py.

ABLATION_CONFIGS = [
    {
        "name":        "abl_spatial_xyz",
        "group":       "spatial_xyz",
        "indices":     list(range(0, 3)),
        "description": "x, y, z coordinates",
    },
    {
        "name":        "abl_role_flags",
        "group":       "role_flags",
        "indices":     list(range(3, 6)),
        "description": "has_ball, is_offense, is_player_node",
    },
    {
        "name":        "abl_geometry",
        "group":       "geometry",
        "indices":     list(range(6, 11)),
        "description": "dist_to_rim, dist_to_ball_handler, angle_to_basket, dist_to_3pt, num_nearby_def",
    },
    {
        "name":        "abl_game_state",
        "group":       "game_state",
        "indices":     list(range(11, 14)),
        "description": "quarter, game_clock, shot_clock",
    },
    {
        "name":        "abl_temporal_id",
        "group":       "temporal_id",
        "indices":     [14],
        "description": "timestamp 0/1/2 (tells GRU snapshot order)",
    },
    {
        "name":        "abl_position_enc",
        "group":       "position_enc",
        "indices":     list(range(15, 20)),
        "description": "position one-hot (G/F/C/G-F/F-C)",
    },
    {
        "name":        "abl_player_stats",
        "group":       "player_stats",
        "indices":     list(range(20, 41)),
        "description": "21 shooting stats",
    },
]


# ── Feature masking ────────────────────────────────────────────────────────────

def zero_features(batch, indices, group):
    """
    Return a batch with the given feature indices masked.

    For all groups: set batch.x[:, indices] = 0.0.

    Special case — temporal_id (index 14):
        Zeroing all timestamps to 0.0 would collapse all nodes to t=0 and make
        timestep subgraphs for t=1 and t=2 empty, causing crashes in the GRU
        model. Instead, we SHUFFLE the timestamp values within each graph so
        that the GRU cannot use temporal ordering — the signal is ablated while
        keeping exactly 11 nodes per timestep across all graphs.
    """
    batch = batch.clone()
    x = batch.x.clone()

    if group == "temporal_id":
        # Shuffle index-14 values within each graph to break temporal ordering
        b = batch.batch
        for g_idx in b.unique():
            node_indices = (b == g_idx).nonzero(as_tuple=True)[0]
            perm = torch.randperm(node_indices.shape[0], device=node_indices.device)
            x[node_indices, 14] = x[node_indices[perm], 14]
    else:
        x[:, indices] = 0.0

    batch.x = x
    return batch


# ── Data ──────────────────────────────────────────────────────────────────────

def load_all_graphs(graph_data_dir):
    print(f"Loading graphs from {graph_data_dir} ...")
    pt_files = sorted(f for f in os.listdir(graph_data_dir) if f.endswith(".pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {graph_data_dir}")
    all_graphs = []
    skipped = 0
    for fname in tqdm(pt_files, desc="Loading game files"):
        path = os.path.join(graph_data_dir, fname)
        try:
            game_graphs = torch.load(path, weights_only=False)
            if isinstance(game_graphs, list):
                all_graphs.extend(game_graphs)
            else:
                all_graphs.append(game_graphs)
        except Exception as e:
            skipped += 1
            tqdm.write(f"  SKIP {fname}: {e}")
    if skipped:
        print(f"Skipped {skipped} unreadable file(s) (legacy format)")
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

def train_epoch(model, loader, optimizer, loss_fn, device, abl_indices, abl_group):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        if abl_indices is not None:
            batch = zero_features(batch, abl_indices, abl_group)
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
def evaluate(model, loader, device, abl_indices, abl_group):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        if abl_indices is not None:
            batch = zero_features(batch, abl_indices, abl_group)
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


def run_ablation(cfg, graphs, labels, device):
    name        = cfg["name"]
    group       = cfg["group"]
    indices     = cfg["indices"]
    description = cfg["description"]

    print(f"\n{'='*70}")
    print(f"  {name}  |  group={group}")
    print(f"  Features zeroed: [{', '.join(map(str, indices))}]  ({description})")
    print(f"  HPs: hd={RANK03_HPS['hidden_dim']} nl={RANK03_HPS['num_layers']} "
          f"wd={RANK03_HPS['weight_decay']:.0e} do={RANK03_HPS['dropout']} "
          f"lr={RANK03_HPS['lr']:.0e} bs={RANK03_HPS['batch_size']}")
    if group == "temporal_id":
        print(f"  [temporal_id: SHUFFLING timestamps instead of zeroing to avoid empty subgraphs]")
    print(f"{'='*70}")

    train_loader, val_loader, test_loader = make_loaders(
        graphs, labels, RANK03_HPS["batch_size"])

    model = TemporalGRUShotPredictor(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=RANK03_HPS["hidden_dim"],
        num_heads=NUM_HEADS,
        num_gnn_layers=RANK03_HPS["num_layers"],
        num_pre_layers=NUM_PRE_LAYERS,
        num_post_layers=NUM_POST_LAYERS,
        dropout=RANK03_HPS["dropout"],
    ).to(device)
    print(f"  Params: {model.param_count():,}")

    loss_fn   = get_loss_fn("ce")
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=RANK03_HPS["lr"],
                                 weight_decay=RANK03_HPS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8)

    best_val_auc = 0.0
    best_epoch   = 0
    patience_ctr = 0
    best_state   = None
    history      = []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, device, indices, group)
        val_auc, val_acc, val_f1 = evaluate(
            model, val_loader, device, indices, group)
        scheduler.step(val_auc)

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_auc": val_auc, "val_acc": val_acc, "val_f1": val_f1,
        })

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

    # ── Test evaluation ────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(device)
    test_auc, test_acc, test_f1 = evaluate(model, test_loader, device, indices, group)
    elapsed = (time.time() - t0) / 60

    delta_auc = test_auc - BASELINE_TEST_AUC
    print(f"\n  RESULT  val_auc={best_val_auc:.4f}@ep{best_epoch}  "
          f"test_auc={test_auc:.4f}  Δ={delta_auc:+.4f}  "
          f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  {elapsed:.1f}m")

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result = {
        "name":           name,
        "group":          group,
        "ablated_indices": indices,
        "description":    description,
        "ablation_method": "shuffle" if group == "temporal_id" else "zero",
        "best_val_auc":   best_val_auc,
        "best_epoch":     best_epoch,
        "test_auc":       test_auc,
        "test_acc":       test_acc,
        "test_f1":        test_f1,
        "delta_auc":      delta_auc,
        "baseline_auc":   BASELINE_TEST_AUC,
        "elapsed_min":    elapsed,
        "params":         model.param_count(),
        "hps":            RANK03_HPS,
        "epochs_run":     epoch,
        "patience":       PATIENCE,
    }
    with open(os.path.join(RESULTS_DIR, f"{name}_result.json"), "w") as f:
        json.dump({**result, "history": history}, f, indent=2)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 7: Feature ablation study")
    parser.add_argument(
        "--ablation", type=str, default=None,
        help="Run only one ablation group (e.g. 'temporal_id', 'player_stats'). "
             "Omit to run all 7."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    graphs, labels = load_all_graphs(GRAPH_DATA_DIR)

    # Filter to requested ablation(s)
    configs = ABLATION_CONFIGS
    if args.ablation is not None:
        configs = [c for c in ABLATION_CONFIGS if c["group"] == args.ablation]
        if not configs:
            valid = [c["group"] for c in ABLATION_CONFIGS]
            raise ValueError(f"Unknown ablation '{args.ablation}'. Valid: {valid}")

    results = []
    for cfg in configs:
        result = run_ablation(cfg, graphs, labels, device)
        results.append(result)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"EXP 7 ABLATION SUMMARY  (baseline test_auc = {BASELINE_TEST_AUC:.4f})")
    print(f"{'='*75}")
    print(f"{'Group':<16} {'test_auc':>9} {'Δ AUC':>8} {'val_auc':>8} "
          f"{'ep':>4} {'method':>8}")
    print("-" * 75)
    # Sort by Δ AUC ascending (most damaging ablation first)
    for r in sorted(results, key=lambda x: x["delta_auc"]):
        method = "shuffle" if r["group"] == "temporal_id" else "zero"
        print(f"{r['group']:<16} {r['test_auc']:>9.4f} {r['delta_auc']:>+8.4f} "
              f"{r['best_val_auc']:>8.4f} {r['best_epoch']:>4} {method:>8}")
    print(f"\n  Negative Δ = feature group hurts performance when removed (important).")
    print(f"  Positive Δ = group may be adding noise (or model adapted without it).")

    # ── Save summary ───────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {
        "baseline_test_auc": BASELINE_TEST_AUC,
        "baseline_name":     "rerun_rank03",
        "hps":               RANK03_HPS,
        "epochs":            EPOCHS,
        "patience":          PATIENCE,
        "results":           sorted(results, key=lambda x: x["delta_auc"]),
    }
    with open(os.path.join(RESULTS_DIR, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

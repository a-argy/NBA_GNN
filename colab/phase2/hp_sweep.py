#!/usr/bin/env python3
"""
Exp 6 — HP search on temporal_gru (SwishNet Phase 2).

Two modes:
  Random mode (default): samples --configs random configs from SEARCH_SPACE.
  Focused mode (--focused): enumerates a tighter grid (lr=5e-4 fixed,
      wd∈{1e-4,1e-3}, nl∈{3,4}, hd∈{64,128,256}, do∈{0.1,0.2}, bs∈{128,256})
      and skips configs already present in sweep_results.json.

Use --run-id-offset N to number runs starting from N+1 (for appending to an
existing sweep_results.json from a prior run).

Usage:
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/hp_sweep.py
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/hp_sweep.py --configs 20
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/hp_sweep.py --focused --run-id-offset 10
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/hp_sweep.py --focused --configs 20 --run-id-offset 10
"""

import argparse
import json
import os
import random
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
RESULTS_DIR    = os.path.join(_PHASE2_DIR, "results", "hp_sweep")

# ── Search spaces ─────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    "hidden_dim":    [64, 128, 256],
    "num_layers":    [1, 2, 3, 4],
    "weight_decay":  [1e-4, 1e-3, 5e-3],
    "dropout":       [0.1, 0.2, 0.3],
    "lr":            [1e-4, 5e-4],
    "batch_size":    [128, 256, 512],
}

# Hand-picked focused configs (lock lr=5e-4, wd=1e-4; based on runs 1-9).
# Rules:
#   - hd=64  → nl=4 only  (nl=3 already covered by run 03)
#   - hd=128 → nl∈{3,4}   (run 01 = hd=128,nl=4,do=0.2,bs=256 auto-skipped)
#   - hd=256 → nl∈{3,4}
#   - do∈{0.1,0.2}, bs∈{256,512}; wd=1e-4, lr=5e-4 fixed
FOCUSED_CONFIGS = [
    # hd=64, nl=4
    {"hidden_dim":  64, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.1, "lr": 5e-4, "batch_size": 256},
    {"hidden_dim":  64, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.1, "lr": 5e-4, "batch_size": 512},
    {"hidden_dim":  64, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.2, "lr": 5e-4, "batch_size": 256},
    {"hidden_dim":  64, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.2, "lr": 5e-4, "batch_size": 512},
    # hd=128, nl=3
    {"hidden_dim": 128, "num_layers": 3, "weight_decay": 1e-4, "dropout": 0.1, "lr": 5e-4, "batch_size": 256},
    {"hidden_dim": 128, "num_layers": 3, "weight_decay": 1e-4, "dropout": 0.1, "lr": 5e-4, "batch_size": 512},
    {"hidden_dim": 128, "num_layers": 3, "weight_decay": 1e-4, "dropout": 0.2, "lr": 5e-4, "batch_size": 256},
    {"hidden_dim": 128, "num_layers": 3, "weight_decay": 1e-4, "dropout": 0.2, "lr": 5e-4, "batch_size": 512},
    # hd=128, nl=4 (do=0.2/bs=256 is run 01 — auto-skipped)
    {"hidden_dim": 128, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.1, "lr": 5e-4, "batch_size": 256},
    {"hidden_dim": 128, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.1, "lr": 5e-4, "batch_size": 512},
    {"hidden_dim": 128, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.2, "lr": 5e-4, "batch_size": 256},  # run 01
    {"hidden_dim": 128, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.2, "lr": 5e-4, "batch_size": 512},
    # hd=256, nl=3
    {"hidden_dim": 256, "num_layers": 3, "weight_decay": 1e-4, "dropout": 0.1, "lr": 5e-4, "batch_size": 256},
    {"hidden_dim": 256, "num_layers": 3, "weight_decay": 1e-4, "dropout": 0.1, "lr": 5e-4, "batch_size": 512},
    {"hidden_dim": 256, "num_layers": 3, "weight_decay": 1e-4, "dropout": 0.2, "lr": 5e-4, "batch_size": 256},
    {"hidden_dim": 256, "num_layers": 3, "weight_decay": 1e-4, "dropout": 0.2, "lr": 5e-4, "batch_size": 512},
    # hd=256, nl=4
    {"hidden_dim": 256, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.1, "lr": 5e-4, "batch_size": 256},
    {"hidden_dim": 256, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.1, "lr": 5e-4, "batch_size": 512},
    {"hidden_dim": 256, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.2, "lr": 5e-4, "batch_size": 256},
    {"hidden_dim": 256, "num_layers": 4, "weight_decay": 1e-4, "dropout": 0.2, "lr": 5e-4, "batch_size": 512},
]

# Fixed across all sweep runs
NUM_NODE_FEATURES = 41
NUM_HEADS         = 4   # fixed; tested separately if needed
NUM_PRE_LAYERS    = 2
NUM_POST_LAYERS   = 2

SWEEP_EPOCHS   = 60
SWEEP_PATIENCE = 15
DATA_SEED      = 42


# ── Data loading ──────────────────────────────────────────────────────────────

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


def run_config(cfg, graphs, labels, device, run_id):
    train_loader, val_loader, _ = make_loaders(graphs, labels, cfg["batch_size"])

    model = TemporalGRUShotPredictor(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=cfg["hidden_dim"],
        num_heads=NUM_HEADS,
        num_gnn_layers=cfg["num_layers"],
        num_pre_layers=NUM_PRE_LAYERS,
        num_post_layers=NUM_POST_LAYERS,
        dropout=cfg["dropout"],
    ).to(device)

    loss_fn   = get_loss_fn("ce")
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6
    )

    best_val_auc = 0.0
    best_epoch   = 0
    patience_ctr = 0
    best_state   = None
    t0 = time.time()

    for epoch in range(1, SWEEP_EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_auc, val_acc, _ = evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            patience_ctr = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = (time.time() - t0) / 60
            print(f"    ep{epoch:3d}  val_auc={val_auc:.4f}  [best={best_val_auc:.4f}@{best_epoch}]  {elapsed:.1f}m")

        if patience_ctr >= SWEEP_PATIENCE:
            break

    elapsed = (time.time() - t0) / 60
    print(f"  [{run_id:02d}] hd={cfg['hidden_dim']} nl={cfg['num_layers']} "
          f"wd={cfg['weight_decay']:.0e} do={cfg['dropout']} "
          f"lr={cfg['lr']:.0e} bs={cfg['batch_size']}  "
          f"val_auc={best_val_auc:.4f} @ ep{best_epoch}  {elapsed:.1f}m")

    return {
        "run_id":        run_id,
        "config":        cfg,
        "best_val_auc":  best_val_auc,
        "best_epoch":    best_epoch,
        "elapsed_min":   elapsed,
        "params":        model.param_count(),
        "best_state":    best_state,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _cfg_key(cfg):
    return tuple(sorted(cfg.items()))


def _load_completed_keys(results_dir):
    """Return a set of config keys already in sweep_results.json, if it exists."""
    path = os.path.join(results_dir, "sweep_results.json")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        existing = json.load(f)
    return {_cfg_key(r["config"]) for r in existing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=int, default=30,
                        help="Max configs to run (default: 30)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for config sampling (default: 0)")
    parser.add_argument("--focused", action="store_true",
                        help="Use focused grid instead of random sampling")
    parser.add_argument("--run-id-offset", type=int, default=0,
                        help="Offset run IDs by N (for appending to prior sweep)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    graphs, labels = load_all_graphs(GRAPH_DATA_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    completed_keys = _load_completed_keys(RESULTS_DIR)
    if completed_keys:
        print(f"Found {len(completed_keys)} already-completed configs in sweep_results.json — skipping.")

    if args.focused:
        # Use hand-picked list, skip completed configs
        candidates = list(FOCUSED_CONFIGS)
        rng.shuffle(candidates)
        configs = [c for c in candidates if _cfg_key(c) not in completed_keys]
        mode_str = f"focused hand-picked ({len(configs)} new / {len(FOCUSED_CONFIGS)} total)"
    else:
        # Random sampling (deduplicate, skip completed)
        seen = set(completed_keys)
        configs = []
        attempts = 0
        while len(configs) < args.configs and attempts < args.configs * 10:
            attempts += 1
            cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
            key = _cfg_key(cfg)
            if key not in seen:
                seen.add(key)
                configs.append(cfg)
        mode_str = "random"

    print(f"\nRunning {len(configs)} configs [{mode_str}] (temporal_gru, CE loss, "
          f"epochs={SWEEP_EPOCHS}, patience={SWEEP_PATIENCE})")
    print(f"Run IDs will start at {args.run_id_offset + 1}")
    print(f"Baseline to beat: val_auc ~0.6368 (temporal_gru default HPs)\n")

    # Load existing results so we can append to them on save
    existing_results_path = os.path.join(RESULTS_DIR, "sweep_results.json")
    if os.path.exists(existing_results_path) and args.run_id_offset > 0:
        with open(existing_results_path) as f:
            prior_summary = json.load(f)
    else:
        prior_summary = []

    results = []

    for i, cfg in enumerate(configs):
        run_id = args.run_id_offset + i + 1
        try:
            result = run_config(cfg, graphs, labels, device, run_id=run_id)
            results.append(result)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [{run_id:02d}] OOM — skipping "
                  f"hd={cfg['hidden_dim']} nl={cfg['num_layers']} bs={cfg['batch_size']}")

        # Save progress after every run (merge with prior results)
        new_summary = [{
            "run_id":       r["run_id"],
            "config":       r["config"],
            "best_val_auc": r["best_val_auc"],
            "best_epoch":   r["best_epoch"],
            "elapsed_min":  r["elapsed_min"],
            "params":       r["params"],
        } for r in results]
        combined = prior_summary + new_summary
        combined_sorted = sorted(combined, key=lambda x: -x["best_val_auc"])
        with open(os.path.join(RESULTS_DIR, "sweep_results.json"), "w") as f:
            json.dump(combined_sorted, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    results_sorted = sorted(results, key=lambda x: -x["best_val_auc"])
    # Re-read combined for the final rank display
    with open(os.path.join(RESULTS_DIR, "sweep_results.json")) as f:
        all_results_json = json.load(f)
    print(f"\nCombined leaderboard ({len(all_results_json)} runs total):")

    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE — Top 10 configs (combined)")
    print(f"{'='*70}")
    print(f"{'#':<4} {'val_auc':>8}  {'hd':>4} {'nl':>3} {'wd':>7} {'do':>5} {'lr':>7} {'bs':>5}  {'ep':>4}")
    print("-" * 70)
    for rank, r in enumerate(all_results_json[:10], 1):
        c = r["config"]
        print(f"{rank:<4} {r['best_val_auc']:>8.4f}  "
              f"{c['hidden_dim']:>4} {c['num_layers']:>3} {c['weight_decay']:>7.0e} "
              f"{c['dropout']:>5.1f} {c['lr']:>7.0e} {c['batch_size']:>5}  "
              f"{r['best_epoch']:>4}")

    # Save top-5 model weights for full reruns
    print(f"\nSaving top-5 model weights ...")
    for rank, r in enumerate(results_sorted[:5], 1):
        path = os.path.join(RESULTS_DIR, f"sweep_rank{rank:02d}_best.pt")
        torch.save(r["best_state"], path)
        print(f"  Rank {rank}: val_auc={r['best_val_auc']:.4f}  → {path}")

    print(f"\nFull results: {os.path.join(RESULTS_DIR, 'sweep_results.json')}")


if __name__ == "__main__":
    main()

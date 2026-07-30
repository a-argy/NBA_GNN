#!/usr/bin/env python3
"""
Exp 9 — Train best HP config from Exp 6b on each temporal graph variant.

Uses the best rerun config (determined by test_auc from rerun_top.py) or
falls back to rerun_rank01 if results aren't available yet.

Trains each variant with full epochs=200/patience=25, evaluates on test set.

Usage:
    # After rerun_top.py completes:
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/train_variants.py

    # Or force a specific config:
    SWISHNET_BASE_DIR=/data/processed python3 colab/phase2/train_variants.py \
        --hidden-dim 256 --num-layers 3 --dropout 0.1 --lr 5e-4 --batch-size 512
"""

import argparse
import json
import os
import sys
import time

import torch
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
BASE_DIR    = os.environ.get("SWISHNET_BASE_DIR", os.path.expanduser("~/data"))
RESULTS_DIR = os.path.join(_PHASE2_DIR, "results", "variants")

VARIANTS = {
    "dense_short": {
        "name":        "dense_short",
        "description": "A — [3,6,9,12] dense 5-timestep short horizon",
        "graph_dir":   os.path.join(BASE_DIR, "variants", "temporal_dense_short", "graph_data"),
    },
    "sparse_long": {
        "name":        "sparse_long",
        "description": "B — [12,50] sparse 3-timestep long horizon",
        "graph_dir":   os.path.join(BASE_DIR, "variants", "temporal_sparse_long", "graph_data"),
    },
    "dense_long": {
        "name":        "dense_long",
        "description": "C — [6,12,25,50] dense 5-timestep long horizon",
        "graph_dir":   os.path.join(BASE_DIR, "variants", "temporal_dense_long", "graph_data"),
    },
}

# ── Fixed architecture HPs ────────────────────────────────────────────────────
NUM_HEADS       = 4
NUM_PRE_LAYERS  = 2
NUM_POST_LAYERS = 2

EPOCHS    = 200
PATIENCE  = 25
DATA_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_all_graphs(graph_dir):
    print(f"  Loading graphs from {graph_dir} ...")
    pt_files = sorted(f for f in os.listdir(graph_dir) if f.endswith(".pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {graph_dir}")
    all_graphs = []
    for fname in tqdm(pt_files, desc="  Loading", leave=False):
        path = os.path.join(graph_dir, fname)
        try:
            game_graphs = torch.load(path, weights_only=False)
        except Exception:
            continue  # skip corrupted legacy-format files
        if isinstance(game_graphs, list):
            all_graphs.extend(game_graphs)
        else:
            all_graphs.append(game_graphs)
    labels = [int(g.y.item()) for g in all_graphs]
    n = len(all_graphs)
    print(f"  Loaded {n:,} graphs — FG% {100*sum(labels)/n:.1f}%")
    return all_graphs, labels


def make_loaders(graphs, labels, batch_size):
    idx = list(range(len(graphs)))
    tr_idx, tmp_idx = train_test_split(idx, train_size=0.7,
                                       stratify=labels, random_state=DATA_SEED)
    tmp_labels = [labels[i] for i in tmp_idx]
    val_idx, te_idx = train_test_split(tmp_idx, train_size=2/3,
                                       stratify=tmp_labels, random_state=DATA_SEED)
    return (
        DataLoader([graphs[i] for i in tr_idx],  batch_size=batch_size, shuffle=True),
        DataLoader([graphs[i] for i in val_idx], batch_size=batch_size, shuffle=False),
        DataLoader([graphs[i] for i in te_idx],  batch_size=batch_size, shuffle=False),
    )


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
    return roc_auc_score(labels, probs), (preds == labels).mean(), f1_score(labels, preds, zero_division=0)


def infer_num_node_features(graphs):
    """Read node feature dim from first graph."""
    return graphs[0].x.shape[1]


def train_variant(variant, cfg, device):
    name = variant["name"]
    print(f"\n{'='*65}")
    print(f"  Variant {variant['description']}")
    print(f"  hd={cfg['hidden_dim']} nl={cfg['num_layers']} "
          f"wd={cfg['weight_decay']:.0e} do={cfg['dropout']} "
          f"lr={cfg['lr']:.0e} bs={cfg['batch_size']}")
    print(f"{'='*65}")

    graphs, labels = load_all_graphs(variant["graph_dir"])
    num_node_features = infer_num_node_features(graphs)
    print(f"  Node features: {num_node_features}")

    train_loader, val_loader, test_loader = make_loaders(graphs, labels, cfg["batch_size"])

    model = TemporalGRUShotPredictor(
        num_node_features=num_node_features,
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

    best_val_auc, best_epoch, patience_ctr = 0.0, 0, 0
    best_state = None
    history    = []
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
            print(f"  Early stop at ep{epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    test_auc, test_acc, test_f1 = evaluate(model, test_loader, device)
    elapsed = (time.time() - t0) / 60

    print(f"\n  RESULT  val_auc={best_val_auc:.4f}@ep{best_epoch}  "
          f"test_auc={test_auc:.4f}  acc={test_acc:.4f}  f1={test_f1:.4f}  {elapsed:.1f}m")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(RESULTS_DIR, f"{name}_best.pt"))

    result = {
        "variant":      name,
        "description":  variant["description"],
        "config":       cfg,
        "best_val_auc": best_val_auc,
        "best_epoch":   best_epoch,
        "test_auc":     test_auc,
        "test_acc":     test_acc,
        "test_f1":      test_f1,
        "elapsed_min":  elapsed,
        "params":       model.param_count(),
    }
    with open(os.path.join(RESULTS_DIR, f"{name}_result.json"), "w") as f:
        json.dump({**result, "history": history}, f, indent=2)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def get_best_config():
    """Load best config from rerun_top results, fall back to rank01."""
    summary_path = os.path.join(_PHASE2_DIR, "results", "rerun_top", "rerun_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            results = json.load(f)
        best = max(results, key=lambda r: r["test_auc"])
        print(f"Using best rerun config: {best['name']} (test_auc={best['test_auc']:.4f})")
        return best["config"]
    # fallback
    print("rerun_summary.json not found — using rank01 config (hd=256, nl=3)")
    return {"hidden_dim": 256, "num_layers": 3, "weight_decay": 1e-4,
            "dropout": 0.1, "lr": 5e-4, "batch_size": 512}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim",  type=int,   default=None)
    parser.add_argument("--num-layers",  type=int,   default=None)
    parser.add_argument("--dropout",     type=float, default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--batch-size",  type=int,   default=None)
    parser.add_argument("--weight-decay",type=float, default=None)
    args = parser.parse_args()

    cfg = get_best_config()
    # CLI overrides
    if args.hidden_dim:   cfg["hidden_dim"]   = args.hidden_dim
    if args.num_layers:   cfg["num_layers"]   = args.num_layers
    if args.dropout:      cfg["dropout"]      = args.dropout
    if args.lr:           cfg["lr"]           = args.lr
    if args.batch_size:   cfg["batch_size"]   = args.batch_size
    if args.weight_decay: cfg["weight_decay"] = args.weight_decay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []
    for variant in VARIANTS.values():
        result = train_variant(variant, cfg, device)
        results.append(result)

        # Save running summary
        with open(os.path.join(RESULTS_DIR, "variants_summary.json"), "w") as f:
            json.dump(results, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    baseline_test_auc = 0.6346  # temporal_gru default HPs, Exp 3
    print(f"\n{'='*65}")
    print(f"VARIANTS SUMMARY  (baseline temporal_gru = {baseline_test_auc:.4f})")
    print(f"{'='*65}")
    print(f"{'Variant':<16} {'val_auc':>8} {'test_auc':>9} {'acc':>7} {'f1':>7}  {'Δbaseline':>9}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: -x["test_auc"]):
        delta = r["test_auc"] - baseline_test_auc
        print(f"{r['variant']:<16} {r['best_val_auc']:>8.4f} {r['test_auc']:>9.4f} "
              f"{r['test_acc']:>7.4f} {r['test_f1']:>7.4f}  {delta:>+9.4f}")


if __name__ == "__main__":
    main()

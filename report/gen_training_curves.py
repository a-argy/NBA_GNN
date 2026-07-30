"""Regenerate training_curves.pdf with all 3 ranks on both panels."""
import json
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "../colab/phase2/results/ext_pca_gmm/rerun_top"
OUT = "figures/training_curves"

configs = {
    "rank01": {"label": "rank01 (nl=3, bs=512)", "color": "#1f77b4", "best_ep": 34},
    "rank02": {"label": "rank02 (nl=3, bs=256)", "color": "#ff7f0e", "best_ep": 59},
    "rank03": {"label": "rank03 (nl=4, bs=512)", "color": "#2ca02c", "best_ep": 49},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for name, cfg in configs.items():
    with open(f"{DATA_DIR}/rerun_{name}_result.json") as f:
        data = json.load(f)
    hist = data["history"]
    epochs = [h["epoch"] for h in hist]
    val_auc = [h["val_auc"] for h in hist]
    train_loss = [h["train_loss"] for h in hist]

    ax1.plot(epochs, val_auc, label=cfg["label"], color=cfg["color"], linewidth=1.2)
    ax1.axvline(cfg["best_ep"], color=cfg["color"], linestyle="--", alpha=0.4, linewidth=0.8)

    ax2.plot(epochs, train_loss, label=cfg["label"], color=cfg["color"], linewidth=1.2)

ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Validation AUC", fontsize=11)
ax1.set_title("(a) Validation AUC", fontsize=12, fontweight="bold")
ax1.legend(fontsize=8, loc="lower right")
ax1.tick_params(labelsize=9)

ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Training Loss", fontsize=11)
ax2.set_title("(b) Training Loss", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8, loc="upper right")
ax2.tick_params(labelsize=9)

fig.tight_layout()
fig.savefig(f"{OUT}.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"{OUT}.png", bbox_inches="tight", dpi=300)
print(f"Saved {OUT}.pdf and {OUT}.png")

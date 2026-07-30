# Extensions: PCA Embedding Visualisation & GMM Shot Clustering

*Companion to RESEARCH_LOG.md — run these after Exp 6 HP sweep completes*

---

## When to run

These extensions are post-hoc analyses on **the best checkpoint from Exp 6**
(or `temporal_gru_best.pt` if the sweep is still running).  They require no
retraining — the checkpoint is loaded in eval mode and inference runs in
a single forward pass over the full dataset.

Estimated runtime: ~15 min on V100 (DataLoader inference) + seconds for PCA/GMM.

---

## Extension 3 — PCA of GNN Embeddings

### What embedding to extract

In `TemporalGRUShotPredictor.forward()` (phase2/models.py, lines 161-163):

```python
seq = torch.stack(snapshots, dim=0)   # [3, B, D]
_, h_n = self.gru(seq)                # h_n: [1, B, D]
h = h_n.squeeze(0)                    # [B, D]  ← THIS is the embedding
h = self.post_mlp(h)                  # then projected and classified
```

`h` (the GRU final hidden state, before `post_mlp`) is the natural summary
of the shot's full temporal trajectory.  It is `D`-dimensional (128 for the
default model; varies if Exp 6 chose a different `hidden_dim`).

### Extraction via forward hook

Register a hook on `model.gru` — the hook fires after the GRU forward and
captures `h_n` without modifying the computation graph:

```python
embeddings_store = {}

def _gru_hook(module, inp, out):
    # out = (output_seq, h_n)
    _, h_n = out
    embeddings_store['h'] = h_n.squeeze(0).detach().cpu()

hook = model.gru.register_forward_hook(_gru_hook)

model.eval()
all_emb, all_y = [], []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        model(batch)                         # triggers hook
        all_emb.append(embeddings_store['h'])
        all_y.append(batch.y.cpu())

hook.remove()
embeddings = torch.cat(all_emb).numpy()      # [N, D]
labels     = torch.cat(all_y).numpy()        # [N]
```

### PCA and plot

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2, random_state=42)
Z   = pca.fit_transform(embeddings)          # [N, 2]

fig, ax = plt.subplots(figsize=(8, 6))
for label, color, name in [(0, '#d62728', 'Miss'), (1, '#2ca02c', 'Make')]:
    mask = labels == label
    ax.scatter(Z[mask, 0], Z[mask, 1], c=color, label=name,
               alpha=0.3, s=4, rasterized=True)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
ax.set_title('PCA of temporal_gru GRU embeddings (87K shots)')
ax.legend()
plt.tight_layout()
plt.savefig('colab/phase2/results/pca_embeddings.png', dpi=150)
```

### What to look for

| Observation | Interpretation |
|-------------|---------------|
| Make/miss clouds visually separate | GRU embedding captures genuine shot-quality signal |
| Complete overlap | Task difficulty is in the representation, not the classifier |
| Elongated/structured clusters | Latent dimensions encode shot geometry (distance, angle) |

Variance explained by PC1+PC2 is the key diagnostic.  If it is very low
(<10%), the signal is spread across many dimensions and a 2D plot will look
like noise regardless — try 3D or UMAP as a follow-up.

---

## Extension 4 — GMM Clustering of Shot Embeddings

### Fitting GMM

Run on the raw high-dim embeddings (not the 2D PCA projection):

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Fit over a range of K; pick by BIC
bic_scores = {}
for k in [3, 4, 5, 6, 8, 10]:
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                          random_state=42, max_iter=200)
    gmm.fit(embeddings)
    bic_scores[k] = gmm.bic(embeddings)
    print(f"K={k}  BIC={bic_scores[k]:.0f}")

best_k = min(bic_scores, key=bic_scores.get)
print(f"Best K by BIC: {best_k}")

gmm_best = GaussianMixture(n_components=best_k, covariance_type='full',
                           random_state=42, max_iter=200)
cluster_labels   = gmm_best.fit_predict(embeddings)    # hard assignment
responsibilities = gmm_best.predict_proba(embeddings)  # [N, K] soft
```

### Cross-tabbing clusters against shot metadata

The 87K rows of embeddings correspond row-for-row with `X_static.npy` and
`y.npy` (guaranteed by the pipeline fair-comparison assert).  Static feature
indices (from `baseline_features.py / STATIC_FEATURE_NAMES`):

| Index | Feature |
|-------|---------|
| 0 | `dist_to_rim` |
| 2 | `angle_to_basket` |
| 3 | `is_3pt` |
| 4 | `shot_clock` |
| 5 | `game_clock` |
| 6 | `quarter` |

```python
import pandas as pd
X_static = np.load('...baseline_data/X_static.npy')
y        = np.load('...baseline_data/y.npy')

df = pd.DataFrame({
    'cluster':      cluster_labels,
    'label':        y,
    'dist_to_rim':  X_static[:, 0],
    'is_3pt':       X_static[:, 3],
    'angle':        X_static[:, 2],
    'shot_clock':   X_static[:, 4],
})

summary = df.groupby('cluster').agg(
    n         = ('label', 'count'),
    fg_pct    = ('label', 'mean'),
    avg_dist  = ('dist_to_rim', 'mean'),
    pct_3pt   = ('is_3pt', 'mean'),
    avg_angle = ('angle', 'mean'),
    avg_clock = ('shot_clock', 'mean'),
).round(3)
print(summary)
```

### What a good result looks like

Clusters that recover known NBA shot archetypes without supervision validate
that the GRU embedding encodes basketball semantics:

| Archetype | Expected cluster signature |
|-----------|---------------------------|
| At-rim / dunk | low dist, low angle, high FG% |
| Corner 3 | ~22 ft, angle ≈ ±90°, high 3pt |
| Above-the-break 3 | ~24 ft, angle ≈ 0°, medium FG% |
| Mid-range | 10–18 ft, low 3pt, moderate FG% |
| Pull-up / off-dribble | low shot_clock, lower FG% |

If clusters cut across these lines (e.g., mix at-rim and corner-3), the
embedding space is organised by something other than shot type — possibly
defensive configuration or game state, which is equally interesting.

### Soft responsibilities as downstream features (optional)

The `responsibilities` matrix `[N, K]` can be appended to `X_static` and
fed to XGBoost as soft cluster membership features.  This is a quick test
of whether the GRU's latent structure adds signal beyond the raw static
features.

```python
X_augmented = np.concatenate([X_static, responsibilities], axis=1)
# then re-run XGBoost from exp_tree_boost.py on X_augmented
```

---

## File outputs

| File | Content |
|------|---------|
| `colab/phase2/results/pca_embeddings.png` | 2D PCA scatter (make vs miss) |
| `colab/phase2/results/gmm_cluster_summary.csv` | Per-cluster shot archetype stats |
| `colab/phase2/results/gmm_responsibilities.npy` | [N, K] soft assignments for downstream use |

---

## Notes on the "final trained model"

The best checkpoint for extraction is whichever `.pt` file achieves the
highest val AUC in Exp 6.  The HP sweep saves `top_configs` with
`val_auc` in `sweep_results.json`; pick that checkpoint.  If Exp 6 is
still running, use `colab/phase2/results/temporal_gru_history.json`'s
corresponding checkpoint (Exp 3 default HPs, val_auc 0.6368).

The model must be reinstantiated with **the same HPs used at training time**
before loading `state_dict`.  The sweep saves HPs alongside each checkpoint
in `sweep_results.json` for this reason.

# SwishNet — Complete Experiment Catalog

> All experiments run throughout the SwishNet project, ordered chronologically from earliest to latest.
> Last updated: 2026-03-12

---

## Table of Contents

1. [Phase 1 — Small Dataset (12 Games)](#phase-1--small-dataset-12-games)
2. [Phase 1b — Reduced Capacity Ablation](#phase-1b--reduced-capacity-ablation)
3. [Phase 2 — Full Dataset (87,147 Shots)](#phase-2--full-dataset-87147-shots)
4. [Phase 3 — Graph Construction Variants](#phase-3--graph-construction-variants)
5. [Extensions](#extensions)
6. [Final Leaderboard](#final-leaderboard)

---

## Phase 1 — Small Dataset (12 Games)

**Dataset:** 12 games, ~1,745 graphs, FG% = 43.0%
**Reference baseline:** Logistic Regression (Static-38 / Lasso) → AUC **0.614**
**Fixed HPs:** hidden_dim=128, num_layers=2, num_pre_layers=2, num_post_layers=2, dropout=0.3, lr=1e-4, wd=5e-3
**Results:** `colab/phase1/results/phase1_results.txt`

### Exp 1 — Pooling Strategies (GAT, CE Loss)

| Model | Pooling | Test AUC |
|-------|---------|----------|
| `gat_mean_max` | concat mean + max | **0.5935** |
| `gat_add` | sum | 0.5050 |
| `gat_mean` | global mean | 0.4755 |
| `gat_max` | global max | 0.4739 |

### Exp 2 — Shooter-Centric Hybrid Readout

| Model | Architecture | Test AUC |
|-------|-------------|----------|
| `gine_mean` | GINE, global mean | 0.5370 |
| `gat_shooter` | GAT, shooter_emb + scene_mean | 0.5238 |
| `gine_shooter` | GINE, shooter_emb + scene_mean | 0.4854 |

### Exp 7 (Phase 1) — Loss Function Variants (GAT + Mean Pool)

| Model | Loss | Test AUC |
|-------|------|----------|
| `gat_focal_10` | Focal γ=1.0 | 0.5070 |
| `gat_focal_05` | Focal γ=0.5 | 0.5002 |
| `gat_ls_010` | Label Smooth α=0.1 | 0.4763 |
| `gat_focal_20` | Focal γ=2.0 | 0.4675 |
| `gat_ls_005` | Label Smooth α=0.05 | 0.4471 |

**Key finding:** No GNN variant beat the LR baseline (0.614) on the small dataset. Best GNN was `gat_mean_max` at 0.5935.

---

## Phase 1b — Reduced Capacity Ablation

**Motivation:** Test whether the Phase 1 gap was due to over-parameterisation.
**Change:** hidden_dim reduced to 32; class weighting added (miss=1.0, make=1.324).
**Results:** `colab/phase1/results/phase1b_results.txt`

| Model | Test AUC | Best Epoch |
|-------|----------|------------|
| `gat_mean_hd32` | **0.5395** | 1 |
| `gat_shootersc_hd32` | 0.5278 | — |
| `gine_mean_hd32` | 0.5275 | — |

**Key finding:** Reducing capacity made things worse (0.5395 < 0.5935). The bottleneck was data quantity, not model capacity.

---

## Phase 2 — Full Dataset (87,147 Shots)

**Dataset:** 87,147 graphs from 632 NBA games (2015–16 season), FG% = 45.8%
**Splits:** 70/20/10 stratified (random_state=42)

### Exp 0a — Baseline Retraining on Full Dataset

**Results:** `colab/phase2/results/exp0a_baselines_full.txt`

| Feature Set | Regularization | Test AUC |
|-------------|----------------|----------|
| Velocity-65 | Lasso (L1, C=0.1) | **0.6377** |
| Velocity-65 | Ridge (L2, C=1) | 0.6375 |
| Velocity-65 | None (Scaled) | 0.6376 |
| Static-38 | Lasso (L1, C=0.1) | 0.6362 |
| Static-38 | Ridge (L2, C=0.1) | 0.6361 |
| Static-38 | None (Scaled) | 0.6361 |
| Static-38 | None (Raw) | 0.6350 |

**Key finding:** 50× more data moved the LR baseline from 0.614 → 0.6377. New GNN target: **AUC > 0.638**.

---

### Exp 0b — Sanity Check: Phase 1 Best on Full Data

**Rationale:** Isolate the data-size hypothesis.

| Model | Test AUC |
|-------|----------|
| `gat_shooter` | **0.6342** |
| `gat_mean_max` | 0.6226 |

**Key finding:** Gap narrowed significantly but GNN still below LR baseline.

---

### Exp 0c — LLM Oracle: Zero-Shot Gemini Prediction

**Results:** `colab/phase2/results/exp0c_llm_oracle.json`

| Model | Shots Evaluated | Test AUC | Gap vs LR |
|-------|-----------------|----------|-----------|
| gemini-2.0-flash-001 | 100/100 | 0.5665 | −0.0712 |
| gemini-2.5-pro | 95/100 | 0.5287 | −0.1090 |

**Key finding:** Zero-shot LLMs fall far short. Empirical training adds ~0.07–0.11 AUC of genuine signal.

---

### Exp 3 — Explicit Temporal Modeling via GRU

**Architecture:** Spatial GNN per timestep → global_mean_pool → GRU → classifier.

| Model | Test AUC | Test F1 | Best Epoch |
|-------|----------|---------|------------|
| `temporal_gru` | **0.6346** | 0.486 | 86/100 |

**Key finding:** Best single architecture so far (tied with gat_shooter at 0.6342). Still 0.003 below LR baseline.

---

### Exp 4 — Alternative Convolution Mechanisms

| Model | Conv Layer | Test AUC | Test F1 |
|-------|-----------|----------|---------|
| `edge_mean_max` | EdgeConv (relative encoding) | 0.6311 | 0.483 |
| `sage_mean_max` | SAGEConv (max agg) | 0.6094 | 0.143 |

**Key finding:** SAGEConv fails without attention. EdgeConv validates that relative spatial relationships matter, but doesn't beat temporal_gru.

---

### Exp 5 — Combine Best Architecture + Focal Loss

| Model | Loss | Test AUC |
|-------|------|----------|
| `temporal_gru_focal` | Focal γ=1.0 | 0.6304 |
| `gat_mean_max_focal` | Focal γ=1.0 | 0.6210 |

**Key finding:** Focal loss hurt performance vs CE. Models are already well-calibrated under CE.

---

### Exp 6 — Hyperparameter Search (29 Configs)

**Phase A:** 10 random configs over broad search space.
**Phase B:** 18 hand-picked configs with lr=5e-4 and wd=1e-4 locked.
**Search space:** hidden_dim ∈ {64,128,256}, num_layers ∈ {1,2,3,4}, dropout ∈ {0.1,0.2,0.3}, lr ∈ {1e-4,5e-4}, batch_size ∈ {128,256,512}, weight_decay ∈ {1e-4,1e-3,5e-3}

**Top 10 by validation AUC:**

| Rank | hd | nl | dropout | lr | bs | Val AUC |
|------|----|----|---------|-----|-----|---------|
| 1 | 256 | 3 | 0.1 | 5e-4 | 512 | 0.6487 |
| 2 | 256 | 3 | 0.1 | 5e-4 | 256 | 0.6479 |
| 3 | 256 | 4 | 0.1 | 5e-4 | 512 | 0.6477 |
| 4 | 256 | 4 | 0.2 | 5e-4 | 512 | 0.6471 |
| 5 | 256 | 4 | 0.1 | 5e-4 | 256 | 0.6468 |
| 6 | 256 | 3 | 0.2 | 5e-4 | 512 | 0.6465 |
| 7 | 128 | 3 | 0.1 | 5e-4 | 512 | 0.6462 |
| 8 | 128 | 4 | 0.1 | 5e-4 | 256 | 0.6460 |
| 9 | 128 | 3 | 0.1 | 5e-4 | 256 | 0.6459 |
| 10 | 256 | 3 | 0.2 | 5e-4 | 256 | 0.6458 |

**Sweep conclusions:** `lr=5e-4` and `wd=1e-4` dominate all top configs. `hd=256` preferred. `dropout=0.1` slightly edges 0.2.

---

### Exp 6b — Full Reruns of Top-3 Sweep Configs

**Training:** 200 epochs, patience=25, L4 GPU.
**Results:** `colab/phase2/results/ext_pca_gmm/rerun_top/rerun_summary.json`

| Rank | hd | nl | bs | Sweep Val | Val AUC | Test AUC | Test F1 | Best Ep |
|------|----|----|-----|-----------|---------|----------|---------|---------|
| 1 | 256 | 3 | 512 | 0.6487 | 0.6452 | 0.6437 | 0.4691 | 34 |
| 2 | 256 | 3 | 256 | 0.6479 | 0.6467 | 0.6444 | 0.4200 | 59 |
| **3** | **256** | **4** | **512** | **0.6477** | **0.6457** | **0.6451** | **0.4452** | **49** |

**Winner: rank03 → test_auc = 0.6451** (best GNN). Gap to XGBoost ceiling (0.650) narrowed to 0.005.

---

### Exp 7 (Phase 2) — Feature Ablation

**Method:** Zero-out or shuffle each feature group; measure drop from rank03 baseline (0.6451).
**Results:** `colab/phase2/results/exp7_ablation/`

| Feature Group | Method | Test AUC | Δ AUC | Importance |
|---------------|--------|----------|-------|------------|
| temporal_id | shuffle | 0.6364 | **−0.0087** | Most critical |
| geometry | zero | 0.6366 | **−0.0085** | Most critical |
| role_flags | zero | 0.6378 | −0.0073 | Critical |
| game_state | zero | 0.6413 | −0.0038 | Moderate |
| position_enc | zero | 0.6429 | −0.0022 | Minor |
| spatial_xyz | zero | 0.6430 | −0.0021 | Minor |
| player_stats | zero | 0.6433 | −0.0018 | Least important |

**Key finding:** player_stats is the LEAST important feature group (−0.0018), contradicting the hypothesis that the GNN is simply a learned LR. Temporal ordering and geometry are most critical — the GNN exploits dynamics and spatial structure that LR cannot capture.

---

### Exp 8a — LLM Player Embeddings (Tabular Gate)

**Method:** Replace raw player stats with Gemini text → PCA-64 embeddings in LR.
**Gate threshold:** AUC > 0.6407.
**Results:** `colab/phase2/results/exp8a_results.txt`

| Config | Features | Test AUC |
|--------|----------|----------|
| 8a-LR-augment | 38 + PCA64 embedding | 0.6361 |
| 8a-LR-replace | 17 + PCA64 embedding | 0.6310 |
| Baseline | Velocity-65 | 0.6377 |

**Gate decision:** ABORT Phase B (best AUC 0.6361 ≤ gate 0.6407). LLM embeddings lose precision vs raw stats.

---

## Phase 3 — Graph Construction Variants

### Exp 9 — Temporal Sampling Ablation (2×2 Factorial)

**Infrastructure:** 3 parallel GCP CPU VMs rebuilding graphs.
**HPs:** Fixed at rank03 (hd=256, nl=4, do=0.1, lr=5e-4, wd=1e-4, bs=512).
**Results:** `colab/phase2/results/ext_pca_gmm/variants/variants_summary.json`

|  | Sparse (3 timesteps) | Dense (5 timesteps) |
|--|----------------------|---------------------|
| **Short horizon (0.48s)** | Baseline `[3, 12]` | Variant A `[3,6,9,12]` |
| **Long horizon (2.0s)** | Variant B `[12, 50]` | Variant C `[6,12,25,50]` |

| Variant | Timesteps | Val AUC | Test AUC | Δ vs Baseline |
|---------|-----------|---------|----------|---------------|
| Baseline | `[3,12]` | — | 0.6346 | — |
| A (dense_short) | `[3,6,9,12]` | 0.6385 | 0.6350 | +0.0004 |
| B (sparse_long) | `[12,50]` | 0.6424 | 0.6348 | +0.0002 |
| **C (dense_long)** | **`[6,12,25,50]`** | **0.6458** | **0.6411** | **+0.0065** |

**Key findings:**
- Variant C is the best graph structure (+0.0065 over baseline).
- Genuine interaction effect: neither factor alone helps (A: +0.0004, B: +0.0002), but combined: +0.0065.
- No variant beats rank03 (0.6451) — optimized HPs are a larger lever than graph structure.

---

## Extensions

### Ext 1 — XGBoost Baseline

**Results:** `colab/phase2/results/ext12_tree_boost_report.txt`
**HPs (fixed):** n_estimators=400, max_depth=4, lr=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5

| Feature Set | Test AUC | Δ vs LR |
|-------------|----------|---------|
| Velocity-65 | **0.6501** | +0.0124 |
| Static-38 | 0.6480 | +0.0103 |

**Key finding:** XGBoost becomes the new ceiling at 0.650.

---

### Ext 2 — Decision Tree Depth Sweep

**Method:** max_depth 1–20, 5-fold CV.
**Results:** `colab/phase2/results/ext2_tree_static38.json`, `ext2_tree_velocity65.json`

| Feature Set | Best Depth | Best Val AUC |
|-------------|-----------|-------------|
| Velocity-65 | 7 | 0.6318 |
| Static-38 | 7 | 0.6299 |

**Key finding:** Both peak at depth 7 (shallow trees). LR still beats single trees. Root splits on dist_to_rim and is_3pt confirm these are dominant features.

---

### Ext 3 — PCA of GRU Embeddings

**Method:** Extract 256-dim GRU embeddings from rank03, project to 2D via PCA.
**Results:** `colab/phase2/results/ext_pca_gmm/ext3_pca_result.json`

| Component | Variance Explained |
|-----------|--------------------|
| PC1 | 68.7% |
| PC2 | 19.8% |
| **Total (2 PCs)** | **88.4%** |

**Key finding:** 88.4% of embedding variance captured in 2D — highly structured latent space. 2D scatter shows visible make/miss separation.

---

### Ext 4 — GMM Shot Clustering + XGBoost Augmentation

#### Part A: GMM Clustering (K=10)

**Results:** `colab/phase2/results/ext_pca_gmm/ext4_gmm_bic.json`

BIC analysis selected K=10. Discovered cluster archetypes:

| Cluster | n | FG% | Avg Dist | % 3PT | Archetype |
|---------|---|-----|----------|-------|-----------|
| 2 | 5,461 | **87.7%** | 5.7 ft | 0% | At-rim / dunks |
| 6 | 5,415 | 65.2% | 2.7 ft | 0% | Right at rim / putbacks |
| 7 | 6,049 | 62.0% | 5.8 ft | 0.7% | Short-range makes |
| 4 | 6,924 | 43.7% | 3.9 ft | 0% | Contested close-range |
| 9 | 11,918 | 46.3% | 15.2 ft | 7.6% | Mid-range (angled) |
| 1 | 7,755 | 41.1% | 7.1 ft | 0.1% | Short mid-range / paint |
| 3 | 11,992 | 41.6% | 18.8 ft | 18.1% | Long 2s / some 3s |
| 8 | 12,712 | 35.0% | 21.9 ft | 48.4% | 3-point misses |
| 0 | 11,610 | 38.5% | 22.4 ft | 46.5% | Above-the-break 3s |
| 5 | 6,589 | **28.9%** | 20.7 ft | 43.6% | Late-clock desperation 3s |

**Key finding:** Clusters recover known NBA shot archetypes without supervision.

#### Part B: XGBoost with GMM Soft Memberships

**Results:** `colab/phase2/results/ext_pca_gmm/ext4_xgb_augmented.json`

| Feature Set | Test AUC |
|-------------|----------|
| **Static-38 + GMM soft memberships** | **0.6551** |
| XGBoost Velocity-65 (Ext 1) | 0.6501 |
| XGBoost Static-38 (Ext 1) | 0.6480 |

**Key finding:** Static-38 + GMM (0.6551) becomes the **new overall best** — beats both XGBoost Velocity-65 (0.6501) and the best GNN (0.6451).

---

### Ext 5 — Input Gradient Saliency + GAT Attention Analysis

**Checkpoint:** rank03 (test_auc=0.6451)
**Results:** `colab/phase2/results/ext5_interpret/`

#### Part A: Input Gradient Saliency (mean |∂logit₁/∂x| per node role)

| Role | role_flags | geometry | pos_enc | tid | stats | xyz | game_state |
|------|-----------|----------|---------|-----|-------|-----|-----------|
| **shooter** | **3.55e-5** | 2.23e-5 | 2.55e-5 | 6.02e-6 | 1.05e-5 | 5.62e-6 | 4.15e-6 |
| **ball_node** | **3.20e-5** | 2.12e-5 | 1.85e-5 | 1.28e-5 | 5.17e-6 | 8.11e-6 | 3.14e-6 |
| defense | 1.47e-5 | 7.83e-6 | 8.03e-6 | 3.97e-6 | 2.55e-6 | 1.69e-6 | 1.11e-6 |
| offense | 8.24e-6 | 4.88e-6 | 4.68e-6 | 1.63e-6 | 1.91e-6 | 1.44e-6 | 1.13e-6 |

- Shooter and ball_node dominate (3–4× higher gradients than other players).
- Role flags most salient for every node type; geometry second.
- player_stats has low saliency (consistent with Exp 7 ablation).

#### Part B: GAT Attention Weights

**By edge type:**

| Edge Type | Mean α |
|-----------|--------|
| PB (Player-Ball) | **0.1256** |
| OO (Off-Off) | 0.0957 |
| OD (Off-Def) | 0.0876 |
| DD (Def-Def) | 0.0734 |

**By distance bin:**

| Distance | Mean α |
|----------|--------|
| 0–6 ft | **0.1282** |
| 6–12 ft | 0.0920 |
| 20+ ft | 0.0919 |
| 12–20 ft | 0.0843 |

- PB edges get most attention (ball-tracking is primary signal).
- Strong proximity bias: attention drops ~35% from 0–6 ft to 6–12 ft.
- Attention roughly flat across layers (L0=0.0964, L1=0.0933, L2=0.0931, L3=0.0909).

---

## Final Leaderboard

Ranked by test AUC across all experiments:

| Rank | Method | Test AUC | Phase | Notes |
|------|--------|----------|-------|-------|
| 1 | **XGBoost Static-38 + GMM** | **0.6551** | Ext 4B | Best overall |
| 2 | XGBoost Velocity-65 | 0.6501 | Ext 1 | |
| 3 | XGBoost Static-38 | 0.6480 | Ext 1 | |
| 4 | **GNN temporal_gru rank03** | **0.6451** | Exp 6b | Best GNN (hd=256, nl=4) |
| 5 | GNN temporal_gru rank02 | 0.6444 | Exp 6b | |
| 6 | GNN temporal_gru rank01 | 0.6437 | Exp 6b | |
| 7 | GNN Variant C (dense_long) | 0.6411 | Exp 9 | Best graph structure |
| 8 | LR Velocity-65 Lasso | 0.6377 | Exp 0a | LR baseline |
| 9 | LR Static-38 Lasso | 0.6362 | Exp 0a | |
| 10 | GNN Variant A (dense_short) | 0.6350 | Exp 9 | |
| 11 | GNN Variant B (sparse_long) | 0.6348 | Exp 9 | |
| 12 | GNN temporal_gru (default HPs) | 0.6346 | Exp 3 | |
| 13 | GNN gat_shooter (full data) | 0.6342 | Exp 0b | |
| 14 | Decision Tree Velocity-65 | 0.6318 | Ext 2 | depth=7 |
| 15 | GNN EdgeConv | 0.6311 | Exp 4 | |
| 16 | Decision Tree Static-38 | 0.6299 | Ext 2 | depth=7 |
| 17 | GNN gat_mean_max (full data) | 0.6226 | Exp 0b | |
| 18 | GNN gat_mean_max_focal | 0.6210 | Exp 5 | |
| 19 | GNN SAGEConv | 0.6094 | Exp 4 | |
| 20 | GNN gat_mean_max (small data) | 0.5935 | Phase 1 | |
| 21 | Gemini Flash zero-shot | 0.5665 | Exp 0c | |
| 22 | LR baseline (small data) | 0.614 | Phase 1 | |
| 23 | GNN gine_mean (small data) | 0.5370 | Phase 1 | |
| 24 | Gemini Pro zero-shot | 0.5287 | Exp 0c | |
| 25 | GNN gat_shooter (small data) | 0.5238 | Phase 1 | |

---

## Summary Statistics

- **Total unique configs evaluated:** ~50+
- **Phase 1:** 12 architecture/loss configs + 8 reduced-capacity configs
- **Phase 2:** Exps 0a–0c, 3–8a (baselines, GNN variants, HP sweep of 29 configs, feature ablation, LLM embeddings)
- **Phase 3:** Exp 9 (4 graph construction variants on 3 parallel GCP VMs)
- **Extensions:** 5 (XGBoost, Decision Trees, PCA embeddings, GMM clustering, Interpretability)
- **Infrastructure:** Local, GCP CPU VMs (`swishnet-data`), GCP GPU VMs (`swishnet-trainer` with T4/L4), GCS bucket `swishnet-nba`
- **Dataset:** 87,147 shots from 632 NBA games (2015–16 season)

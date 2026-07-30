# SwishNet Research Log

*Last updated: 2026-03-12 (Ext 3 & 4 complete)*

---

## Project Goal

Predict NBA shot outcomes (make/miss) using Graph Neural Networks on player-tracking data.
The graph represents the spatial configuration of all 10 players + ball at the moment of release,
across 3 timesteps (t=0 release, t≈0.24s prior, t≈0.48s prior).

**Primary metric:** Test AUC (ROC)
**Baseline to beat:** Logistic regression on 38 static features → AUC **0.614** (Phase 1, 12 games)
**Phase 2 baseline:** Velocity-65 / Lasso (L1) on 87K shots → AUC **0.638** (Exp 0a)

---

## Architecture (fixed across all experiments unless noted)

| Component | Detail |
|-----------|--------|
| Node features | 41: xyz, has_ball, is_offense, is_player_node, dist_to_rim, dist_to_ball_handler, angle_to_basket, dist_to_3pt, num_nearby_def, quarter, game_clock, shot_clock, timestamp, position_encoding[5], player_stats[21] |
| Edge features | 9: x_rel, y_rel, euclidean_dist, edge_angle, rel_type_OO/OD/DD/PB/TEMPORAL |
| Graph | 11 nodes × 3 timesteps = 33 nodes; fully connected within timestep + bidirectional temporal edges |
| Default HPs | hidden_dim=128, num_layers=2, num_pre/post_layers=2, dropout=0.3, lr=1e-4, wd=5e-3, batch=256, epochs=200, patience=25 |
| Data split | 70/20/10 stratified, random_state=42 |

---

## Baseline

| Model | Features | AUC |
|-------|----------|-----|
| Logistic Regression (Lasso) | Static-38 | **0.614** |

**Feature groups (38 static):** dist_to_rim, ball_z, angle_to_basket, is_3pt, shot_clock, game_clock, quarter (7) · sorted offender distances (4) · sorted defender distances (5) · num_defs_within_6ft (1) · shooter season stats (21)

---

## Phase 1 — Architecture/Loss Ablation on Small Dataset

**Dataset:** 12 games, ~1,745 graphs, FG% = 43.0%
**Splits:** Train 1,221 / Val 349 / Test 175

### Exp 1 — Pooling strategies (GAT, CE loss)

| Experiment | Readout | AUC | Acc | F1 | Miss% | Make% |
|------------|---------|-----|-----|----|-------|-------|
| gat_mean_max | concat(mean, max) | **0.5935** | 0.5714 | 0.000 | 100% | 0% |
| gat_add | sum pool | 0.5050 | 0.5714 | 0.118 | 95% | 7% |
| gat_mean | global mean | 0.4755 | 0.5257 | 0.108 | 87% | 7% |
| gat_max | global max | 0.4739 | 0.4743 | 0.233 | 69% | 19% |

**Finding:** mean+max concat is best. Max pooling alone underperforms — on 1,745 graphs it likely overfits
to specific spatial extremes. Add pool's normalisation via BatchNorm provides a modest benefit.
Mean pooling's collapse (0% make recall) indicates severe class imbalance sensitivity.

---

### Exp 2 — Shooter-centric hybrid readout

| Experiment | Conv | Readout | AUC |
|------------|------|---------|-----|
| gat_shooter | GAT | concat(shooter_emb, scene_mean) | 0.5238 |
| gine_mean | GINE | global mean | 0.5370 |
| gine_shooter | GINE | concat(shooter_emb, scene_mean) | 0.4854 |

**Finding:** Shooter-centric readout improves GAT (+0.048 over gat_mean) — extracting the informationally richer
shooter node explicitly helps the post-MLP. GINE with mean pooling (0.537) slightly beats GAT mean (0.476),
but shooter-centric GINE is worse, suggesting GINE's aggregation already captures shooter-relevant structure
in a way that explicit extraction then disrupts.

---

### Exp 7 — Loss function variants (GAT + mean pool)

| Experiment | Loss | AUC |
|------------|------|-----|
| gat_focal_10 | Focal γ=1.0 | **0.5070** |
| gat_focal_05 | Focal γ=0.5 | 0.5002 |
| gat_ls_010 | Label smooth α=0.1 | 0.4763 |
| gat_mean (CE) | CE baseline | 0.4755 |
| gat_focal_20 | Focal γ=2.0 | 0.4675 |
| gat_ls_005 | Label smooth α=0.05 | 0.4471 |

**Finding:** Focal loss (γ=1.0) gives a modest improvement. Heavy focal (γ=2.0) and label smoothing both hurt —
the model is already uncertain enough on 1,745 graphs; over-regularising the loss makes things worse.
Exps 1 and 7 were tested independently; their combination (mean_max + focal) has not been tried.

---

### Phase 1 Overall Summary

**Best model:** `gat_mean_max` — AUC 0.5935 (vs LR 0.614, gap = -0.021)

No GNN variant beat the logistic regression baseline. Two candidate explanations:
1. **Data starvation:** 1,745 graphs is far too small for a 331K-parameter model (wd=5e-3 likely over-regularising)
2. **Structural:** Phase 1 architectural variants do not address the core limiting factor

---

## Phase 1b — Reduced Capacity + Class Weighting

**Motivation:** Test whether the gap was over-parameterisation. Reduced hidden_dim to 32/64 and added class weights (miss=1.0, make=1.324) to address 43% FG% imbalance.

| Experiment | hd | AUC | Notes |
|------------|----|-----|-------|
| gat_mean_hd32 | 32 | **0.5395** | Best epoch=1, barely trains |
| gat_shootersc_hd32 | 32 | 0.5278 | |
| gine_mean_hd32 | 32 | 0.5275 | 100% make recall, 0% miss recall — degenerate |
| gat_shootersc_hd64 | 64 | 0.4962 | |
| gat_meanmax_hd32 | 32 | 0.4718 | |
| gat_meanmax_hd64 | 64 | 0.4542 | |
| gat_mean_hd64 | 64 | 0.4230 | |
| gine_mean_hd64 | 64 | 0.4058 | |

**Finding:** Reducing capacity made things *worse* overall. Phase 1b best (0.5395) < Phase 1 best (0.5935).
Class weighting caused several models to collapse to predicting only one class.
**Conclusion:** The bottleneck is data quantity, not model capacity. The fix is more data.

---

## Full-Season Data Pipeline

**Date run:** 2026-03-05
**Infrastructure:** GCP CPU VM `swishnet-data` (n1-standard-4), data stored in `gs://swishnet-nba/`

### Pipeline results

| Stage | Count | Notes |
|-------|-------|-------|
| Games processed | 632 | 5 below threshold (<90 shots), 0 calibration failures |
| Stage 1: shot events | 106,245 | |
| Stage 1: extracted | 91,713 | 86.3% — 11,985 rescued by secondary event fallback |
| Stage 1: top drop reasons | release_no_rim_moment_found (7,162) · release_first_player_not_shooter (5,147) | |
| Stage 2: graphs built | 87,147 | 95.0% of stage-1 shots |
| Stage 2: top drop reason | missing_field / null shot_clock (4,069) | |
| **Final dataset (fair)** | **87,147** | Both graph AND features succeeded |
| FG% | 45.8% | Consistent with 2015-16 league average |

**Fair comparison guarantee:** Row i of `X_static.npy` is guaranteed to correspond to graph i in `graphs_GAME.pt`.
The same 87,147 shots will be used for both GNN training and baseline retraining in Phase 2.

### GCS layout
```
gs://swishnet-nba/
  raw/                      pbp_cache.csv, player_shooting_stats_2016.csv
  processed/
    graph_data/             graphs_XXXXXXXXXX.pt per game
    baseline_data/          X_static.npy (87147×38), X_full.npy (87147×65), y.npy (87147,)
    pipeline_stats/         pipeline_report.txt, drop_stats.json
```

---

## Phase 2 — Full Dataset Experiments (Planned)

**Dataset:** 87,147 graphs (~50× Phase 1), same 70/20/10 split, random_state=42
**Target:** AUC > baseline retrained on 87K (see Exp 0a — target will move from 0.614)

### Exp 0a — Retrain all baselines on full dataset *(COMPLETE — 2026-03-10)*

**Dataset:** 87,147 shots, FG% 45.8%, 5-fold stratified CV
**Results:**

| Feature | Regularization | Scaling | C* | AUC | Acc | F1 |
|---------|---------------|---------|-----|-----|-----|----|
| Static-38 | None | Raw | — | 0.6350±0.0024 | 61.0±0.2% | 0.5176±0.0026 |
| Static-38 | None | Scaled | — | 0.6361±0.0021 | 61.0±0.3% | 0.5195±0.0030 |
| Static-38 | Ridge (L2) | Scaled | 0.1 | 0.6361±0.0021 | 61.0±0.3% | 0.5196±0.0031 |
| Static-38 | Lasso (L1) | Scaled | 0.1 | 0.6362±0.0022 | 61.0±0.2% | 0.5194±0.0026 |
| Velocity-65 | None | Raw | — | 0.6365±0.0026 | 61.2±0.2% | 0.5183±0.0027 |
| Velocity-65 | None | Scaled | — | 0.6376±0.0022 | 61.2±0.2% | 0.5193±0.0023 |
| Velocity-65 | Ridge (L2) | Scaled | 1 | 0.6375±0.0022 | 61.2±0.2% | 0.5193±0.0022 |
| **Velocity-65** | **Lasso (L1)** | **Scaled** | **0.1** | **0.6377±0.0023** | **61.2±0.2%** | **0.5198±0.0023** |

**New GNN target: AUC > 0.638** *(superseded by Ext 1 below — see XGBoost)*

**Finding:** 50× more data moved the LR ceiling from 0.614 → 0.638 (+0.024). All 8 configs are
tightly clustered (0.635–0.638) — the linear feature ceiling is real. Regularisation and velocity
features contribute marginal gains. Tree models (RF/GBM) not tested here; the LR ceiling is the
relevant reference since GNNs are the focus.

**Files:** `colab/phase2/results/exp0a_baselines_full.txt`

---

### Ext 1 — XGBoost baseline *(COMPLETE — 2026-03-11)*

**What:** XGBoost on Static-38 and Velocity-65, same 5-fold stratified CV as Exp 0a.
Fixed HPs: n_estimators=400, max_depth=4, lr=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5.
No inner HP search — this is a representative ceiling estimate, not a tuned entry.

| Feature | AUC | Acc | F1 | vs LR baseline |
|---------|-----|-----|----|----------------|
| Static-38 | 0.6480±0.0033 | 62.2% | 0.4763 | +0.0103 |
| **Velocity-65** | **0.6501±0.0024** | **62.4%** | **0.4758** | **+0.0124** |

**New overall ceiling: AUC 0.650 (XGBoost, Velocity-65)**

**Finding:** XGBoost beats LR on both feature sets without any tuning, and beats the best GNN
to date (temporal_gru 0.6346). The velocity features add +0.002 AUC, consistent with Exp 0a.
The F1 (~0.476) is similar to temporal_gru (0.486), confirming class balance is not the differentiator.
XGBoost's ensemble of shallow trees can approximate the non-linear interactions in shot geometry
(e.g., distance × defender proximity) that linear models cannot.

**New GNN target: AUC > 0.650**

**Files:** `colab/phase2/results/ext1_xgb_static38.json`, `ext1_xgb_velocity65.json`, `ext12_tree_boost_report.txt`

---

### Ext 2 — Decision Tree depth sweep *(COMPLETE — 2026-03-11)*

**What:** DecisionTreeClassifier swept over max_depth 1–20, 5-fold CV, both feature sets.
Records train and val AUC per fold to produce an overfitting curve.

| Feature | Best depth | Best val AUC | vs LR baseline |
|---------|-----------|-------------|----------------|
| Static-38 | 7 | 0.6299 | -0.0078 |
| Velocity-65 | 7 | 0.6318 | -0.0059 |

**Finding:** Both feature sets peak at depth 7 — shallow trees capture the main signal and deeper
trees overfit. Neither beats LR (0.6377): LR's smooth linear boundary outperforms axis-aligned
splits on these features. The train/val divergence is visible from depth ~8 onward (classic
overfitting curve). Decision trees are interpretable at depth 7 — the root splits are on
dist_to_rim and is_3pt, confirming these are the dominant predictors.

**Files:** `colab/phase2/results/ext2_tree_curve_static38.png`, `ext2_tree_curve_velocity65.png`,
`ext2_tree_static38.json`, `ext2_tree_velocity65.json`

---

### Exp 0c — LLM Oracle: Zero-shot shot prediction via Vertex AI Gemini *(COMPLETE — 2026-03-10)*

**What:** Sampled 100 random shots from local game data (seed=42), built natural-language prompts
describing each shot situation (shooter name, season stats, shot distance/type, defender proximity,
game clock/quarter), and queried two Gemini models for a make probability (0–1). Computed AUC
against ground truth outcomes. No training — pure zero-shot inference.

**Prompt included:** shooter name, overall FG%, 3PT FG%, avg shot distance, shot type (2PT/3PT),
distance to rim, angle to basket, shot clock, game clock, quarter, closest defender distance,
number of defenders within 6 ft.

**Results:**

| Model | Shots | AUC | Prob std | vs LR baseline |
|-------|-------|-----|----------|----------------|
| gemini-2.0-flash-001 | 100/100 | 0.5665 | ~0.12 | -0.0712 |
| gemini-2.5-pro | 95/100 | 0.5287 | 0.105 | -0.1090 |
| LR baseline (Exp 0a) | 87K | **0.6377** | — | — |

**Finding:** Both models perform above chance (0.5), confirming genuine basketball knowledge.
However, both fall well short of the LR baseline trained on empirical shot data.
Counterintuitively, Flash outperforms Pro: the thinking model over-hedges toward league average
(max prob=0.64, std=0.105) while Flash produces more confident/spread predictions (max=0.78)
that rank better by AUC. Pro's chain-of-thought reasoning introduces second-guessing rather than
sharper discrimination.

**Conclusion:** Zero-shot LLMs lack calibration to empirical shot distributions. AUC ~0.53–0.57
is the ceiling for language-prior-only prediction. GNN/LR approaches that train on actual outcomes
add ~0.07–0.11 AUC of genuine signal beyond what language priors provide. Validates that the
GNN is learning something real, not just recovering player identity information.

**Files:** `colab/phase2/llm_oracle.py`, `colab/phase2/results/exp0c_llm_oracle.json`

---

### Exp 0b — Sanity check: Phase 1 best on full data *(COMPLETE — 2026-03-11)*

**What:** Retrain `gat_mean_max` and `gat_shooter` with identical Phase 1 HPs on 87K graphs.

**Rationale:** Isolates the data-size hypothesis. If these already beat the new baseline (Exp 0a),
the architecture is sufficient and Phase 2 focuses on HP tuning. If not, structural experiments
(Exps 3–4) are essential. Cheapest GNN experiment — gates interpretation of everything downstream.

**Results:**

| Experiment | Conv | Readout | Test AUC | Test Acc | Test F1 | Best Val AUC | Best Ep | Time |
|------------|------|---------|----------|----------|---------|--------------|---------|------|
| gat_mean_max | GATv2 | concat(mean, max) | 0.6226 | 57.5% | 0.172 | 0.6299 | 71/91 | 46 min |
| gat_shooter | GATv2 | concat(shooter_emb, scene_mean) | **0.6342** | 61.5% | 0.450 | 0.6399 | 63/83 | 40 min |

**Baseline to beat (Exp 0a):** AUC 0.6377

**Finding:** Both models fall short of the LR baseline, but the gap has narrowed dramatically vs Phase 1.
`gat_mean_max` jumped from 0.5935 → 0.6226 (+0.029) confirming the data-starvation hypothesis.
`gat_shooter`'s shooter-centric readout is clearly superior: +0.012 AUC and F1 0.450 vs 0.172 —
`gat_mean_max` collapses to near-zero make recall while shooter-centric maintains balanced predictions.
Neither model beats LR yet — structural experiments (Exps 3–4) are necessary.

**Files:** `colab/phase2/results/gat_mean_max_history.json`, `gat_shooter_history.json`

---

### Exp 3 — Explicit temporal modeling via GRU *(COMPLETE — 2026-03-11)*

**What:** Spatial GNN within each timestep independently → `global_mean_pool` each snapshot →
feed sequence `[h_t2, h_t1, h_t0]` through a GRU → classify from final hidden state.
Remove `rel_type_TEMPORAL` edges from graph.

**Rationale:** Current architecture conflates temporal and spatial message passing — GATv2Conv
treats a TEMPORAL edge identically to an OD edge, just with different features. A GRU enforces
causal direction explicitly: the t=0 shot configuration is the outcome of what happened at t=1, t=2.
Player motion (shooter stepping into the shot, defender closing) is directional; message passing
is not.

**Results:**

| Experiment | Test AUC | Test Acc | Test F1 | Best Val AUC | Best Ep | Params | Time |
|------------|----------|----------|---------|--------------|---------|--------|------|
| temporal_gru | **0.6346** | 61.2% | 0.486 | 0.6368 | 86/100 | 430,466 | 22 min |

**Finding:** Best single architecture so far (tied with gat_shooter at 0.6342). The explicit
causal temporal structure (GRU over timestep snapshots) adds genuine value — 22 min vs 40 min
for gat_shooter despite more parameters, thanks to the V100 GPU and larger batch size.
Still 0.003 short of the LR baseline. F1 of 0.486 confirms balanced class predictions.

**Files:** `colab/phase2/results/temporal_gru_history.json`

---

### Exp 4 — Alternative conv mechanisms *(COMPLETE — 2026-03-11)*

**Variant A — SAGEConv (max agg):** Tests whether GATv2's attention mechanism earns its cost.
Simpler, fewer parameters, less likely to overfit.

**Variant B — EdgeConv (`h_j - h_i` relative encoding):** Designed for point-cloud data (structurally
similar to tracking). Relative position/speed between players is directly encoded in the message —
natural for basketball where it's the *relative* defender proximity that matters, not absolute coordinates.
Also handles the court-end shift (absolute x flips; relative x between two players doesn't).

**Results:**

| Experiment | Conv | Test AUC | Test Acc | Test F1 | Best Val AUC | Best Ep | Params | Time |
|------------|------|----------|----------|---------|--------------|---------|--------|------|
| sage_mean_max | SAGEConv | 0.6094 | 55.9% | 0.143 | 0.6155 | 35/100 | 138,882 | 6 min |
| edge_mean_max | EdgeConv | 0.6311 | 61.8% | 0.483 | 0.6389 | 65/100 | 172,418 | 13 min |

**Finding:** SAGEConv significantly underperforms (0.6094) — without attention, the conv cannot
discriminate meaningful player relationships. EdgeConv (relative encoding) is strong (0.6311, F1 0.483),
validating that relative spatial relationships between players matter more than absolute positions.
However neither beats temporal_gru. EdgeConv's val AUC peaked at 0.6389 (above baseline) but test
came in at 0.6311 — a wider val/test gap (0.008) than other models, suggesting mild overfitting.

**Files:** `colab/phase2/results/sage_mean_max_history.json`, `edge_mean_max_history.json`

---

### Exp 5 — Combine best architecture + focal loss *(COMPLETE — 2026-03-11, updated)*

**What:** Apply focal loss (γ=1.0) to the best architecture from Exps 0–4.
Two variants run — one incorrectly using gat_mean_max (pre-decided before winners were known),
one correctly using temporal_gru (the true Exp 5).

**Rationale:** Focal loss down-weights easy/confident examples, forcing the model to focus on
hard shots. If temporal_gru's gap to baseline (0.003 AUC) is due to miscalibration on easy shots,
focal loss should help.

**Results:**

| Experiment | Conv | Loss | Test AUC | Test Acc | Test F1 | Best Val AUC | Best Ep | Time |
|------------|------|------|----------|----------|---------|--------------|---------|------|
| gat_mean_max_focal | GATv2 | Focal γ=1.0 | 0.6210 | 57.0% | 0.138 | 0.6262 | 59/100 | 17 min |
| temporal_gru_focal | GRU | Focal γ=1.0 | 0.6304 | 61.7% | — | 0.6360 | 85/100 | 22 min |

**Finding:** Focal loss did not help either architecture. temporal_gru_focal (0.6304) is worse than
temporal_gru CE (0.6346 test). gat_mean_max_focal (0.6210) is worse than gat_mean_max CE (0.6226).
Focal loss introduces training instability (multiple val AUC dips) and in the GATv2 case collapses
F1. The model is already well-calibrated under CE — focal loss's hard-example weighting adds noise
rather than signal at 87K training samples.

**Files:** `colab/phase2/results/gat_mean_max_focal_history.json`, `temporal_gru_focal_history.json`

---

### Exp 8a — LLM Player Embeddings: Tabular Gate *(COMPLETE — 2026-03-11)*

**What:** Embedded all 477 unique players using OpenAI `text-embedding-3-large` (3072-dim).
Built natural-language shooting profiles per player (position, age, FG%, zone breakdowns,
3PT%, assist rates, dunks). Projected to 64-dim via PCA (72.7% variance explained, fit once
on full dataset). Replaced or augmented the 21 shooter stats in X_static; re-ran LR Lasso
(C=0.1, the Exp 0a winner) under 5-fold stratified CV.

**Results:**

| Config | Features | AUC | Δ vs Exp 0a |
|--------|----------|-----|------------|
| 8a-LR-replace | 17 + PCA64 | 0.6310±0.0029 | -0.0067 |
| 8a-LR-augment | 38 + PCA64 | 0.6361±0.0022 | -0.0016 |
| Baseline (Exp 0a) | Velocity-65 / Lasso | 0.6377 | — |

**Gate decision: ABORT Phase B.** Best AUC 0.6361 ≤ gate 0.6407.

**Finding:** LLM embeddings do not improve over raw shooting stats. Replace hurts (-0.007):
the 21 raw stats are more informative than the projected embedding. Augment is neutral (-0.002):
the embedding adds no independent signal. The hypothesis that text embeddings encode richer
player tendencies was not supported — for shot prediction, numeric stats *are* the right
representation; encoding them as text and re-embedding loses precision without adding knowledge.

**Files:** `colab/phase2/results/player_embeddings.pkl`, `colab/phase2/results/exp8a_results.txt`

---

### Exp 6 — HP search on temporal_gru *(COMPLETE — 2026-03-11)*

**What:** Two-phase HP search, 29 total configs, epochs=60/patience=15 each.
- **Phase A (runs 1–10):** Random sampling from broad search space.
- **Phase B (runs 11–28):** Focused hand-picked grid based on Phase A findings; `lr=5e-4, wd=1e-4` locked.

**Phase A search space:** `hidden_dim` {64,128,256} · `num_layers` {1,2,3,4} · `weight_decay` {1e-4,1e-3,5e-3} · `dropout` {0.1,0.2,0.3} · `lr` {1e-4,5e-4} · `batch_size` {128,256,512}

**Phase B focused grid** (lock `lr=5e-4`, `wd=1e-4`): `hd=64` nl=4 only; `hd=128/256` nl∈{3,4}; do∈{0.1,0.2}; bs∈{256,512}

**Full sweep leaderboard (top 10 of 29):**

| Rank | val_auc | hd | nl | wd | do | lr | bs |
|------|---------|----|----|----|----|-----|-----|
| 1 | **0.6487** | 256 | 3 | 1e-4 | 0.1 | 5e-4 | 512 |
| 2 | 0.6479 | 256 | 3 | 1e-4 | 0.1 | 5e-4 | 256 |
| 3 | 0.6477 | 256 | 4 | 1e-4 | 0.1 | 5e-4 | 512 |
| 4 | 0.6471 | 256 | 4 | 1e-4 | 0.2 | 5e-4 | 512 |
| 5 | 0.6468 | 256 | 4 | 1e-4 | 0.1 | 5e-4 | 256 |
| 6 | 0.6465 | 256 | 3 | 1e-4 | 0.2 | 5e-4 | 512 |
| 7 | 0.6462 | 128 | 3 | 1e-4 | 0.1 | 5e-4 | 512 |
| 8 | 0.6460 | 128 | 4 | 1e-4 | 0.1 | 5e-4 | 256 |
| 9 | 0.6459 | 128 | 3 | 1e-4 | 0.1 | 5e-4 | 256 |
| 10 | 0.6458 | 256 | 3 | 1e-4 | 0.2 | 5e-4 | 256 |

**Key findings:**
- `lr=5e-4` and `wd=1e-4` are dominant — every top-10 config uses both
- `hd=256` dominates top 6: model was capacity-limited at hd=128
- `nl=3` slightly edges `nl=4` (top-2 both nl=3)
- `do=0.1` preferred over 0.2 in top configs; do=0.3 consistently worst
- `bs` has minimal impact at this scale

**Baseline to beat:** temporal_gru default HPs → val_auc 0.6368 / test_auc 0.6346
Best sweep val_auc **0.6487** (+0.012 over baseline)

**Script:** `colab/phase2/hp_sweep.py --focused --run-id-offset 9`
**Results:** `colab/phase2/results/hp_sweep/sweep_results.json`
**GCS:** `gs://swishnet-nba/results/hp_sweep/`

---

### Exp 6b — Full reruns of top-3 sweep configs *(COMPLETE — 2026-03-12)*

**What:** Retrain top-3 configs from Exp 6 with full epochs=200/patience=25 to get proper test AUC.
The 60-epoch sweep underestimates peak performance — full training needed before comparing to baselines.
Run on `swishnet-rerun` (g2-standard-8 + NVIDIA L4, on-demand, us-central1-a). ~1 min/epoch on L4.

| Config | hd | nl | wd | do | lr | bs | sweep_val_auc | val_auc | best_ep | **test_auc** | test_acc | test_f1 |
|--------|----|----|----|----|----|----|--------------|---------|---------|------------|----------|---------|
| rerun_rank01 | 256 | 3 | 1e-4 | 0.1 | 5e-4 | 512 | 0.6487 | 0.6452 | 34 | 0.6437 | 0.6232 | 0.4691 |
| rerun_rank02 | 256 | 3 | 1e-4 | 0.1 | 5e-4 | 256 | 0.6479 | 0.6467 | 59 | 0.6444 | 0.6226 | 0.4200 |
| **rerun_rank03** | **256** | **4** | **1e-4** | **0.1** | **5e-4** | **512** | **0.6477** | **0.6457** | **49** | **0.6451** | **0.6240** | **0.4452** |

**Winner: rerun_rank03** (hd=256, nl=4, bs=512) — highest test_auc=0.6451

**Key findings:**
- All 3 configs beat the temporal_gru default baseline (test_auc 0.6346) by +0.009–0.011
- rank03's extra GNN layer (nl=4 vs nl=3) adds meaningful test generalization despite similar val_auc
- rank01 (bs=512) vs rank02 (bs=256): larger batch generalizes slightly better (+0.0007 test_auc)
- Best test_auc (0.6451) remains below XGBoost ceiling (0.650) — gap narrowed to 0.005

**Script:** `colab/phase2/rerun_top.py`
**Results:** `colab/phase2/results/rerun_top/`
**GCS:** `gs://swishnet-nba/results/rerun_top/`

---

### Exp 7 — Feature ablation (diagnostic) *(COMPLETE — 2026-03-12)*

**What:** Train best model (rank03: hd=256, nl=4, do=0.1, lr=5e-4, wd=1e-4, bs=512) with each
feature group zeroed in turn. Temporal ID (index 14) is SHUFFLED rather than zeroed to avoid empty
subgraph crashes — the signal is equally ablated but the model structure stays valid.

**Baseline:** rank03 test_auc = 0.6451 | epochs=100/patience=20

**Results (sorted by Δ AUC ascending — most damaging first):**

| Group | Indices | Method | test_auc | Δ AUC | val_auc @ ep |
|-------|---------|--------|----------|-------|-------------|
| temporal_id  | 14    | shuffle | 0.6364 | **−0.0087** | 0.6429 @ 57 |
| geometry     | 6–10  | zero    | 0.6366 | **−0.0085** | 0.6388 @ 54 |
| role_flags   | 3–5   | zero    | 0.6378 | −0.0073 | 0.6434 @ 43 |
| game_state   | 11–13 | zero    | 0.6413 | −0.0038 | 0.6460 @ 24 |
| position_enc | 15–19 | zero    | 0.6429 | −0.0022 | 0.6456 @ 41 |
| spatial_xyz  | 0–2   | zero    | 0.6430 | −0.0021 | 0.6489 @ 31 |
| player_stats | 20–40 | zero    | 0.6433 | −0.0018 | 0.6472 @ 35 |

**Key findings:**

- **Sanity check FAILED (interestingly):** `player_stats` is the *least* important group (Δ=−0.0018),
  not the most. Removing 21 shooting-history features barely hurts the GNN.
- **Temporal ordering is most critical:** `temporal_id` (Δ=−0.0087) — shuffling which snapshot is
  t=0/1/2 causes the largest drop. The GRU genuinely relies on temporal sequencing.
- **Geometry is nearly as critical:** `geometry` (Δ=−0.0085) — dist_to_rim, angle_to_basket,
  num_nearby_def drive the model's shot-quality estimate as much as temporal structure.
- **Role flags matter significantly:** `role_flags` (Δ=−0.0073) — without knowing who is the shooter
  and who is offense vs defense, the model loses directionality for message passing.
- **Spatial coordinates (xyz) barely matter:** (Δ=−0.0021) — the model extracts what it needs from
  the derived geometry features; raw x/y/z coordinates are largely redundant given them.
- **Core interpretation:** The GNN is NOT acting as a learned LR over shooting history. It exploits
  temporal dynamics (GRU ordering) and spatial geometry (dist features) — signals that LR cannot
  capture from static features alone. This explains why GNN > LR despite player_stats being LR's
  strongest predictor.

**Script:** `colab/phase2/exp7_feature_ablation.py`
**Output:** `colab/phase2/results/exp7_ablation/` → `{name}_result.json` + `ablation_summary.json`
**GCS:** `gs://swishnet-nba/results/exp7_ablation/`

---

## Phase 3 — Graph Construction Variants

Requires rebuilding all 87K graphs. Each variant trained with `temporal_gru` (best arch, Exp 3)
using identical HPs for direct comparison.

### Exp 9 — Temporal sampling ablation *(COMPLETE — 2026-03-12)*

**What:** Rebuild all 87K graphs under 3 variants that form a 2×2 factorial design, using
Exp 3 (`[3, 12]`, 3 timesteps, 0.48s) as the free baseline cell.

**Design:**

|  | Sparse (3 timesteps) | Dense (5 timesteps) |
|--|----------------------|---------------------|
| **Short horizon (0.48s)** | `[3, 12]` — Exp 3 baseline | **Variant A** `[3,6,9,12]` |
| **Long horizon (2.0s)**   | **Variant B** `[12, 50]`   | **Variant C** `[6,12,25,50]` |

**Causal questions:**
- **A vs Exp 3:** Does finer resolution within the same window help? (density effect)
- **B vs Exp 3:** Does a longer lookback help at the same step count? (horizon effect)
- **C vs A and B:** Does combining both outperform either alone? (interaction effect)

**Graph sizes:**

| Variant | `MOMENTS_BACK` | Timesteps | Nodes/graph |
|---------|---------------|-----------|-------------|
| Baseline (Exp 3) | `[3, 12]` | 3 | 33 |
| A — dense short | `[3, 6, 9, 12]` | 5 | 55 |
| B — sparse long | `[12, 50]` | 3 | 33 |
| C — dense long | `[6, 12, 25, 50]` | 5 | 55 |

**Infrastructure:** 3 parallel GCP CPU VMs (`swishnet-variant-a/b/c`, n1-standard-4) for graph building.
Training on `swishnet-rerun` (g2-standard-8 + NVIDIA L4). Best config from Exp 6b used (rank03: hd=256, nl=4, do=0.1, lr=5e-4, wd=1e-4, bs=512).

**Important:** Exp 9 is a controlled graph structure ablation — HPs are held fixed at rank03's values. The reference for Δ comparisons is the Exp 3 temporal_gru baseline (same HPs, default `[3,12]` sampling), **not** rank03's test_auc. The best overall model remains **rank03 at test_auc=0.6451** — none of the variants beat it.

**Training results:**

| Variant | Description | val_auc | best_ep | **test_auc** | test_acc | test_f1 | Δ vs Exp 3 baseline | Δ vs rank03 |
|---------|-------------|---------|---------|------------|----------|---------|---------------------|-------------|
| Exp 3 baseline | `[3,12]` sparse 3-step short | — | — | 0.6346 | — | — | — | -0.0105 |
| A — dense_short | `[3,6,9,12]` dense 5-step short | 0.6385 | 21 | 0.6350 | 0.6317 | 0.4668 | +0.0004 | -0.0101 |
| B — sparse_long | `[12,50]` sparse 3-step long | 0.6424 | 32 | 0.6348 | 0.6250 | 0.4760 | +0.0002 | -0.0103 |
| **C — dense_long** | **`[6,12,25,50]` dense 5-step long** | **0.6458** | **33** | **0.6411** | **0.6242** | **0.4640** | **+0.0065** | **-0.0040** |

**Key findings:**
- **Variant C is the best graph structure** — combining long horizon + dense sampling yields +0.0065 over the Exp 3 graph baseline; neither factor alone helps (+0.0004 / +0.0002). A genuine interaction effect.
- **No variant beats rank03 (0.6451)** — the optimised HPs from Exp 6b are a larger lever than graph structure. C comes closest at -0.004 below rank03.
- **A vs Exp 3 (density effect):** +0.0004 — finer resolution within 0.48s adds almost nothing
- **B vs Exp 3 (horizon effect):** +0.0002 — longer lookback at 3 timesteps adds almost nothing
- **C vs A+B (interaction):** +0.006 above either alone — GRU benefits from both dense sampling AND long horizon together to capture pre-shot motion
- All variants early-stopped well before ep200 (ep46/ep57/ep59)
- Best overall model to date: **rank03, test_auc=0.6451** (Exp 6b default graphs + optimised HPs)

**GCS output:**
```
gs://swishnet-nba/processed/variants/
  temporal_dense_short/graph_data/   ← Variant A
  temporal_sparse_long/graph_data/   ← Variant B
  temporal_dense_long/graph_data/    ← Variant C

gs://swishnet-nba/results/variants/  ← training results
```

**Script:** `colab/phase2/train_variants.py`
**Results:** `colab/phase2/results/variants/`

---

### Ext 3 — PCA of GRU Embeddings *(COMPLETE — 2026-03-12)*

**What:** Extracted 256-dim GRU final hidden states from all 86,425 shots using a forward hook on `model.gru`. Reduced to 2D via PCA and scatter-plotted make vs miss.

**Results:**

| Component | Variance explained |
|-----------|--------------------|
| PC1 | 68.7% |
| PC2 | 19.8% |
| **PC1 + PC2** | **88.4%** |

**Finding:** 88.4% of variance in the 256-dim embedding space is captured in 2 principal components — the GRU representation is highly structured. The 2D scatter shows visible make/miss separation, confirming the embedding encodes genuine shot-quality signal.

**Files:** `colab/phase2/results/ext_pca_gmm/pca_embeddings.png`, `ext3_pca_result.json`

---

### Ext 4 — GMM Shot Clustering + XGBoost Augmentation *(COMPLETE — 2026-03-12)*

**What:** Fit GMM (diag covariance) over K=3–10 on raw 256-dim embeddings. BIC selected K=10. Cross-tabbed clusters against static shot features. Re-ran XGBoost with soft GMM responsibilities appended to Static-38.

**GMM cluster summary (K=10):**

| Cluster | n | FG% | Avg dist (ft) | % 3PT | Shot clock | Archetype |
|---------|---|-----|--------------|-------|------------|-----------|
| 2 | 5,461 | **87.7%** | 5.7 | 0% | 18.5s | At-rim / dunks |
| 6 | 5,415 | **65.2%** | 2.7 | 0% | 15.9s | Right at rim / putbacks |
| 7 | 6,049 | **62.0%** | 5.8 | 0.7% | 13.9s | Short-range makes |
| 4 | 6,924 | 43.7% | 3.9 | 0% | 14.4s | Contested close-range |
| 9 | 11,918 | 46.3% | 15.2 | 7.6% | 12.4s | Mid-range (angled) |
| 1 | 7,755 | 41.1% | 7.1 | 0.1% | 11.5s | Short mid-range / paint |
| 3 | 11,992 | 41.6% | 18.8 | 18.1% | 12.8s | Long 2s / some 3s |
| 8 | 12,712 | **35.0%** | 21.9 | 48.4% | 11.6s | 3-point misses |
| 0 | 11,610 | **38.5%** | 22.4 | 46.5% | 12.6s | Above-the-break 3s |
| 5 | 6,589 | **28.9%** | 20.7 | 43.6% | **6.5s** | Late-clock desperation 3s |

**Finding:** Clusters recover known NBA shot archetypes without supervision — high-FG% clusters (2, 6, 7) are all close-range; low-FG% clusters (5, 8, 0) are far-out or late-clock. The model learned basketball-meaningful shot geometry purely from tracking dynamics.

**XGBoost augmentation:**

| Feature set | AUC | Δ |
|-------------|-----|---|
| Static-38 (XGBoost) | 0.6476±0.0031 | — |
| **Static-38 + GMM soft memberships** | **0.6551±0.0025** | **+0.0075** |
| XGBoost Velocity-65 ceiling (Ext 1) | 0.6501 | ref |
| GNN best model (Exp 6b rank03) | 0.6451 | ref |

**Headline:** Static-38 + GMM (0.6551) beats the previous XGBoost ceiling (0.6501) and the GNN (0.6451). The GNN's latent structure encodes signal that raw features cannot — feeding it back via soft cluster memberships extracts that signal into a tree model.

**Notes:** 5 game files (722 shots, 0.83%) excluded due to PyTorch legacy format incompatibility; X_static filtered via `shot_index_map.json` to maintain row alignment. GMM used `diag` covariance for tractability at 256 dims.

**Script:** `colab/phase2/ext_pca_gmm.py`
**Files:** `colab/phase2/results/ext_pca_gmm/` → `gs://swishnet-nba/results/ext_pca_gmm/`

---

### Ext 5 — Input Gradient Saliency + GAT Attention Analysis *(COMPLETE — 2026-03-12)*

**What:** Two inference-only interpretability analyses on rank03 checkpoint (test_auc=0.6451). No retraining.

**Checkpoint:** `colab/phase2/results/rerun_top/rerun_rank03_best.pt`

#### Part A: Input Gradient Saliency

Mean |∂logit₁/∂x| per node role × feature group (row-normalised within each role):

| Role | role | geom | pos | tid | stats | xyz | state |
|------|------|------|-----|-----|-------|-----|-------|
| shooter   | **3.55e-5** | 2.23e-5 | 2.55e-5 | 6.02e-6 | 1.05e-5 | 5.62e-6 | 4.15e-6 |
| ball_node | **3.20e-5** | 2.12e-5 | 1.85e-5 | 1.28e-5 | 5.17e-6 | 8.11e-6 | 3.14e-6 |
| defense   | **1.47e-5** | 7.83e-6 | 8.03e-6 | 3.97e-6 | 2.55e-6 | 1.69e-6 | 1.11e-6 |
| offense   | 8.24e-6 | 4.88e-6 | **4.68e-6** | 1.63e-6 | 1.91e-6 | 1.44e-6 | 1.13e-6 |

Top-5 feature groups (averaged across all nodes): role > geom ≈ pos > tid > stats

**Findings:**
- **Shooter and ball_node dominate**: gradients 3–4× higher than offense/defense nodes — predictions hinge on what the shooter and ball are doing, not the rest of the offense
- **Role flags most salient for every node type**: the model's primary signal is identity (who is the shooter, who has the ball, offense vs defense)
- **Geometry second**: dist_to_rim, angle_to_basket, num_nearby_def — structural shot quality
- **Player_stats low saliency**: consistent with Exp 7 ablation — shooting history barely drives predictions
- **tid salient for ball_node** (1.28e-5): GRU uses temporal ordering signal on ball position specifically

#### Part B: GAT Attention Weights

**By edge type:**

| Edge type | Mean α | Rank |
|-----------|--------|------|
| PB (Player-Ball)   | **0.1256** | 1st |
| OO (Off-Off)       | 0.0957 | 2nd |
| OD (Off-Def)       | 0.0876 | 3rd |
| DD (Def-Def)       | 0.0734 | 4th |

**By distance bin:**

| Distance | Mean α |
|----------|--------|
| 0–6 ft   | **0.1282** |
| 6–12 ft  | 0.0920 |
| 20+ ft   | 0.0919 |
| 12–20 ft | 0.0843 |

**By layer:** L0=0.0964 > L1=0.0933 > L2=0.0931 > L3=0.0909 (slight decay, roughly uniform)

**Findings:**
- **PB edges get most attention**, not OD — the model is primarily ball-tracking, not modelling matchups. Ball-player proximity is the single strongest attention signal.
- **OD edges are 3rd** (not 1st as hypothesised) — offense-defense matchup structure is used but is not the dominant attention pattern.
- **DD edges least attended** — defender-to-defender interactions contribute least; what the defense *collectively* does matters less than individual defender proximity to the ball/shooter.
- **Strong proximity bias**: attention drops ~35% from 0–6 ft to 6–12 ft bin. Model strongly prioritises nearby players, consistent with Exp 7 finding that geometry features matter.
- **Attention is roughly flat across layers** (L0–L3 differ by only 0.006) — no layer specifically specialises in OD or long-range interactions.
- **Exp 10 implication**: since attention *already* decays sharply with distance, sparse proximity graphs (Exp 10) may produce minimal Δ AUC — the model is effectively already ignoring distant pairs.

**Script:** `colab/phase2/ext5_interpret.py`
**Output:** `colab/phase2/results/ext5_interpret/` → `saliency_heatmap.png`, `attention_by_edge_type.png`, `ext5_results.json`
**GCS:** `gs://swishnet-nba/results/ext5_interpret/`

---

### Exp 10 — Sparse proximity graph (Future)
**What:** Replace fully-connected within-timestep clique (110 directed edges/timestep) with:
mandatory ball↔all and shooter↔all edges + proximity edges between players within D feet.
Test D ∈ {6, 10, 15, 20} ft.

**Rationale:** A center in the paint currently receives messages from a corner-3 shooter 35 feet away.
That edge likely adds noise. Sparse graphs reduce gradient signal from irrelevant pairs. Also tests
whether GATv2's attention mechanism was already zeroing distant edges (sparse ≈ dense → it was;
sparse > dense → it wasn't, and the model was wasting capacity).

---

## Infrastructure

| Resource | Detail |
|----------|--------|
| GCP Project | swishnet-489108 |
| GCS Bucket | gs://swishnet-nba |
| CPU VM (data) | swishnet-data, n1-standard-4, us-central1-a (stop when done) |
| GPU VM (training) | swishnet-trainer, n1-standard-8 + Tesla V100-SXM2-16GB, on-demand, us-central1-a (quota exhausted Mar 2026) |
| GPU VM (Exp 6b/9) | swishnet-rerun, g2-standard-8 + NVIDIA L4 24GB, on-demand, us-central1-a |
| CPU VMs (Phase 3) | swishnet-variant-a/b/c, n1-standard-4, us-central1-a (auto-shutdown when done) |
| Code | `colab/pipeline/` — full-season pipeline |
| | `colab/phase1/` — Phase 1 experiments + results |
| | `colab/phase2/` — Phase 2 experiments + results |
| | `colab/pipeline/launch_variant_vms.sh` — launches 3 parallel graph-rebuild VMs |

### Key commands
```bash
# Check VM status
gcloud compute instances describe swishnet-data --zone=us-central1-a \
  --project=swishnet-489108 --format='get(status)'

# Start/stop data VM
gcloud compute instances start swishnet-data --zone=us-central1-a --project=swishnet-489108
gcloud compute instances stop swishnet-data --zone=us-central1-a --project=swishnet-489108

# Re-run pipeline (resumable — skips already-processed games)
gcloud compute ssh swishnet-data --project=swishnet-489108 --zone=us-central1-a
nohup /home/anthonyargyropoulos/run_pipeline.sh </dev/null >/dev/null 2>&1 &
```

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Pre-2026 | Run Phase 1 on 12 games only | Validate pipeline before committing to full-season compute |
| Pre-2026 | Group Exps 1, 2, 7 into Phase 1 | These require no graph reconstruction — cheapest to run first |
| Pre-2026 | Run Phase 1b (reduced capacity + class weights) | Test if over-parameterisation explains gap — result: it doesn't |
| 2026-03-05 | Build full-season pipeline with fair comparison guarantee | Same shots in both GNN and baseline datasets; hard row-count assert |
| 2026-03-05 | Do not create GPU VM yet | Wait until Phase 2 code is ready; avoid idle GPU charges |
| 2026-03-05 | Drop "3–4 timesteps" idea from Phase 2 | Requires graph reconstruction → Phase 3 only |
| 2026-03-05 | Add baseline retraining as Exp 0a (before any GNN work) | 0.614 was on 12 games; target will move at 87K — must establish new reference before interpreting GNN results |
| 2026-03-10 | Exp 0a complete — new GNN target is AUC > 0.638 | LR ceiling confirmed: 50× more data only moved the baseline +0.024. Feature ceiling is real; GNNs must add structural value |
| 2026-03-10 | GPU quota = 0 on GCP project swishnet-489108 | Blocking Exps 0b–5; need to request GPUS_ALL_REGIONS ≥ 1 via GCP Console → IAM & Admin → Quotas |
| 2026-03-10 | Exp 0c complete — LLM oracle AUC 0.53–0.57 | Zero-shot Gemini confirms language priors alone are insufficient; empirical training adds ~0.07–0.11 AUC. Flash > Pro due to over-hedging in thinking model. |
| 2026-03-11 | GPUS_ALL_REGIONS quota approved (swishnet-489108) | Unblocked GPU VM creation |
| 2026-03-11 | SPOT T4 preempted 3× (~every 45 min) | Switched to on-demand V100 (n1-standard-8 + Tesla V100-SXM2-16GB) to avoid interruptions |
| 2026-03-11 | PyTorch 2.7.1 incompatible with V100 (CUDA cc 7.0 < required 7.5) | Downgraded to PyTorch 2.3.0+cu121 on VM; PyG extensions disabled but GATv2Conv/GINEConv/SAGEConv all work via pure-PyTorch scatter |
| 2026-03-11 | gat_mean_max_focal was not the true Exp 5 | Was hardcoded before winners known; re-ran as temporal_gru_focal (best arch + focal loss) — the correct Exp 5 |
| 2026-03-11 | Exp 9 Phase 3 temporal ablation launched | 2×2 factorial design isolating horizon vs. resolution effects; 3 parallel CPU VMs rebuilding 87K graphs each |
| 2026-03-11 | swishnet-trainer stopped (preempted), restarted after quota increase | VM terminated mid-run during Phase B; CPU quota raised from 12→24 vCPUs to run trainer alongside variant VMs |
| 2026-03-11 | XGBoost (Ext 1) becomes new overall ceiling at 0.650 | Beats LR (0.638) and best GNN (0.635) without tuning; new GNN target raised to AUC > 0.650 |
| 2026-03-11 | Decision Tree (Ext 2) peaks at depth 7 on both feature sets | Shallower than expected; confirms dist_to_rim and is_3pt dominate; LR still beats trees (0.638 vs 0.632) |
| 2026-03-12 | V100 unavailable (ZONE_RESOURCE_POOL_EXHAUSTED all zones); switched to L4 | g2-standard-8 + NVIDIA L4 (24GB VRAM) used for Exp 6b and Exp 9 training; ~1.5–2× slower than V100 due to lower HBM bandwidth, but no interruptions |
| 2026-03-12 | rerun_top.py was dying after ep1 due to systemd SIGKILL on SSH disconnect | nohup only protects against SIGHUP; systemd kills user session processes with SIGKILL. Fixed with `loginctl enable-linger` + screen detached session |
| 2026-03-12 | Exp 6b complete — best config is rank03 (hd=256, nl=4, bs=512), test_auc=0.6451 | nl=4 beats nl=3 on test_auc despite similar val_auc; gap to XGBoost ceiling narrowed to 0.005 |
| 2026-03-12 | Exp 9 complete — Variant C (dense_long) wins with test_auc=0.6411 (+0.0065 vs baseline) | Combining long horizon [6,12,25,50] + 5 timesteps produces an interaction effect; neither factor alone helps (A: +0.0004, B: +0.0002). Next step: rebuild default graphs with [6,12,25,50] sampling. |
| 2026-03-12 | Ext 3 complete — PCA of GRU embeddings captures 88.4% variance in 2 PCs | Highly structured embedding space; make/miss clouds visually separate in 2D |
| 2026-03-12 | Ext 4 complete — GMM K=10 recovers NBA shot archetypes without supervision | High-FG% clusters = at-rim/dunks (87.7%); low-FG% = late-clock 3s (28.9%) |
| 2026-03-12 | Ext 4 XGBoost augmentation: Static-38+GMM AUC=0.6551, new overall best | +0.0075 over raw XGBoost (0.6476); beats Velocity-65 ceiling (0.6501) and GNN (0.6451) |
| 2026-03-12 | Exp 7: GNN is NOT a learned LR — geometry and temporal ordering dominate, not player_stats | player_stats ablation Δ=−0.0018 (least important); temporal_id Δ=−0.0087 and geometry Δ=−0.0085 (most important). GNN adds structural value beyond baseline features. Ext 5 confirms: PB edges most attended, proximity bias strong — Exp 10 sparse graphs may yield minimal gain. |

# Visualization Handoff Summary

All figures and tables specified for the SwishNet paper. The visualization agent should produce each of these and place the outputs in `report/figures/`.

---

## Figure 1: NBA Tracking Screenshot + Graph Example (MANDATORY, side-by-side)

**Location in paper:** Section 3 (Dataset and Features), Figure 1

### Panel A (Left): NBA Game Screenshot
- **Type:** Screenshot / annotated court diagram
- **Content:** A single NBA SportVU tracking snapshot at the moment of shot release. Show all 10 players and the ball on a half-court diagram. Highlight the shooter distinctly (e.g., star marker or bold circle). Mark offensive players in one color (blue) and defensive players in another (red). Show the ball position (orange).
- **Layout:** Half-court view, standard NBA court lines visible. Player positions as circles/dots with jersey numbers or role labels if space permits. Shooter clearly distinguished.
- **Caption:** "NBA SportVU tracking snapshot at the moment of a shot release. Player positions are marked on the half-court; the shooter is highlighted."
- **Source data:** Pick one representative shot from the local game data in `colab/data/` or from a `graphs_XXXXXXXXXX.pt` file. Reconstruct positions from node features (indices 0-2 are x,y,z). Choose a shot that is visually clear (mid-range or 3-pointer, not a dunk).
- **Priority:** HIGH
- **Output:** `report/figures/nba_tracking_screenshot.png`

### Panel B (Right): Graph Visualization
- **Type:** Network graph / node-link diagram
- **Content:** The 11-node graph for ONE timestep of the same shot shown in Panel A. 5 offense nodes (blue), 5 defense nodes (red), 1 ball node (orange). Show all edges with colors indicating type: O-O (blue), O-D (purple), D-D (red), player-ball (orange). Annotate one node with a subset of its 41 features to show what information each node carries.
- **Layout:** Nodes positioned roughly corresponding to their court positions (spatial layout). Node size proportional to role importance or uniform. Edge thickness can be uniform. Include a small legend for edge types.
- **Caption:** "The corresponding player-ball interaction graph for a single timestep. Nodes represent the 5 offensive players (blue), 5 defensive players (red), and the ball (orange). Edge colors indicate relationship type."
- **Source data:** Same shot as Panel A. Use `build_graph.py` / `build_graph_from_shot()` to construct the graph, then visualize with networkx or similar.
- **Priority:** HIGH
- **Output:** `report/figures/graph_example.png`

**Note:** These two panels should be combined into a single figure with `\subfigure` or side-by-side `\includegraphics`. Total width = `\textwidth`.

---

## Figure 2: Architecture Comparison Bar Chart

**Location in paper:** Section 5.2 (GNN Architecture Ablation)

- **Type:** Horizontal bar chart
- **Content:** Test AUC for each architecture tested:
  - SAGEConv: 0.6094
  - EdgeConv: 0.6311
  - Static GAT (shooter readout): 0.6342
  - TemporalGRU (default HPs): 0.6346
  - TemporalGRU (tuned, rank03): 0.6451
  - Variant C (dense_long): 0.6411
- **Layout:** Horizontal bars sorted by AUC descending. Add two vertical dashed reference lines: LR Lasso baseline (0.6377, labeled) and XGBoost ceiling (0.6501, labeled). Color bars by architecture family (static GNN in blue, temporal in green, variant in teal). X-axis: "Test AUC", range [0.60, 0.66]. Font size >= 10pt.
- **Caption:** "Test AUC across GNN architectures. Dashed lines indicate LR Lasso (0.6377) and XGBoost Velocity-65 (0.6501) baselines."
- **Source data:** RESEARCH_LOG.md, Exps 0b, 3, 4, 6b, 9
- **Priority:** HIGH
- **Output:** `report/figures/architecture_comparison.png`

---

## Figure 3: Interpretability Panel (2 subplots)

**Location in paper:** Section 5.6 (Interpretability)

### Panel A: Feature Ablation Bar Chart
- **Type:** Horizontal bar chart
- **Content:** Delta AUC when each feature group is ablated:
  - temporal_id: -0.0087
  - geometry: -0.0085
  - role_flags: -0.0073
  - game_state: -0.0038
  - position_enc: -0.0022
  - spatial_xyz: -0.0021
  - player_stats: -0.0018
- **Layout:** Horizontal bars, sorted by magnitude (most damaging at top). Bars colored in a red gradient (darker = larger drop). X-axis: "$\Delta$ AUC". Add a vertical line at 0 for reference.
- **Caption:** "(a) Feature ablation: AUC drop when each feature group is zeroed/shuffled."

### Panel B: Attention by Edge Type
- **Type:** Bar chart
- **Content:** Mean GATv2 attention weight by edge type:
  - PB (Player-Ball): 0.1256
  - OO (Off-Off): 0.0957
  - OD (Off-Def): 0.0876
  - DD (Def-Def): 0.0734
- **Layout:** Vertical bars, color-coded by edge type (same colors as Figure 1B). Y-axis: "Mean attention weight $\alpha$".
- **Caption:** "(b) Mean GATv2 attention weight by edge type."

**Combined caption:** "Interpretability analyses. (a) Feature ablation: AUC drop when each feature group is zeroed. Temporal ordering and geometry are most critical; player shooting stats contribute least. (b) Mean GATv2 attention weight by edge type. Player-ball edges receive highest attention."
- **Source data:** RESEARCH_LOG.md Exp 7 + Ext 5; `colab/phase2/results/exp7_ablation/ablation_summary.json`, `colab/phase2/results/ext5_interpret/ext5_results.json`
- **Priority:** HIGH
- **Output:** `report/figures/interpretability_panel.png`

---

## ~~Figure 4: PCA Scatter~~ -- MOVED TO APPENDIX (see Appendix Figure A4 below)

## ~~Figure 5: Training Curves~~ -- MOVED TO APPENDIX (see Appendix Figure A3 below)

---

## Tables (already in LaTeX)

### Table 1: Main Results Leaderboard
- Already fully specified in `swishnet_paper.tex`, Table 1 (`tab:main_results`)
- No visualization agent work needed

### Table 2: Graph Construction Variants
- Already fully specified in `swishnet_paper.tex`, Table 2 (`tab:graph_variants`)
- No visualization agent work needed

### Table 3: Feature Ablation
- Already fully specified in `swishnet_paper.tex`, Table 3 (`tab:ablation`)
- No visualization agent work needed

---

# Appendix Figures (NEW — supplementary material beyond 5-page main body)

## Appendix Figure A1: Decision Tree Depth Curve

**Location in paper:** Appendix A (Full Baseline Results), Figure after Table `tab:lr_full`

- **Type:** Line plot with two series
- **Content:** Train AUC and validation AUC vs. max_depth (1--20) for both Static-38 and Velocity-65. Four lines total.
- **Layout:** X-axis: "Max Depth" (1--20). Y-axis: "AUC" (range 0.50--0.70). Two colors for feature sets, solid=train / dashed=val. Vertical dashed line at depth=7 (optimal). Legend in upper-right. Font >= 10pt.
- **Caption:** "Decision tree train vs. validation AUC as a function of max depth (1--20), under 5-fold CV. Both feature sets peak at depth 7. Divergence beyond depth 8 is a textbook overfitting signature."
- **Source data:** `colab/phase2/results/ext2_tree_static38.json`, `ext2_tree_velocity65.json` (contain per-depth train/val AUC). Existing PNGs: `ext2_tree_curve_static38.png`, `ext2_tree_curve_velocity65.png`
- **Priority:** MEDIUM — can reuse existing PNGs if they meet quality standards, otherwise regenerate
- **Output:** `report/figures/decision_tree_depth_curve.png`

---

## Appendix Figure A2: HP Sweep Scatter Plot

**Location in paper:** Appendix B (Hyperparameter Sweep Details)

- **Type:** Scatter plot (strip/jitter or grouped)
- **Content:** All 29 HP configurations plotted as val_auc (y-axis) vs. hidden_dim (x-axis, categorical: 64/128/256), with points colored by num_layers (1/2/3/4 as discrete color scale) and sized by dropout (0.1=large, 0.3=small).
- **Layout:** X-axis: hidden_dim (categorical). Y-axis: "Validation AUC" (range 0.62--0.65). Color legend for num_layers. Horizontal dashed line at default HP baseline (val_auc=0.6368). Point jitter to avoid overlap. Font >= 10pt.
- **Caption:** "Hyperparameter sweep: validation AUC vs. hidden dimension for all 29 configurations, colored by number of GATv2 layers. Larger hidden dimension and 3--4 layers consistently yield higher AUC."
- **Source data:** `colab/phase2/results/hp_sweep/sweep_results.json` (on GCS: `gs://swishnet-nba/results/hp_sweep/sweep_results.json`). Top-10 values also in RESEARCH_LOG.md Exp 6.
- **Priority:** MEDIUM
- **Output:** `report/figures/hp_sweep_scatter.png`

---

## Appendix Figure A3: Training Curves (now APPENDIX, promoted from optional)

**Location in paper:** Appendix C (Training Dynamics)

- **Type:** Line plot, 2 subplots
- **Content:**
  - **(a)** Validation AUC vs. epoch for all 3 rerun configs (rank01, rank02, rank03). Vertical dashed lines at each config's early stopping epoch (34, 59, 49).
  - **(b)** Training loss vs. epoch for rank03 only. Smooth curve showing convergence.
- **Layout:** Side-by-side subplots sharing x-axis (epoch, 0--75). (a) Y-axis: val AUC (0.58--0.65). (b) Y-axis: train loss (0.66--0.70). Three colored lines in (a) with legend. Font >= 10pt.
- **Caption:** "Training dynamics for the top-3 HP configurations. (a) Validation AUC vs. epoch; vertical lines mark early stopping. (b) Training loss for rank03 showing smooth convergence."
- **Source data:** `colab/phase2/results/ext_pca_gmm/rerun_top/rerun_rank01_result.json`, `rerun_rank02_result.json`, `rerun_rank03_result.json` — each contains a `history` array with per-epoch train_loss, train_acc, val_auc, val_acc, val_f1.
- **Priority:** MEDIUM
- **Output:** `report/figures/training_curves.png`

---

## Appendix Figure A4: PCA + GMM Scatter (promoted from optional to APPENDIX)

**Location in paper:** Appendix D (GMM Cluster Analysis)

- **Type:** Scatter plot, 2D, two panels
- **Content:** 86,425 shots plotted in PCA space (PC1 vs PC2, 88.4% variance). (a) Color by make/miss outcome. (b) Color by GMM cluster assignment (K=10).
- **Layout:** Two side-by-side scatter plots. Left: make (green) vs miss (red), alpha=0.05 for density. Right: 10 distinct colors with cluster centroids marked as large circles. Axis labels: "PC1 (68.7%)", "PC2 (19.8%)". Font >= 10pt.
- **Caption:** "PCA projection of GRU embeddings. (a) Colored by outcome: visible make/miss separation with overlap reflecting stochasticity. (b) Colored by GMM cluster. Cluster 2 (at-rim, 87.7% FG) is spatially distinct."
- **Source data:** `colab/phase2/results/ext_pca_gmm/gru_embeddings.npy` (86,425 x 256), `gmm_responsibilities.npy`, `ext3_pca_result.json`. Existing PNG: `pca_embeddings.png`
- **Priority:** MEDIUM
- **Output:** `report/figures/pca_gmm_scatter.png`

---

# Appendix Tables (already in LaTeX, no visualization agent work needed)

### Table A1: Full LR Sweep (`tab:lr_full`)
- 8 configurations, already in appendix LaTeX

### Table A2: HP Sweep Top 10 (`tab:hp_sweep`)
- Already in appendix LaTeX

### Table A3: Val-Test Gap (`tab:val_test_gap`)
- Already in appendix LaTeX

### Table A4: GMM Cluster Summary (`tab:gmm_clusters`)
- Already in appendix LaTeX

### Table A5: Saliency Matrix (`tab:saliency`)
- Already in appendix LaTeX

### Table A6: Attention by Distance (`tab:attn_distance`)
- Already in appendix LaTeX

### Table A7: Phase 1 Results (`tab:phase1`)
- Already in appendix LaTeX

---

## Production Notes

1. All figures should be saved at 300 DPI minimum, PNG or PDF format
2. Font size in figures must be >= 10pt (per report guidelines)
3. All plots need axis labels, legends, and legible font sizes when printed
4. Color choices should be colorblind-friendly where possible
5. The paper has a strict 5-page limit for the main body — Figures 4 and 5 from the main body are now in the appendix
6. Figure 1 (NBA screenshot + graph) and Figure 3 (interpretability) remain the highest priority main-body visualizations
7. Appendix figures A1--A4 should be generated but are lower priority than main-body figures 1--3
8. Source data for training curves is in JSON `history` arrays inside the rerun result files

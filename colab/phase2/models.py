"""
Phase 2 GNN models.

New architectures beyond Phase 1:
  TemporalGRUShotPredictor  — per-timestep spatial GNN + GRU (Exp 3)
  GNNShotPredictorV2        — extends Phase 1 with SAGEConv and EdgeConv (Exp 4)

Phase 1 GAT/GINE model is imported directly from phase1/models.py for Exps 0b, 5.

Node feature index constants (see build_graph.py):
  _HAS_BALL_IDX  = 3   (1.0 → shooter)
  _TIMESTAMP_IDX = 14  (0.0=current, 1.0=prev, 2.0=two-back)
Edge attr index:
  _TEMPORAL_IDX  = 8   (1.0 → edge is cross-timestep TEMPORAL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    SAGEConv,
    EdgeConv,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.utils import subgraph

_HAS_BALL_IDX  = 3
_TIMESTAMP_IDX = 14
_TEMPORAL_IDX  = 8   # index in edge_attr for the TEMPORAL relation type flag


# ---------------------------------------------------------------------------
# Exp 3 — Temporal GRU predictor
# ---------------------------------------------------------------------------

class TemporalGRUShotPredictor(nn.Module):
    """
    Per-timestep spatial GNN → GRU over snapshot sequence → classifier.

    Architecture:
        pre-MLP (shared across timesteps)
        → for each t in [t=2, t=1, t=0]  (oldest → newest):
              spatial GATv2Conv on subgraph of timestep-t nodes
              global_mean_pool  →  snapshot embedding h_t  [B, D]
        → GRU([h_2, h_1, h_0])  →  final hidden state  [B, D]
        → post-MLP  →  Linear(D, 2)  →  raw logits

    TEMPORAL edges are removed before the per-timestep GNNs so that
    temporal information flows only through the GRU, not message passing.

    Args:
        num_node_features : input node feature dim (default 41)
        hidden_dim        : width of all hidden layers
        num_heads         : GATv2 attention heads
        num_gnn_layers    : spatial GNN layers applied within each timestep
        num_pre_layers    : depth of shared pre-MLP
        num_post_layers   : depth of post-GRU MLP
        dropout           : dropout rate
    """

    def __init__(
        self,
        num_node_features=41,
        hidden_dim=128,
        num_heads=4,
        num_gnn_layers=2,
        num_pre_layers=2,
        num_post_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout    = dropout

        # ── Pre-MLP (shared; projects raw node features → hidden_dim) ────────
        pre = []
        in_dim = num_node_features
        for _ in range(num_pre_layers):
            pre.append(self._mlp_block(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim
        self.pre_mlp = nn.Sequential(*pre) if pre else nn.Identity()

        # ── Spatial GNN layers (shared across timesteps) ───────────────────────
        self.spatial_convs = nn.ModuleList()
        self.spatial_bns   = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_c = hidden_dim if (i > 0 or num_pre_layers > 0) else num_node_features
            self.spatial_convs.append(
                GATv2Conv(in_c, hidden_dim, heads=num_heads,
                          dropout=dropout, edge_dim=9, concat=False)
            )
            self.spatial_bns.append(nn.BatchNorm1d(hidden_dim))

        # ── GRU over 3-timestep sequence ─────────────────────────────────────
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                          num_layers=1, batch_first=False)

        # ── Post-MLP + classifier ─────────────────────────────────────────────
        post = []
        in_dim = hidden_dim
        for _ in range(num_post_layers):
            post.append(self._mlp_block(in_dim, hidden_dim, dropout))
        self.post_mlp = nn.Sequential(*post) if post else nn.Identity()
        self.fc = nn.Linear(hidden_dim, 2)

    @staticmethod
    def _mlp_block(in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, data):
        x_raw, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        n_nodes = x_raw.shape[0]

        # ── Remove TEMPORAL edges ──────────────────────────────────────────────
        spatial_mask   = (edge_attr[:, _TEMPORAL_IDX] != 1.0)
        sp_edge_index  = edge_index[:, spatial_mask]
        sp_edge_attr   = edge_attr[spatial_mask]

        # ── Pre-MLP (all nodes jointly) ────────────────────────────────────────
        x = self.pre_mlp(x_raw)

        # ── Per-timestep GNN + pool ────────────────────────────────────────────
        snapshots = []   # [h_t2, h_t1, h_t0]

        for t in [2, 1, 0]:   # chronological: oldest → newest
            t_mask = (x_raw[:, _TIMESTAMP_IDX] == float(t))  # [N] bool

            # Subgraph: only edges where both endpoints are at timestep t
            ei_t, ea_t = subgraph(
                t_mask, sp_edge_index, sp_edge_attr,
                relabel_nodes=True, num_nodes=n_nodes,
            )
            x_t     = x[t_mask]           # [B*11, D]
            batch_t = batch[t_mask]        # [B*11]

            # Spatial GNN layers (shared weights)
            for conv, bn in zip(self.spatial_convs, self.spatial_bns):
                x_res = x_t
                x_t   = conv(x_t, ei_t, ea_t)
                x_t   = bn(x_t)
                x_t   = F.relu(x_t)
                if x_res.shape == x_t.shape:
                    x_t = x_t + x_res

            h_t = global_mean_pool(x_t, batch_t)  # [B, D]
            snapshots.append(h_t)

        # ── GRU over snapshot sequence [h_2, h_1, h_0] ────────────────────────
        seq = torch.stack(snapshots, dim=0)      # [3, B, D]
        _, h_n = self.gru(seq)                   # h_n: [1, B, D]
        h = h_n.squeeze(0)                       # [B, D]

        # ── Post-MLP + classify ────────────────────────────────────────────────
        h = self.post_mlp(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc(h)


# ---------------------------------------------------------------------------
# Exp 4 — SAGEConv and EdgeConv variants
# ---------------------------------------------------------------------------

class GNNShotPredictorV2(nn.Module):
    """
    GAT/GINE/SAGE/EDGE conv with mean_max readout.

    Extends Phase 1 GNNShotPredictor with two new conv types:
      'SAGE' — SAGEConv (max aggregation, no edge features)
      'EDGE' — EdgeConv (max aggregation, relative node features h_j - h_i)

    Readout is fixed to mean_max (Phase 1 best) for Exp 4.

    Note: SAGE and EDGE convolutions do not use edge_attr.  This is
    intentional — the experiment tests whether topology + relative node
    embeddings can match or exceed attention-based message passing that
    uses explicit edge features.
    """

    VALID_CONVS = ('SAGE', 'EDGE')

    def __init__(
        self,
        conv_type='SAGE',
        num_node_features=41,
        hidden_dim=128,
        num_layers=2,
        num_pre_layers=2,
        num_post_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        assert conv_type in self.VALID_CONVS, \
            f"conv_type must be one of {self.VALID_CONVS}"

        self.conv_type  = conv_type
        self.hidden_dim = hidden_dim
        self.dropout    = dropout

        # ── Pre-MLP ───────────────────────────────────────────────────────────
        self.pre_layers = nn.ModuleList()
        in_dim = num_node_features
        for _ in range(num_pre_layers):
            self.pre_layers.append(self._mlp_block(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim

        # ── Conv layers ───────────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for i in range(num_layers):
            in_c = hidden_dim if (i > 0 or num_pre_layers > 0) else num_node_features
            if conv_type == 'SAGE':
                self.convs.append(SAGEConv(in_c, hidden_dim, aggr='max'))
            else:  # EDGE
                # EdgeConv MLP input: [h_i || h_j - h_i] → 2 * in_c
                edge_mlp = nn.Sequential(
                    nn.Linear(2 * in_c, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(EdgeConv(nn=edge_mlp, aggr='max'))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ── Post-MLP (input is 2×hidden for mean_max readout) ─────────────────
        post_in = 2 * hidden_dim
        self.post_layers = nn.ModuleList()
        for i in range(num_post_layers):
            self.post_layers.append(
                self._mlp_block(post_in if i == 0 else hidden_dim, hidden_dim, dropout)
            )

        self.fc = nn.Linear(hidden_dim if num_post_layers > 0 else post_in, 2)

    @staticmethod
    def _mlp_block(in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # SAGE and EDGE convolutions do not use edge_attr

        # Pre-MLP
        for layer in self.pre_layers:
            x = layer(x)

        # Conv layers
        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if x_res.shape == x.shape:
                x = x + x_res

        # mean_max readout
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=-1)

        # Post-MLP
        for layer in self.post_layers:
            x = layer(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)

"""
Phase 1 GNN models: unified architecture with pluggable readout strategies.

Supports GAT (GATv2Conv) and GINE (GINEConv) convolutions with five readouts:
  'mean'            – global mean pool (baseline)
  'max'             – global max pool
  'add'             – global add pool (sum)
  'mean_max'        – concat of mean and max (doubles post-MLP input dim)
  'shooter_centric' – concat of shooter-node embedding and scene mean pool.
                      Shooter is identified by has_ball==1 AND timestamp==0,
                      guaranteed present in every graph by build_graph.py.

All models return raw logits (shape [B, 2]).  Do NOT apply log_softmax here.
Use F.cross_entropy for training and torch.softmax for probability outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, GINEConv,
    global_mean_pool, global_max_pool, global_add_pool,
)

# Indices into the 41-dimensional node feature vector (see build_graph.py)
_HAS_BALL_IDX  = 3   # x[:, 3]  == 1.0  →  this player is the shooter
_TIMESTAMP_IDX = 14  # x[:, 14] == 0.0  →  current moment (t=0)


def _apply_readout(x, batch, readout, shooter_mask):
    """
    Apply the chosen readout strategy after GNN layers.

    Args:
        x            : node embeddings [N, D]
        batch        : graph assignment vector [N]
        readout      : one of ('mean', 'max', 'add', 'mean_max', 'shooter_centric')
        shooter_mask : bool tensor [N], True for the shooter node at t=0

    Returns:
        graph-level embedding [B, D] for mean/max/add, [B, 2D] for the rest
    """
    if readout == 'mean':
        return global_mean_pool(x, batch)
    elif readout == 'max':
        return global_max_pool(x, batch)
    elif readout == 'add':
        return global_add_pool(x, batch)
    elif readout == 'mean_max':
        return torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1
        )
    elif readout == 'shooter_centric':
        shooter_emb = x[shooter_mask]              # [B, D] — one row per graph
        scene_emb   = global_mean_pool(x, batch)   # [B, D]
        return torch.cat([shooter_emb, scene_emb], dim=-1)  # [B, 2D]
    else:
        raise ValueError(f"Unknown readout: '{readout}'")


class GNNShotPredictor(nn.Module):
    """
    Unified GNN shot predictor supporting GAT and GINE conv types plus five
    readout strategies.

    Architecture:
        pre-MLP  →  GNN conv layers (with BatchNorm + ReLU)  →  readout
                 →  post-MLP  →  Linear(hidden_dim, 2)  →  raw logits

    Residual skip connections are applied in GAT layers when dimensions match.
    GINE layers rely on the learnable epsilon (train_eps) instead.

    Args:
        conv_type       : 'GAT' or 'GINE'
        readout         : 'mean' | 'max' | 'add' | 'mean_max' | 'shooter_centric'
        num_node_features: input node feature dimension (default 41)
        hidden_dim      : width of all hidden layers
        num_heads       : GAT attention heads (ignored for GINE)
        num_layers      : number of GNN conv layers
        num_pre_layers  : depth of pre-processing MLP (0 = no pre-MLP)
        num_post_layers : depth of post-processing MLP (0 = no post-MLP)
        dropout         : dropout rate applied in MLP blocks and before final fc
        train_eps       : GINE learnable epsilon flag (ignored for GAT)
    """

    VALID_CONVS    = ('GAT', 'GINE')
    VALID_READOUTS = ('mean', 'max', 'add', 'mean_max', 'shooter_centric')

    def __init__(
        self,
        conv_type='GAT',
        readout='mean',
        num_node_features=41,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_pre_layers=2,
        num_post_layers=2,
        dropout=0.3,
        train_eps=True,
    ):
        super().__init__()
        assert conv_type in self.VALID_CONVS,    f"conv_type must be one of {self.VALID_CONVS}"
        assert readout   in self.VALID_READOUTS, f"readout must be one of {self.VALID_READOUTS}"

        self.conv_type = conv_type
        self.readout   = readout
        self.hidden_dim = hidden_dim
        self.dropout    = dropout
        self.num_layers = num_layers

        # Readouts that concat two vectors have 2× the post-MLP input dim
        post_in_dim = (
            2 * hidden_dim if readout in ('mean_max', 'shooter_centric')
            else hidden_dim
        )

        # ── Pre-processing MLP ────────────────────────────────────────────────
        self.pre_layers = nn.ModuleList()
        if num_pre_layers > 0:
            self.pre_layers.append(self._mlp_block(num_node_features, hidden_dim, dropout))
            for _ in range(num_pre_layers - 1):
                self.pre_layers.append(self._mlp_block(hidden_dim, hidden_dim, dropout))
            current_dim = hidden_dim
        else:
            current_dim = num_node_features

        # ── GNN conv layers ───────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for i in range(num_layers):
            in_dim = current_dim if i == 0 else hidden_dim
            if conv_type == 'GAT':
                self.convs.append(
                    GATv2Conv(
                        in_dim, hidden_dim,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=9,
                        concat=False,
                    )
                )
            else:  # GINE
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINEConv(nn=mlp, edge_dim=9, train_eps=train_eps))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ── Post-processing MLP ───────────────────────────────────────────────
        self.post_layers = nn.ModuleList()
        if num_post_layers > 0:
            self.post_layers.append(self._mlp_block(post_in_dim, hidden_dim, dropout))
            for _ in range(num_post_layers - 1):
                self.post_layers.append(self._mlp_block(hidden_dim, hidden_dim, dropout))
            fc_in = hidden_dim
        else:
            fc_in = post_in_dim

        self.fc = nn.Linear(fc_in, 2)

    # ── Helpers ───────────────────────────────────────────────────────────────

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

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, data):
        """
        Args:
            data: PyG Data (or Batch) with fields x, edge_index, edge_attr, batch

        Returns:
            logits: [B, 2] raw (un-normalised) class scores
        """
        x_raw, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Shooter mask computed from ORIGINAL features before any transform.
        # has_ball==1 at timestamp==0 is guaranteed to be exactly one node per
        # graph by build_graph.py (shooter_not_found causes the shot to be
        # dropped before saving, so no incomplete graphs reach this point).
        shooter_mask = (
            (x_raw[:, _HAS_BALL_IDX]  == 1.0) &
            (x_raw[:, _TIMESTAMP_IDX] == 0.0)
        )

        x = x_raw
        # Pre-processing MLP
        for layer in self.pre_layers:
            x = layer(x)

        # GNN conv layers
        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            # Residual skip for GAT (GINE uses learnable epsilon instead)
            if self.conv_type == 'GAT' and x_res.shape == x.shape:
                x = x + x_res

        # Readout → graph-level embedding
        x = _apply_readout(x, batch, self.readout, shooter_mask)

        # Post-processing MLP
        for layer in self.post_layers:
            x = layer(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)   # raw logits — caller decides loss / softmax

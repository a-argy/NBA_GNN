"""
GAT-based model for NBA shot prediction (graph classification).
Simple architecture with GAT layers, pre/post MLPs, batch normalization, and mean pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATShotPredictor(torch.nn.Module):
    """
    Graph Attention Network (GAT) model for predicting shot outcomes.

    Architecture:
    - Pre-processing MLP layers
    - Configurable number of GATv2Conv layers with multi-head attention
    - Batch normalization after each layer
    - Global mean pooling to aggregate node features
    - Post-processing MLP layers
    - Final classification layer

    Attributes:
        hidden_dim (int): Dimension of hidden layers
        dropout (float): Dropout rate for regularization
        num_layers (int): Number of GAT layers
        num_pre_layers (int): Number of pre-processing MLP layers
        num_post_layers (int): Number of post-processing MLP layers
        pre_layers (nn.ModuleList): Pre-processing MLP layers
        convs (nn.ModuleList): GAT convolutional layers
        bns (nn.ModuleList): Batch normalization layers
        post_layers (nn.ModuleList): Post-processing MLP layers
        fc (nn.Linear): Final classification layer
    """

    def __init__(self, num_node_features=41, hidden_dim=16, num_heads=4,
                 num_layers=2, num_pre_layers=1, num_post_layers=1, dropout=0.3):
        """
        Initialize the GAT model.

        Args:
            num_node_features (int): Number of input node features (default: 41)
            hidden_dim (int): Dimension of hidden layers (default: 16)
            num_heads (int): Number of attention heads (default: 4)
            num_layers (int): Number of GAT layers (default: 2)
            num_pre_layers (int): Number of pre-processing MLP layers (default: 1)
            num_post_layers (int): Number of post-processing MLP layers (default: 1)
            dropout (float): Dropout rate (default: 0.3)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_pre_layers = num_pre_layers
        self.num_post_layers = num_post_layers

        # Pre-processing MLP layers
        self.pre_layers = nn.ModuleList()
        if num_pre_layers > 0:
            # First pre-processing layer
            self.pre_layers.append(nn.Sequential(
                nn.Linear(num_node_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            # Additional pre-processing layers
            for _ in range(num_pre_layers - 1):
                self.pre_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
            current_dim = hidden_dim
        else:
            current_dim = num_node_features

        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer
                self.convs.append(GATv2Conv(current_dim, hidden_dim, heads=num_heads,
                                           dropout=dropout, edge_dim=9, concat=False))
            else:
                # Hidden layers
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=num_heads,
                                           dropout=dropout, edge_dim=9, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Post-processing MLP layers
        self.post_layers = nn.ModuleList()
        for _ in range(num_post_layers):
            self.post_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        # Final classification layer
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        """
        Forward pass through the model.

        Args:
            data: PyTorch Geometric Data object with x, edge_index, edge_attr, batch

        Returns:
            Log softmax probabilities for each class [batch_size, 2]
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Pre-processing MLP layers
        for pre_layer in self.pre_layers:
            x = pre_layer(x)

        # GAT layers with skip connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_residual = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)

            # Skip connection (if dimensions match)
            if x_residual.shape == x.shape:
                x = x + x_residual

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Post-processing MLP layers
        for post_layer in self.post_layers:
            x = post_layer(x)

        # Dropout before final classification
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Final classification
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
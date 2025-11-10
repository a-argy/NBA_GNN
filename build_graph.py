import json
import torch
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

"""
Build a graph representation from shot data.

Each shot has:
- 5 offensive players with positions (x, y)
- 5 defensive players with positions (x, y)

Graph structure:
- 10 nodes (5 offense + 5 defense)
- Node features: [x_pos, y_pos]
- 45 edges (complete graph: 10 choose 2 = 10*9/2 = 45)
- Edge features: [x_rel, y_rel, euclidean_distance]
  where for edge (u, v): [x_u - x_v, y_u - y_v, sqrt((x_u - x_v)^2 + (y_u - y_v)^2)]
"""


def build_graph_from_shot(shot_data):
    """
    Build a PyTorch Geometric graph from a single shot.
    
    Args:
        shot_data: Dictionary containing 'offense_position' and 'defense_position'
        
    Returns:
        Data: PyTorch Geometric Data object with nodes, edges, and edge attributes
    """
    # Extract player positions
    offense_positions = shot_data['offense_position']
    defense_positions = shot_data['defense_position']
    
    # Build node features: [10, 2] tensor with (x, y) positions
    node_features = []
    
    # Add offensive players (indices 0-4)
    for player in offense_positions:
        x, y = player[2], player[3]  # x and y are at indices 2 and 3
        node_features.append([x, y])
    
    # Add defensive players (indices 5-9)
    for player in defense_positions:
        x, y = player[2], player[3]
        node_features.append([x, y])
    
    # Convert to tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Build edge index for complete graph (all pairs of nodes)
    edge_index = []
    for i in range(10):
        for j in range(i + 1, 10):
            # Add edge in both directions for undirected graph
            edge_index.append([i, j])
            edge_index.append([j, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Build edge attributes: [x_rel, y_rel, euclidean_distance]
    edge_attr = []
    
    for i in range(10):
        for j in range(i + 1, 10):
            # Get positions
            x_i, y_i = node_features[i]
            x_j, y_j = node_features[j]
            
            # Calculate relative position and distance
            x_rel = x_i - x_j
            y_rel = y_i - y_j
            distance = np.sqrt(x_rel**2 + y_rel**2)
            
            # Add for both directions (i->j and j->i)
            edge_attr.append([x_rel, y_rel, distance])
            edge_attr.append([-x_rel, -y_rel, distance])  # Reverse direction
    
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


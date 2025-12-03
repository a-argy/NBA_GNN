import json
import torch
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

"""
Build a graph representation from shot data.

Each shot has:
- 5 offensive players with positions (x, y, z)
- 5 defensive players with positions (x, y, z)
- Ball position (x, y, z)

Graph structure:
- 10 nodes (5 offense + 5 defense)
- Node features: [player_id, x_pos, y_pos, has_ball]
  - player_id: unique identifier for the player
  - has_ball: 1 if player has possession, 0 otherwise
- 45 edges (complete graph: 10 choose 2 = 10*9/2 = 45)
- Edge features: [x_rel, y_rel, euclidean_distance]
  where for edge (u, v): [x_u - x_v, y_u - y_v, sqrt((x_u - x_v)^2 + (y_u - y_v)^2)]
"""


def build_graph_from_shot(shot_data, possession_threshold=4.0):
    """
    Build a PyTorch Geometric graph from a single shot.
    """
    # Extract player positions and ball position
    offense_positions = shot_data['offense_position']
    defense_positions = shot_data['defense_position']
    ball_position = shot_data['ball_position']

    # Validate each team has 5 players
    if len(offense_positions) != 5 or len(defense_positions) != 5:
        return None
    
    # Combine all player positions
    all_players = offense_positions + defense_positions
    offensive_players = all_players[:5]
    
    # Find player with ball possession
    ball_x, ball_y, ball_z = ball_position
    min_distance = float('inf')
    player_with_ball = -1
    
    for idx, player in enumerate(offensive_players):
        player_x, player_y, player_z = player[2], player[3], player[4]
        distance = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2 + (player_z - ball_z)**2)
        
        if distance < min_distance:
            min_distance = distance
            player_with_ball = idx
    
    if min_distance > possession_threshold:
        player_with_ball = -1
    
    # Build node features: [10, 3] tensor with (x, y, has_ball)
    node_features = []

    # Add offensive players
    for idx, player in enumerate(offense_positions):
        x, y = player[2], player[3]
        has_ball = 1.0 if idx == player_with_ball else 0.0
        node_features.append([x, y, has_ball])  # REMOVED player_id

    # Add defensive players  
    for idx, player in enumerate(defense_positions):
        x, y = player[2], player[3]
        has_ball = 0.0
        node_features.append([x, y, has_ball])  # REMOVED player_id
    
    # Convert to tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Build edge index for complete graph
    edge_index = []
    for i in range(10):
        for j in range(i + 1, 10):
            edge_index.append([i, j])
            edge_index.append([j, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Build edge attributes
    edge_attr = []
    
    for i in range(10):
        for j in range(i + 1, 10):
            x_i, y_i, _ = node_features[i]
            x_j, y_j, _ = node_features[j]
            
            x_rel = x_i - x_j
            y_rel = y_i - y_j
            distance = np.sqrt(x_rel**2 + y_rel**2)
            
            edge_attr.append([x_rel, y_rel, distance])
            edge_attr.append([-x_rel, -y_rel, distance])
    
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
   
    return data


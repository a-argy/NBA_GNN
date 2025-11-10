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


def visualize_graph(graph, shot_data, save_path='shot_graph.png'):
    """
    Visualize the graph with labeled nodes showing player positions.
    
    Args:
        graph: PyTorch Geometric Data object
        shot_data: Original shot data dictionary
        save_path: Path to save the visualization
    """
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(graph.num_nodes):
        G.add_node(i)
    
    # Add edges (only unique edges for undirected graph)
    edge_index = graph.edge_index.numpy()
    for i in range(0, edge_index.shape[1], 2):  # Skip every other edge (bidirectional)
        source, target = edge_index[0, i], edge_index[1, i]
        G.add_edge(source, target)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use actual court positions for node layout
    pos = {}
    for i in range(graph.num_nodes):
        x, y = graph.x[i].numpy()
        pos[i] = (x, y)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)
    
    # Draw nodes with different colors for offense and defense
    offense_nodes = list(range(5))  # Nodes 0-4
    defense_nodes = list(range(5, 10))  # Nodes 5-9
    
    nx.draw_networkx_nodes(G, pos, nodelist=offense_nodes, 
                          node_color='red', node_size=500, 
                          alpha=0.8, label='Offense', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=defense_nodes, 
                          node_color='blue', node_size=500, 
                          alpha=0.8, label='Defense', ax=ax)
    
    # Create labels with player IDs
    labels = {}
    for i in range(5):
        player_id = shot_data['offense_position'][i][1]
        labels[i] = f'O{i}\n{player_id}'
    for i in range(5, 10):
        player_id = shot_data['defense_position'][i-5][1]
        labels[i] = f'D{i-5}\n{player_id}'
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                           font_weight='bold', ax=ax)
    
    # Add ball position
    ball_x, ball_y = shot_data['ball_position'][0], shot_data['ball_position'][1]
    ax.scatter(ball_x, ball_y, color='orange', s=200, marker='*', 
              edgecolors='black', linewidths=2, label='Ball', zorder=5)
    
    # Formatting
    ax.set_xlabel('X Position (feet)', fontsize=12)
    ax.set_ylabel('Y Position (feet)', fontsize=12)
    ax.set_xlim(0, 94)  # Court length
    ax.set_ylim(0, 50)  # Court width
    ax.set_title(f'Shot Graph Visualization\nComplete Graph: {graph.num_nodes} Nodes, {graph.num_edges//2} Edges\nShot Made: {shot_data["made"]}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved to {save_path}")
    plt.close()  # Close instead of show to avoid blocking
    
    return fig
    # Load the shot data
    with open('shot_data.json', 'r') as f:
        shots = json.load(f)
    
    # Build graph for the first shot
    first_shot = shots[0]
    graph = build_graph_from_shot(first_shot)
    
    print("Graph built successfully!")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Node features shape: {graph.x.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    print(f"Edge attributes shape: {graph.edge_attr.shape}")
    print(f"\nFirst few nodes (x, y positions):")
    print(graph.x[:3])
    print(f"\nFirst few edges:")
    print(graph.edge_index[:, :6])
    print(f"\nFirst few edge attributes (x_rel, y_rel, distance):")
    print(graph.edge_attr[:3])
    
    # Visualize the graph
    print("\nGenerating visualization...")
    visualize_graph(graph, first_shot)

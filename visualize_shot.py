import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from build_graph import build_graph_from_shot

# Load the shot data
with open('shot_data_new_temp.json', 'r') as f:
    shots = json.load(f)

# Build graph for the first shot
first_shot = shots[0]
print("First shot info: \n", first_shot)
graph = build_graph_from_shot(first_shot)
print("All node features:")
print("Node features: [x, y, z, has_ball, is_offense, dist_to_rim, dist_to_ball_handler,")
print("               angle_to_basket, dist_to_3pt_line, num_nearby_defenders,")
print("               pos_G, pos_F, pos_C, pos_GF, pos_FC,")
print("               age, fg_pct, avg_dist, pct_fga_fg2a, pct_fga_00_03, pct_fga_03_10,")
print("               pct_fga_10_16, pct_fga_16_xx, pct_fga_fg3a, fg2_pct, fg_pct_00_03,")
print("               fg_pct_03_10, fg_pct_10_16, fg_pct_16_xx, fg3_pct, pct_ast_fg2, pct_ast_fg3,")
print("               pct_fga_dunk, fg_pct_dunk, pct_fga_corner3, fg_pct_corner3, games]")
print("  (For offense: dist_to_ball_handler = distance to nearest defender)")
print("  (For defense: dist_to_ball_handler = distance to ball handler)")
for i in range(graph.num_nodes):
    features = graph.x[i].numpy()
    x, y, z = features[0], features[1], features[2]
    has_ball, is_offense = features[3], features[4]
    dist_to_rim, dist_metric = features[5], features[6]
    angle_to_basket, dist_to_3pt = features[7], features[8]
    num_nearby_def = features[9]
    position_encoding = features[10:15]
    player_stats = features[15:37]  # 22 stats features (indices 15-36)
    role = "OFFENSE" if is_offense == 1 else "DEFENSE"
    metric_name = "dist_to_nearest_def" if is_offense == 1 else "dist_to_ball_handler"
    
    # Get player ID and name
    if i < 5:  # Offensive player
        player_id = first_shot['offense_position'][i][1]
    else:  # Defensive player
        player_id = first_shot['defense_position'][i-5][1]
    
    player_name = "Unknown"
    if 'player_names' in first_shot and str(player_id) in first_shot['player_names']:
        player_name = first_shot['player_names'][str(player_id)].get('full_name', 'Unknown')
    
    # Get position type
    pos_types = ['G', 'F', 'C', 'G-F', 'F-C']
    pos_idx = position_encoding.argmax() if position_encoding.sum() > 0 else 0
    pos_type = pos_types[int(pos_idx)]
    
    # Extract all 22 stats
    age = player_stats[0] if len(player_stats) > 0 else 0
    fg_pct = player_stats[1] if len(player_stats) > 1 else 0
    avg_dist = player_stats[2] if len(player_stats) > 2 else 0
    pct_fga_fg2a = player_stats[3] if len(player_stats) > 3 else 0
    pct_fga_00_03 = player_stats[4] if len(player_stats) > 4 else 0
    pct_fga_03_10 = player_stats[5] if len(player_stats) > 5 else 0
    pct_fga_10_16 = player_stats[6] if len(player_stats) > 6 else 0
    pct_fga_16_xx = player_stats[7] if len(player_stats) > 7 else 0
    pct_fga_fg3a = player_stats[8] if len(player_stats) > 8 else 0
    fg2_pct = player_stats[9] if len(player_stats) > 9 else 0
    fg_pct_00_03 = player_stats[10] if len(player_stats) > 10 else 0
    fg_pct_03_10 = player_stats[11] if len(player_stats) > 11 else 0
    fg_pct_10_16 = player_stats[12] if len(player_stats) > 12 else 0
    fg_pct_16_xx = player_stats[13] if len(player_stats) > 13 else 0
    fg3_pct = player_stats[14] if len(player_stats) > 14 else 0
    pct_ast_fg2 = player_stats[15] if len(player_stats) > 15 else 0
    pct_ast_fg3 = player_stats[16] if len(player_stats) > 16 else 0
    pct_fga_dunk = player_stats[17] if len(player_stats) > 17 else 0
    fg_pct_dunk = player_stats[18] if len(player_stats) > 18 else 0
    pct_fga_corner3 = player_stats[19] if len(player_stats) > 19 else 0
    fg_pct_corner3 = player_stats[20] if len(player_stats) > 20 else 0
    games = player_stats[21] if len(player_stats) > 21 else 0
    
    print(f"\nNode {i} ({role}, {pos_type}) - {player_name} (ID: {player_id}):")
    print(f"  Position: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    print(f"  Ball & Role: has_ball={int(has_ball)}, is_offense={int(is_offense)}")
    print(f"  Distances: dist_to_rim={dist_to_rim:.2f}, {metric_name}={dist_metric:.2f}")
    print(f"  Court: angle={angle_to_basket:.2f}, dist_3pt={dist_to_3pt:.2f}, nearby_def={int(num_nearby_def)}")
    print(f"  Position Encoding: {position_encoding.tolist()}")
    print(f"  Player Stats (22 features):")
    print(f"    Age: {age:.4f} (norm), Overall FG%: {fg_pct:.3f}, Avg Dist: {avg_dist:.4f} (norm)")
    print(f"    Shot Distribution: 2PT%={pct_fga_fg2a:.3f}, 3PT%={pct_fga_fg3a:.3f}")
    print(f"    2PT Zones: 0-3ft={pct_fga_00_03:.3f}, 3-10ft={pct_fga_03_10:.3f}, 10-16ft={pct_fga_10_16:.3f}, 16+ft={pct_fga_16_xx:.3f}")
    print(f"    2PT Accuracy: Overall={fg2_pct:.3f}, 0-3ft={fg_pct_00_03:.3f}, 3-10ft={fg_pct_03_10:.3f}, 10-16ft={fg_pct_10_16:.3f}, 16+ft={fg_pct_16_xx:.3f}")
    print(f"    3PT: FG%={fg3_pct:.3f}, Corner%={pct_fga_corner3:.3f}, Corner FG%={fg_pct_corner3:.3f}")
    print(f"    Assists: 2PT={pct_ast_fg2:.3f}, 3PT={pct_ast_fg3:.3f}")
    print(f"    Dunks: Freq={pct_fga_dunk:.3f}, FG%={fg_pct_dunk:.3f}")
    print(f"    Games: {games:.4f} (norm)")

print("\nEdge features: [x_rel, y_rel, distance, edge_angle, O-O, O-D, D-D]")
for i in range(min(10, graph.num_edges)):  # Show first 10 edges
    edge_features = graph.edge_attr[i]
    x_rel, y_rel, distance = edge_features[0], edge_features[1], edge_features[2]
    edge_angle = edge_features[3]
    edge_type = edge_features[4:7]
    edge_type_str = "O-O" if edge_type[0] == 1 else ("O-D" if edge_type[1] == 1 else "D-D")
    print(f'Edge {i}: x_rel={x_rel:.2f}, y_rel={y_rel:.2f}, distance={distance:.2f}, '
          f'angle={edge_angle:.2f}, type={edge_type_str}')

print("Graph built successfully!")
print(f"Number of nodes: {graph.num_nodes}")
print(f"Number of edges: {graph.num_edges}")

# Create visualization
G = nx.Graph()

# Add nodes
for i in range(graph.num_nodes):
    G.add_node(i)

# Add edges (only unique edges for undirected graph)
edge_index = graph.edge_index.numpy()
for i in range(0, edge_index.shape[1], 2):
    source, target = edge_index[0, i], edge_index[1, i]
    G.add_edge(source, target)

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 10))

# Use actual court positions for node layout
pos = {}
for i in range(graph.num_nodes):
    x, y = graph.x[i][0].item(), graph.x[i][1].item()
    pos[i] = (x, y)

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)

# Draw nodes with different colors
offense_nodes = list(range(5))
defense_nodes = list(range(5, 10))

nx.draw_networkx_nodes(G, pos, nodelist=offense_nodes, 
                      node_color='red', node_size=500, 
                      alpha=0.8, label='Offense', ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=defense_nodes, 
                      node_color='blue', node_size=500, 
                      alpha=0.8, label='Defense', ax=ax)

# Create labels
labels = {}
for i in range(5):
    player_id = first_shot['offense_position'][i][1]
    labels[i] = f'O{i}\n{player_id}'
for i in range(5, 10):
    player_id = first_shot['defense_position'][i-5][1]
    labels[i] = f'D{i-5}\n{player_id}'

nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                       font_weight='bold', ax=ax)

# Add ball position
ball_x, ball_y = first_shot['ball_position'][0], first_shot['ball_position'][1]
ax.scatter(ball_x, ball_y, color='orange', s=200, marker='*', 
          edgecolors='black', linewidths=2, label='Ball', zorder=5)

# Formatting
ax.set_xlabel('X Position (feet)', fontsize=12)
ax.set_ylabel('Y Position (feet)', fontsize=12)
ax.set_xlim(0, 94)  # Court length
ax.set_ylim(50, 0)  # Court width
ax.set_title(f'Shot Graph Visualization\nComplete Graph: {graph.num_nodes} Nodes, {graph.num_edges//2} Edges\nShot Made: {first_shot["made"]}', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
save_path = 'shot_graph.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nGraph visualization saved to {save_path}")
plt.close()

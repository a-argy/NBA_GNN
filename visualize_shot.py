import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from build_graph import build_graph_from_shot

# Load the shot data
with open('shot_data_warriors_game.json', 'r') as f:
    shots = json.load(f)

# Build graph for the first shot
first_shot = shots[3]
print("First shot info: \n", first_shot)
graph = build_graph_from_shot(first_shot)
print("All node features:")
for i in range(graph.num_nodes):
    player_id, x, y, has_ball = graph.x[i].numpy()
    print(f"Node {i}: player_id={int(player_id)}, x={x:.2f}, y={y:.2f}, has_ball={int(has_ball)}")

for i in range(graph.num_edges):
    x_rel, y_rel, distance = graph.edge_attr[i]
    print(f'Edge {i}: x_rel={x_rel}, y_rel={y_rel}, distance={distance}')

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
    _, x, y,_ = graph.x[i].numpy()
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

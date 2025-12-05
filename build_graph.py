import json
import torch
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

# Load player stats mapping (will be loaded once when module is imported)
PLAYER_STATS = None

def load_player_stats(stats_file='player_stats_2016.json'):
    """Load player statistics mapping from JSON file."""
    global PLAYER_STATS
    if PLAYER_STATS is None and os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            PLAYER_STATS = json.load(f)
        print(f"Loaded stats for {len(PLAYER_STATS)} players from {stats_file}")
    return PLAYER_STATS

"""
Build a graph representation from shot data.

Each shot has:
- 5 offensive players with positions (x, y, z)
- 5 defensive players with positions (x, y, z)
- Ball position (x, y, z)

Graph structure:
- 10 nodes (5 offense + 5 defense)
- Node features: [x_pos, y_pos, z_pos, has_ball, is_offense, dist_to_rim, dist_to_ball_handler, 
                  angle_to_basket, dist_to_3pt_line, num_nearby_defenders, quarter, game_clock, shot_clock,
                  position_encoding, player_stats]
  - x_pos, y_pos, z_pos: player coordinates (3 features)
  - has_ball: 1 if player has possession, 0 otherwise (1 feature)
  - is_offense: 1 if offensive player, 0 if defensive (1 feature)
  - dist_to_rim: Euclidean distance to the rim (1 feature)
  - dist_to_ball_handler: For offense: distance to nearest defender; For defense: distance to ball handler (1 feature)
  - angle_to_basket: Angle from player to basket in radians (1 feature)
  - dist_to_3pt_line: Distance to 3-point arc, negative=inside, positive=outside (1 feature)
  - num_nearby_defenders: Count of defenders within 6 feet (1 feature)
  - quarter: quarter of the game (1-4) (1 feature)
  - game_clock: game clock in seconds (0-720) (1 feature)
  - shot_clock: shot clock in seconds (0-24) (1 feature)
  - position_encoding: one-hot encoding of player position (G, F, C, G-F, F-C) (5 features)
  - player_stats: season shooting stats - 22 features:
      age, fg_pct, avg_dist, pct_fga_fg2a, pct_fga_00_03, pct_fga_03_10, pct_fga_10_16, pct_fga_16_xx,
      pct_fga_fg3a, fg2_pct, fg_pct_00_03, fg_pct_03_10, fg_pct_10_16, fg_pct_16_xx, fg3_pct,
      pct_ast_fg2, pct_ast_fg3, pct_fga_dunk, fg_pct_dunk, pct_fga_corner3, fg_pct_corner3, games
  Total: 40 features per node
- Edges: complete graph (10 nodes = 90 directed edges)
- Edge features: [x_rel, y_rel, euclidean_distance, edge_angle, rel_type_OO, rel_type_OD, rel_type_DD]
  - x_rel, y_rel: relative position (2 features)
  - euclidean_distance: distance between nodes (1 feature)
  - edge_angle: angle/bearing from source to target node in radians (1 feature)
  - rel_type_*: one-hot encoding of edge type (Offense-Offense, Offense-Defense, Defense-Defense) (3 features)
  Total: 7 features per edge
"""

# Court dimensions (in feet)
COURT_LENGTH = 94
COURT_WIDTH = 50
# Rim positions (standard NBA: 5.25 feet from baseline, centered at 25 feet)
LEFT_RIM = (5.25, 25.0)
RIGHT_RIM = (88.75, 25.0)
# 3-point line distance from basket (NBA: ~23.75 feet, corner: 22 feet)
THREE_POINT_DISTANCE = 23.75
THREE_POINT_CORNER_DISTANCE = 22.0
# Defensive contest distance threshold (feet)
CONTEST_DISTANCE = 6.0

def parse_game_clock(game_clock_str):
    """
    Parse game clock string (MM:SS) to seconds.
    Returns float seconds, or 0.0 if parsing fails.
    """
    if game_clock_str is None:
        return 0.0
    try:
        if isinstance(game_clock_str, (int, float)):
            return float(game_clock_str)
        parts = str(game_clock_str).split(':')
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), float(parts[1])
            return minutes * 60 + seconds
        return float(game_clock_str)
    except (ValueError, AttributeError):
        return 0.0

# Position encoding mapping
POSITION_MAP = {
    'G': 0,
    'F': 1,
    'C': 2,
    'G-F': 3,
    'F-G': 3,  # Treat F-G same as G-F
    'F-C': 4,
}

def determine_target_rim(home_team_id, poss_team_id, quarter):
    """
    Determine which rim the offensive team is shooting at.
    
    In NBA tracking data, typically:
    - Home team shoots at right rim (88.75, 25) in Q1, Q3
    - Home team shoots at left rim (5.25, 25) in Q2, Q4
    - Away team shoots at the opposite rim
    """
    is_home_team = (poss_team_id == home_team_id)
    
    # Quarters 1 and 3: home shoots right, away shoots left
    # Quarters 2 and 4: home shoots left, away shoots right
    if quarter in [1, 3]:
        return RIGHT_RIM if is_home_team else LEFT_RIM
    else:  # quarters 2, 4
        return LEFT_RIM if is_home_team else RIGHT_RIM


def encode_position(position_str):
    """
    Encode player position as one-hot vector.
    Returns: [is_G, is_F, is_C, is_GF, is_FC]
    """
    encoding = [0.0] * 5
    pos_idx = POSITION_MAP.get(position_str, -1)
    if pos_idx >= 0:
        encoding[pos_idx] = 1.0
    return encoding


def calculate_angle_to_basket(x, y, rim_pos):
    """
    Calculate angle from player position to basket.
    Returns angle in radians from -π to π.
    """
    dx = rim_pos[0] - x
    dy = rim_pos[1] - y
    return np.arctan2(dy, dx)


def calculate_distance_to_three_point_line(x, y, rim_pos):
    """
    Calculate approximate distance to 3-point line.
    Simplified: distance from rim minus 3PT distance.
    Negative = inside the arc, positive = outside.
    """
    dist_to_rim = np.sqrt((x - rim_pos[0])**2 + (y - rim_pos[1])**2)
    # Simple approximation: distance beyond/within 3PT arc
    return dist_to_rim - THREE_POINT_DISTANCE


def count_nearby_defenders(x, y, defense_positions, threshold=CONTEST_DISTANCE):
    """
    Count number of defenders within threshold distance.
    """
    count = 0
    for def_player in defense_positions:
        def_x, def_y = def_player[2], def_player[3]
        dist = np.sqrt((x - def_x)**2 + (y - def_y)**2)
        if dist <= threshold:
            count += 1
    return count


def calculate_edge_angle(x_i, y_i, x_j, y_j):
    """
    Calculate angle of edge from node i to node j.
    Returns angle in radians from -π to π.
    """
    dx = x_j - x_i
    dy = y_j - y_i
    return np.arctan2(dy, dx)


def get_player_stats_features(player_id):
    """
    Get player statistics features for a given player ID.
    Returns a list of 22 stat features, or zeros if player not found.
    """
    # Try both string and int versions of player_id
    stats = None
    if PLAYER_STATS:
        stats = PLAYER_STATS.get(str(player_id))
        if stats is None:
            stats = PLAYER_STATS.get(int(player_id))
    
    if stats:
        features = [
            stats.get('age', 0) / 40.0,  # Normalize age by ~40
            stats.get('fg_pct', 0),  # Already 0-1
            stats.get('avg_dist', 0) / 30.0,  # Normalize by max distance
            stats.get('pct_fga_fg2a', 0),  # % of FGA that are 2PT
            stats.get('pct_fga_00_03', 0),  # % at rim (0-3 ft)
            stats.get('pct_fga_03_10', 0),  # % short mid-range
            stats.get('pct_fga_10_16', 0),  # % mid-range
            stats.get('pct_fga_16_xx', 0),  # % long 2PT
            stats.get('pct_fga_fg3a', 0),  # % of FGA that are 3PT
            stats.get('fg2_pct', 0),  # 2PT FG%
            stats.get('fg_pct_00_03', 0),  # FG% at rim
            stats.get('fg_pct_03_10', 0),  # FG% short mid
            stats.get('fg_pct_10_16', 0),  # FG% mid-range
            stats.get('fg_pct_16_xx', 0),  # FG% long 2PT
            stats.get('fg3_pct', 0),  # 3PT FG%
            stats.get('pct_ast_fg2', 0),  # % of 2PT assisted
            stats.get('pct_ast_fg3', 0),  # % of 3PT assisted
            stats.get('pct_fga_dunk', 0),  # % dunks
            stats.get('fg_pct_dunk', 0),  # Dunk FG%
            stats.get('pct_fga_corner3', 0),  # % corner 3s
            stats.get('fg_pct_corner3', 0),  # Corner 3 FG%
            stats.get('games', 0) / 82.0,  # Normalize by season length
        ]
        return features
    else:
        # Return zeros if player not found (e.g., rookie, two-way player)
        return [0.0] * 22


def build_graph_from_shot(shot_data, possession_threshold=4.0):
    """
    Build a PyTorch Geometric graph from a single shot with NORMALIZED features.
    
    Node features (40 total):
    - x, y, z: position (3)
    - has_ball: binary flag (1)
    - is_offense: binary flag (1)
    - dist_to_rim: distance to rim (1)
    - dist_to_ball_handler: For offense: distance to nearest defender; For defense: distance to ball handler (1)
    - angle_to_basket: angle from player to basket in radians (1)
    - dist_to_3pt_line: distance to 3-point line, negative=inside (1)
    - num_nearby_defenders: count of defenders within 6 feet (1)
    - quarter: quarter of the game (1-4) (1)
    - game_clock: game clock in seconds (0-720) (1)
    - shot_clock: shot clock in seconds (0-24) (1)
    - position_encoding: one-hot for G, F, C, G-F, F-C (5)
    - player_stats: season shooting statistics (22)
    
    Edge features (7 total):
    - x_rel, y_rel: relative position (2)
    - euclidean_distance: distance between nodes (1)
    - edge_angle: bearing from source to target in radians (1)
    - rel_type_OO, rel_type_OD, rel_type_DD: one-hot edge type (3)
    """
    # Load player stats if not already loaded
    load_player_stats()
    
    # Extract player positions and ball position
    offense_positions = shot_data['offense_position']
    defense_positions = shot_data['defense_position']
    ball_position = shot_data['ball_position']
    player_positions_map = shot_data.get('player_positions', {})
    
    # Get rim position for distance calculation
    home_team_id = shot_data.get('home_team_id')
    poss_team_id = shot_data.get('poss_team_id')
    quarter = shot_data.get('quarter', 1)
    
    # Extract game state features (raw values, not normalized)
    game_clock_seconds = shot_data.get('game_clock', 0.0) or 0.0  # Already in seconds
    shot_clock = shot_data.get('shot_clock', 24.0) or 24.0  # Default to 24 if None
    
    if home_team_id and poss_team_id and quarter:
        target_rim = determine_target_rim(home_team_id, poss_team_id, quarter)
    else:
        # Default to right rim if metadata missing
        target_rim = RIGHT_RIM

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
    
    # Build node features with enriched information
    node_features = []
    
    # Get ball handler position for defender distance calculations
    ball_handler_pos = None
    if player_with_ball >= 0:
        ball_handler_pos = (offense_positions[player_with_ball][2], offense_positions[player_with_ball][3])

    # Normalization constants
    COURT_LENGTH_NORM = 94.0
    COURT_WIDTH_NORM = 50.0
    MAX_HEIGHT = 10.0
    MAX_DISTANCE = 100.0  # Approximate max court diagonal
    MAX_GAME_CLOCK = 720.0  # 12 minutes
    MAX_SHOT_CLOCK = 24.0
    
    # Add offensive players (nodes 0-4) with NORMALIZED features
    for idx, player in enumerate(offense_positions):
        team_id, player_id, x, y, z = player[0], player[1], player[2], player[3], player[4]
        
        # Normalize positions to [0, 1]
        x_norm = x / COURT_LENGTH_NORM
        y_norm = y / COURT_WIDTH_NORM
        z_norm = z / MAX_HEIGHT
        
        # Basic features
        has_ball = 1.0 if idx == player_with_ball else 0.0
        is_offense = 1.0
        
        # Normalize distance to rim
        dist_to_rim = np.sqrt((x - target_rim[0])**2 + (y - target_rim[1])**2)
        dist_to_rim_norm = dist_to_rim / MAX_DISTANCE
        
        # Normalize distance to nearest defender
        min_def_dist = float('inf')
        for def_player in defense_positions:
            def_x, def_y = def_player[2], def_player[3]
            dist = np.sqrt((x - def_x)**2 + (y - def_y)**2)
            min_def_dist = min(min_def_dist, dist)
        min_def_dist_norm = min_def_dist / MAX_DISTANCE
        
        # Normalize angle to [-1, 1]
        angle_to_basket = calculate_angle_to_basket(x, y, target_rim)
        angle_norm = angle_to_basket / np.pi
        
        # Normalize distance to 3-point line
        dist_to_3pt = calculate_distance_to_three_point_line(x, y, target_rim)
        dist_to_3pt_norm = np.tanh(dist_to_3pt / 10.0)  # Use tanh for unbounded values
        
        # Normalize nearby defenders count
        num_nearby_def = count_nearby_defenders(x, y, defense_positions)
        num_nearby_def_norm = num_nearby_def / 5.0  # Max 5 defenders
        
        # Normalize clock features
        quarter_norm = quarter / 4.0
        game_clock_norm = game_clock_seconds / MAX_GAME_CLOCK
        shot_clock_norm = shot_clock / MAX_SHOT_CLOCK
        
        # Get position encoding and player stats (already normalized)
        position_str = player_positions_map.get(player_id, 'G')
        position_encoding = encode_position(position_str)
        player_stats_features = get_player_stats_features(player_id)
        
        # Combine NORMALIZED features (40 total)
        features = [x_norm, y_norm, z_norm, has_ball, is_offense, dist_to_rim_norm, min_def_dist_norm,
                   angle_norm, dist_to_3pt_norm, num_nearby_def_norm,
                   quarter_norm, game_clock_norm, shot_clock_norm] + position_encoding + player_stats_features
        node_features.append(features)

    # Add defensive players (nodes 5-9) with NORMALIZED features
    for idx, player in enumerate(defense_positions):
        team_id, player_id, x, y, z = player[0], player[1], player[2], player[3], player[4]
        
        # Normalize positions
        x_norm = x / COURT_LENGTH_NORM
        y_norm = y / COURT_WIDTH_NORM
        z_norm = z / MAX_HEIGHT
        
        has_ball = 0.0
        is_offense = 0.0
        
        # Normalize distance to rim
        dist_to_rim = np.sqrt((x - target_rim[0])**2 + (y - target_rim[1])**2)
        dist_to_rim_norm = dist_to_rim / MAX_DISTANCE
        
        # Normalize distance to ball handler
        if ball_handler_pos is not None:
            dist_to_ball_handler = np.sqrt((x - ball_handler_pos[0])**2 + (y - ball_handler_pos[1])**2)
        else:
            dist_to_ball_handler = np.sqrt((x - ball_x)**2 + (y - ball_y)**2)
        dist_to_ball_handler_norm = dist_to_ball_handler / MAX_DISTANCE
        
        # Normalize angle
        angle_to_basket = calculate_angle_to_basket(x, y, target_rim)
        angle_norm = angle_to_basket / np.pi
        
        # Normalize distance to 3pt
        dist_to_3pt = calculate_distance_to_three_point_line(x, y, target_rim)
        dist_to_3pt_norm = np.tanh(dist_to_3pt / 10.0)
        
        num_nearby_def_norm = 0.0
        
        # Normalize clock
        quarter_norm = quarter / 4.0
        game_clock_norm = game_clock_seconds / MAX_GAME_CLOCK
        shot_clock_norm = shot_clock / MAX_SHOT_CLOCK
        
        position_str = player_positions_map.get(player_id, 'G')
        position_encoding = encode_position(position_str)
        player_stats_features = get_player_stats_features(player_id)
        
        # Combine NORMALIZED features (40 total)
        features = [x_norm, y_norm, z_norm, has_ball, is_offense, dist_to_rim_norm, dist_to_ball_handler_norm,
                   angle_norm, dist_to_3pt_norm, num_nearby_def_norm,
                   quarter_norm, game_clock_norm, shot_clock_norm] + position_encoding + player_stats_features
        node_features.append(features)
    
    # Convert to tensor
    x = torch.tensor(node_features, dtype=torch.float)
    # Verify node feature dimensions    
    assert x.shape[0] == 10, f"Expected 10 nodes (5 offense + 5 defense), got {x.shape[0]}"
    assert x.shape[1] == 40, f"Expected 40 node features (3 pos + 7 basic + 3 clock + 5 position + 22 stats), got {x.shape[1]}"
  
    # Build edge index for complete directed graph
    edge_index = []
    for i in range(10):
        for j in range(10):
            if i != j:  # No self-loops
                edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Build edge attributes with edge type encoding
    edge_attr = []
    
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
                
            x_i, y_i = node_features[i][0], node_features[i][1]
            x_j, y_j = node_features[j][0], node_features[j][1]
            
            # Relative position
            x_rel = x_i - x_j
            y_rel = y_i - y_j
            distance = np.sqrt(x_rel**2 + y_rel**2)
            
            # Edge angle/bearing
            edge_angle = calculate_edge_angle(x_i, y_i, x_j, y_j)
            
            # Determine edge type
            is_i_offense = (i < 5)
            is_j_offense = (j < 5)
            
            if is_i_offense and is_j_offense:
                # Offense-Offense
                rel_type = [1.0, 0.0, 0.0]
            elif is_i_offense and not is_j_offense:
                # Offense-Defense
                rel_type = [0.0, 1.0, 0.0]
            elif not is_i_offense and is_j_offense:
                # Defense-Offense (also count as O-D)
                rel_type = [0.0, 1.0, 0.0]
            else:
                # Defense-Defense
                rel_type = [0.0, 0.0, 1.0]
            
            edge_attr.append([x_rel, y_rel, distance, edge_angle] + rel_type)
    
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Verify edge feature dimensions
    assert edge_attr.shape[0] == 90, f"Expected 90 edges (10 nodes * 9 connections each), got {edge_attr.shape[0]}"
    assert edge_attr.shape[1] == 7, f"Expected 7 edge features (2 rel pos + 1 distance + 1 angle + 3 edge type), got {edge_attr.shape[1]}"
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
   
    return data


import json
import torch
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
from pathlib import Path

# Add local_tests to path for imports
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from scrape_player_stats import build_player_stats_for_game

# Player stats for current game (set by process_single_game)
PLAYER_STATS = None

def set_player_stats(stats_dict):
    """Set the player stats dict directly (used when building from shot data)."""
    global PLAYER_STATS
    PLAYER_STATS = stats_dict
    return PLAYER_STATS

"""
Build a graph representation from shot data.

Each shot has:
- 5 offensive players with positions (x, y, z)
- 5 defensive players with positions (x, y, z)
- Ball position (x, y, z)
- Previous moments: list of historical snapshots with same structure

Graph structure:
- 11 nodes per timestep (5 offense + 5 defense + 1 ball) x num_timesteps
- Node features: [x_pos, y_pos, z_pos, has_ball, is_offense, is_player_node, dist_to_rim, dist_to_ball_handler, 
                  angle_to_basket, dist_to_3pt_line, num_nearby_defenders, quarter, game_clock, shot_clock,
                  timestamp, position_encoding, player_stats]
  - x_pos, y_pos, z_pos: coordinates (3 features)
  - has_ball: 1 if player has possession, 0 otherwise (1 feature)
  - is_offense: 1 if offensive player, 0 if defensive/ball (1 feature)
  - is_player_node: 1 if player node, 0 if ball node (1 feature)
  - dist_to_rim: Euclidean distance to the rim (1 feature)
  - dist_to_ball_handler: For offense: distance to nearest defender; For defense: distance to ball handler; For ball: 0 (1 feature)
  - angle_to_basket: Angle to basket in radians (1 feature)
  - dist_to_3pt_line: Distance to 3-point arc, negative=inside, positive=outside (1 feature)
  - num_nearby_defenders: Count of defenders within 6 feet (1 feature)
  - quarter: quarter of the game (1-4) (1 feature)
  - game_clock: game clock in seconds (0-720) (1 feature)
  - shot_clock: shot clock in seconds (0-24) (1 feature)
  - timestamp: temporal identifier (0=current, 1=previous, 2=two steps back) (1 feature)
  - position_encoding: one-hot encoding of player position (G, F, C, G-F, F-C); zeros for ball (5 features)
  - player_stats: season shooting stats - 21 features (zeros for ball nodes):
      age, fg_pct, avg_dist, pct_fga_fg2a, pct_fga_00_03, pct_fga_03_10, pct_fga_10_16, pct_fga_16_xx,
      pct_fga_fg3a, fg2_pct, fg_pct_00_03, fg_pct_03_10, fg_pct_10_16, fg_pct_16_xx, fg3_pct,
      pct_ast_fg2, pct_ast_fg3, pct_fga_dunk, pct_fga_corner3, fg_pct_corner3, games
  Total: 41 features per node
- Edges: 
  - Within each timestep: complete graph (11 nodes = 110 directed edges)
  - Temporal edges: each player/ball connected to themselves in adjacent timesteps
- Edge features: [x_rel, y_rel, euclidean_distance, edge_angle, rel_type_OO, rel_type_OD, rel_type_DD, rel_type_PB, rel_type_TEMPORAL]
  - x_rel, y_rel: relative position (2 features)
  - euclidean_distance: distance between nodes (1 feature)
  - edge_angle: angle/bearing from source to target node in radians (1 feature)
  - rel_type_*: one-hot encoding of edge type (Offense-Offense, Offense-Defense, Defense-Defense, Player-Ball, Temporal) (5 features)
  Total: 9 features per edge
"""

# Default rim position (used as fallback if target_rim not in shot data)
RIGHT_RIM = (88.75, 25.0)
# 3-point line distance from basket (NBA: ~23.75 feet)
THREE_POINT_DISTANCE = 23.75
# Defensive contest distance threshold (feet)
CONTEST_DISTANCE = 6.0

# Position encoding mapping
POSITION_MAP = {
    'G': 0,
    'F': 1,
    'C': 2,
    'G-F': 3,
    'F-G': 3,  # Treat F-G same as G-F
    'F-C': 4,
}

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
    Returns a list of 21 stat features, or None if player not found.
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
            stats.get('pct_fga_corner3', 0),  # % corner 3s
            stats.get('fg_pct_corner3', 0),  # Corner 3 FG%
            stats.get('games', 0) / 82.0,  # Normalize by season length
        ]
        return features
    else:
        # Return None if player not found (e.g., rookie, two-way player)
        return None


def build_graph_from_shot(shot_data):
    """
    Build a PyTorch Geometric graph from a single shot with enriched features.
    Includes temporal nodes from previous moments connected via temporal edges.
    Includes ball as its own node connected to all players within each timestep.
    
    Node features (41 total):
    - x, y, z: position (3)
    - has_ball: binary flag (1)
    - is_offense: binary flag (1)
    - is_player_node: 1 for player, 0 for ball (1)
    - dist_to_rim: distance to rim (1)
    - dist_to_ball_handler: For offense: distance to nearest defender; For defense: distance to ball handler; For ball: 0 (1)
    - angle_to_basket: angle to basket in radians (1)
    - dist_to_3pt_line: distance to 3-point line, negative=inside (1)
    - num_nearby_defenders: count of defenders within 6 feet (1)
    - quarter: quarter of the game (1-4) (1)
    - game_clock: game clock in seconds (0-720) (1)
    - shot_clock: shot clock in seconds (0-24) (1)
    - timestamp: temporal identifier (0=current, 1=previous, 2=two steps back) (1)
    - position_encoding: one-hot for G, F, C, G-F, F-C; zeros for ball (5)
    - player_stats: season shooting statistics (21) - zeros for ball nodes
    
    Edge features (9 total):
    - x_rel, y_rel: relative position (2)
    - euclidean_distance: distance between nodes (1)
    - edge_angle: bearing from source to target in radians (1)
    - rel_type_OO, rel_type_OD, rel_type_DD, rel_type_PB, rel_type_TEMPORAL: one-hot edge type (5)
    
    Returns:
        tuple: (graph, skip_reason) where:
            - graph is the PyTorch Geometric Data object or None if failed
            - skip_reason is None on success, or dict with reason info:
              Current moment issues:
              - {'reason': 'missing_field', 'field': field_name}
              - {'reason': 'invalid_player_count'}
              - {'reason': 'missing_player_stats', 'player_id': id, 'player_name': name}
              - {'reason': 'shooter_not_found', 'shooter_id': id}
              Previous moment issues (skipping individual moments would break temporal graph structure):
              - {'reason': 'prev_moment_missing_field', 'timestep': idx, 'fields': [field_names]}
              - {'reason': 'prev_moment_none_field', 'timestep': idx, 'fields': [field_names]}
              - {'reason': 'prev_moment_invalid_player_count', 'timestep': idx, 'offense_count': n, 'defense_count': n}
              - {'reason': 'prev_moment_shooter_not_found', 'timestep': idx, 'shooter_id': id}
    """
    # Required fields - return early if any are missing
    required_fields = [
        'offense_position', 'defense_position', 'ball_position',
        'player_positions', 'player_names', 'target_rim',
        'quarter', 'game_clock', 'shot_clock', 'shooter_id', 'previous_moments'
    ]
    
    for field in required_fields:
        if field not in shot_data or shot_data[field] is None:
            return None, {'reason': 'missing_field', 'field': field}
    
    # Extract fields (all validated as present)
    offense_positions = shot_data['offense_position']
    defense_positions = shot_data['defense_position']
    player_positions_map = shot_data['player_positions']
    player_names_map = shot_data['player_names']
    previous_moments = shot_data['previous_moments']
    target_rim = tuple(shot_data['target_rim'])
    quarter = shot_data['quarter']
    game_clock_seconds = shot_data['game_clock']
    shot_clock = shot_data['shot_clock']
    shooter_id = shot_data['shooter_id']

    # Validate each team has 5 players
    if len(offense_positions) != 5 or len(defense_positions) != 5:
        return None, {'reason': 'invalid_player_count'}
    
    # Find shooter index in offense_positions using shooter_id
    player_with_ball = -1
    for idx, player in enumerate(offense_positions):
        if player[1] == shooter_id:
            player_with_ball = idx
            break
    
    if player_with_ball == -1:
        return None, {'reason': 'shooter_not_found', 'shooter_id': shooter_id}
    
    # Build node features with enriched information
    node_features = []
    # Track player IDs for temporal edge matching
    # node_player_ids[i] = player_id of node i
    node_player_ids = []
    # node_timestep[i] = timestep of node i (0 = current, 1 = first previous, etc.)
    node_timesteps = []
    
    # Get ball handler position for defender distance calculations
    ball_handler_pos = (offense_positions[player_with_ball][2], offense_positions[player_with_ball][3])

    # Add offensive players (nodes 0-4) - CURRENT MOMENT (timestep 0)
    for idx, player in enumerate(offense_positions):
        team_id, player_id, x, y, z = player[0], player[1], player[2], player[3], player[4]
        
        # Player stats features - check if player has stats
        player_stats_features = get_player_stats_features(player_id)
        if player_stats_features is None:
            player_name = player_names_map.get(str(player_id), f"Unknown (ID: {player_id})")
            return None, {'reason': 'missing_player_stats', 'player_id': player_id, 'player_name': player_name}
        
        # Basic features (shooter_id determines who has the ball)
        has_ball = 1.0 if idx == player_with_ball else 0.0
        is_offense = 1.0
        is_player_node = 1.0  # This is a player node
        
        # Distance to rim
        dist_to_rim = np.sqrt((x - target_rim[0])**2 + (y - target_rim[1])**2)
        
        # Distance to nearest defender
        min_def_dist = float('inf')
        for def_player in defense_positions:
            def_x, def_y = def_player[2], def_player[3]
            dist = np.sqrt((x - def_x)**2 + (y - def_y)**2)
            min_def_dist = min(min_def_dist, dist)
        
        # Angle to basket
        angle_to_basket = calculate_angle_to_basket(x, y, target_rim)
        
        # Distance to 3-point line
        dist_to_3pt = calculate_distance_to_three_point_line(x, y, target_rim)
        
        # Number of nearby defenders (within 6 feet)
        num_nearby_def = count_nearby_defenders(x, y, defense_positions)
        
        # Position encoding (convert player_id to string for lookup)
        position_str = player_positions_map.get(str(player_id), 'G')
        position_encoding = encode_position(position_str)
        
        # Timestamp feature (0 = current moment)
        timestamp = 0.0

        # Combine all features (41 total)
        features = [x, y, z, has_ball, is_offense, is_player_node, dist_to_rim, min_def_dist, 
                   angle_to_basket, dist_to_3pt, num_nearby_def,
                   quarter, game_clock_seconds, shot_clock, timestamp] + position_encoding + player_stats_features
        node_features.append(features)
        node_player_ids.append(player_id)
        node_timesteps.append(0)

    # Add defensive players (nodes 5-9) - CURRENT MOMENT (timestep 0)
    for idx, player in enumerate(defense_positions):
        team_id, player_id, x, y, z = player[0], player[1], player[2], player[3], player[4]
        
        # Player stats features - check if player has stats
        player_stats_features = get_player_stats_features(player_id)
        if player_stats_features is None:
            player_name = player_names_map.get(str(player_id), f"Unknown (ID: {player_id})")
            return None, {'reason': 'missing_player_stats', 'player_id': player_id, 'player_name': player_name}
        
        # Basic features
        has_ball = 0.0
        is_offense = 0.0
        is_player_node = 1.0  # This is a player node
        
        # Distance to rim
        dist_to_rim = np.sqrt((x - target_rim[0])**2 + (y - target_rim[1])**2)
        
        # Distance to ball handler (shooter)
        dist_to_ball_handler = np.sqrt((x - ball_handler_pos[0])**2 + (y - ball_handler_pos[1])**2)
        
        # Angle to basket
        angle_to_basket = calculate_angle_to_basket(x, y, target_rim)
        
        # Distance to 3-point line
        dist_to_3pt = calculate_distance_to_three_point_line(x, y, target_rim)
        
        # Number of nearby defenders (0 for defenders, but keep feature for consistency)
        num_nearby_def = 0.0
        
        # Position encoding (convert player_id to string for lookup)
        position_str = player_positions_map.get(str(player_id), 'G')
        position_encoding = encode_position(position_str)
        
        # Timestamp feature (0 = current moment)
        timestamp = 0.0

        # Combine all features (41 total)
        features = [x, y, z, has_ball, is_offense, is_player_node, dist_to_rim, dist_to_ball_handler,
                   angle_to_basket, dist_to_3pt, num_nearby_def,
                   quarter, game_clock_seconds, shot_clock, timestamp] + position_encoding + player_stats_features
        node_features.append(features)
        node_player_ids.append(player_id)
        node_timesteps.append(0)
    
    # Add BALL NODE for current moment (node 10)
    ball_pos = shot_data['ball_position']
    ball_x, ball_y, ball_z = ball_pos[0], ball_pos[1], ball_pos[2]
    
    # Ball features: x, y, z position + padded features
    ball_has_ball = 0.0  # Ball doesn't "have" itself in the traditional sense
    ball_is_offense = 0.0  # Ball is neither offense nor defense
    ball_is_player_node = 0.0  # This is NOT a player node
    ball_dist_to_rim = np.sqrt((ball_x - target_rim[0])**2 + (ball_y - target_rim[1])**2)
    ball_dist_metric = 0.0  # Not applicable for ball
    ball_angle_to_basket = calculate_angle_to_basket(ball_x, ball_y, target_rim)
    ball_dist_to_3pt = calculate_distance_to_three_point_line(ball_x, ball_y, target_rim)
    ball_nearby_def = 0.0  # Not applicable for ball
    ball_timestamp = 0.0
    ball_position_encoding = [0.0] * 5  # No position for ball
    ball_stats = [0.0] * 21  # No player stats for ball
    
    ball_features = [ball_x, ball_y, ball_z, ball_has_ball, ball_is_offense, ball_is_player_node,
                     ball_dist_to_rim, ball_dist_metric, ball_angle_to_basket, ball_dist_to_3pt,
                     ball_nearby_def, quarter, game_clock_seconds, shot_clock, ball_timestamp] + ball_position_encoding + ball_stats
    node_features.append(ball_features)
    node_player_ids.append(-1)  # Special ID for ball
    node_timesteps.append(0)
    
    # Process PREVIOUS MOMENTS (timesteps 1, 2, ...)
    for timestep_idx, prev_moment in enumerate(previous_moments, start=1):
        # Return None if required fields are missing in previous moment
        # We cannot skip individual moments as this would create gaps in the temporal graph structure
        if ('offense_position' not in prev_moment or 'defense_position' not in prev_moment or
            'ball_position' not in prev_moment or 'quarter' not in prev_moment or 
            'game_clock' not in prev_moment or 'shot_clock' not in prev_moment):
            missing = [f for f in ['offense_position', 'defense_position', 'ball_position', 'quarter', 'game_clock', 'shot_clock'] 
                       if f not in prev_moment]
            return None, {'reason': 'prev_moment_missing_field', 'timestep': timestep_idx, 'fields': missing}
        
        prev_offense = prev_moment['offense_position']
        prev_defense = prev_moment['defense_position']
        prev_ball_position = prev_moment['ball_position']
        prev_quarter = prev_moment['quarter']
        prev_game_clock = prev_moment['game_clock']
        prev_shot_clock = prev_moment['shot_clock']
        
        # Return None if any required field is None
        if (prev_offense is None or prev_defense is None or prev_ball_position is None or
            prev_quarter is None or prev_game_clock is None or prev_shot_clock is None):
            none_fields = [f for f, v in [('offense_position', prev_offense), ('defense_position', prev_defense),
                                           ('ball_position', prev_ball_position), ('quarter', prev_quarter), 
                                           ('game_clock', prev_game_clock), ('shot_clock', prev_shot_clock)] if v is None]
            return None, {'reason': 'prev_moment_none_field', 'timestep': timestep_idx, 'fields': none_fields}
        
        # Return None if previous moment doesn't have 5 players per team
        if len(prev_offense) != 5 or len(prev_defense) != 5:
            return None, {'reason': 'prev_moment_invalid_player_count', 'timestep': timestep_idx,
                          'offense_count': len(prev_offense), 'defense_count': len(prev_defense)}
        
        # Find shooter index in previous moment using shooter_id
        prev_player_with_ball = -1
        for idx, player in enumerate(prev_offense):
            if player[1] == shooter_id:
                prev_player_with_ball = idx
                break
        
        # Return None if shooter not found in previous moment
        if prev_player_with_ball == -1:
            return None, {'reason': 'prev_moment_shooter_not_found', 'timestep': timestep_idx, 'shooter_id': shooter_id}
        
        # Ball handler position for defender distance
        prev_ball_handler_pos = (prev_offense[prev_player_with_ball][2], prev_offense[prev_player_with_ball][3])
        
        # Add offensive players for this previous moment
        for idx, player in enumerate(prev_offense):
            team_id, player_id, x, y, z = player[0], player[1], player[2], player[3], player[4]
            
            # Basic features (same logic as current moment)
            has_ball = 1.0 if idx == prev_player_with_ball else 0.0
            is_offense = 1.0
            is_player_node = 1.0  # This is a player node
            
            dist_to_rim = np.sqrt((x - target_rim[0])**2 + (y - target_rim[1])**2)
            
            # Distance to nearest defender in this previous moment
            min_def_dist = float('inf')
            for def_player in prev_defense:
                def_x, def_y = def_player[2], def_player[3]
                dist = np.sqrt((x - def_x)**2 + (y - def_y)**2)
                min_def_dist = min(min_def_dist, dist)
            
            angle_to_basket = calculate_angle_to_basket(x, y, target_rim)
            dist_to_3pt = calculate_distance_to_three_point_line(x, y, target_rim)
            num_nearby_def = count_nearby_defenders(x, y, prev_defense)
            
            position_str = player_positions_map.get(str(player_id), 'G')
            position_encoding = encode_position(position_str)

            # Timestamp feature (1, 2, etc. for previous moments)
            timestamp = float(timestep_idx)
            
            # Include the player stats for previous moments
            features = [x, y, z, has_ball, is_offense, is_player_node, dist_to_rim, min_def_dist,
                        angle_to_basket, dist_to_3pt, num_nearby_def,
                        prev_quarter, prev_game_clock, prev_shot_clock, timestamp] + position_encoding + player_stats_features
            node_features.append(features)
            node_player_ids.append(player_id)
            node_timesteps.append(timestep_idx)
        
        # Add defensive players for this previous moment
        for idx, player in enumerate(prev_defense):
            team_id, player_id, x, y, z = player[0], player[1], player[2], player[3], player[4]
            
            has_ball = 0.0
            is_offense = 0.0
            is_player_node = 1.0  # This is a player node
            
            dist_to_rim = np.sqrt((x - target_rim[0])**2 + (y - target_rim[1])**2)
            
            # Distance to ball handler (shooter) in this previous moment
            dist_to_ball_handler = np.sqrt((x - prev_ball_handler_pos[0])**2 + (y - prev_ball_handler_pos[1])**2)
            
            angle_to_basket = calculate_angle_to_basket(x, y, target_rim)
            dist_to_3pt = calculate_distance_to_three_point_line(x, y, target_rim)
            num_nearby_def = 0.0
            
            position_str = player_positions_map.get(str(player_id), 'G')
            position_encoding = encode_position(position_str)
            
             # Timestamp feature (1, 2, etc. for previous moments)
            timestamp = float(timestep_idx)
            
            # Include the player stats for previous moments
            features = [x, y, z, has_ball, is_offense, is_player_node, dist_to_rim, dist_to_ball_handler,
                        angle_to_basket, dist_to_3pt, num_nearby_def,
                        prev_quarter, prev_game_clock, prev_shot_clock, timestamp] + position_encoding + player_stats_features
            node_features.append(features)
            node_player_ids.append(player_id)
            node_timesteps.append(timestep_idx)
        
        # Add BALL NODE for this previous moment
        # Ball position is validated as not None above, so we can safely use it
        prev_ball_x, prev_ball_y, prev_ball_z = prev_ball_position[0], prev_ball_position[1], prev_ball_position[2]
        
        # Ball features for previous moment
        prev_ball_has_ball = 0.0
        prev_ball_is_offense = 0.0
        prev_ball_is_player_node = 0.0  # This is NOT a player node
        prev_ball_dist_to_rim = np.sqrt((prev_ball_x - target_rim[0])**2 + (prev_ball_y - target_rim[1])**2)
        prev_ball_dist_metric = 0.0
        prev_ball_angle_to_basket = calculate_angle_to_basket(prev_ball_x, prev_ball_y, target_rim)
        prev_ball_dist_to_3pt = calculate_distance_to_three_point_line(prev_ball_x, prev_ball_y, target_rim)
        prev_ball_nearby_def = 0.0
        prev_ball_timestamp = float(timestep_idx)
        prev_ball_position_encoding = [0.0] * 5
        prev_ball_stats = [0.0] * 21
        
        prev_ball_features = [prev_ball_x, prev_ball_y, prev_ball_z, prev_ball_has_ball, prev_ball_is_offense, prev_ball_is_player_node,
                              prev_ball_dist_to_rim, prev_ball_dist_metric, prev_ball_angle_to_basket, prev_ball_dist_to_3pt,
                              prev_ball_nearby_def, prev_quarter, prev_game_clock, prev_shot_clock, prev_ball_timestamp] + prev_ball_position_encoding + prev_ball_stats
        node_features.append(prev_ball_features)
        node_player_ids.append(-1)  # Special ID for ball
        node_timesteps.append(timestep_idx)
        
    
    # Convert to tensor
    num_nodes = len(node_features)
    num_timesteps = 1 + len(previous_moments)
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Build edge index for complete graph WITHIN each timestep + bidirectional temporal edges
    edge_index = []
    edge_attr = []
    
    # Edges WITHIN each timestep (fully connected within each 11-node group)
    for t in range(num_timesteps):
        base_idx = t * 11  # Starting node index for this timestep (now 11 nodes per timestep)
        
        for i in range(11):  # Now 11 nodes per timestep
            for j in range(11):  # Now 11 nodes per timestep
                if i != j:  # No self-loops
                    node_i = base_idx + i
                    node_j = base_idx + j
                    
                    # Skip if nodes don't exist (in case previous moment was skipped)
                    if node_i >= num_nodes or node_j >= num_nodes:
                        continue
                    
                    edge_index.append([node_i, node_j])
                    
                    x_i, y_i = node_features[node_i][0], node_features[node_i][1]
                    x_j, y_j = node_features[node_j][0], node_features[node_j][1]
                    
                    # Relative position
                    x_rel = x_i - x_j
                    y_rel = y_i - y_j
                    distance = np.sqrt(x_rel**2 + y_rel**2)
                    
                    # Edge angle/bearing
                    edge_angle = calculate_edge_angle(x_i, y_i, x_j, y_j)
                    
                    # Determine edge type based on node types
                    # Node 0-4: offense, 5-9: defense, 10: ball
                    is_i_player = node_features[node_i][5] > 0.5  # is_player_node flag
                    is_j_player = node_features[node_j][5] > 0.5  # is_player_node flag
                    
                    if is_i_player and is_j_player:
                        # Both are players, determine O-O, O-D, or D-D
                        is_i_offense = (i < 5)
                        is_j_offense = (j < 5)
                        
                        if is_i_offense and is_j_offense:
                            rel_type = [1.0, 0.0, 0.0, 0.0, 0.0]  # O-O
                        elif is_i_offense != is_j_offense:
                            rel_type = [0.0, 1.0, 0.0, 0.0, 0.0]  # O-D
                        else:
                            rel_type = [0.0, 0.0, 1.0, 0.0, 0.0]  # D-D
                    else:
                        # At least one is ball, so it's Player-Ball edge
                        rel_type = [0.0, 0.0, 0.0, 1.0, 0.0]  # P-B (Player-Ball)
                    
                    edge_attr.append([x_rel, y_rel, distance, edge_angle] + rel_type)
        
    # TEMPORAL edges: connect players/ball to themselves across adjacent timesteps
    # Connect from current to previous: node at t=0 connects to node at t=1    
    for t in range(num_timesteps - 1):
        curr_base = t * 11  # Now 11 nodes per timestep
        next_base = (t + 1) * 11  # Now 11 nodes per timestep
        
        for i in range(11):  # Now 11 nodes per timestep (including ball at i=10)
            curr_node = curr_base + i
            next_node = next_base + i
            
            # Skip if nodes don't exist
            if curr_node >= num_nodes or next_node >= num_nodes:
                continue
            
            # Only connect if same player/ball (by player_id, -1 for ball)
            if node_player_ids[curr_node] == node_player_ids[next_node]:
                # Bidirectional temporal edges between adjacent timesteps
                x_i, y_i = node_features[curr_node][0], node_features[curr_node][1]
                x_j, y_j = node_features[next_node][0], node_features[next_node][1]
                
                # Temporal edge type
                rel_type = [0.0, 0.0, 0.0, 0.0, 1.0]  # TEMPORAL
                
                # Edge from current to previous (t -> t+1)
                edge_index.append([curr_node, next_node])
                x_rel = x_i - x_j
                y_rel = y_i - y_j
                distance = np.sqrt(x_rel**2 + y_rel**2)
                edge_angle = calculate_edge_angle(x_i, y_i, x_j, y_j)
                edge_attr.append([x_rel, y_rel, distance, edge_angle] + rel_type)
                
                # Reverse edge from previous to current (t+1 -> t)
                edge_index.append([next_node, curr_node])
                x_rel_rev = x_j - x_i
                y_rel_rev = y_j - y_i
                edge_angle_rev = calculate_edge_angle(x_j, y_j, x_i, y_i)
                edge_attr.append([x_rel_rev, y_rel_rev, distance, edge_angle_rev] + rel_type)
    
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Store temporal info as metadata
    data.num_timesteps = num_timesteps
    data.nodes_per_timestep = 11  # Updated to 11 (10 players + 1 ball)
    
    # Validate graph structure
    assert data.x.shape[1] == 41, f"Expected 41 node features, got {data.x.shape[1]}"
    assert data.edge_attr.shape[1] == 9, f"Expected 9 edge features, got {data.edge_attr.shape[1]}"
    assert data.num_nodes == num_timesteps * 11, f"Expected {num_timesteps * 11} nodes, got {data.num_nodes}"
    
    # Validate node feature ranges
    assert torch.all((data.x[:, 5] == 0.0) | (data.x[:, 5] == 1.0)), "is_player_node flag must be 0 or 1"
    assert torch.sum(data.x[:, 5] == 0.0) == num_timesteps, f"Expected {num_timesteps} ball nodes, got {torch.sum(data.x[:, 5] == 0.0)}"
    assert torch.sum(data.x[:, 5] == 1.0) == num_timesteps * 10, f"Expected {num_timesteps * 10} player nodes, got {torch.sum(data.x[:, 5] == 1.0)}"
    
    # Validate edge type one-hot encoding (indices 4-8)
    edge_type_sum = torch.sum(data.edge_attr[:, 4:9], dim=1)
    assert torch.allclose(edge_type_sum, torch.ones_like(edge_type_sum)), "Each edge must have exactly one type"

    return data, None

def process_single_game(shot_file, csv_path, output_dir):
    """
    Process a single game's shot file and save graphs to output directory.
    
    This function is called from a Colab cell for each game file.
    It calls scrape_player_stats.py once to get player stats for this game.
    
    Args:
        shot_file: Path to the shots JSON file (e.g., "NBA_GNN_files/shots_data/shots_0021500001.json")
        csv_path: Path to the player shooting stats CSV
        output_dir: Directory to save graph .pt files
    
    Returns:
        dict: Results with keys:
            - 'game_id': The game ID
            - 'num_graphs': Number of graphs built
            - 'num_made': Number of made shots
            - 'num_missed': Number of missed shots
            - 'skip_stats': Dict of skip reason counts
            - 'skip_details': List of dicts with detailed skip information
            - 'unmatched_players': List of dicts with unmatched player details
            - 'output_path': Path to saved .pt file (or None if no graphs)
    """
    # Initialize result
    result = {
        'game_id': None,
        'num_graphs': 0,
        'num_made': 0,
        'num_missed': 0,
        'skip_stats': {
            'missing_field': 0,
            'invalid_player_count': 0,
            'missing_player_stats': 0,
            'shooter_not_found': 0,
            'prev_moment_missing_field': 0,
            'prev_moment_none_field': 0,
            'prev_moment_invalid_player_count': 0,
            'prev_moment_shooter_not_found': 0,
        },
        'skip_details': [],  # Store detailed skip information
        'unmatched_players': [],
        'output_path': None
    }
    
    # Load shot data
    with open(shot_file, 'r') as f:
        shots = json.load(f)
    
    if not shots:
        print(f"  ⚠️  Empty shots file: {shot_file}")
        return result
    
    # Get player stats for this game (calls scrape_player_stats.py)
    first_shot = shots[0]
    player_stats, unmatched_players, game_id = build_player_stats_for_game(
        first_shot=first_shot,
        csv_path=csv_path
    )
    
    result['game_id'] = game_id
    result['unmatched_players'] = unmatched_players  # This now contains detailed info
    
    if not player_stats:
        print(f"  ⚠️  Failed to build player stats for game {game_id}")
        return result
    
    # Set player stats for graph building
    set_player_stats(player_stats)
    
    # Build graphs for all shots
    graphs = []
    labels = []
    
    for shot_idx, shot in enumerate(shots):
        graph, skip_reason = build_graph_from_shot(shot)
        
        if graph is not None:
            # Add label
            graph.y = torch.tensor([1 if shot['made'] else 0], dtype=torch.long)
            
            # Add metadata
            graph.shooter_id = shot['shooter_id']
            graph.game_id = shot.get('game_id')
            graph.quarter = shot['quarter']
            graph.game_clock = shot['game_clock']
            graph.shot_clock = shot['shot_clock']
            
            # Player info (only for first 10 players per timestep, excluding ball nodes)
            player_names_map = shot['player_names']
            offense_positions = shot['offense_position']
            defense_positions = shot['defense_position']
            node_player_ids = [p[1] for p in offense_positions] + [p[1] for p in defense_positions]
            node_player_names = [player_names_map.get(str(pid), f"Unknown ({pid})") for pid in node_player_ids]
            graph.player_ids = node_player_ids
            graph.player_names = node_player_names
            
            graphs.append(graph)
            labels.append(1 if shot['made'] else 0)
        elif skip_reason is not None:
            reason = skip_reason['reason']
            result['skip_stats'][reason] = result['skip_stats'].get(reason, 0) + 1
            # Store detailed skip info
            skip_reason['shot_index'] = shot_idx
            result['skip_details'].append(skip_reason)
    
    result['num_graphs'] = len(graphs)
    result['num_made'] = sum(labels)
    result['num_missed'] = len(labels) - sum(labels)
    
    if graphs:
        # Save graphs
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"graphs_{game_id}.pt")
        
        torch.save({
            'graphs': graphs,
            'labels': torch.tensor(labels, dtype=torch.long),
            'num_graphs': len(graphs),
            'num_node_features': 41,  
            'num_edge_features': 9,   
        }, output_path)
        
        result['output_path'] = output_path
    
    return result


def print_game_result(result):
    """Print formatted result for a single game."""
    game_id = result['game_id'] or 'Unknown'
    
    if result['num_graphs'] > 0:
        print(f"  ✅ Game {game_id}: {result['num_graphs']} graphs (made: {result['num_made']}, missed: {result['num_missed']})")
        if result['output_path']:
            print(f"     Saved to: {result['output_path']}")
    else:
        print(f"  ⚠️  Game {game_id}: No graphs built")
    
    # Report skip stats with details
    total_skipped = sum(result['skip_stats'].values())
    if total_skipped > 0:
        print(f"     Skipped {total_skipped} shots:")
        for reason, count in result['skip_stats'].items():
            if count > 0:
                print(f"       - {reason}: {count}")
        
        # Print detailed skip information (limit to first few for brevity)
        if result['skip_details']:
            print(f"     Skip details (showing first 3):")
            for detail in result['skip_details'][:3]:
                reason = detail['reason']
                shot_idx = detail.get('shot_index', '?')
                
                if reason == 'missing_field':
                    field = detail.get('field', 'unknown')
                    print(f"       Shot {shot_idx}: missing field '{field}'")
                elif reason == 'prev_moment_missing_field':
                    timestep = detail.get('timestep', '?')
                    fields = detail.get('fields', [])
                    print(f"       Shot {shot_idx} (t={timestep}): missing fields {fields}")
                elif reason == 'prev_moment_none_field':
                    timestep = detail.get('timestep', '?')
                    fields = detail.get('fields', [])
                    print(f"       Shot {shot_idx} (t={timestep}): None fields {fields}")
                elif reason == 'missing_player_stats':
                    player_id = detail.get('player_id', '?')
                    player_name = detail.get('player_name', 'Unknown')
                    print(f"       Shot {shot_idx}: missing stats for {player_name} (ID: {player_id})")
                elif reason == 'shooter_not_found':
                    shooter_id = detail.get('shooter_id', '?')
                    print(f"       Shot {shot_idx}: shooter not found (ID: {shooter_id})")
                else:
                    print(f"       Shot {shot_idx}: {reason} - {detail}")
    
    # Report unmatched players with details
    if result['unmatched_players']:
        print(f"     Unmatched players: {len(result['unmatched_players'])}")
        # Print details for first few unmatched players
        for player in result['unmatched_players'][:3]:
            orig_name = player.get('original_name', 'Unknown')
            norm_name = player.get('normalized_name', 'Unknown')
            team_abbr = player.get('team_abbr', '?')
            player_id = player.get('player_id', '?')
            similar = player.get('similar_csv_names', [])
            
            print(f"       • {orig_name} (ID: {player_id}, Team: {team_abbr})")
            print(f"         Normalized: '{norm_name}'")
            if similar:
                print(f"         Similar in CSV: {', '.join(similar[:2])}")
        
        if len(result['unmatched_players']) > 3:
            print(f"       ... and {len(result['unmatched_players']) - 3} more")

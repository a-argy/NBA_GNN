#!/usr/bin/env python3
"""
Example usage of the NBA tracking data with play-by-play context
"""

from load_with_context import load_nba_data_with_context
import math
import pandas as pd

def get_description(event):
    """Get event description, handling both home and away team events"""
    desc_home = event['event_info']['desc_home']
    desc_away = event['event_info']['desc_away']
    
    if pd.notna(desc_home):
        return desc_home
    elif pd.notna(desc_away):
        return desc_away
    else:
        return "No description"

# Load the enriched data
print("Loading data...")
events = load_nba_data_with_context()
print(f"Loaded {len(events)} events\n")

# Example 1: Show all events
print("=" * 60)
print("Example 1: All Events (First 50)")
print("=" * 60)
print(f"Total events loaded: {len(events)}")

# Event type names
event_type_names = {
    1: "MADE SHOT",
    2: "MISSED SHOT",
    3: "FREE THROW",
    4: "REBOUND",
    5: "TURNOVER",
    6: "FOUL",
    10: "JUMP BALL",
    8: "SUBSTITUTION",
    9: "TIMEOUT",
    12: "START PERIOD",
    13: "END PERIOD",
    18: "INSTANT REPLAY"
}

# Show first 50 events
for i, event in enumerate(events[:50], 1):
    event_type_code = event['event_info']['type']
    event_type_name = event_type_names.get(event_type_code, f"TYPE {event_type_code}")
    desc = get_description(event)
    print(f"{i}. [{event_type_name}] {desc}")

# Still keep shots list for later examples
shots = [e for e in events if e['event_info']['type'] in [1, 2]]  # 1=made, 2=missed

# Example 2: Calculate player movement distance for one event
print(f"\n{'=' * 60}")
print("Example 2: Player Movement Analysis")
print("=" * 60)

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Pick an event with lots of movement
sample_event = shots[0]  # First shot
print(f"Analyzing: {get_description(sample_event)}")
print(f"Duration: {len(sample_event['moments'])} moments (~{len(sample_event['moments'])/25:.1f} seconds at 25fps)")

# Calculate distance traveled for each player
if len(sample_event['moments']) > 1:
    player_distances = {}
    
    # Get first moment to initialize
    first_moment = sample_event['moments'][0]
    for player in first_moment['player_coordinates']:
        player_id = player['playerid']
        player_distances[player_id] = {
            'distance': 0.0,
            'positions': [(player['x'], player['y'])]
        }
    
    # Sum up distances across all moments
    for moment in sample_event['moments'][1:]:
        for player in moment['player_coordinates']:
            player_id = player['playerid']
            if player_id in player_distances:
                prev_pos = player_distances[player_id]['positions'][-1]
                curr_pos = (player['x'], player['y'])
                
                dist = calculate_distance(prev_pos[0], prev_pos[1], curr_pos[0], curr_pos[1])
                player_distances[player_id]['distance'] += dist
                player_distances[player_id]['positions'].append(curr_pos)
    
    # Find player who moved the most
    max_distance = 0
    most_active_player_id = None
    for player_id, data in player_distances.items():
        if data['distance'] > max_distance:
            max_distance = data['distance']
            most_active_player_id = player_id
    
    print(f"\nMost active player ID: {most_active_player_id}")
    print(f"Distance traveled: {max_distance:.2f} feet")

# Example 3: Get ball height at shot moment
print(f"\n{'=' * 60}")
print("Example 3: Ball Height Analysis")
print("=" * 60)

# For made shots, track ball height
made_shots = [e for e in events if e['event_info']['type'] == 1]
print(f"Analyzing {len(made_shots)} made shots")

for shot in made_shots[:3]:
    desc = get_description(shot)
    if shot['moments']:
        # Get ball height at last moment (near basket)
        last_moment = shot['moments'][-1]
        ball_height = last_moment['ball_coordinates']['z']
        print(f"  Shot: {desc}")
        print(f"  Ball height at final moment: {ball_height:.2f} feet")

# Example 4: Team possession statistics
print(f"\n{'=' * 60}")
print("Example 4: Team Possession Statistics")
print("=" * 60)

home_team_id = events[0]['home']['teamid']
visitor_team_id = events[0]['visitor']['teamid']

home_poss = sum(1 for e in events if e['event_info']['possession_team_id'] == home_team_id)
visitor_poss = sum(1 for e in events if e['event_info']['possession_team_id'] == visitor_team_id)

print(f"{events[0]['home']['name']}: {home_poss} possessions")
print(f"{events[0]['visitor']['name']}: {visitor_poss} possessions")

# Example 5: Event type distribution
print(f"\n{'=' * 60}")
print("Example 5: Event Type Distribution")
print("=" * 60)

event_types = {
    1: "Made Shot",
    2: "Missed Shot",
    3: "Free Throw",
    4: "Rebound",
    5: "Turnover",
    6: "Foul",
    10: "Jump Ball"
}

type_counts = {}
for event in events:
    event_type = event['event_info']['type']
    type_counts[event_type] = type_counts.get(event_type, 0) + 1

for event_type, count in sorted(type_counts.items()):
    type_name = event_types.get(event_type, f"Type {event_type}")
    print(f"{type_name}: {count}")

print(f"\n{'=' * 60}")
print("✓ Examples complete!")
print("=" * 60)


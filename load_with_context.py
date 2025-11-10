#!/usr/bin/env python3
"""
Direct loader for NBA tracking data combined with play-by-play context
"""

import json
import os
import glob
import pandas as pd
import requests
import math

# Configuration
# CHANGE BASED ON WHERE YOUR LOCAL DATA IS AT
LOCAL_DATA_DIR = "/Users/ianlasic/224W NBA_GNN/NBA_GNN/data"
PBP_URL = "https://github.com/sumitrodatta/nba-alt-awards/raw/main/Historical/PBP%20Data/2015-16_pbp.csv"

# Court dimensions (in feet)
COURT_LENGTH = 94
COURT_WIDTH = 50
# Rim positions (standard NBA: 5.25 feet from baseline, centered at 25 feet)
LEFT_RIM = (5.25, 25.0)
RIGHT_RIM = (88.75, 25.0)

def home_away_event_conversion(number):
    """Convert PERSON type to home/away"""
    if pd.isna(number):
        return None
    if int(number) == 4:
        return "home"
    elif int(number) == 5:
        return "away"
    else:
        return None

def identify_offense(row):
    """Identify which team has possession"""
    identified_offense_events = [1, 2, 3, 4, 5]
    if int(row['EVENTMSGTYPE']) in identified_offense_events:
        poss_team_id = row['PLAYER1_TEAM_ID']
    elif ("OFF.FOUL" in str(row["HOMEDESCRIPTION"])) or ("OFF.FOUL" in str(row["VISITORDESCRIPTION"])):
        poss_team_id = row['PLAYER1_TEAM_ID']
    elif int(row['EVENTMSGTYPE']) == 6:
        poss_team_id = row['PLAYER2_TEAM_ID']
    else:
        poss_team_id = None
    return poss_team_id

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def format_game_clock(seconds):
    """
    Convert game clock seconds to MM:SS format.
    
    Args:
        seconds: Time in seconds (can be float)
    
    Returns:
        String in MM:SS format (e.g., "11:38")
    """
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02d}"

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

def find_moment_at_rim(moments, target_rim, rim_height=10.0, xy_tolerance=2.0, z_tolerance=1.0):
    """
    Find the moment when the ball is at the rim position and rim height.
    
    The rim is at 10 feet height, so we look for when the ball is:
    - At approximately rim height (z ≈ 10 feet)
    - At approximately rim x,y coordinates
    
    Args:
        moments: List of moment dictionaries
        target_rim: Tuple (x, y) of rim coordinates
        rim_height: Height of rim in feet (default 10.0)
        xy_tolerance: Acceptable x,y distance from rim in feet (default 2.0)
        z_tolerance: Acceptable height difference from rim in feet (default 1.0)
    
    Returns:
        The moment when ball is closest to being at the rim, or None if not found within tolerance
    """
    closest_moment = None
    closest_score = float('inf')
    
    for moment in moments:
        ball_x = moment['ball_coordinates']['x']
        ball_y = moment['ball_coordinates']['y']
        ball_z = moment['ball_coordinates']['z']
        
        # Calculate distance from rim position (x, y)
        xy_distance = calculate_distance(ball_x, ball_y, target_rim[0], target_rim[1])
        
        # Calculate height difference from rim height
        z_diff = abs(ball_z - rim_height)
        
        # Combined score (both need to be small)
        # Weight them equally - both need to be close
        score = xy_distance + z_diff
        
        if score < closest_score:
            closest_score = score
            closest_moment = moment
    
    # Only return if we found something reasonably close
    # Check both xy and z are within tolerances
    if closest_moment:
        ball_x = closest_moment['ball_coordinates']['x']
        ball_y = closest_moment['ball_coordinates']['y']
        ball_z = closest_moment['ball_coordinates']['z']
        
        xy_distance = calculate_distance(ball_x, ball_y, target_rim[0], target_rim[1])
        z_diff = abs(ball_z - rim_height)
        
        if xy_distance <= xy_tolerance and z_diff <= z_tolerance:
            return closest_moment
    
    return None

def load_nba_data_with_context(pbp_cache_file="pbp_cache.csv"):
    """
    Load local NBA tracking data and combine with play-by-play context
    
    Returns:
        list: List of enriched events with tracking data and contextual information
    """
    # Download or load cached PBP data
    if os.path.exists(pbp_cache_file):
        print(f"Loading cached PBP data from {pbp_cache_file}...")
        pbp = pd.read_csv(pbp_cache_file)
    else:
        print(f"Downloading PBP data from {PBP_URL}...")
        pbp = pd.read_csv(PBP_URL)
        pbp.to_csv(pbp_cache_file, index=False)
        print(f"Cached PBP data to {pbp_cache_file}")
    
    # Find all local JSON files
    json_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.json"))
    print(f"\nFound {len(json_files)} local tracking data files")
    
    all_events = []
    
    for json_file in json_files:
        print(f"\nProcessing: {os.path.basename(json_file)}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            game = json.load(f)
        
        game_id = game['gameid']
        game_date = game['gamedate']
        
        for event in game['events']:
            event_id = event['eventId']
            
            # Look up matching PBP row
            event_row = pbp.loc[(pbp.GAME_ID == int(game_id)) & (pbp.EVENTNUM == int(event_id))]
            
            if len(event_row) != 1:
                # No matching PBP data, skip
                continue
            
            # Extract PBP context
            event_row = event_row.iloc[0]
            
            event_type = event_row["EVENTMSGTYPE"]
            event_home_desc = event_row["HOMEDESCRIPTION"]
            event_away_desc = event_row["VISITORDESCRIPTION"]
            
            primary_home_away = home_away_event_conversion(event_row["PERSON1TYPE"])
            primary_player_id = event_row["PLAYER1_ID"]
            primary_team_id = event_row["PLAYER1_TEAM_ID"]
            
            secondary_home_away = home_away_event_conversion(event_row["PERSON2TYPE"])
            secondary_player_id = event_row["PLAYER2_ID"]
            secondary_team_id = event_row["PLAYER2_TEAM_ID"]
            
            poss_team_id = identify_offense(event_row)
            
            # Restructure moments with named fields
            moments = []
            for moment in event['moments']:
                moments.append({
                    "quarter": moment[0],
                    "game_clock": moment[2],
                    "shot_clock": moment[3],
                    "ball_coordinates": {
                        "x": moment[5][0][2],
                        "y": moment[5][0][3],
                        "z": moment[5][0][4]
                    },
                    "player_coordinates": [
                        {
                            "teamid": player[0],
                            "playerid": player[1],
                            "x": player[2],
                            "y": player[3],
                            "z": player[4]
                        } for player in moment[5][1:]
                    ]
                })
            
            # Build enriched event
            enriched_event = {
                "gameid": game_id,
                "gamedate": game_date,
                "event_info": {
                    "id": event_id,
                    "type": int(event_type),
                    "possession_team_id": poss_team_id,
                    "desc_home": event_home_desc,
                    "desc_away": event_away_desc
                },
                "primary_info": {
                    "team": primary_home_away,
                    "player_id": primary_player_id,
                    "team_id": primary_team_id
                },
                "secondary_info": {
                    "team": secondary_home_away,
                    "player_id": secondary_player_id,
                    "team_id": secondary_team_id
                },
                "visitor": event['visitor'],
                "home": event['home'],
                "moments": moments
            }
            
            all_events.append(enriched_event)
    
    return all_events


def load_shot_attempts(pbp_cache_file="pbp_cache.csv", rim_height=10.0):
    """
    Load NBA tracking data and extract shot attempts with player positions
    at the moment when the ball is at the rim (10 feet height and at rim coordinates).
    
    Args:
        pbp_cache_file: Path to cached play-by-play CSV file
        rim_height: Height of the rim in feet (default 10.0)
    
    Returns:
        list: List of shot attempt dictionaries with format:
            {
                "made": bool,
                "game_clock": str (time in MM:SS format, e.g., "11:38"),
                "offense_position": [[teamid, playerid, x, y, z], ...],
                "defense_position": [[teamid, playerid, x, y, z], ...],
                "ball_position": [x, y, z]
            }
    """
    # Download or load cached PBP data
    if os.path.exists(pbp_cache_file):
        print(f"Loading cached PBP data from {pbp_cache_file}...")
        pbp = pd.read_csv(pbp_cache_file)
    else:
        print(f"Downloading PBP data from {PBP_URL}...")
        pbp = pd.read_csv(PBP_URL)
        pbp.to_csv(pbp_cache_file, index=False)
        print(f"Cached PBP data to {pbp_cache_file}")
    
    # Find all local JSON files
    json_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.json"))
    print(f"\nFound {len(json_files)} local tracking data files")
    
    all_shots = []
    
    for json_file in json_files:
        print(f"\nProcessing: {os.path.basename(json_file)}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            game = json.load(f)
        
        game_id = game['gameid']
        
        # Get home team ID from first event (all events have same home team)
        home_team_id = None
        if game['events']:
            home_team_id = game['events'][0]['home']['teamid']
        
        if home_team_id is None:
            print(f"  Warning: Could not determine home team ID, skipping game")
            continue
        
        for event in game['events']:
            event_id = event['eventId']
            
            # Look up matching PBP row
            event_row = pbp.loc[(pbp.GAME_ID == int(game_id)) & (pbp.EVENTNUM == int(event_id))]
            
            if len(event_row) != 1:
                continue
            
            event_row = event_row.iloc[0]
            event_type = int(event_row["EVENTMSGTYPE"])
            
            # Filter for shot attempts only (1 = made, 2 = missed)
            if event_type not in [1, 2]:
                continue
            
            # Determine make/miss
            is_made = (event_type == 1)
            
            # Get possession team
            poss_team_id = identify_offense(event_row)
            if poss_team_id is None:
                continue
            
            # Restructure moments with named fields
            moments = []
            for moment in event['moments']:
                moments.append({
                    "quarter": moment[0],
                    "game_clock": moment[2],
                    "shot_clock": moment[3],
                    "ball_coordinates": {
                        "x": moment[5][0][2],
                        "y": moment[5][0][3],
                        "z": moment[5][0][4]
                    },
                    "player_coordinates": [
                        {
                            "teamid": player[0],
                            "playerid": player[1],
                            "x": player[2],
                            "y": player[3],
                            "z": player[4]
                        } for player in moment[5][1:]
                    ]
                })
            
            if not moments:
                continue
            
            # Determine which rim to use
            quarter = moments[0]['quarter']
            target_rim = determine_target_rim(home_team_id, poss_team_id, quarter)
            
            # Find moment when ball is at rim (at rim height and rim coordinates)
            snapshot_moment = find_moment_at_rim(moments, target_rim, rim_height)
            
            if snapshot_moment is None:
                continue
            
            # Extract ball position
            ball_pos = [
                snapshot_moment['ball_coordinates']['x'],
                snapshot_moment['ball_coordinates']['y'],
                snapshot_moment['ball_coordinates']['z']
            ]
            
            # Separate offensive and defensive players
            offense_positions = []
            defense_positions = []
            
            for player in snapshot_moment['player_coordinates']:
                player_pos = [
                    player['teamid'],
                    player['playerid'],
                    player['x'],
                    player['y'],
                    player['z']
                ]
                
                if player['teamid'] == poss_team_id:
                    offense_positions.append(player_pos)
                else:
                    defense_positions.append(player_pos)
            
            # Create shot entry
            shot_entry = {
                "made": is_made,
                "game_clock": format_game_clock(snapshot_moment['game_clock']),
                "offense_position": offense_positions,
                "defense_position": defense_positions,
                "ball_position": ball_pos
            }
            
            all_shots.append(shot_entry)
    print('all shots', all_shots)
    return all_shots


if __name__ == "__main__":
    # Load shot attempt data
    print("=" * 60)
    print("Loading NBA Shot Attempts with Position Data")
    print("=" * 60)
    
    shots = load_shot_attempts()
    
    print(f"\n" + "=" * 60)
    print(f"Successfully loaded {len(shots)} shot attempts!")
    print("=" * 60)
    
    # Calculate make/miss statistics
    if shots:
        made_shots = sum(1 for shot in shots if shot['made'])
        missed_shots = len(shots) - made_shots
        fg_pct = (made_shots / len(shots) * 100) if shots else 0
        
        print(f"\nShot Statistics:")
        print(f"  Made: {made_shots}")
        print(f"  Missed: {missed_shots}")
        print(f"  FG%: {fg_pct:.1f}%")
    
    # Display first shot details
    if shots:
        first_shot = shots[0]
        print(f"\n{'=' * 60}")
        print("First Shot Attempt Details")
        print("=" * 60)
        print(f"Made: {first_shot['made']}")
        print(f"Game Clock: {first_shot['game_clock']}")
        print(f"\nBall Position (at rim - 10ft height):")
        print(f"  x={first_shot['ball_position'][0]:.2f}, "
              f"y={first_shot['ball_position'][1]:.2f}, "
              f"z={first_shot['ball_position'][2]:.2f} ft")
        
        print(f"\nOffensive Players ({len(first_shot['offense_position'])}):")
        for i, player in enumerate(first_shot['offense_position'][:3]):  # Show first 3
            print(f"  Player {i+1}: Team={player[0]}, ID={player[1]}, "
                  f"pos=({player[2]:.2f}, {player[3]:.2f}, {player[4]:.2f})")
        if len(first_shot['offense_position']) > 3:
            print(f"  ... and {len(first_shot['offense_position']) - 3} more")
        
        print(f"\nDefensive Players ({len(first_shot['defense_position'])}):")
        for i, player in enumerate(first_shot['defense_position'][:3]):  # Show first 3
            print(f"  Player {i+1}: Team={player[0]}, ID={player[1]}, "
                  f"pos=({player[2]:.2f}, {player[3]:.2f}, {player[4]:.2f})")
        if len(first_shot['defense_position']) > 3:
            print(f"  ... and {len(first_shot['defense_position']) - 3} more")
    
    print(f"\n{'=' * 60}")
    print("✓ Shot data loaded successfully!")
    print("=" * 60)
    print(f"\nTo use this data in your own code:")
    print("  from load_with_context import load_shot_attempts")
    print("  shots = load_shot_attempts()")
    print("  # shots is a list of dictionaries with shot outcomes and positions")
    print("\nData format:")
    print("  Each shot contains:")
    print("    - 'made': boolean (True/False)")
    print("    - 'game_clock': string (time in MM:SS format, e.g., '11:38')")
    print("    - 'offense_position': list of [teamid, playerid, x, y, z]")
    print("    - 'defense_position': list of [teamid, playerid, x, y, z]")
    print("    - 'ball_position': [x, y, z]")
    print("\nTo save as JSON:")
    print("  import json")
    print("  with open('shot_data.json', 'w') as f:")
    print("      json.dump(shots, f, indent=2)")

with open('shot_data.json', 'w') as f:
      json.dump(shots, f, indent=2)


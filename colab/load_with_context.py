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
# LOCAL_DATA_DIR points to the game_data folder
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "game_data")
PBP_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "supplemental_data", "pbp_cache.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "shot_data")
PBP_URL = "https://github.com/sumitrodatta/nba-alt-awards/raw/main/Historical/PBP%20Data/2015-16_pbp.csv"

# ============ PROCESSING PARAMETERS (CHANGE THESE) ============
START_INDEX = 0           # ← CHANGE THIS TO RESUME
NUM_GAMES = None          # Number of games to process (None = all remaining)
STATS_GROUP_SIZE = 5      # Print stats for groups of N games

# Parameters for find_moment_of_release:
RIM_HEIGHT = 9.7
RIM_XY_THRESHOLD = 4.0
PLAYER_XYZ_THRESHOLD = 9.0
MIN_RELEASE_HEIGHT = 6.5
NUM_SNAPSHOTS = 5
MOMENTS_BACK = [6, 12, 25, 50]

# Time-based matching tolerance (seconds) for PBP to JSON event matching
TIME_MATCH_TOLERANCE = 5.0
# ===============================================================

# Court dimensions (in feet)
COURT_LENGTH = 94
COURT_WIDTH = 50
COURT_CENTER_X = 47.0  # Center of 94ft court
COURT_CENTER_Y = 25.0  # Center of 50ft court
# Rim positions (standard NBA: 5.25 feet from baseline, centered at 25 feet)
LEFT_RIM = (5.25, 25.0)
RIGHT_RIM = (88.75, 25.0)

# Per-game calibration state (set by calibrate_rim_assignments)
_CURRENT_HOME_TEAM_ID = None
_CURRENT_HOME_Q1Q2_RIM = None


def parse_pbp_time(time_str):
    """Convert PBP time string (MM:SS) to seconds"""
    parts = str(time_str).split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0


def find_candidate_events(json_events, quarter, pbp_time_seconds, tolerance=TIME_MATCH_TOLERANCE):
    """
    Find all JSON events that contain the given quarter and time, sorted by center_diff ascending.

    Returns:
        List of matching event dicts sorted by center_diff (best match first), or empty list.
    """
    matches = []

    for event in json_events:
        moments = event.get('moments', [])
        if not moments:
            continue

        event_quarter = moments[0][0]
        if event_quarter != quarter:
            continue

        start_time = moments[0][2]  # First moment game clock
        end_time = moments[-1][2]   # Last moment game clock

        # Game clock counts DOWN, so start_time > end_time
        if end_time - tolerance <= pbp_time_seconds <= start_time + tolerance:
            matches.append({
                'event': event,
                'center_diff': abs((start_time + end_time) / 2 - pbp_time_seconds)
            })

    matches.sort(key=lambda x: x['center_diff'])
    return [m['event'] for m in matches]


def find_matching_json_event(json_events, quarter, pbp_time_seconds, tolerance=TIME_MATCH_TOLERANCE):
    """
    Find the JSON event that contains the given quarter and time.

    Returns:
        The best-matching event dict, or None if not found.
    """
    candidates = find_candidate_events(json_events, quarter, pbp_time_seconds, tolerance)
    return candidates[0] if candidates else None


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

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def determine_target_rim(poss_team_id, quarter):
    """
    Determine which rim the offensive team is shooting at.
    
    Uses module-level globals set by calibrate_rim_assignments():
        _CURRENT_HOME_TEAM_ID: Team ID of the home team
        _CURRENT_HOME_Q1Q2_RIM: The rim the home team shoots at in Q1/Q2
    
    Args:
        poss_team_id: Team ID of the team with possession (shooting)
        quarter: Current quarter (1-4)
    
    Returns:
        Tuple (x, y) of the target rim coordinates
    """
    is_home_team = (poss_team_id == _CURRENT_HOME_TEAM_ID)
    
    if quarter in [1, 2]:
        # Q1/Q2: Home shoots at _CURRENT_HOME_Q1Q2_RIM, Away shoots at opposite
        if is_home_team:
            return _CURRENT_HOME_Q1Q2_RIM
        else:
            return LEFT_RIM if _CURRENT_HOME_Q1Q2_RIM == RIGHT_RIM else RIGHT_RIM
    else:
        # Q3/Q4: Teams switch sides - Home shoots at opposite of _CURRENT_HOME_Q1Q2_RIM
        if is_home_team:
            return LEFT_RIM if _CURRENT_HOME_Q1Q2_RIM == RIGHT_RIM else RIGHT_RIM
        else:
            return _CURRENT_HOME_Q1Q2_RIM


def _fallback_rim_calibration(game, home_team_id, pbp):
    """
    Fallback method to calibrate rim assignments by analyzing shot events in Q1.
    
    Used when the primary calibration method fails (same_side_error or invalid_first_moment).
    
    Logic:
    1. Get all Q1 shot events from PBP for this game
    2. For each shot event, use time-based matching to find the corresponding JSON event
    3. For each shot event with moments:
       - Search FORWARD from start for first rim moment (ball above rim + descending)
       - Search BACKWARD from end for last rim moment (ball above rim + descending)
    4. If both find the same rim and moments are within 2 seconds of each other:
       - Trace back to find the nearest OFFENSIVE player (on shooter's team)
       - If that player IS the labeled shooter, use to calibrate
       - If it's a teammate, skip to next event
    5. Continue until valid shot found or end of Q1 reached
    
    Args:
        game: Game data dictionary
        home_team_id: Team ID of the home team
        pbp: Play-by-play DataFrame
    
    Returns:
        tuple: (success, error_type, details_dict)
    """
    global _CURRENT_HOME_TEAM_ID, _CURRENT_HOME_Q1Q2_RIM
    
    game_id = game['gameid']
    _CURRENT_HOME_TEAM_ID = home_team_id
    
    rim_height = 9.7
    rim_xy_threshold = 4.0
    player_xyz_threshold = 9.0
    
    # Get Q1 shot events from PBP for this game
    game_pbp = pbp[pbp.GAME_ID == int(game_id)]
    q1_shots = game_pbp[(game_pbp.PERIOD == 1) & (game_pbp.EVENTMSGTYPE.isin([1, 2]))]
    
    for _, shot_row in q1_shots.iterrows():
        # Get shooter info from PBP
        poss_team_id = shot_row['PLAYER1_TEAM_ID']
        shooter_id = shot_row['PLAYER1_ID']
        
        if pd.isna(poss_team_id) or pd.isna(shooter_id):
            continue
        
        poss_team_id = int(poss_team_id)
        shooter_id = int(shooter_id)
        
        # Use time-based matching to find the corresponding JSON event
        pbp_time_seconds = parse_pbp_time(shot_row['PCTIMESTRING'])
        quarter = int(shot_row['PERIOD'])
        
        event = find_matching_json_event(game['events'], quarter, pbp_time_seconds)
        if event is None:
            continue
        
        moments = event.get('moments', [])
        if len(moments) < 2:
            continue
        
        # Search from START (forward) for first rim moment
        start_rim_moment = None
        start_rim_idx = None
        start_rim = None
        
        for idx in range(1, len(moments)):
            ball = moments[idx][5][0]  # [team_id, -1, x, y, z]
            ball_x, ball_y, ball_z = ball[2], ball[3], ball[4]
            prev_ball_z = moments[idx - 1][5][0][4]
            
            is_descending = ball_z < prev_ball_z
            
            # Check distance to each rim
            dist_to_left = math.sqrt((ball_x - LEFT_RIM[0])**2 + (ball_y - LEFT_RIM[1])**2)
            dist_to_right = math.sqrt((ball_x - RIGHT_RIM[0])**2 + (ball_y - RIGHT_RIM[1])**2)
            
            if dist_to_left <= rim_xy_threshold and ball_z >= rim_height and is_descending:
                start_rim_moment = moments[idx]
                start_rim_idx = idx
                start_rim = LEFT_RIM
                break
            elif dist_to_right <= rim_xy_threshold and ball_z >= rim_height and is_descending:
                start_rim_moment = moments[idx]
                start_rim_idx = idx
                start_rim = RIGHT_RIM
                break
        
        # Search from END (backward) for last rim moment
        end_rim_moment = None
        end_rim_idx = None
        end_rim = None
        
        for idx in range(len(moments) - 1, 0, -1):
            ball = moments[idx][5][0]
            ball_x, ball_y, ball_z = ball[2], ball[3], ball[4]
            prev_ball_z = moments[idx - 1][5][0][4]
            
            is_descending = ball_z < prev_ball_z
            
            dist_to_left = math.sqrt((ball_x - LEFT_RIM[0])**2 + (ball_y - LEFT_RIM[1])**2)
            dist_to_right = math.sqrt((ball_x - RIGHT_RIM[0])**2 + (ball_y - RIGHT_RIM[1])**2)
            
            if dist_to_left <= rim_xy_threshold and ball_z >= rim_height and is_descending:
                end_rim_moment = moments[idx]
                end_rim_idx = idx
                end_rim = LEFT_RIM
                break
            elif dist_to_right <= rim_xy_threshold and ball_z >= rim_height and is_descending:
                end_rim_moment = moments[idx]
                end_rim_idx = idx
                end_rim = RIGHT_RIM
                break
        
        # Check if both found the same rim
        if start_rim_moment is None or end_rim_moment is None:
            continue
        
        if start_rim != end_rim:
            continue
        
        # Check if within 2 seconds of each other
        start_game_clock = start_rim_moment[2]
        end_game_clock = end_rim_moment[2]
        time_diff = abs(start_game_clock - end_game_clock)
        
        if time_diff > 2.0:
            continue
        
        # Found a valid rim convergence! Now verify the shooter using PBP-labeled shooter
        target_rim = start_rim
        
        # Use the earlier rim moment (lower index = earlier in the list)
        rim_idx = min(start_rim_idx, end_rim_idx)
        
        # Trace back to find the nearest OFFENSIVE player (on shooter's team)
        found_shooter = False
        
        for back_idx in range(rim_idx, -1, -1):
            moment = moments[back_idx]
            ball = moment[5][0]
            ball_x, ball_y, ball_z = ball[2], ball[3], ball[4]
            
            # Find closest player on the OFFENSIVE team (poss_team_id) using 3D distance
            closest_offensive_player_id = None
            closest_dist = float('inf')
            
            for player in moment[5][1:]:
                team_id, player_id, px, py, pz = player
                # Only consider players on the offensive team
                if team_id != poss_team_id:
                    continue
                
                dist = math.sqrt((ball_x - px)**2 + (ball_y - py)**2 + (ball_z - pz)**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_offensive_player_id = player_id
            
            if closest_dist <= player_xyz_threshold and closest_offensive_player_id is not None:
                # Check if this is the labeled shooter
                if closest_offensive_player_id == shooter_id:
                    found_shooter = True
                # If it's a teammate (not the shooter), skip this event
                break
        
        if not found_shooter:
            # The nearest offensive player was not the labeled shooter, try next event
            continue
        
        # Verified! The labeled shooter was the nearest offensive player
        # Determine rim assignment based on shooter's team
        if poss_team_id == home_team_id:
            # Home team shot at this rim in Q1, so this is their Q1/Q2 target
            _CURRENT_HOME_Q1Q2_RIM = target_rim
        else:
            # Away team shot at this rim, so home shoots at the opposite
            _CURRENT_HOME_Q1Q2_RIM = LEFT_RIM if target_rim == RIGHT_RIM else RIGHT_RIM
        
        home_shoots_at_q1q2 = 'right' if _CURRENT_HOME_Q1Q2_RIM == RIGHT_RIM else 'left'
        
        # Success!
        return True, None, {
            'game_id': game_id,
            'method': 'fallback_shot_analysis',
            'event_quarter': quarter,
            'shooter_id': shooter_id,
            'shooter_team_id': poss_team_id,
            'shooter_is_home': poss_team_id == home_team_id,
            'target_rim_found': 'right' if target_rim == RIGHT_RIM else 'left',
            'home_shoots_at_q1q2': home_shoots_at_q1q2,
            'time_diff_seconds': time_diff,
            'calibration': 'SUCCESS_FALLBACK'
        }
    
    # Reached end of Q1 without finding valid calibration
    return False, 'fallback_failed', {
        'game_id': game_id,
        'message': 'Could not calibrate rim using fallback shot analysis in Q1'
    }


def calibrate_rim_assignments(game, home_team_id, pbp=None):
    """
    Determine which rim each team shoots at by analyzing the first moment of the game.
    
    Sets module-level globals for use by determine_target_rim():
        _CURRENT_HOME_TEAM_ID: Set to home_team_id
        _CURRENT_HOME_Q1Q2_RIM: Set to LEFT_RIM or RIGHT_RIM based on player positions
    
    Logic:
    - Get the first moment from the game data
    - Validate it's in Q1 with no more than 1 second elapsed (game_clock >= 719.0)
    - Calculate average x position for each team
    - The team with average on LEFT side (x < 47) shoots at RIGHT rim in Q1/Q2
    - The team with average on RIGHT side (x > 47) shoots at LEFT rim in Q1/Q2
    - If both teams average on the same side, or first moment is invalid, fall back to
      analyzing shot events in Q1 to determine rim assignments (requires pbp data)
    
    Args:
        game: Game data dictionary
        home_team_id: Team ID of the home team
        pbp: Play-by-play DataFrame (required for fallback method)
    
    Returns:
        tuple: (success, error_type, details_dict)
        - success: True if calibration succeeded, False otherwise
        - error_type: None if successful, or one of:
            'no_first_moment' - couldn't find any moments in the game
            'invalid_first_moment' - first moment is not Q1 or more than 1 second elapsed (before fallback)
            'same_side_error' - both teams have average position on the same side (before fallback)
            'fallback_failed' - fallback shot analysis also failed
        - details_dict: Dictionary with debugging details including average positions
    """
    global _CURRENT_HOME_TEAM_ID, _CURRENT_HOME_Q1Q2_RIM
    
    game_id = game['gameid']
    _CURRENT_HOME_TEAM_ID = home_team_id
    
    # Step 1: Find the first moment in the game
    first_moment = None
    away_team_id = None
    
    for event in game['events']:
        if not event['moments']:
            continue
        
        for moment in event['moments']:
            first_moment = moment
            
            # Identify away team from players
            for player in moment[5][1:]:
                if player[0] != home_team_id and player[0] != -1:
                    away_team_id = player[0]
                    break
            break
        
        if first_moment is not None:
            break
    
    if first_moment is None:
        return False, 'no_first_moment', {
            'game_id': game_id,
            'message': 'Could not find any moments in game data'
        }
    
    # Step 2: Validate the first moment is in Q1 with no more than 1 second elapsed
    quarter = first_moment[0]
    game_clock = first_moment[2]
    
    needs_fallback = False
    fallback_reason = None
    fallback_details = None
    
    if quarter != 1 or game_clock < 719.0:
        needs_fallback = True
        fallback_reason = 'invalid_first_moment'
        fallback_details = {
            'game_id': game_id,
            'message': f'First moment is not at start of Q1 (Q{quarter}, game_clock={game_clock:.2f}s)',
            'quarter': quarter,
            'game_clock': game_clock
        }
    else:
        # Step 3: Calculate average x position for each team
        players = first_moment[5][1:]  # Skip ball
        
        home_x_positions = []
        away_x_positions = []
        
        for player in players:
            team_id, player_id, x, y, z = player
            
            if team_id == home_team_id:
                home_x_positions.append(x)
            elif team_id != -1:  # Skip ball marker
                away_x_positions.append(x)
        
        # Calculate averages
        home_avg_x = sum(home_x_positions) / len(home_x_positions) if home_x_positions else COURT_CENTER_X
        away_avg_x = sum(away_x_positions) / len(away_x_positions) if away_x_positions else COURT_CENTER_X
        
        # Determine which side each team is on based on average position
        home_side = 'left' if home_avg_x < COURT_CENTER_X else 'right'
        away_side = 'left' if away_avg_x < COURT_CENTER_X else 'right'
        
        # Step 4: Check if both teams are on the same side (needs fallback)
        if home_side == away_side:
            needs_fallback = True
            fallback_reason = 'same_side_error'
            fallback_details = {
                'game_id': game_id,
                'message': f'Both teams have average position on {home_side} side of court',
                'home_avg_x': home_avg_x,
                'away_avg_x': away_avg_x,
                'home_side': home_side,
                'away_side': away_side,
                'home_player_count': len(home_x_positions),
                'away_player_count': len(away_x_positions)
            }
        else:
            # Step 5: Determine rim assignments and set global
            # Team on LEFT side of court (x < 47) shoots at RIGHT rim in Q1/Q2
            # Team on RIGHT side of court (x > 47) shoots at LEFT rim in Q1/Q2
            if home_side == 'left':
                _CURRENT_HOME_Q1Q2_RIM = RIGHT_RIM  # Home shoots at right rim (88.75, 25) in Q1/Q2
            else:
                _CURRENT_HOME_Q1Q2_RIM = LEFT_RIM   # Home shoots at left rim (5.25, 25) in Q1/Q2
            
            home_shoots_at_q1q2 = 'right' if _CURRENT_HOME_Q1Q2_RIM == RIGHT_RIM else 'left'
            away_shoots_at_q1q2 = 'left' if home_shoots_at_q1q2 == 'right' else 'right'
            
            # Success - return True with calibration details
            return True, None, {
                'game_id': game_id,
                'home_avg_x': home_avg_x,
                'away_avg_x': away_avg_x,
                'home_side': home_side,
                'away_side': away_side,
                'home_player_count': len(home_x_positions),
                'away_player_count': len(away_x_positions),
                'home_shoots_at_q1q2': home_shoots_at_q1q2,
                'away_shoots_at_q1q2': away_shoots_at_q1q2,
                'calibration': 'SUCCESS'
            }
    
    # If we need fallback, try the shot analysis method
    if needs_fallback:
        if pbp is None:
            # Can't use fallback without PBP data
            return False, fallback_reason, fallback_details
        
        fallback_success, fallback_error, fallback_result = _fallback_rim_calibration(game, home_team_id, pbp)
        
        if fallback_success:
            # Add info about why fallback was needed
            fallback_result['primary_failure_reason'] = fallback_reason
            fallback_result['primary_failure_details'] = fallback_details
            return True, None, fallback_result
        else:
            # Fallback also failed - include both failure details
            fallback_result['primary_failure_reason'] = fallback_reason
            fallback_result['primary_failure_details'] = fallback_details
            return False, 'fallback_failed', fallback_result
    
    # Should not reach here, but just in case
    return False, 'unknown_error', {'game_id': game_id}


def find_moment_of_release(moments, shooter_id, poss_team_id, target_rim,
                           rim_height=9.7,
                           rim_xy_threshold=4.0,
                           player_xyz_threshold=9.0,
                           min_release_height=6.5,
                           num_snapshots=3,
                           moments_back=[3, 12],
                           pbp_time_seconds=None,
                           pbp_proximity_threshold=None):
    """
    Find the moment when the ball was released by the shooter.
    
    Logic:
    1. Loop backwards from the end to find the moment where ball is:
       - Within XY threshold of rim position
       - Above the rim height
       - Descending (z decreasing)
    2. From that rim moment, go back in time to find the first OFFENSIVE player within XYZ threshold
       (using 3D distance to avoid counting ball passing over defender's head)
    3. If that player is the shooter:
       - Continue going back in time while ball is getting closer to shooter each moment
       - Find the moment with minimum XY distance to shooter
       - But ball must remain above min_release_height
       - If min XY occurs when ball is below min_release_height, use the moment just before
         the ball went below that height
    4. If that player is NOT the shooter, skip this shot (return None)
    
    Args:
        moments: List of moment dictionaries with ball and player coordinates
        shooter_id: Player ID of the shooter from play-by-play data
        poss_team_id: Team ID of the offensive team (possession)
        target_rim: Tuple (x, y) of the rim coordinates being shot at
        rim_height: Height of rim in feet (default 10.0)
        rim_xy_threshold: Max XY distance from rim to be considered "near rim" (default 5.0)
        player_xyz_threshold: Max 3D distance to consider ball "with a player" (default 10.0)
        min_release_height: Minimum ball height for valid release point (default 5.0)
        num_snapshots: Number of snapshots to return (default 1). Should equal 1 + len(moments_back).
        moments_back: List of integers representing how many moments to go back in time from the 
                      release moment. E.g., [5, 10] means return release moment plus moments 
                      at result_idx-5 and result_idx-10. Default is None (returns only release moment).
    
    Returns:
        Tuple of (moments_list, drop_reason, debug_info):
        - (list_of_moments, None, None) on success - list contains release moment followed by 
          historical moments as specified by moments_back
        - (None, reason_string, debug_dict) on failure
    """
    
    if moments_back is None:
        moments_back = []
    
    if len(moments) < 2:
        return None, 'not_enough_moments', None
    
    def get_shooter_position(moment):
        """Get the shooter's position from the moment, or None if not found"""
        for player in moment['player_coordinates']:
            if player['playerid'] == shooter_id:
                return player
        return None
    
    def get_xy_distance_to_shooter(moment):
        """Calculate XY distance from ball to shooter"""
        shooter = get_shooter_position(moment)
        if shooter is None:
            return float('inf')
        ball = moment['ball_coordinates']
        return math.sqrt(
            (ball['x'] - shooter['x'])**2 + 
            (ball['y'] - shooter['y'])**2
        )
    
    def get_closest_offensive_player_3d(moment):
        """Find the closest OFFENSIVE player to the ball using 3D distance"""
        ball = moment['ball_coordinates']
        closest_player_id = None
        closest_distance = float('inf')
        
        for player in moment['player_coordinates']:
            # Only consider players on the offensive team
            if player['teamid'] != poss_team_id:
                continue
                
            dist_3d = math.sqrt(
                (ball['x'] - player['x'])**2 + 
                (ball['y'] - player['y'])**2 +
                (ball['z'] - player['z'])**2
            )
            if dist_3d < closest_distance:
                closest_distance = dist_3d
                closest_player_id = player['playerid']
        
        return closest_player_id, closest_distance
    
    def is_descending(idx):
        """Check if ball is descending at this moment (z decreasing over time)"""
        if idx <= 0 or idx >= len(moments):
            return False
        curr_z = moments[idx]['ball_coordinates']['z']
        prev_z = moments[idx - 1]['ball_coordinates']['z']
        return curr_z < prev_z
    
    # Step 1: Find the LAST moment where ball is near rim, above rim, and descending
    # Iterate backwards through time (from end of event toward start)
    rim_idx = None
    for idx in range(len(moments) - 1, 0, -1):
        ball = moments[idx]['ball_coordinates']
        xy_dist_to_rim = math.sqrt(
            (ball['x'] - target_rim[0])**2 + 
            (ball['y'] - target_rim[1])**2
        )
        
        # Check: within XY threshold of rim, above rim height, and descending
        if (xy_dist_to_rim <= rim_xy_threshold and 
            ball['z'] >= rim_height and 
            is_descending(idx)):
            rim_idx = idx
            break
    
    if rim_idx is None:
        # Return debug info: start and end game clocks
        debug_info = {
            'start_game_clock': moments[0]['game_clock'],
            'end_game_clock': moments[-1]['game_clock'],
            'quarter': moments[0]['quarter']
        }
        return None, 'no_rim_moment_found', debug_info
    
    # Step 2: From rim moment, go back in time to find first OFFENSIVE player within XYZ threshold
    # We loop from rim_idx down to 0, which goes backward in time (earlier moments have lower indices)
    first_player_idx = None
    first_player_id = None
    
    for idx in range(rim_idx, -1, -1):
        closest_player_id, closest_distance = get_closest_offensive_player_3d(moments[idx])
        
        if closest_distance <= player_xyz_threshold:
            first_player_idx = idx
            first_player_id = closest_player_id
            break
    
    if first_player_idx is None:
        return None, 'no_player_within_threshold', None
    
    # Step 3: Verify the first player encountered is the shooter
    if first_player_id != shooter_id:
        # Ball was close to someone else first — save debug info then try forward scan fallback.
        # The backward scan finds Shot B's rim approach in multi-shot events; the forward scan
        # finds Shot A's rim approach (earlier in the moments list) instead.
        debug_info = {
            'game_clock': moments[first_player_idx]['game_clock'],
            'quarter': moments[first_player_idx]['quarter'],
            'expected_shooter_id': shooter_id,
            'found_player_id': first_player_id
        }

        # Forward scan: find the FIRST (earliest) rim approach in this event
        fwd_rim_idx = None
        for idx in range(1, len(moments)):
            ball = moments[idx]['ball_coordinates']
            xy_dist_to_rim = math.sqrt(
                (ball['x'] - target_rim[0])**2 +
                (ball['y'] - target_rim[1])**2
            )
            if (xy_dist_to_rim <= rim_xy_threshold and
                    ball['z'] >= rim_height and
                    is_descending(idx)):
                fwd_rim_idx = idx
                break

        fwd_success = False
        if fwd_rim_idx is not None and fwd_rim_idx != rim_idx:
            # Re-run Step 2 from the forward-scan rim moment
            fwd_first_player_idx = None
            fwd_first_player_id = None
            for idx in range(fwd_rim_idx, -1, -1):
                closest_player_id, closest_distance = get_closest_offensive_player_3d(moments[idx])
                if closest_distance <= player_xyz_threshold:
                    fwd_first_player_idx = idx
                    fwd_first_player_id = closest_player_id
                    break

            if fwd_first_player_idx is not None and fwd_first_player_id == shooter_id:
                # Forward scan found the correct shooter — redirect Step 4 to use this index
                first_player_idx = fwd_first_player_idx
                fwd_success = True

        if not fwd_success:
            return None, 'first_player_not_shooter', debug_info
    
    # Step 4: Continue going back to find the moment with minimum XY distance to shooter
    # while ball is getting closer each moment and remains above min_release_height
    
    min_xy_distance = get_xy_distance_to_shooter(moments[first_player_idx])
    min_xy_idx = first_player_idx
    last_valid_idx = first_player_idx  # Last moment where ball was above min_release_height
    prev_xy_distance = min_xy_distance
    
    for idx in range(first_player_idx - 1, -1, -1):
        moment = moments[idx]
        ball_z = moment['ball_coordinates']['z']
        current_xy_distance = get_xy_distance_to_shooter(moment)
        
        # Stop if ball is no longer getting closer to shooter
        if current_xy_distance >= prev_xy_distance:
            break
        
        prev_xy_distance = current_xy_distance
        
        # Check if ball is within valid release height range
        if ball_z >= min_release_height:
            last_valid_idx = idx
            if current_xy_distance < min_xy_distance:
                min_xy_distance = current_xy_distance
                min_xy_idx = idx
        elif ball_z < min_release_height:
            # Ball went below min height - use the last valid moment
            break
    
    # Return the moment with minimum XY distance (if valid) or last valid moment
    # If min_xy_idx moment has ball below min_release_height, use last_valid_idx instead
    result_idx = min_xy_idx
    if moments[min_xy_idx]['ball_coordinates']['z'] < min_release_height:
        result_idx = last_valid_idx

    # Optional proximity gate: drop if release clock deviates too far from PBP-logged time
    if pbp_time_seconds is not None and pbp_proximity_threshold is not None:
        release_clock = moments[result_idx]['game_clock']
        if abs(release_clock - pbp_time_seconds) > pbp_proximity_threshold:
            return None, 'time_proximity_failed', None

    # Build list of moments: release moment + historical moments going back in time
    result_moments = [moments[result_idx]]
    
    for offset in moments_back:
        historical_idx = max(0, result_idx - offset)
        result_moments.append(moments[historical_idx])
    
    return result_moments, None, None


def load_shot_attempts(json_file, pbp, drop_stats, calibration_errors,
                       rim_height=RIM_HEIGHT,
                       rim_xy_threshold=RIM_XY_THRESHOLD,
                       player_xyz_threshold=PLAYER_XYZ_THRESHOLD,
                       min_release_height=MIN_RELEASE_HEIGHT,
                       num_snapshots=NUM_SNAPSHOTS,
                       moments_back=MOMENTS_BACK,
                       time_match_tolerance=TIME_MATCH_TOLERANCE,
                       pbp_proximity_threshold=None):
    """
    Load NBA tracking data from a single JSON file and extract shot attempts with player positions
    at the moment of release (when ball left shooter's hands).
    
    Args:
        json_file: Path to a single game JSON file
        pbp: Pre-loaded play-by-play DataFrame
        drop_stats: Dictionary to accumulate drop statistics (modified in place)
        calibration_errors: Dictionary to accumulate calibration errors (modified in place)
        rim_height: Height of rim in feet for find_moment_of_release
        rim_xy_threshold: Max XY distance from rim to be considered "near rim"
        player_xyz_threshold: Max 3D distance to consider ball "with a player"
        min_release_height: Minimum ball height for valid release point
        num_snapshots: Number of snapshots to return
        moments_back: List of integers for how many moments to go back
        time_match_tolerance: Seconds of tolerance for PBP to JSON event matching
    
    Returns:
        list: List of shot attempt dictionaries with format:
            {
                "made": bool,
                "game_clock": float (time in seconds),
                "offense_position": [[teamid, playerid, x, y, z], ...],
                "defense_position": [[teamid, playerid, x, y, z], ...],
                "ball_position": [x, y, z],
                "game_id": str,
                "home_team_id": int,
                "poss_team_id": int,
                "shooter_id": int,
                "quarter": int,
                "target_rim": [x, y] (either LEFT_RIM or RIGHT_RIM coordinates),
                "player_positions": {playerid: position_str, ...}
            }
    """
    all_shots = []
    
    try:
        with open(json_file, 'r') as f:
            game = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        return all_shots
    except Exception as e:
        return all_shots
        
    game_id = game['gameid']
    
    # Get home team ID from first event (all events have same home team)
    home_team_id = None
    if game['events']:
        home_team_id = game['events'][0]['home']['teamid']
    
    if home_team_id is None:
        return all_shots
    
    # Calibrate rim assignments using first moment player positions
    calibration_success, calibration_error, calibration_details = calibrate_rim_assignments(
        game, home_team_id, pbp
    )
    
    if not calibration_success:
        calibration_errors[calibration_error].append(calibration_details)
        # Skip this game - can't determine which rim teams are shooting at
        return all_shots
    else:
        # Check if this was a fallback success or primary success
        if calibration_details.get('calibration') == 'SUCCESS_FALLBACK':
            calibration_errors['calibrated_games_fallback'].append(calibration_details)
        else:
            calibration_errors['calibrated_games'].append(calibration_details)
    
    # Build full game player position and name mappings (for lookups)
    all_player_positions = {}
    all_player_names = {}
    if game['events']:
        for player in game['events'][0]['home']['players']:
            all_player_positions[player['playerid']] = player['position']
            all_player_names[player['playerid']] = f"{player.get('firstname', '')} {player.get('lastname', '')}"
        for player in game['events'][0]['visitor']['players']:
            all_player_positions[player['playerid']] = player['position']
            all_player_names[player['playerid']] = f"{player.get('firstname', '')} {player.get('lastname', '')}"
    
    # Get all shot events from PBP for this game
    game_pbp = pbp[pbp.GAME_ID == int(game_id)]
    shot_events = game_pbp[game_pbp.EVENTMSGTYPE.isin([1, 2])]
    
    for _, event_row in shot_events.iterrows():
        drop_stats['total_shot_events_processed'] += 1
        
        # Determine make/miss
        event_type = int(event_row["EVENTMSGTYPE"])
        is_made = (event_type == 1)
        
        # Get possession team and shooter (for shots, PLAYER1 is the shooter)
        poss_team_id = event_row['PLAYER1_TEAM_ID']
        shooter_id = event_row['PLAYER1_ID']
        if pd.isna(poss_team_id) or pd.isna(shooter_id):
            drop_stats['missing_team_or_shooter'] += 1
            continue
        
        poss_team_id = int(poss_team_id)
        shooter_id = int(shooter_id)
        
        # Use time-based matching to find all candidate JSON events (sorted best-first)
        pbp_time_seconds = parse_pbp_time(event_row['PCTIMESTRING'])
        quarter = int(event_row['PERIOD'])

        candidates = find_candidate_events(game['events'], quarter, pbp_time_seconds, tolerance=time_match_tolerance)
        if not candidates:
            drop_stats['no_tracking_match'] += 1
            continue

        def restructure_event_moments(event):
            """Convert raw moment arrays to named-field dicts."""
            result = []
            for moment in event.get('moments', []):
                try:
                    result.append({
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
                except (IndexError, TypeError):
                    pass
            return result

        # Try each candidate event until one yields a valid release moment
        release_moments = None
        drop_reason = None
        debug_info = None

        for attempt_num, candidate_event in enumerate(candidates):
            moments = restructure_event_moments(candidate_event)
            if not moments:
                if attempt_num == 0:
                    drop_stats['no_moments'] += 1
                continue

            quarter_from_moments = moments[0]['quarter']
            target_rim = determine_target_rim(poss_team_id, quarter_from_moments)

            release_moments, drop_reason, debug_info = find_moment_of_release(
                moments, shooter_id, poss_team_id, target_rim,
                rim_height=rim_height,
                rim_xy_threshold=rim_xy_threshold,
                player_xyz_threshold=player_xyz_threshold,
                min_release_height=min_release_height,
                num_snapshots=num_snapshots,
                moments_back=moments_back,
                pbp_time_seconds=pbp_time_seconds,
                pbp_proximity_threshold=pbp_proximity_threshold,
            )

            if release_moments is not None:
                if attempt_num > 0:
                    drop_stats['rescued_by_secondary_event'] += 1
                break

        if release_moments is None:
            # Track the specific reason for the drop (from the last attempt)
            if drop_reason == 'not_enough_moments':
                drop_stats['release_not_enough_moments'] += 1
            elif drop_reason == 'no_rim_moment_found':
                drop_stats['release_no_rim_moment_found'] += 1
                # Collect debug info for first 10
                if len(drop_stats['debug_no_rim_moment']) < 10 and debug_info:
                    debug_info['pbp_row'] = event_row.to_dict()
                    drop_stats['debug_no_rim_moment'].append(debug_info)
            elif drop_reason == 'no_player_within_threshold':
                drop_stats['release_no_player_within_threshold'] += 1
            elif drop_reason == 'first_player_not_shooter':
                drop_stats['release_first_player_not_shooter'] += 1
                # Collect debug info for first 10
                if len(drop_stats['debug_first_player_not_shooter']) < 10 and debug_info:
                    debug_info['expected_shooter_name'] = all_player_names.get(
                        debug_info['expected_shooter_id'], 'Unknown')
                    debug_info['found_player_name'] = all_player_names.get(
                        debug_info['found_player_id'], 'Unknown')
                    drop_stats['debug_first_player_not_shooter'].append(debug_info)
            elif drop_reason == 'not_enough_history':
                drop_stats['release_not_enough_history'] += 1
            elif drop_reason == 'time_proximity_failed':
                drop_stats['release_time_proximity_failed'] += 1
            continue

        # Use the tracked quarter from the winning candidate's moments
        quarter = release_moments[0]['quarter']
        
        # Use the first moment (release moment) for primary data extraction
        release_moment = release_moments[0]
        
        # Extract ball position at RELEASE TIME (for shooter identification)
        ball_pos = [
            release_moment['ball_coordinates']['x'],
            release_moment['ball_coordinates']['y'],
            release_moment['ball_coordinates']['z']
        ]
        
        # Separate offensive and defensive players AT RELEASE TIME
        offense_positions = []
        defense_positions = []
        
        for player in release_moment['player_coordinates']:
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
        
        # Process previous moments (historical snapshots)
        previous_moments = []
        for prev_moment in release_moments[1:]:
            prev_ball_pos = [
                prev_moment['ball_coordinates']['x'],
                prev_moment['ball_coordinates']['y'],
                prev_moment['ball_coordinates']['z']
            ]
            
            prev_offense_positions = []
            prev_defense_positions = []
            
            for player in prev_moment['player_coordinates']:
                player_pos = [
                    player['teamid'],
                    player['playerid'],
                    player['x'],
                    player['y'],
                    player['z']
                ]
                
                if player['teamid'] == poss_team_id:
                    prev_offense_positions.append(player_pos)
                else:
                    prev_defense_positions.append(player_pos)
            
            prev_entry = {
                "quarter": int(prev_moment['quarter']),
                "game_clock": float(prev_moment['game_clock']),
                "shot_clock": float(prev_moment['shot_clock']) if prev_moment['shot_clock'] is not None else None,
                "offense_position": prev_offense_positions,
                "defense_position": prev_defense_positions,
                "ball_position": prev_ball_pos,
            }
            previous_moments.append(prev_entry)

        # Create shot entry (convert numpy int64 to native Python int for JSON serialization)
        shot_entry = {
            "made": is_made,
            "quarter": int(release_moment['quarter']),
            "game_clock": float(release_moment['game_clock']),
            "shot_clock": float(release_moment['shot_clock']) if release_moment['shot_clock'] is not None else None,
            "shooter_id": int(shooter_id),
            "offense_position": offense_positions,
            "defense_position": defense_positions,
            "ball_position": ball_pos,
            "game_id": game_id,
            "home_team_id": int(home_team_id),
            "poss_team_id": int(poss_team_id),
            "target_rim": list(target_rim),
            "player_positions": all_player_positions,
            "player_names": all_player_names,
            "previous_moments": previous_moments
        }
        
        all_shots.append(shot_entry)
        drop_stats['time_diffs'].append(pbp_time_seconds - release_moment['game_clock'])

    return all_shots


def format_game_clock(seconds, quarter):
    """Convert game clock seconds to human-readable format (Q# MM:SS)"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"Q{quarter} {minutes}:{secs:05.2f}"

def create_empty_drop_stats():
    """Create a fresh drop_stats dictionary"""
    return {
        'total_shot_events_processed': 0,
        'missing_team_or_shooter': 0,
        'no_tracking_match': 0,  # No matching JSON event found by time
        'no_moments': 0,  # JSON event found but has no moments
        'release_not_enough_moments': 0,
        'release_no_rim_moment_found': 0,
        'release_no_player_within_threshold': 0,
        'release_first_player_not_shooter': 0,
        'release_not_enough_history': 0,
        'release_time_proximity_failed': 0,
        'rescued_by_secondary_event': 0,
        'debug_no_rim_moment': [],
        'debug_first_player_not_shooter': [],
        'time_diffs': [],
    }

def create_empty_calibration_errors():
    """Create a fresh calibration_errors dictionary"""
    return {
        'no_first_moment': [],      # Games where no moments were found
        'fallback_failed': [],      # Games where both primary and fallback calibration failed
        'calibrated_games': [],     # Games that were successfully calibrated (primary method)
        'calibrated_games_fallback': []  # Games that were successfully calibrated (fallback method)
    }

def print_group_stats(drop_stats, calibration_errors, extracted_count, group_label, 
                      game_shot_counts=None, dropped_games=None):
    """Print stats for a group of games
    
    Args:
        drop_stats: Dictionary of drop statistics
        calibration_errors: Dictionary of calibration results
        extracted_count: Total shots extracted in the group
        group_label: Label for this group (e.g., "Games 0-4")
        game_shot_counts: Optional dict of {game_id: shot_count} for games in this group
        dropped_games: Optional list of game_ids dropped due to insufficient shots
    """
    total_in_pbp = drop_stats['total_shot_events_processed']
    total_dropped = total_in_pbp - extracted_count
    
    print(f"\n{'=' * 60}")
    print(f"Stats for {group_label}")
    print("=" * 60)
    print(f"Total shot events in play-by-play: {total_in_pbp}")
    print(f"Successfully extracted: {extracted_count}")
    print(f"Total dropped: {total_dropped}")
    
    # Breakdown of drops in load_shot_attempts
    print(f"\nDrops in load_shot_attempts:")
    print(f"  - Missing team or shooter ID: {drop_stats['missing_team_or_shooter']}")
    print(f"  - No tracking match found (time-based): {drop_stats['no_tracking_match']}")
    print(f"  - No moments in matched event: {drop_stats['no_moments']}")
    
    # Breakdown of drops in find_moment_of_release
    release_drops = (drop_stats['release_not_enough_moments'] + 
                     drop_stats['release_no_rim_moment_found'] + 
                     drop_stats['release_no_player_within_threshold'] + 
                     drop_stats['release_first_player_not_shooter'] +
                     drop_stats['release_not_enough_history'])
    print(f"\nDrops in find_moment_of_release ({release_drops} total):")
    print(f"  - Not enough moments (<2): {drop_stats['release_not_enough_moments']}")
    print(f"  - No rim moment found: {drop_stats['release_no_rim_moment_found']}")
    print(f"  - No player within XYZ threshold: {drop_stats['release_no_player_within_threshold']}")
    print(f"  - First player near ball was not shooter: {drop_stats['release_first_player_not_shooter']}")
    print(f"  - Not enough history: {drop_stats['release_not_enough_history']}")
    
    # Calibration stats summary
    total_calibrated = len(calibration_errors['calibrated_games']) + len(calibration_errors['calibrated_games_fallback'])
    total_failed = len(calibration_errors['no_first_moment']) + len(calibration_errors['fallback_failed'])
    total_games = total_calibrated + total_failed
    
    print(f"\nRim Calibration Summary:")
    print(f"  - Games processed: {total_games}")
    print(f"  - Calibrated successfully (total): {total_calibrated}")
    print(f"      - Primary method: {len(calibration_errors['calibrated_games'])}")
    print(f"      - Fallback method: {len(calibration_errors['calibrated_games_fallback'])}")
    print(f"  - Failed to calibrate (total): {total_failed}")
    print(f"      - No moments found: {len(calibration_errors['no_first_moment'])}")
    print(f"      - Fallback failed: {len(calibration_errors['fallback_failed'])}")
    
    # Detailed calibration info for each game in this group
    print(f"\n--- Calibration Details for Each Game ---")
    
    # Primary method successes
    for details in calibration_errors['calibrated_games']:
        shot_count = game_shot_counts.get(details['game_id'], 0) if game_shot_counts else 'N/A'
        was_dropped = details['game_id'] in (dropped_games or [])
        dropped_str = " [DROPPED: <100 shots]" if was_dropped else ""
        print(f"  Game {details['game_id']}: PRIMARY - Home shoots at {details['home_shoots_at_q1q2'].upper()} rim (Q1/Q2), {shot_count} shots{dropped_str}")
    
    # Fallback method successes
    for details in calibration_errors['calibrated_games_fallback']:
        shot_count = game_shot_counts.get(details['game_id'], 0) if game_shot_counts else 'N/A'
        was_dropped = details['game_id'] in (dropped_games or [])
        dropped_str = " [DROPPED: <100 shots]" if was_dropped else ""
        print(f"  Game {details['game_id']}: FALLBACK ({details.get('primary_failure_reason', 'N/A')}) - Home shoots at {details['home_shoots_at_q1q2'].upper()} rim (Q1/Q2), {shot_count} shots{dropped_str}")
    
    # Failures
    for details in calibration_errors['no_first_moment']:
        print(f"  Game {details['game_id']}: FAILED - No moments found")
    
    for details in calibration_errors['fallback_failed']:
        primary_reason = details.get('primary_failure_reason', 'unknown')
        print(f"  Game {details['game_id']}: FAILED - Primary: {primary_reason}, Fallback also failed")
    
    # Games dropped due to insufficient shots
    if dropped_games:
        print(f"\n--- Games Dropped (<100 shots) ---")
        for game_id in dropped_games:
            shot_count = game_shot_counts.get(game_id, 0) if game_shot_counts else 'N/A'
            print(f"  Game {game_id}: {shot_count} shots extracted, output file NOT saved")
    
    print(f"\n{'=' * 60}\n")

# Minimum shots threshold - games with fewer than this many successful extractions are dropped
MIN_SHOTS_THRESHOLD = 100

if __name__ == "__main__":
    # Load PBP data once
    if os.path.exists(PBP_CACHE_FILE):
        pbp = pd.read_csv(PBP_CACHE_FILE)
    else:
        pbp = pd.read_csv(PBP_URL)
        pbp.to_csv(PBP_CACHE_FILE, index=False)
    
    # Find all local JSON files and apply START_INDEX and NUM_GAMES
    all_json_files = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, "*.json")))
    
    end_index = START_INDEX + NUM_GAMES if NUM_GAMES is not None else len(all_json_files)
    json_files = all_json_files[START_INDEX:end_index]
    
    print(f"Processing games {START_INDEX} to {START_INDEX + len(json_files) - 1} "
          f"(total: {len(json_files)} games)")
    print(f"Stats will be printed every {STATS_GROUP_SIZE} games")
    print(f"Games with fewer than {MIN_SHOTS_THRESHOLD} shots will be dropped (no output file saved)")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize per-group stats (reset every STATS_GROUP_SIZE)
    group_drop_stats = create_empty_drop_stats()
    group_calibration_errors = create_empty_calibration_errors()
    group_extracted = 0
    group_game_shot_counts = {}  # {game_id: shot_count}
    group_dropped_games = []  # game_ids dropped due to insufficient shots
    
    # Initialize overall totals
    overall_extracted = 0
    overall_shot_events = 0
    overall_games_calibrated_primary = 0
    overall_games_calibrated_fallback = 0
    overall_games_failed_calibration = 0
    overall_games_dropped_insufficient_shots = 0
    overall_games_saved = 0
    
    # Process each JSON file
    for i, json_file in enumerate(json_files):
        game_shots = load_shot_attempts(
            json_file, pbp, group_drop_stats, group_calibration_errors,
            rim_height=RIM_HEIGHT,
            rim_xy_threshold=RIM_XY_THRESHOLD,
            player_xyz_threshold=PLAYER_XYZ_THRESHOLD,
            min_release_height=MIN_RELEASE_HEIGHT,
            num_snapshots=NUM_SNAPSHOTS,
            moments_back=MOMENTS_BACK,
            time_match_tolerance=TIME_MATCH_TOLERANCE
        )
        
        # Track shot count for this game
        game_id = None
        if game_shots:
            game_id = game_shots[0]['game_id']
            group_game_shot_counts[game_id] = len(game_shots)
        
        group_extracted += len(game_shots)
        
        # Save output file for this game ONLY if it has >= MIN_SHOTS_THRESHOLD shots
        if game_shots:
            if len(game_shots) >= MIN_SHOTS_THRESHOLD:
                input_basename = os.path.basename(json_file)
                output_filename = f"shots_{input_basename}"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                with open(output_path, 'w') as f:
                    json.dump(game_shots, f, indent=2)
            else:
                # Game has fewer than threshold shots - don't save and track as dropped
                group_dropped_games.append(game_id)
        
        # Check if we should print group stats (every STATS_GROUP_SIZE games)
        games_processed_in_group = (i + 1) % STATS_GROUP_SIZE
        if games_processed_in_group == 0 or i == len(json_files) - 1:
            global_idx_start = START_INDEX + (i // STATS_GROUP_SIZE) * STATS_GROUP_SIZE
            global_idx_end = START_INDEX + i
            group_label = f"Games {global_idx_start}-{global_idx_end}"
            
            print_group_stats(
                group_drop_stats, group_calibration_errors, group_extracted, group_label,
                game_shot_counts=group_game_shot_counts,
                dropped_games=group_dropped_games
            )
            
            # Accumulate to overall totals before resetting
            overall_extracted += group_extracted
            overall_shot_events += group_drop_stats['total_shot_events_processed']
            overall_games_calibrated_primary += len(group_calibration_errors['calibrated_games'])
            overall_games_calibrated_fallback += len(group_calibration_errors['calibrated_games_fallback'])
            overall_games_failed_calibration += (
                len(group_calibration_errors['no_first_moment']) +
                len(group_calibration_errors['fallback_failed'])
            )
            overall_games_dropped_insufficient_shots += len(group_dropped_games)
            
            # Count games that were actually saved (calibrated successfully AND >= threshold shots)
            games_saved_this_group = 0
            for details in group_calibration_errors['calibrated_games']:
                if details['game_id'] not in group_dropped_games:
                    games_saved_this_group += 1
            for details in group_calibration_errors['calibrated_games_fallback']:
                if details['game_id'] not in group_dropped_games:
                    games_saved_this_group += 1
            overall_games_saved += games_saved_this_group
            
            # Reset group stats for next group (unless this is the last game)
            if i < len(json_files) - 1:
                group_drop_stats = create_empty_drop_stats()
                group_calibration_errors = create_empty_calibration_errors()
                group_extracted = 0
                group_game_shot_counts = {}
                group_dropped_games = []
    
    # Print overall totals
    print(f"\n{'=' * 60}")
    print("OVERALL TOTALS")
    print("=" * 60)
    print(f"Games processed: {len(json_files)} (indices {START_INDEX} to {START_INDEX + len(json_files) - 1})")
    print(f"Total shot events in play-by-play: {overall_shot_events}")
    print(f"Total shots extracted: {overall_extracted}")
    print(f"Total shots dropped: {overall_shot_events - overall_extracted}")
    print(f"\nCalibration Results:")
    print(f"  - Calibrated via primary method: {overall_games_calibrated_primary}")
    print(f"  - Calibrated via fallback method: {overall_games_calibrated_fallback}")
    print(f"  - Failed calibration: {overall_games_failed_calibration}")
    print(f"\nOutput Files:")
    print(f"  - Games with output files saved: {overall_games_saved}")
    print(f"  - Games dropped (<{MIN_SHOTS_THRESHOLD} shots): {overall_games_dropped_insufficient_shots}")
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    

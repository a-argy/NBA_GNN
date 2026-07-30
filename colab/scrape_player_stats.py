"""
Build player stats from basketball-reference CSV data.
Maps player names from shot data to their shooting statistics.

This module is called by build_graph.py for each game.
It does NOT save to disk - it returns the stats dict directly.

Expects CSV file at: NBA_GNN_files/supplemental_data/player_shooting_stats_2016.csv
"""

import pandas as pd
import os
import unicodedata
import re

# NBA Team ID to abbreviation mapping (2015-16 season)
TEAM_ID_TO_ABBR = {
    1610612737: "ATL",  # Atlanta Hawks
    1610612738: "BOS",  # Boston Celtics
    1610612739: "CLE",  # Cleveland Cavaliers
    1610612740: "NOP",  # New Orleans Pelicans
    1610612741: "CHI",  # Chicago Bulls
    1610612742: "DAL",  # Dallas Mavericks
    1610612743: "DEN",  # Denver Nuggets
    1610612744: "GSW",  # Golden State Warriors
    1610612745: "HOU",  # Houston Rockets
    1610612746: "LAC",  # Los Angeles Clippers
    1610612747: "LAL",  # Los Angeles Lakers
    1610612748: "MIA",  # Miami Heat
    1610612749: "MIL",  # Milwaukee Bucks
    1610612750: "MIN",  # Minnesota Timberwolves
    1610612751: "BRK",  # Brooklyn Nets
    1610612752: "NYK",  # New York Knicks
    1610612753: "ORL",  # Orlando Magic
    1610612754: "IND",  # Indiana Pacers
    1610612755: "PHI",  # Philadelphia 76ers
    1610612756: "PHO",  # Phoenix Suns
    1610612757: "POR",  # Portland Trail Blazers
    1610612758: "SAC",  # Sacramento Kings
    1610612759: "SAS",  # San Antonio Spurs
    1610612760: "OKC",  # Oklahoma City Thunder
    1610612761: "TOR",  # Toronto Raptors
    1610612762: "UTA",  # Utah Jazz
    1610612763: "MEM",  # Memphis Grizzlies
    1610612764: "WAS",  # Washington Wizards
    1610612765: "DET",  # Detroit Pistons
    1610612766: "CHO",  # Charlotte Hornets (was CHA, now CHO)
}

# Cache the CSV dataframe and name mapping to avoid reloading for each game
_CSV_CACHE = None
_NAME_TO_ROWS_CACHE = None


def normalize_name(name):
    """Normalize player name for matching."""
    if not name:
        return ""
    
    # Remove common suffixes (Jr., Sr., II, III, IV, V) - do this BEFORE other normalization
    suffixes = [r'\s+Jr\.?', r'\s+Sr\.?', r'\s+II', r'\s+III', r'\s+IV', r'\s+V']
    for suffix in suffixes:
        name = re.sub(suffix, '', name, flags=re.IGNORECASE)
    
    # Handle special characters that don't normalize well
    replacements = {
        'ß': 'ss',  # German sharp S
        'ø': 'o',   # Nordic o
        'æ': 'ae',  # Nordic ae
        'þ': 'th',  # Icelandic thorn
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
        name = name.replace(old.upper(), new.upper())
    
    # Remove accents and normalize unicode
    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])
    
    # Remove all non-alphanumeric characters and convert to lowercase
    name = re.sub(r'[^a-zA-Z0-9]', '', name).lower()
    
    return name


def _load_csv_cache(csv_path):
    """
    Load and cache the CSV data and name-to-rows mapping.
    Only loads once per session.
    """
    global _CSV_CACHE, _NAME_TO_ROWS_CACHE
    
    if _CSV_CACHE is not None:
        return _CSV_CACHE, _NAME_TO_ROWS_CACHE
    
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found at {csv_path}")
        return None, None
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Create normalized name column in CSV
    df['normalized_name'] = df['name_display'].apply(normalize_name)
    
    # Build mapping of normalized_name -> list of rows (for handling duplicates/traded players)
    name_to_rows = {}
    for idx, row in df.iterrows():
        norm_name = row['normalized_name']
        if norm_name not in name_to_rows:
            name_to_rows[norm_name] = []
        name_to_rows[norm_name].append(row)
    
    _CSV_CACHE = df
    _NAME_TO_ROWS_CACHE = name_to_rows
    
    print(f"Loaded player stats CSV with {len(df)} entries")
    
    return df, name_to_rows


def extract_player_team_mapping(shot_data):
    """
    Extract a mapping of player_id -> team_id from a shot's position data.
    
    Args:
        shot_data: A single shot dict with offense_position and defense_position
    
    Returns:
        Dict mapping player_id (int) -> team_id (int)
    """
    player_team_mapping = {}
    
    # Extract from offense positions
    for player in shot_data.get('offense_position', []):
        if len(player) >= 2:
            team_id, player_id = player[0], player[1]
            player_team_mapping[player_id] = team_id
    
    # Extract from defense positions
    for player in shot_data.get('defense_position', []):
        if len(player) >= 2:
            team_id, player_id = player[0], player[1]
            player_team_mapping[player_id] = team_id
    
    return player_team_mapping


def build_player_stats_for_game(first_shot, csv_path):
    """
    Build player stats dict for a game by matching player names from shot data
    to entries in the shooting stats CSV.
    
    This function does NOT save to disk - it returns the stats dict directly.
    
    Args:
        first_shot: Dict containing first shot data with 'game_id', 'player_names', 
                   'offense_position', and 'defense_position'
        csv_path: Path to the player_shooting_stats_2016.csv file
    
    Returns:
        tuple: (player_id_to_stats, unmatched_players, game_id) where:
            - player_id_to_stats: Dict mapping player_id (str) -> stats dict
            - unmatched_players: List of dicts with unmatched player details
            - game_id: The game ID string
    """
    if not first_shot:
        print("ERROR: No shot data provided")
        return {}, [], None
    
    game_id = first_shot.get('game_id', 'unknown')
    
    # Get player names and team mapping from first shot
    all_player_names = first_shot.get('player_names', {})
    if not all_player_names:
        print(f"WARNING: No player_names found in first shot for game {game_id}")
        return {}, [], game_id
    
    player_team_mapping = extract_player_team_mapping(first_shot)
    
    # Load CSV (cached after first call)
    df, name_to_rows = _load_csv_cache(csv_path)
    if df is None or name_to_rows is None:
        return {}, [], game_id
    
    player_id_to_stats = {}
    unmatched_players = []
    matched_count = 0
    
    for player_id, player_name in all_player_names.items():
        player_id_str = str(player_id)
        norm_name = normalize_name(player_name)
        
        if norm_name not in name_to_rows:
            # No match found - collect detailed info for debugging
            team_id = player_team_mapping.get(player_id) or player_team_mapping.get(int(player_id) if isinstance(player_id, str) else player_id)
            team_abbr = TEAM_ID_TO_ABBR.get(team_id, 'UNKNOWN')
            
            # Find similar names in CSV (for typo detection)
            similar_names = []
            for csv_norm_name in name_to_rows.keys():
                # Check if names share significant overlap (simple heuristic)
                if len(csv_norm_name) >= 4 and len(norm_name) >= 4:
                    # Check for substring match or edit distance
                    if csv_norm_name[:4] == norm_name[:4] or norm_name[:4] == csv_norm_name[:4]:
                        # Get original display name from CSV
                        original_csv_name = name_to_rows[csv_norm_name][0]['name_display']
                        similar_names.append(original_csv_name)
            
            unmatched_info = {
                'player_id': player_id,
                'original_name': player_name,
                'normalized_name': norm_name,
                'team_id': team_id,
                'team_abbr': team_abbr,
                'game_id': game_id,
                'similar_csv_names': similar_names[:5]  # Limit to 5 suggestions
            }
            unmatched_players.append(unmatched_info)
            
            # Use all zeros for stats
            player_id_to_stats[player_id_str] = {
                'name': player_name,
                'age': 0.0,
                'games': 0,
                'minutes': 0.0,
                'fg_pct': 0.0,
                'avg_dist': 0.0,
                'pct_fga_fg2a': 0.0,
                'pct_fga_00_03': 0.0,
                'pct_fga_03_10': 0.0,
                'pct_fga_10_16': 0.0,
                'pct_fga_16_xx': 0.0,
                'pct_fga_fg3a': 0.0,
                'fg2_pct': 0.0,
                'fg_pct_00_03': 0.0,
                'fg_pct_03_10': 0.0,
                'fg_pct_10_16': 0.0,
                'fg_pct_16_xx': 0.0,
                'fg3_pct': 0.0,
                'pct_ast_fg2': 0.0,
                'pct_ast_fg3': 0.0,
                'pct_fga_dunk': 0.0,
                'pct_fga_corner3': 0.0,
                'fg_pct_corner3': 0.0,
            }
            continue
        
        matching_rows = name_to_rows[norm_name]
        
        # If only one match, use it
        if len(matching_rows) == 1:
            row = matching_rows[0]
        else:
            # Multiple matches - could be traded player OR multiple players with same name
            # Priority: team match first (handles same-name players correctly),
            # then combined stats (2TM), then first entry as fallback
            selected_row = None
            
            # Step 1: Try exact team match first (critical for same-name disambiguation)
            team_id = player_team_mapping.get(player_id) or player_team_mapping.get(int(player_id) if isinstance(player_id, str) else player_id)
            team_abbr = TEAM_ID_TO_ABBR.get(team_id, None)
            
            if team_abbr:
                for r in matching_rows:
                    row_team = r.get('team_name_abbr', '')
                    if row_team == team_abbr:
                        selected_row = r
                        break
            
            # Step 2: If no team match, look for combined stats entry (2TM, 3TM, etc.)
            if selected_row is None:
                for r in matching_rows:
                    row_team = r.get('team_name_abbr', '')
                    if row_team in ('2TM', '3TM'): # Only 2TM and 3TM appear in season stats data
                        selected_row = r
                        break
            
            # Step 3: Last resort - take the first entry
            if selected_row is None:
                selected_row = matching_rows[0]
            
            row = selected_row
        
        # Helper to safely convert to float, handling NaN
        def safe_float(val, default=0.0):
            if pd.isna(val) or val is None or val == '':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Extract stats from the row
        player_id_to_stats[player_id_str] = {
            'name': player_name,
            'age': safe_float(row.get('age', 0)),
            'games': int(safe_float(row.get('games', 0))),
            'minutes': safe_float(row.get('mp', 0)),
            'fg_pct': safe_float(row.get('fg_pct', 0)),
            'avg_dist': safe_float(row.get('avg_dist', 0)),
            'pct_fga_fg2a': safe_float(row.get('pct_fga_fg2a', 0)),
            'pct_fga_00_03': safe_float(row.get('pct_fga_00_03', 0)),
            'pct_fga_03_10': safe_float(row.get('pct_fga_03_10', 0)),
            'pct_fga_10_16': safe_float(row.get('pct_fga_10_16', 0)),
            'pct_fga_16_xx': safe_float(row.get('pct_fga_16_xx', 0)),
            'pct_fga_fg3a': safe_float(row.get('pct_fga_fg3a', 0)),
            'fg2_pct': safe_float(row.get('fg_pct_fg2a', 0)),
            'fg_pct_00_03': safe_float(row.get('fg_pct_00_03', 0)),
            'fg_pct_03_10': safe_float(row.get('fg_pct_03_10', 0)),
            'fg_pct_10_16': safe_float(row.get('fg_pct_10_16', 0)),
            'fg_pct_16_xx': safe_float(row.get('fg_pct_16_xx', 0)),
            'fg3_pct': safe_float(row.get('fg_pct_fg3a', 0)),
            'pct_ast_fg2': safe_float(row.get('pct_ast_fg2', 0)),
            'pct_ast_fg3': safe_float(row.get('pct_ast_fg3', 0)),
            'pct_fga_dunk': safe_float(row.get('pct_fga_dunk', 0)),
            'pct_fga_corner3': safe_float(row.get('pct_fg3a_corner3', 0)),
            'fg_pct_corner3': safe_float(row.get('fg_pct_corner3', 0)),
        }
        matched_count += 1
    
    # Print summary
    print(f"\n📊 Player Matching Summary for Game {game_id}:")
    print(f"   Total players: {len(all_player_names)}")
    print(f"   Matched: {matched_count}")
    print(f"   Unmatched: {len(unmatched_players)}")
    
    return player_id_to_stats, unmatched_players, game_id

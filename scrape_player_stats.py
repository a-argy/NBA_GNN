"""
Scrape NBA player shooting stats from basketball-reference.com
and create a mapping from player names to their stats.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time

def scrape_shooting_stats(year=2016):
    """
    Scrape shooting stats from basketball-reference.com
    
    Args:
        year: NBA season year (e.g., 2016 for 2015-16 season)
    
    Returns:
        DataFrame with player shooting statistics
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_shooting.html"
    
    print(f"Scraping {url}...")
    
    # Add headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: Failed to fetch data (status code: {response.status_code})")
        return None
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the shooting stats table
    table = soup.find('table', {'id': 'shooting'})
    
    if not table:
        print("Error: Could not find shooting stats table")
        return None
    
    # Extract headers
    headers_row = table.find('thead').find_all('tr')[-1]  # Get last header row
    headers = [th.get('data-stat') for th in headers_row.find_all('th')]
    
    # Extract data rows
    tbody = table.find('tbody')
    rows = []
    
    for tr in tbody.find_all('tr', class_=lambda x: x != 'thead'):  # Skip header rows in tbody
        if tr.find('th', {'scope': 'row'}):  # Valid data row
            row_data = {}
            for th in tr.find_all(['th', 'td']):
                stat_name = th.get('data-stat')
                stat_value = th.text.strip()
                row_data[stat_name] = stat_value
            rows.append(row_data)
    
    df = pd.DataFrame(rows)
    
    print(f"Successfully scraped {len(df)} players")
    
    return df


def clean_shooting_stats(df):
    """
    Clean and process the shooting stats dataframe.
    Convert percentages and numeric fields to proper types.
    """
    if df is None or df.empty:
        return None
    
    # Numeric columns to convert
    numeric_cols = [
        'age', 'g', 'mp',
        'fg_pct', 'avg_dist',
        'pct_fga_fg2a', 'pct_fga_00_03', 'pct_fga_03_10', 
        'pct_fga_10_16', 'pct_fga_16_xx', 'pct_fga_fg3a',
        'fg2_pct', 'pct_fg2a_00_03', 'pct_fg2a_03_10',
        'pct_fg2a_10_16', 'pct_fg2a_16_xx',
        'fg3_pct', 'pct_fg3a_corner3', 'fg3_pct_corner3',
        'pct_ast_fg2', 'pct_ast_fg3',
        'pct_fga_dunk', 'fg_pct_dunk',
        'pct_fga_layup', 'fg_pct_layup',
        'pct_fg3a_heave', 'fg3_pct_heave'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values with 0 (players with no attempts in certain categories)
    df = df.fillna(0)
    
    return df


def map_player_names_to_ids(stats_df, shot_data_file='shot_data_new_temp.json'):
    """
    Create a mapping from player names to player IDs using shot data.
    
    Args:
        stats_df: DataFrame with player stats from basketball-reference
        shot_data_file: Path to shot data JSON file with player_names field
    
    Returns:
        Dictionary mapping player_id to stats
    """
    import os
    
    if not os.path.exists(shot_data_file):
        print(f"Error: {shot_data_file} not found. Run load_with_context.py first to generate it.")
        return {}
    
    # Collect all unique player name -> ID mappings from shot data
    player_name_to_id = {}
    
    print(f"\nLoading player mappings from {shot_data_file}...")
    
    with open(shot_data_file, 'r') as f:
        shots = json.load(f)
    
    print(f"Found {len(shots)} shots in dataset")
    
    # Extract player names and IDs from shots
    for shot in shots:
        if 'player_names' in shot:
            for player_id, name_info in shot['player_names'].items():
                full_name = name_info.get('full_name', '').strip()
                if full_name and player_id:
                    player_name_to_id[full_name] = int(player_id)
    
    print(f"Found {len(player_name_to_id)} unique player name -> ID mappings")

    # Now map stats to plxxayer IDs
    player_id_to_stats = {}
    unmatched_players = []
    
    for _, row in stats_df.iterrows():
        player_name = row.get('name_display', '')
        
        if player_name in player_name_to_id:
            player_id = player_name_to_id[player_name]
            
            # Store relevant stats as a dict - ALL available shooting stats
            player_id_to_stats[player_id] = {
                'name': player_name,
                'age': row.get('age', 0),
                'games': row.get('g', 0),
                'minutes': row.get('mp', 0),
                'fg_pct': row.get('fg_pct', 0),
                'avg_dist': row.get('avg_dist', 0),
                'pct_fga_fg2a': row.get('pct_fga_fg2a', 0),
                'pct_fga_00_03': row.get('pct_fga_00_03', 0),
                'pct_fga_03_10': row.get('pct_fga_03_10', 0),
                'pct_fga_10_16': row.get('pct_fga_10_16', 0),
                'pct_fga_16_xx': row.get('pct_fga_16_xx', 0),
                'pct_fga_fg3a': row.get('pct_fga_fg3a', 0),
                'fg2_pct': row.get('fg2_pct', 0),
                'fg_pct_00_03': row.get('pct_fg2a_00_03', 0),
                'fg_pct_03_10': row.get('pct_fg2a_03_10', 0),
                'fg_pct_10_16': row.get('pct_fg2a_10_16', 0),
                'fg_pct_16_xx': row.get('pct_fg2a_16_xx', 0),
                'fg3_pct': row.get('fg3_pct', 0),
                'pct_ast_fg2': row.get('pct_ast_fg2', 0),
                'pct_ast_fg3': row.get('pct_ast_fg3', 0),
                'pct_fga_dunk': row.get('pct_fga_dunk', 0),
                'fg_pct_dunk': row.get('fg_pct_dunk', 0),
                'pct_fga_corner3': row.get('pct_fg3a_corner3', 0),
                'fg_pct_corner3': row.get('fg3_pct_corner3', 0),
            }
        else:
            unmatched_players.append(player_name)
    
    print(f"\nMatched {len(player_id_to_stats)} players to their stats")
    print(f"Unmatched: {len(unmatched_players)} players")
    
    if unmatched_players[:10]:
        print(f"Sample unmatched: {unmatched_players[:10]}")
    
    return player_id_to_stats


def save_player_stats(player_id_to_stats, output_file='player_stats_2016.json'):
    """Save player stats mapping to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(player_id_to_stats, f, indent=2)
    
    print(f"\nSaved player stats to {output_file}")
    print(f"Total players: {len(player_id_to_stats)}")


if __name__ == "__main__":
    # Scrape shooting stats
    df = scrape_shooting_stats(year=2016)
    
    if df is not None:
        # Clean the data
        df = clean_shooting_stats(df)
        
        # Save raw stats
        df.to_csv('player_shooting_stats_2016.csv', index=False)
        print(f"\nSaved raw stats to player_shooting_stats_2016.csv")
        
        # Map to player IDs
        player_id_to_stats = map_player_names_to_ids(df)
        
        # Save the mapping
        save_player_stats(player_id_to_stats)
        
        print("\nDone! You can now use player_stats_2016.json in build_graph.py")

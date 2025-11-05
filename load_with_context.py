#!/usr/bin/env python3
"""
Direct loader for NBA tracking data combined with play-by-play context
No Hugging Face datasets required - just pure Python
"""

import json
import os
import glob
import pandas as pd
import requests

# Configuration
LOCAL_DATA_DIR = "/Users/anthonyargyropoulos/Documents/GitHub/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs"
PBP_URL = "https://github.com/sumitrodatta/nba-alt-awards/raw/main/Historical/PBP%20Data/2015-16_pbp.csv"

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


if __name__ == "__main__":
    # Load the data
    print("=" * 60)
    print("Loading NBA Tracking Data with Play-by-Play Context")
    print("=" * 60)
    
    events = load_nba_data_with_context()
    
    print(f"\n" + "=" * 60)
    print(f"Successfully loaded {len(events)} events with context!")
    print("=" * 60)
    
    # Display first event details
    if events:
        first_event = events[0]
        print(f"\n{'=' * 60}")
        print("First Event Details")
        print("=" * 60)
        print(f"Game ID: {first_event['gameid']}")
        print(f"Game Date: {first_event['gamedate']}")
        print(f"Event ID: {first_event['event_info']['id']}")
        print(f"Event Type: {first_event['event_info']['type']}")
        print(f"Home Description: {first_event['event_info']['desc_home']}")
        print(f"Away Description: {first_event['event_info']['desc_away']}")
        print(f"Possession Team ID: {first_event['event_info']['possession_team_id']}")
        print(f"Number of moments: {len(first_event['moments'])}")
        
        # Show team info
        print(f"\nHome Team: {first_event['home']['name']} ({first_event['home']['abbreviation']})")
        print(f"Visitor Team: {first_event['visitor']['name']} ({first_event['visitor']['abbreviation']})")
        
        # Show first moment
        if first_event['moments']:
            first_moment = first_event['moments'][0]
            print(f"\n{'=' * 60}")
            print("First Moment")
            print("=" * 60)
            print(f"Quarter: {first_moment['quarter']}")
            print(f"Game Clock: {first_moment['game_clock']:.2f}s")
            print(f"Shot Clock: {first_moment['shot_clock']:.2f}s")
            print(f"Ball Position: "
                  f"x={first_moment['ball_coordinates']['x']:.2f}, "
                  f"y={first_moment['ball_coordinates']['y']:.2f}, "
                  f"z={first_moment['ball_coordinates']['z']:.2f} ft")
            print(f"Number of players tracked: {len(first_moment['player_coordinates'])}")
            
            # Show a player position
            if first_moment['player_coordinates']:
                player = first_moment['player_coordinates'][0]
                print(f"\nExample Player Position:")
                print(f"  Team ID: {player['teamid']}")
                print(f"  Player ID: {player['playerid']}")
                print(f"  Position: x={player['x']:.2f}, y={player['y']:.2f}, z={player['z']:.2f} ft")
    
    print(f"\n{'=' * 60}")
    print("✓ Data loaded successfully!")
    print("=" * 60)
    print(f"\nTo use this data in your own code:")
    print("  from load_with_context import load_nba_data_with_context")
    print("  events = load_nba_data_with_context()")
    print("  # events is a list of dictionaries with tracking + context")


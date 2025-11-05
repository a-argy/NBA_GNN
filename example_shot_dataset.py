#!/usr/bin/env python3
"""
Example script to load shot attempt data and save it as JSON
"""

import json
from load_with_context import load_shot_attempts

def main():
    # Load shot attempts
    print("Loading shot attempts...")
    shots = load_shot_attempts()
    
    print(f"\nLoaded {len(shots)} shot attempts")
    
    # Show some statistics
    if shots:
        made_shots = sum(1 for shot in shots if shot['made'])
        missed_shots = len(shots) - made_shots
        fg_pct = (made_shots / len(shots) * 100) if shots else 0
        
        print(f"\nOverall Statistics:")
        print(f"  Total Shots: {len(shots)}")
        print(f"  Made: {made_shots}")
        print(f"  Missed: {missed_shots}")
        print(f"  FG%: {fg_pct:.1f}%")
        
        # Analyze player counts
        offense_counts = [len(shot['offense_position']) for shot in shots]
        defense_counts = [len(shot['defense_position']) for shot in shots]
        
        print(f"\nPlayer Position Data:")
        print(f"  Avg offensive players per shot: {sum(offense_counts)/len(offense_counts):.1f}")
        print(f"  Avg defensive players per shot: {sum(defense_counts)/len(defense_counts):.1f}")
    
    # Save to JSON
    output_file = "shot_dataset.json"
    print(f"\nSaving data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(shots, f, indent=2)
    
    print(f"✓ Successfully saved {len(shots)} shots to {output_file}")
    
    # Show example of first shot
    if shots:
        print(f"\n{'='*60}")
        print("Example Shot Entry (First Shot):")
        print("Ball captured at rim height (10ft) and rim coordinates")
        print(f"{'='*60}")
        print(json.dumps(shots[0], indent=2))

if __name__ == "__main__":
    main()


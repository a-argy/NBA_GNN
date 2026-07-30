import glob
import os
from build_graph import process_single_game, print_game_result

# Paths
BASE_DIR = "/content/drive/MyDrive/NBA_GNN_files/"
SHOTS_DIR = os.path.join(BASE_DIR, "shots_data")
CSV_PATH = os.path.join(BASE_DIR, "supplemental_data", "player_shooting_stats_2016.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "graph_data")

# Get all shot files
shot_files = sorted(glob.glob(os.path.join(SHOTS_DIR, "shots_*.json")))
print(f"Found {len(shot_files)} games to process")

# Cell 2: Process all games
total_graphs = 0
total_made = 0
total_missed = 0
total_skip_stats = {}
all_unmatched = {}  # Dict with player name as key

# Load all the shot files
for i, shot_file in enumerate(shot_files):
    print(f"\n[{i+1}/{len(shot_files)}] {os.path.basename(shot_file)}")

    result = process_single_game(shot_file, CSV_PATH, OUTPUT_DIR)
    print_game_result(result)

    # Aggregate
    total_graphs += result['num_graphs']
    total_made += result['num_made']
    total_missed += result['num_missed']

    for reason, count in result['skip_stats'].items():
        total_skip_stats[reason] = total_skip_stats.get(reason, 0) + count

    # Aggregate unmatched players by name
    for player_info in result['unmatched_players']:
        player_name = player_info['original_name']
        all_unmatched[player_name] = all_unmatched.get(player_name, 0) + 1

# Final summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Total graphs: {total_graphs}")
if total_graphs > 0:
    print(f"  Made: {total_made} ({100*total_made/total_graphs:.1f}%)")
    print(f"  Missed: {total_missed} ({100*total_missed/total_graphs:.1f}%)")

total_skipped = sum(total_skip_stats.values())
if total_skipped > 0:
    print(f"\nTotal skipped: {total_skipped}")
    for reason, count in sorted(total_skip_stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  - {reason}: {count}")

if all_unmatched:
    print(f"\nUnmatched players across all games: {len(all_unmatched)} unique players")
    print("Top 10 most frequent:")
    for name, count in sorted(all_unmatched.items(), key=lambda x: -x[1])[:10]:
        print(f"  - {name}: appeared in {count} games")

print(f"\n✅ Graphs saved to: {OUTPUT_DIR}")
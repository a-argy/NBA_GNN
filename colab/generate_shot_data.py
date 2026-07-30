import os
import sys
import json
import time
import glob
import pandas as pd
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Configuration - EDIT THESE
DATA_DIR = "/content/drive/MyDrive/NBA_GNN_files/game_data"
OUTPUT_DIR = "/content/drive/MyDrive/NBA_GNN_files/shots_data"
PBP_FILE = "/content/drive/MyDrive/NBA_GNN_files/pbp_cache.csv"

# Failure settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries
RETRY_EXCEPTIONS = (OSError, IOError, TimeoutError, ConnectionError)

# Processing configuration
START_INDEX = 14
NUM_GAMES = None           # Number of games to process (None = all remaining)
STATS_GROUP_SIZE = 5     # Print stats for groups of N games
DELAY_BETWEEN_GAMES = 0.5  # Seconds between games (helps Drive stability)
MIN_SHOTS_THRESHOLD = 90  # Games with fewer shots are dropped (no output file saved)

# Parameters for find_moment_of_release:
RIM_HEIGHT = 9.7
RIM_XY_THRESHOLD = 4.0
PLAYER_XYZ_THRESHOLD = 9.0
MIN_RELEASE_HEIGHT = 6.5
NUM_SNAPSHOTS = 3
MOMENTS_BACK = [3, 12]

# Add script location to path
sys.path.insert(0, "/content/drive/MyDrive/NBA_GNN_files")
import load_with_context

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all game files and load PBP data ONCE
all_json_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
total_games = len(all_json_files)
pbp = pd.read_csv(PBP_FILE)

# Determine which games to process
end_index = START_INDEX + NUM_GAMES if NUM_GAMES is not None else len(all_json_files)
json_files = all_json_files[START_INDEX:end_index]

print(f"Total games available: {total_games}")
print(f"Processing games {START_INDEX} to {START_INDEX + len(json_files) - 1} (total: {len(json_files)} games)")
print(f"Stats will be printed every {STATS_GROUP_SIZE} games")
print(f"Games with fewer than {MIN_SHOTS_THRESHOLD} shots will be dropped (no output file saved)")
print("=" * 60)

# Initialize per-group stats (reset every STATS_GROUP_SIZE)
group_drop_stats = load_with_context.create_empty_drop_stats()
group_calibration_errors = load_with_context.create_empty_calibration_errors()
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

try:
    for i, json_file in enumerate(json_files):
        current_index = START_INDEX + i
        filename = os.path.basename(json_file)

        # Check Drive connection every 5 games
        if i % 5 == 0 and not os.path.exists(DATA_DIR):
            print(f"\n⚠️ Drive disconnected at index {current_index}!")
            break

        # Retry logic for processing
        game_shots = []
        for attempt in range(MAX_RETRIES):
            try:
                game_shots = load_with_context.load_shot_attempts(
                    json_file, pbp, group_drop_stats, group_calibration_errors,
                    rim_height=RIM_HEIGHT,
                    rim_xy_threshold=RIM_XY_THRESHOLD,
                    player_xyz_threshold=PLAYER_XYZ_THRESHOLD,
                    min_release_height=MIN_RELEASE_HEIGHT,
                    num_snapshots=NUM_SNAPSHOTS,
                    moments_back=MOMENTS_BACK
                )
                break  # Success - exit retry loop
            except RETRY_EXCEPTIONS as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  ⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  ❌ All {MAX_RETRIES} attempts failed for {filename}: {e}")
                    game_shots = []  # Treat as no shots extracted

        # Track shot count for this game (OUTSIDE the retry loop)
        game_id = None
        if game_shots:
            game_id = game_shots[0]['game_id']
            group_game_shot_counts[game_id] = len(game_shots)

        group_extracted += len(game_shots)

        # Save output file for this game ONLY if it has >= MIN_SHOTS_THRESHOLD shots
        if game_shots:
            if len(game_shots) >= MIN_SHOTS_THRESHOLD:
                output_filename = f"shots_{filename}"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                with open(output_path, 'w') as f:
                    json.dump(game_shots, f, indent=2)
                status = "✅"
            else:
                # Game has fewer than threshold shots - don't save and track as dropped
                group_dropped_games.append(game_id)
                status = "⚠️ DROPPED"
        else:
            status = "❌"

        # Status update
        print(f"{status} [{current_index + 1}/{total_games}] {filename}: {len(game_shots)} shots")

        # Check if we should print group stats (every STATS_GROUP_SIZE games)
        games_processed_in_group = (i + 1) % STATS_GROUP_SIZE
        if games_processed_in_group == 0 or i == len(json_files) - 1:
            global_idx_start = START_INDEX + (i // STATS_GROUP_SIZE) * STATS_GROUP_SIZE
            global_idx_end = START_INDEX + i
            group_label = f"Games {global_idx_start}-{global_idx_end}"

            load_with_context.print_group_stats(
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
                group_drop_stats = load_with_context.create_empty_drop_stats()
                group_calibration_errors = load_with_context.create_empty_calibration_errors()
                group_extracted = 0
                group_game_shot_counts = {}
                group_dropped_games = []

        # Small delay
        if DELAY_BETWEEN_GAMES > 0 and i < len(json_files) - 1:
            time.sleep(DELAY_BETWEEN_GAMES)

except Exception as e:
    print(f"\n❌ Error at index {START_INDEX + i}: {e}")
    import traceback
    traceback.print_exc()

# Print overall totals
print(f"\n{'=' * 60}")
print("🏁 OVERALL TOTALS")
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
print(f"\n📁 Output files saved to: {OUTPUT_DIR}")
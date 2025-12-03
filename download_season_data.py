import requests
import os
import py7zr
from tqdm import tqdm
import time

def download_nba_season_data(output_dir='data', limit=None, extract=True, keep_7z=False):
    """
    Download and extract NBA tracking data files from GitHub repository.
    
    Files are stored as .7z compressed archives and need to be extracted.
    
    Args:
        output_dir: Directory to save files
        limit: Maximum number of files to download (None for all)
        extract: Whether to extract .7z files to JSON
        keep_7z: Whether to keep .7z files after extraction
    """
    # GitHub API endpoint for the directory
    api_url = "https://api.github.com/repos/linouk23/NBA-Player-Movements/contents/data/2016.NBA.Raw.SportVU.Game.Logs"
    
    print("Fetching file list from GitHub...")
    response = requests.get(api_url)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return
    
    files = response.json()
    seven_z_files = [f for f in files if f['name'].endswith('.7z')]
    
    print(f"Found {len(seven_z_files)} compressed game files")
    
    if limit:
        seven_z_files = seven_z_files[:limit]
        print(f"Limiting to {limit} files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_count = 0
    extracted_count = 0
    
    # Download and extract files
    for file_info in tqdm(seven_z_files, desc="Processing files"):
        file_name = file_info['name']
        download_url = file_info['download_url']
        seven_z_path = os.path.join(output_dir, file_name)
        
        # Expected JSON filename (remove .7z extension)
        json_name = file_name.replace('.7z', '.json')
        json_path = os.path.join(output_dir, json_name)
        
        # Skip if JSON already exists
        if os.path.exists(json_path):
            tqdm.write(f"  ✓ {json_name} already extracted")
            continue
        
        # Download .7z file if not exists
        if not os.path.exists(seven_z_path):
            try:
                tqdm.write(f"  Downloading {file_name}...")
                file_response = requests.get(download_url, timeout=30)
                if file_response.status_code == 200:
                    with open(seven_z_path, 'wb') as f:
                        f.write(file_response.content)
                    downloaded_count += 1
                    time.sleep(0.5)  # Be nice to GitHub
                else:
                    tqdm.write(f"  ✗ Failed to download {file_name}")
                    continue
            except Exception as e:
                tqdm.write(f"  ✗ Error downloading {file_name}: {e}")
                continue
        
        # Extract .7z file
        if extract and os.path.exists(seven_z_path):
            try:
                tqdm.write(f"  Extracting {file_name}...")
                with py7zr.SevenZipFile(seven_z_path, mode='r') as archive:
                    archive.extractall(path=output_dir)
                extracted_count += 1
                
                # Remove .7z file if requested
                if not keep_7z:
                    os.remove(seven_z_path)
                    
            except Exception as e:
                tqdm.write(f"  ✗ Error extracting {file_name}: {e}")
                continue
    
            
    print(f"\n{'='*60}")
    print("Download Summary")
    print('='*60)
    print(f"Downloaded: {downloaded_count} files")
    print(f"Extracted: {extracted_count} files")
    print(f"Output directory: {output_dir}/")
    print('='*60)


def check_dependencies():
    """Check if py7zr is installed"""
    try:
        import py7zr
        return True
    except ImportError:
        print("Error: py7zr not installed")
        print("\nInstall it with:")
        print("  pip install py7zr")
        return False


if __name__ == "__main__":
    if not check_dependencies():
        exit(1)
    
    print("="*60)
    print("NBA Season Data Downloader")
    print("="*60)
    print("\nThis will download .7z files from GitHub and extract to JSON")
    print("Each file is ~6MB compressed, ~20-30MB extracted\n")
    
    # Download first 25 games
    print("Starting download (first 25 games)...")
    download_nba_season_data(
        output_dir='data',
        limit=200,
        extract=True,
        keep_7z=False  # Delete .7z after extraction to save space
    )
    
    # To download ALL games (~600+ files, ~15-20GB total):
    # Uncomment below and comment out the above
    #download_nba_season_data(output_dir='data', limit=None, extract=True, keep_7z=False)

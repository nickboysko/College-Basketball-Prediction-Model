"""
Download historical team stats from Barttorvik Time Machine

This downloads the exact stats as they were on specific dates,
eliminating data leakage completely.
"""

import pandas as pd
import requests
import gzip
import json
from datetime import datetime, timedelta
import time
import os

print("=" * 80)
print("BARTTORVIK TIME MACHINE DOWNLOADER")
print("=" * 80)

def download_date_stats(date_str):
    """
    Download team stats for a specific date from Barttorvik Time Machine
    
    Args:
        date_str: Date in YYYYMMDD format (e.g., "20220115")
    
    Returns:
        DataFrame with team stats, or None if download fails
    """
    url = f"https://barttorvik.com/timemachine/team_results/{date_str}_team_results.json.gz"
    
    print(f"[FETCH] {date_str}...", end=" ")
    
    try:
        # Download the file
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Try to decompress first (in case it's gzipped)
            try:
                json_data = gzip.decompress(response.content)
                data = json.loads(json_data)
            except:
                # If decompression fails, treat as raw JSON
                data = json.loads(response.content)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Add metadata
            df['stats_date'] = date_str
            
            print(f"✓ ({len(df)} teams)")
            return df
            
        elif response.status_code == 404:
            print(f"✗ (not found)")
            return None
        else:
            print(f"✗ (error {response.status_code})")
            return None
            
    except Exception as e:
        print(f"✗ ({str(e)})")
        return None

def get_unique_game_dates(games_file):
    """
    Extract unique game dates from historical games file
    
    Returns:
        List of dates in YYYYMMDD format
    """
    print("\n[ANALYZE] Finding unique game dates...")
    
    # Try to load as Excel first, then CSV
    try:
        if games_file.endswith('.xlsx') or games_file.endswith('.xls'):
            games = pd.read_excel(games_file)
        else:
            games = pd.read_csv(games_file)
    except Exception as e:
        print(f"[ERROR] Could not load {games_file}: {e}")
        return []
    
    # Convert game_date to datetime
    games['game_date'] = pd.to_datetime(games['game_date'])
    
    # Get unique dates and format as YYYYMMDD
    unique_dates = games['game_date'].dt.strftime('%Y%m%d').unique()
    unique_dates = sorted(unique_dates)
    
    print(f"[OK] Found {len(unique_dates)} unique game dates")
    print(f"[INFO] Date range: {unique_dates[0]} to {unique_dates[-1]}")
    
    return list(unique_dates)

def download_checkpoint_dates(year, season_start_month=11):
    """
    Download stats for standard checkpoint dates within a season
    
    Args:
        year: End year of season (e.g., 2022 for 2021-2022 season)
        season_start_month: Month season starts (default: November)
    
    Returns:
        Dictionary of {checkpoint_name: DataFrame}
    """
    # Define checkpoint dates for a season
    start_year = year - 1
    
    checkpoints = {
        'early': f"{start_year}1201",      # Dec 1
        'mid': f"{year}0115",               # Jan 15
        'late': f"{year}0301",              # Mar 1
        'end': f"{year}0401"                # Apr 1
    }
    
    stats = {}
    
    for name, date_str in checkpoints.items():
        df = download_date_stats(date_str)
        if df is not None:
            stats[name] = df
            
            # Save to CSV
            output_file = f"{year}_stats_{name}.csv"
            df.to_csv(output_file, index=False)
            print(f"  [SAVED] {output_file}")
        
        # Be nice to the server
        time.sleep(1)
    
    return stats

def download_for_all_game_dates(games_file, output_dir="historical_stats"):
    """
    Download stats for every unique date in the games file
    
    This is the most accurate method but requires many downloads
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique dates
    dates = get_unique_game_dates(games_file)
    
    if not dates:
        print("[ERROR] No dates found in games file")
        return
    
    print(f"\n[DOWNLOAD] Downloading stats for {len(dates)} dates...")
    print("[INFO] This will take a while (~{:.1f} minutes)".format(len(dates) * 2 / 60))
    print("[INFO] Being respectful to server with 2-second delays")
    
    successful = 0
    failed = 0
    
    for i, date_str in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] ", end="")
        
        df = download_date_stats(date_str)
        
        if df is not None:
            # Save to CSV
            output_file = os.path.join(output_dir, f"{date_str}_stats.csv")
            df.to_csv(output_file, index=False)
            successful += 1
        else:
            failed += 1
        
        # Be respectful - 2 second delay between requests
        time.sleep(2)
    
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"[OK] Successfully downloaded: {successful}/{len(dates)}")
    if failed > 0:
        print(f"[WARN] Failed downloads: {failed}")
    print(f"[SAVED] Files in: {output_dir}/")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Option 1 (Checkpoints): python download_time_machine.py checkpoint <year>")
        print("    Example: python download_time_machine.py checkpoint 2022")
        print("    Downloads 4 checkpoint dates for the 2021-2022 season")
        print()
        print("  Option 2 (All dates): python download_time_machine.py alldates <games_file>")
        print("    Example: python download_time_machine.py alldates teamrankings_2021-2024.xlsx")
        print("    Downloads stats for every unique game date in the file")
        print()
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "checkpoint":
        if len(sys.argv) < 3:
            print("[ERROR] Please specify year: python download_time_machine.py checkpoint 2022")
            return
        
        year = int(sys.argv[2])
        print(f"\n[MODE] Checkpoint download for {year-1}-{year} season")
        download_checkpoint_dates(year)
        
    elif mode == "alldates":
        if len(sys.argv) < 3:
            print("[ERROR] Please specify games file: python download_time_machine.py alldates games.xlsx")
            return
        
        games_file = sys.argv[2]
        print(f"\n[MODE] Download for all unique dates in {games_file}")
        download_for_all_game_dates(games_file)
        
    else:
        print(f"[ERROR] Unknown mode: {mode}")
        print("[INFO] Use 'checkpoint' or 'alldates'")

if __name__ == "__main__":
    main()
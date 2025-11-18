"""
Download 2024-2025 Season Stats ONLY

This fills the critical gap between April 2024 and current season.
Much faster than downloading all historical dates!
"""

import pandas as pd
import requests
import gzip
import json
from datetime import datetime, timedelta
import time
import os

print("=" * 80)
print("DOWNLOAD 2024-2025 SEASON STATS")
print("=" * 80)

# Create directory if needed
STATS_DIR = 'historical_stats'
os.makedirs(STATS_DIR, exist_ok=True)

def download_date_stats(date_str):
    """Download stats for a specific date"""
    url = f"https://barttorvik.com/timemachine/team_results/{date_str}_team_results.json.gz"
    
    print(f"[FETCH] {date_str}...", end=" ", flush=True)
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Try to decompress first (in case it's gzipped)
            try:
                json_data = gzip.decompress(response.content)
                data = json.loads(json_data)
            except:
                data = json.loads(response.content)
            
            df = pd.DataFrame(data)
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
        print(f"✗ (error)")
        return None

def get_2024_25_game_dates():
    """Get unique game dates from 2024-25 season file"""
    try:
        games = pd.read_excel('teamrankings_2024_25.csv.xlsx')
        games['game_date'] = pd.to_datetime(games['game_date'])
        dates = games['game_date'].dt.strftime('%Y%m%d').unique()
        return sorted(dates)
    except Exception as e:
        print(f"[ERROR] Could not load 2024-25 games: {e}")
        return []

print("\n[LOAD] Loading 2024-25 season games...")
dates_to_download = get_2024_25_game_dates()

if not dates_to_download:
    print("[ERROR] No dates found in teamrankings_2024_25.csv.xlsx")
    exit(1)

print(f"[OK] Found {len(dates_to_download)} unique dates")
print(f"[INFO] Range: {dates_to_download[0]} to {dates_to_download[-1]}")

# Check which dates we already have
existing_files = set(os.listdir(STATS_DIR)) if os.path.exists(STATS_DIR) else set()
existing_dates = {f.replace('_stats.csv', '') for f in existing_files if f.endswith('_stats.csv')}

dates_needed = [d for d in dates_to_download if d not in existing_dates]

if not dates_needed:
    print(f"\n[OK] All {len(dates_to_download)} dates already downloaded!")
    print("[INFO] No action needed")
    exit(0)

print(f"\n[INFO] Already have: {len(existing_dates)} dates")
print(f"[INFO] Need to download: {len(dates_needed)} dates")
print(f"[INFO] Estimated time: ~{len(dates_needed) * 2 / 60:.1f} minutes")

input("\nPress Enter to start downloading...")

print("\n" + "=" * 80)
print("DOWNLOADING")
print("=" * 80)

successful = 0
failed = 0

for i, date_str in enumerate(dates_needed, 1):
    print(f"[{i}/{len(dates_needed)}] ", end="")
    
    df = download_date_stats(date_str)
    
    if df is not None:
        output_file = os.path.join(STATS_DIR, f"{date_str}_stats.csv")
        df.to_csv(output_file, index=False)
        successful += 1
    else:
        failed += 1
    
    # Be respectful - 2 second delay
    time.sleep(2)

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"[OK] Successfully downloaded: {successful}/{len(dates_needed)}")
if failed > 0:
    print(f"[WARN] Failed: {failed}")
print(f"[SAVED] Files in: {STATS_DIR}/")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Run: python merge_data_exact_dates.py")
print("2. Run: python add_momentum_features_clean.py")
print("3. Run: python train_model_comparison.py")
print("\nYour model will now have proper 2024-25 season context!")

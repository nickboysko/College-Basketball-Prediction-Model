import pandas as pd
from datetime import datetime
import sys
import os
from team_name_mapping import map_team_name

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("MERGE TODAY'S GAMES WITH TEAM STATS")
print("="*80)

# Get current season
today = datetime.now()
current_year = today.year
next_year = current_year + 1 if today.month >= 11 else current_year

print("\nStep 1: Loading current team stats...")
print("-"*80)

# Load team stats
team_stats_file = f'{next_year}_team_results.csv'
if not os.path.exists(team_stats_file):
    print(f"[!] Error: {team_stats_file} not found!")
    print(f"[!] Please download current season stats from Barttorvik first.")
    sys.exit(1)

print(f"[OK] Loaded {next_year}_team_results.csv")
team_stats = pd.read_csv(team_stats_file)

print("\nStep 2: Loading today's games...")
print("-"*80)

# Load today's games
if not os.path.exists('todays_games_raw.csv'):
    print("[!] Error: todays_games_raw.csv not found!")
    print("[!] Please create this file manually with today's games.")
    print("\nFormat:")
    print("team,opp,spread,game_date,season")
    print("Duke,North Carolina,-5.5,2024-11-07,2025-2026")
    sys.exit(1)

print("[OK] Loaded todays_games_raw.csv")
todays_games = pd.read_csv('todays_games_raw.csv')

print(f"[OK] Found {len(todays_games)} games for today")

# ============================================================================
# APPLY TEAM NAME MAPPING (THIS IS THE FIX!)
# ============================================================================
print("\nStep 2.5: Applying team name mapping...")
print("-"*80)

# Store original names for reference
todays_games['team_original'] = todays_games['team']
todays_games['opp_original'] = todays_games['opp']

# Apply mapping
todays_games['team'] = todays_games['team'].apply(map_team_name)
todays_games['opp'] = todays_games['opp'].apply(map_team_name)

# Show what was mapped
mapped_teams = todays_games[todays_games['team'] != todays_games['team_original']]
mapped_opps = todays_games[todays_games['opp'] != todays_games['opp_original']]

if len(mapped_teams) > 0:
    print(f"[OK] Mapped {len(mapped_teams)} team names:")
    for _, row in mapped_teams.iterrows():
        print(f"  {row['team_original']} → {row['team']}")

if len(mapped_opps) > 0:
    print(f"[OK] Mapped {len(mapped_opps)} opponent names:")
    for _, row in mapped_opps.iterrows():
        print(f"  {row['opp_original']} → {row['opp']}")

print("\nStep 3: Merging team stats...")
print("-"*80)

# Merge team stats for both teams
merged = todays_games.merge(
    team_stats.add_suffix('_team'),
    left_on='team',
    right_on='team_team',
    how='left'
)

merged = merged.merge(
    team_stats.add_suffix('_opp'),
    left_on='opp',
    right_on='team_opp',
    how='left'
)

# Check for missing stats
missing_teams = merged[merged['adjoe_team'].isna()]['team'].unique()
missing_opps = merged[merged['adjoe_opp'].isna()]['opp'].unique()

if len(missing_teams) > 0:
    print(f"[!] Warning: Missing stats for teams: {', '.join(missing_teams)}")
if len(missing_opps) > 0:
    print(f"[!] Warning: Missing stats for opponents: {', '.join(missing_opps)}")

# Drop rows with missing stats
before_count = len(merged)
merged = merged.dropna(subset=['adjoe_team', 'adjoe_opp'])
after_count = len(merged)

if before_count > after_count:
    print(f"[!] Dropped {before_count - after_count} games due to missing stats")

# Calculate differential features
merged['adjoe_diff'] = merged['adjoe_team'] - merged['adjoe_opp']
merged['adjde_diff'] = merged['adjde_team'] - merged['adjde_opp']
merged['barthag_diff'] = merged['barthag_team'] - merged['barthag_opp']

print(f"[OK] Successfully merged {len(merged)} games")

# Save
output_file = 'todays_games_merged.csv'
merged.to_csv(output_file, index=False)
print(f"[OK] Saved to {output_file}")

print("\n" + "="*80)
print("MERGE COMPLETE")
print("="*80)
print(f"\nNext step: Run predictions with predict_today.py")
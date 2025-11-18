import pandas as pd
from datetime import datetime
import sys
import os
import io
from team_name_mapping import map_team_name

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("MERGE TODAY'S GAMES WITH TEAM STATS (FIXED FOUR FACTORS)")
print("="*80)

# Get current season
today = datetime.now()
current_year = today.year
next_year = current_year + 1 if today.month >= 11 else current_year

print("\nStep 1: Loading current team stats...")
print("-"*80)

# Load team rankings/efficiency stats
team_results_file = f'{next_year}_team_results.csv'
if not os.path.exists(team_results_file):
    print(f"[!] Error: {team_results_file} not found!")
    print(f"[!] Please download current season stats from Barttorvik first.")
    sys.exit(1)

print(f"[OK] Loaded {team_results_file}")
team_results = pd.read_csv(team_results_file)

# Load Four Factors stats - CRITICAL FIX FOR MALFORMED CSV
fffinal_file = f'{next_year}_fffinal.csv'
if not os.path.exists(fffinal_file):
    print(f"[!] Error: {fffinal_file} not found!")
    print(f"[!] Please download Four Factors from: https://barttorvik.com/{next_year}_fffinal.csv")
    sys.exit(1)

print(f"[FIX] Loading {fffinal_file} (malformed CSV with 41 data columns, 37 header columns)...")

# The CSV has 37 header columns but 41 data columns - pandas gets confused
# Solution: Read only the first 37 columns (usecols) and handle duplicate names
with open(fffinal_file, 'r') as f:
    headers = f.readline().strip().split(',')

# Make column names unique
col_names = []
seen = {}
for h in headers:
    clean_h = h.strip()
    if clean_h in seen:
        seen[clean_h] += 1
        col_names.append(f"{clean_h}_{seen[clean_h]}")
    else:
        seen[clean_h] = 0
        col_names.append(clean_h)

print(f"[INFO] Reading first {len(col_names)} columns only...")

# Read with usecols to only get first 37 columns
fffinal = pd.read_csv(fffinal_file, names=col_names, skiprows=1, usecols=range(len(col_names)))

print(f"[OK] Loaded Four Factors with {len(fffinal)} teams")

# Verify we have actual team names now
if len(fffinal) > 0:
    sample_team = fffinal['TeamName'].iloc[0]
    print(f"[CHECK] Sample team from Four Factors: '{sample_team}'")
    
    if not isinstance(sample_team, str) or len(str(sample_team)) < 3:
        print("[ERROR] Four Factors doesn't have valid team names!")
        print(f"[DEBUG] First 5 'teams': {fffinal['TeamName'].head().tolist()}")
        print(f"[DEBUG] This means the CSV structure is still wrong")
        sys.exit(1)
else:
    print("[ERROR] Four Factors file is empty!")
    sys.exit(1)

# Ensure both team columns are strings for merging
print(f"[MERGE] Combining rankings and Four Factors data...")
team_results['team'] = team_results['team'].astype(str).str.strip()
fffinal['TeamName'] = fffinal['TeamName'].astype(str).str.strip()

team_stats = team_results.merge(
    fffinal,
    left_on='team',
    right_on='TeamName',
    how='left',
    suffixes=('', '_ff')
)

# Drop duplicate team name column
if 'TeamName' in team_stats.columns:
    team_stats = team_stats.drop(columns=['TeamName'])

print(f"[OK] Combined stats for {len(team_stats)} teams")
print(f"[OK] Total columns: {len(team_stats.columns)}")

# Verify Four Factors merged
ff_cols_check = ['eFG%', 'TO%', 'OR%', 'FTR']
found_ff = sum(1 for col in ff_cols_check if col in team_stats.columns)
print(f"[VERIFY] Four Factors columns found: {found_ff}/{len(ff_cols_check)}")

if found_ff < len(ff_cols_check):
    print(f"[ERROR] Only found {found_ff}/{len(ff_cols_check)} Four Factors columns!")
    sys.exit(1)

# Check that values aren't NaN for a sample team
sample_check = team_stats[team_stats['team'] == 'Duke']
if len(sample_check) > 0:
    duke_efg = sample_check.iloc[0]['eFG%']
    print(f"[VERIFY] Duke eFG% = {duke_efg} (should be a number, not NaN)")
    if pd.isna(duke_efg):
        print("[ERROR] Four Factors data is NaN - team names not matching!")
        sys.exit(1)
else:
    print("[WARN] Duke not found in stats - can't verify Four Factors")

print("\nStep 2: Loading today's games...")
print("-"*80)

# Load today's games
if not os.path.exists('todays_games_raw.csv'):
    print("[!] Error: todays_games_raw.csv not found!")
    print("[!] Please run scrape_today_odds_api.py first")
    sys.exit(1)

print("[OK] Loaded todays_games_raw.csv")
todays_games = pd.read_csv('todays_games_raw.csv')

print(f"[OK] Found {len(todays_games)} games for today")

# ============================================================================
# APPLY TEAM NAME MAPPING
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

print("\nStep 3: Merging team stats (including Four Factors)...")
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

# CRITICAL CHECK: Verify Four Factors data is present and not NaN
print("\n[VERIFY] Checking Four Factors data merged correctly...")
ff_check_cols = ['eFG%_team', 'TO%_team', 'OR%_team']
all_good = True
for col in ff_check_cols:
    if col in merged.columns:
        non_nan = merged[col].notna().sum()
        print(f"  {col}: {non_nan}/{len(merged)} games have data")
        if non_nan == 0:
            print(f"    ❌ WARNING: All NaN! Four Factors not merging correctly!")
            all_good = False
    else:
        print(f"  ❌ {col} not found in merged data!")
        all_good = False

if not all_good:
    print("\n[ERROR] Four Factors merge failed!")
    print("[DEBUG] Sample team names:")
    print(f"  From games: {todays_games['team'].head(3).tolist()}")
    print(f"  From stats: {team_stats['team'].head(3).tolist()}")

# Drop rows with missing core stats
before_count = len(merged)
merged = merged.dropna(subset=['adjoe_team', 'adjoe_opp'])
after_count = len(merged)

if before_count > after_count:
    print(f"[!] Dropped {before_count - after_count} games due to missing stats")

if len(merged) == 0:
    print("[ERROR] No games left after dropping those with missing stats!")
    sys.exit(1)

print(f"[OK] Successfully merged {len(merged)} games")

# Verify at least one game has Four Factors data
if len(merged) > 0:
    sample_game = merged.iloc[0]
    print("\n[SAMPLE] First game verification:")
    print(f"  Team: {sample_game['team']}")
    print(f"  adjoe_team: {sample_game.get('adjoe_team', 'NOT FOUND'):.2f}")
    print(f"  eFG%_team: {sample_game.get('eFG%_team', 'NOT FOUND')}")
    
    if pd.notna(sample_game.get('eFG%_team')):
        print("  ✅ Four Factors data is present!")
    else:
        print("  ❌ Four Factors data is NaN!")

# Save
output_file = 'todays_games_merged.csv'
merged.to_csv(output_file, index=False)
print(f"\n[OK] Saved to {output_file}")

print("\n" + "="*80)
print("MERGE COMPLETE")
print("="*80)

# Final verification
if len(merged) > 0 and 'eFG%_team' in merged.columns:
    has_ff_data = merged['eFG%_team'].notna().sum()
    if has_ff_data > 0:
        print(f"\n✅ SUCCESS! Four Factors data present in {has_ff_data}/{len(merged)} games")
        print(f"\nNext step: python calculate_momentum_features.py")
    else:
        print(f"\n⚠️ WARNING! Four Factors columns exist but {len(merged) - has_ff_data} games have NaN")
        print("This may be due to team name mismatches")
else:
    print("\n⚠️  WARNING: Could not verify Four Factors data")
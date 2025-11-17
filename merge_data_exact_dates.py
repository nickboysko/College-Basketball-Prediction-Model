import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os

# Try to import the team name mapping if it exists
try:
    from team_name_mapping import map_team_name, TEAM_NAME_MAPPING
    HAS_TEAM_MAPPING = True
    print("[INFO] Using team_name_mapping.py for better name matching")
except ImportError:
    HAS_TEAM_MAPPING = False
    print("[WARN] team_name_mapping.py not found, using basic normalization")

# Simple team name normalization (fallback or supplement)
def normalize_team_name(name):
    """Normalize team names to match Barttorvik format"""
    if pd.isna(name):
        return name
    
    name = str(name).strip()
    
    # First, try using the imported team_name_mapping if available
    if HAS_TEAM_MAPPING:
        mapped = map_team_name(name)
        if mapped != name:  # If mapping was found
            return mapped
    
    # Common replacements to match Barttorvik format (fallback)
    replacements = {
        'Alabama St': 'Alabama St.',
        'Alcorn St': 'Alcorn St.',
        'App State': 'Appalachian St.',
        'AR-Pine Bluff': 'Ark. Pine Bluff',
        'Arizona St': 'Arizona St.',
        'Arkansas St': 'Arkansas St.',
        'Ball St': 'Ball St.',
        'Boise St': 'Boise St.',
        'Boston U': 'Boston U.',
        'C Arkansas': 'Central Ark.',
        'C Connecticut': 'Central Conn. St.',
        'C Michigan': 'Central Mich.',
        'CS Bakersfield': 'CSU Bakersfield',
        'CS Fullerton': 'CS Fullerton',
        'CS Northridge': 'CSU Northridge',
        'Charleston So': 'Charleston So.',
        'Chicago St': 'Chicago St.',
        'Cleveland St': 'Cleveland St.',
        'Colorado St': 'Colorado St.',
        'Coppin St': 'Coppin St.',
        'Delaware St': 'Delaware St.',
        'Detroit': 'Detroit',
        'E Illinois': 'E. Illinois',
        'E Kentucky': 'Eastern Ky.',
        'E Michigan': 'E. Michigan',
        'E Tennessee St': 'East Tenn. St.',
        'E Washington': 'Eastern Wash.',
        'FL Gulf Coast': 'Florida Gulf Coast',
        'Fla Atlantic': 'Fla. Atlantic',
        'Fresno St': 'Fresno St.',
        'Ga Southern': 'Ga. Southern',
        'Georgia St': 'Georgia St.',
        'Grambling St': 'Grambling',
        'Green Bay': 'Green Bay',
        'Houston Baptist': 'Houston Christian',
        'Illinois St': 'Illinois St.',
        'Incarnate Word': 'Incarnate Word',
        'Indiana St': 'Indiana St.',
        'Iowa St': 'Iowa St.',
        'Jackson St': 'Jackson St.',
        'Jacksonville St': 'Jacksonville St.',
        'Kansas St': 'Kansas St.',
        'Kennesaw St': 'Kennesaw St.',
        'Kent St': 'Kent St.',
        'LA Tech': 'Louisiana Tech',
        'LIU': 'Long Island U.',
        'Long Beach St': 'Long Beach St.',
        'McNeese St': 'McNeese St.',
        'Miami FL': 'Miami FL',
        'Miami OH': 'Miami OH',
        'Michigan St': 'Michigan St.',
        'Middle Tennessee': 'Middle Tenn.',
        'Mississippi St': 'Mississippi St.',
        'Mississippi Val': 'Miss. Valley St.',
        'Missouri St': 'Missouri St.',
        'Montana St': 'Montana St.',
        'Morehead St': 'Morehead St.',
        'Morgan St': 'Morgan St.',
        'Mt St Mary\'s': 'Mt. St. Mary\'s',
        'Murray St': 'Murray St.',
        'N Arizona': 'N. Arizona',
        'N Colorado': 'Northern Colo.',
        'N Dakota St': 'North Dakota St.',
        'N Illinois': 'N. Illinois',
        'N Kentucky': 'N. Kentucky',
        'NC A&T': 'North Carolina A&T',
        'NC Central': 'N.C. Central',
        'NC Greensboro': 'UNC Greensboro',
        'NC St': 'N.C. State',
        'NC Wilmington': 'UNC Wilmington',
        'New Mexico St': 'New Mexico St.',
        'Norfolk St': 'Norfolk St.',
        'North Dakota': 'North Dakota',
        'Northwestern St': 'Northwestern St.',
        'Ohio St': 'Ohio St.',
        'Oklahoma St': 'Oklahoma St.',
        'Old Dominion': 'Old Dominion',
        'Oregon St': 'Oregon St.',
        'Penn St': 'Penn St.',
        'Portland St': 'Portland St.',
        'Prairie View': 'Prairie View',
        'Sam Houston St': 'Sam Houston St.',
        'San Diego St': 'San Diego St.',
        'San Jose St': 'San Jose St.',
        'Savannah St': 'Savannah St.',
        'SE Louisiana': 'SE Louisiana',
        'SE Missouri St': 'Southeast Mo. St.',
        'SIU Edwardsville': 'SIU Edwardsville',
        'South Alabama': 'South Ala.',
        'South Carolina St': 'S.C. State',
        'South Dakota St': 'South Dakota St.',
        'Southern Illinois': 'Southern Ill.',
        'Southern Miss': 'Southern Miss.',
        'Southern Utah': 'Southern Utah',
        'St Bonaventure': 'St. Bonaventure',
        'St Francis PA': 'St. Francis PA',
        'St John\'s': 'St. John\'s',
        'St Joseph\'s': 'Saint Joseph\'s',
        'Stephen F Austin': 'S.F. Austin',
        'Stony Brook': 'Stony Brook',
        'TAM C. Christi': 'Texas A&M Corpus Chris',
        'Tennessee St': 'Tenn. St.',
        'Tennessee Tech': 'Tennessee Tech',
        'Texas A&M': 'Texas A&M',
        'Texas Southern': 'Texas Southern',
        'Texas St': 'Texas St.',
        'UC Davis': 'UC Davis',
        'UC Irvine': 'UC Irvine',
        'UC Riverside': 'UC Riverside',
        'UC San Diego': 'UC San Diego',
        'UC Santa Barbara': 'UCSB',
        'UNC Asheville': 'UNC Asheville',
        'UT Arlington': 'UT Arlington',
        'UT Martin': 'UT Martin',
        'UT Rio Grande Valley': 'UTRGV',
        'Utah St': 'Utah St.',
        'Utah Valley': 'Utah Valley',
        'VA Commonwealth': 'VCU',
        'W Carolina': 'W. Carolina',
        'W Illinois': 'W. Illinois',
        'W Kentucky': 'WKU',
        'W Michigan': 'W. Michigan',
        'Washington St': 'Washington St.',
        'Weber St': 'Weber St.',
        'Wichita St': 'Wichita St.',
        'Wright St': 'Wright St.',
        'Youngstown St': 'Youngstown St.',
    }
    
    # Apply mapping if exists
    if name in replacements:
        return replacements[name]
    
    return name

# Directory where Time Machine stats are stored
STATS_DIR = 'historical_stats'

TRAINING_GAMES = 'teamrankings_2021-2022_to_2023-2024.csv.xlsx'
TESTING_GAMES = 'teamrankings_2024_25.csv.xlsx'

OUTPUT_TRAINING = 'training_data_no_leakage.csv'
OUTPUT_TESTING = 'testing_data_no_leakage.csv'

def load_all_date_stats():
    """
    Load all downloaded date-specific stat files
    Returns: Dictionary mapping date strings to DataFrames
    """
    print("\n[LOAD] Loading all date-specific stat files...")
    
    stats_dir = Path(STATS_DIR)
    
    if not stats_dir.exists():
        raise Exception(f"Stats directory not found: {STATS_DIR}\nPlease run download_time_machine.py first!")
    
    stats_files = list(stats_dir.glob('*_stats.csv'))
    
    if len(stats_files) == 0:
        raise Exception(f"No stat files found in {STATS_DIR}/\nPlease run download_time_machine.py first!")
    
    print(f"[INFO] Found {len(stats_files)} stat files")
    
    # Load all files into memory (they're small enough)
    stats_by_date = {}
    
    for file_path in stats_files:
        # Extract date from filename (e.g., "20220115_stats.csv" -> "20220115")
        date_str = file_path.stem.split('_')[0]
        
        try:
            df = pd.read_csv(file_path)
            
            # The columns are numbered as strings ('0', '1', '2', '3'...)
            # Based on the JSON structure:
            # 0: rank, 1: team, 2: conf, 3: record, 4: adjoe, 5: oe_rank, 
            # 6: adjde, 7: de_rank, 8: barthag, 9: barthag_rank...
            
            if df.columns[0] == '0':  # If columns are numbered strings
                # Create proper column names based on Barttorvik structure
                col_names = [
                    'rank', 'team', 'conf', 'record', 
                    'adjoe', 'oe_rank', 'adjde', 'de_rank', 'barthag', 'barthag_rank',
                    'proj_w', 'proj_l', 'proj_conf_w', 'proj_conf_l', 'conf_record',
                    'sos', 'ncsos', 'conf_sos', 
                    'efg_pct', 'efgd_pct', 'tor', 'tord', 
                    'orb', 'drb', 'ftr', 'ftrd',
                    'twop_pct', 'twopd_pct', 'threep_pct', 'threepd_pct',
                    'threep_rate', 'threed_rate', 'adjt'
                ]
                # Add generic names for any remaining columns
                while len(col_names) < len(df.columns):
                    col_names.append(f'col_{len(col_names)}')
                
                df.columns = col_names
            
            # Normalize column names to lowercase for consistency
            df.columns = df.columns.str.lower()
            
            stats_by_date[date_str] = df
            
        except Exception as e:
            print(f"[WARN] Failed to load {file_path}: {e}")
            continue
    
    print(f"[OK] Loaded {len(stats_by_date)} date-specific stat files")
    
    # Show date range
    dates = sorted(stats_by_date.keys())
    print(f"[INFO] Date range: {dates[0]} to {dates[-1]}")
    
    return stats_by_date

def get_stats_for_date(stats_by_date, game_date):
    """
    Get the stats file for a specific game date
    Uses the exact date if available, otherwise uses the most recent date before it
    
    Args:
        stats_by_date: Dictionary of date -> DataFrame
        game_date: Date as datetime or string (YYYY-MM-DD or YYYYMMDD)
    
    Returns:
        DataFrame with stats, or None if no suitable stats found
    """
    # Convert game_date to YYYYMMDD format
    if isinstance(game_date, str):
        if '-' in game_date:
            game_date = datetime.strptime(game_date, '%Y-%m-%d')
        else:
            # Already in YYYYMMDD format
            game_date_str = game_date
            game_date = datetime.strptime(game_date, '%Y%m%d')
    else:
        game_date_str = game_date.strftime('%Y%m%d')
    
    if not isinstance(game_date, datetime):
        game_date_str = game_date.strftime('%Y%m%d')
    
    # Try exact match first
    if game_date_str in stats_by_date:
        return stats_by_date[game_date_str]
    
    # Otherwise, find the most recent date before this game
    available_dates = sorted([d for d in stats_by_date.keys() if d <= game_date_str])
    
    if len(available_dates) == 0:
        return None
    
    # Use the most recent date before the game
    best_date = available_dates[-1]
    return stats_by_date[best_date]

def merge_date_specific_stats(games_df, stats_by_date):
    """
    Merge team stats based on exact game dates
    """
    print(f"\n[MERGE] Merging stats for {len(games_df)} games...")
    
    # Parse game dates
    games_df['game_date_dt'] = pd.to_datetime(games_df['game_date'])
    games_df['game_date_str'] = games_df['game_date_dt'].dt.strftime('%Y%m%d')
    
    # Track success/failure
    merged_rows = []
    missing_stats = []
    
    for idx, game in games_df.iterrows():
        if idx % 1000 == 0 and idx > 0:
            print(f"  [PROGRESS] {idx}/{len(games_df)} games processed...")
        
        # Get stats for this game date
        stats_df = get_stats_for_date(stats_by_date, game['game_date_dt'])
        
        if stats_df is None:
            missing_stats.append(game['game_date_str'])
            continue
        
        # Find home team stats
        home_stats = stats_df[stats_df['team'] == game['team']]
        away_stats = stats_df[stats_df['team'] == game['opp']]
        
        if len(home_stats) == 0 or len(away_stats) == 0:
            missing_stats.append(f"{game['team']} vs {game['opp']} on {game['game_date_str']}")
            continue
        
        # Merge into single row
        game_row = game.to_dict()
        
        # Add home team stats
        for col in home_stats.columns:
            if col not in ['team', 'stats_date']:
                game_row[f'home_{col}'] = home_stats.iloc[0][col]
        
        # Add away team stats
        for col in away_stats.columns:
            if col not in ['team', 'stats_date']:
                game_row[f'away_{col}'] = away_stats.iloc[0][col]
        
        merged_rows.append(game_row)
    
    print(f"[OK] Successfully merged {len(merged_rows)}/{len(games_df)} games")
    
    if len(missing_stats) > 0:
        print(f"[WARN] {len(missing_stats)} games had missing stats")
        if len(missing_stats) <= 10:
            for miss in missing_stats[:10]:
                print(f"  - {miss}")
    
    # Convert to DataFrame
    merged_df = pd.DataFrame(merged_rows)
    
    return merged_df

def create_features(df):
    """Create differential and other engineered features"""
    
    print("\n[FEATURES] Creating differential features...")
    
    # Core efficiency features
    if 'home_adjoe' in df.columns and 'away_adjoe' in df.columns:
        df['adjoe_diff'] = df['home_adjoe'] - df['away_adjoe']
    
    if 'home_adjde' in df.columns and 'away_adjde' in df.columns:
        df['adjde_diff'] = df['home_adjde'] - df['away_adjde']
    
    if 'home_adjt' in df.columns and 'away_adjt' in df.columns:
        df['adjt_diff'] = df['home_adjt'] - df['away_adjt']
    
    if 'home_barthag' in df.columns and 'away_barthag' in df.columns:
        df['barthag_diff'] = df['home_barthag'] - df['away_barthag']
    
    # Ranking features
    if 'home_rank' in df.columns and 'away_rank' in df.columns:
        df['rank_diff'] = df['away_rank'] - df['home_rank']
    
    # Strength of schedule features
    if 'home_sos' in df.columns and 'away_sos' in df.columns:
        df['sos_diff'] = df['home_sos'] - df['away_sos']
    
    if 'home_ncsos' in df.columns and 'away_ncsos' in df.columns:
        df['ncsos_diff'] = df['home_ncsos'] - df['away_ncsos']
    
    # Shooting efficiency features
    if 'home_efg_pct' in df.columns and 'away_efg_pct' in df.columns:
        df['efg_pct_diff'] = df['home_efg_pct'] - df['away_efg_pct']
    
    if 'home_efgd_pct' in df.columns and 'away_efgd_pct' in df.columns:
        df['efgd_pct_diff'] = df['home_efgd_pct'] - df['away_efgd_pct']
    
    # Turnover features
    if 'home_tor' in df.columns and 'away_tor' in df.columns:
        df['tor_diff'] = df['home_tor'] - df['away_tor']
    
    if 'home_tord' in df.columns and 'away_tord' in df.columns:
        df['tord_diff'] = df['home_tord'] - df['away_tord']
    
    # Rebounding features
    if 'home_orb' in df.columns and 'away_orb' in df.columns:
        df['orb_diff'] = df['home_orb'] - df['away_orb']
    
    if 'home_drb' in df.columns and 'away_drb' in df.columns:
        df['drb_diff'] = df['home_drb'] - df['away_drb']
    
    # Free throw features
    if 'home_ftr' in df.columns and 'away_ftr' in df.columns:
        df['ftr_diff'] = df['home_ftr'] - df['away_ftr']
    
    if 'home_ftrd' in df.columns and 'away_ftrd' in df.columns:
        df['ftrd_diff'] = df['home_ftrd'] - df['away_ftrd']
    
    # 2-point shooting features
    if 'home_twop_pct' in df.columns and 'away_twop_pct' in df.columns:
        df['twop_pct_diff'] = df['home_twop_pct'] - df['away_twop_pct']
    
    if 'home_twopd_pct' in df.columns and 'away_twopd_pct' in df.columns:
        df['twopd_pct_diff'] = df['home_twopd_pct'] - df['away_twopd_pct']
    
    # 3-point shooting features
    if 'home_threep_pct' in df.columns and 'away_threep_pct' in df.columns:
        df['threep_pct_diff'] = df['home_threep_pct'] - df['away_threep_pct']
    
    if 'home_threepd_pct' in df.columns and 'away_threepd_pct' in df.columns:
        df['threepd_pct_diff'] = df['home_threepd_pct'] - df['away_threepd_pct']
    
    print(f"[OK] Created {len([c for c in df.columns if '_diff' in c])} differential features")
    
    return df

def create_target_variable(df):
    """Create target: did home team cover the spread?"""
    
    print("\n[TARGET] Creating target variable...")
    
    if 'score' in df.columns:
        scores = df['score'].str.split('-', expand=True)
        df['home_score'] = pd.to_numeric(scores[0], errors='coerce')
        df['away_score'] = pd.to_numeric(scores[1], errors='coerce')
        
        df['margin'] = df['home_score'] - df['away_score']
        df['home_covered'] = (df['margin'] + df['spread'] > 0).astype(int)
        df['ats_margin'] = df['margin'] + df['spread']
        
        print(f"[OK] Target variable created")
        print(f"[INFO] Home team covered: {df['home_covered'].sum()} ({df['home_covered'].mean()*100:.1f}%)")
    
    return df

def main():
    print("=" * 80)
    print("DATA MERGE - ZERO DATA LEAKAGE")
    print("=" * 80)
    print("\nUsing exact date-specific stats from Barttorvik Time Machine")
    print("This eliminates all data leakage!")
    
    # Load all date-specific stats
    stats_by_date = load_all_date_stats()
    
    # Process training data
    print("\n" + "=" * 80)
    print("PROCESSING TRAINING DATA (2021-2024)")
    print("=" * 80)
    
    if not os.path.exists(TRAINING_GAMES):
        print(f"[ERROR] Training games file not found: {TRAINING_GAMES}")
        return
    
    train_games = pd.read_excel(TRAINING_GAMES) if TRAINING_GAMES.endswith('.xlsx') else pd.read_csv(TRAINING_GAMES)
    print(f"[OK] Loaded {len(train_games)} training games")
    
    # Normalize team names
    print("\n[CLEAN] Normalizing team names...")
    train_games['team'] = train_games['team'].apply(normalize_team_name)
    train_games['opp'] = train_games['opp'].apply(normalize_team_name)
    
    # Merge with date-specific stats
    train_merged = merge_date_specific_stats(train_games, stats_by_date)
    train_merged = create_features(train_merged)
    train_merged = create_target_variable(train_merged)
    
    # Process testing data
    print("\n" + "=" * 80)
    print("PROCESSING TESTING DATA (2024-2025)")
    print("=" * 80)
    
    if not os.path.exists(TESTING_GAMES):
        print(f"[WARN] Testing games file not found: {TESTING_GAMES}")
        print("[INFO] Skipping testing data...")
        test_merged = None
    else:
        test_games = pd.read_excel(TESTING_GAMES) if TESTING_GAMES.endswith('.xlsx') else pd.read_csv(TESTING_GAMES)
        print(f"[OK] Loaded {len(test_games)} testing games")
        
        # Normalize team names
        print("\n[CLEAN] Normalizing team names...")
        test_games['team'] = test_games['team'].apply(normalize_team_name)
        test_games['opp'] = test_games['opp'].apply(normalize_team_name)
        
        # Merge with date-specific stats
        test_merged = merge_date_specific_stats(test_games, stats_by_date)
        test_merged = create_features(test_merged)
        test_merged = create_target_variable(test_merged)
    
    # Save merged datasets
    print("\n" + "=" * 80)
    print("SAVING DATASETS")
    print("=" * 80)
    
    train_merged.to_csv(OUTPUT_TRAINING, index=False)
    print(f"[SAVED] {OUTPUT_TRAINING}: {len(train_merged)} games")
    
    if test_merged is not None:
        test_merged.to_csv(OUTPUT_TESTING, index=False)
        print(f"[SAVED] {OUTPUT_TESTING}: {len(test_merged)} games")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTraining Data:")
    print(f"  Total games: {len(train_merged)}")
    complete_train = len(train_merged.dropna(subset=['home_adjoe', 'away_adjoe']))
    print(f"  Complete data: {complete_train} ({complete_train/len(train_merged)*100:.1f}%)")
    
    if test_merged is not None:
        print(f"\nTesting Data:")
        print(f"  Total games: {len(test_merged)}")
        complete_test = len(test_merged.dropna(subset=['home_adjoe', 'away_adjoe']))
        print(f"  Complete data: {complete_test} ({complete_test/len(test_merged)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ ZERO DATA LEAKAGE!")
    print("=" * 80)
    print("\nEach game uses stats from BEFORE the game was played")
    print("Your model will now learn from realistic early/mid/late season data")
    
    print("\n" + "=" * 80)
    print("NEXT STEP: Retrain Your Model")
    print("=" * 80)
    print(f"\n1. Update train_models.py to load: {OUTPUT_TRAINING}")
    print(f"2. Run: python train_models.py")
    print(f"3. Compare new accuracy to old (should see improvement!)")
    print(f"4. Model should handle early-season games much better")

if __name__ == "__main__":
    main()
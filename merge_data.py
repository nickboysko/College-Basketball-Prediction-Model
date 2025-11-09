import pandas as pd
import numpy as np
from pathlib import Path
from team_name_mapping import normalize_team_name, create_full_mapping_from_data, TEAM_NAME_MAPPING

# File paths
TEAM_STATS_FILES = {
    2022: '2022_team_results.csv',
    2023: '2023_team_results.csv',
    2024: '2024_team_results.csv',
    2025: '2025_team_results.csv'
}

TRAINING_GAMES = 'teamrankings_2021-2022_to_2023-2024.csv.xlsx'
TESTING_GAMES = 'teamrankings_2024_25.csv.xlsx'

# Output paths - save to current directory
OUTPUT_TRAINING = 'training_data.csv'
OUTPUT_TESTING = 'testing_data.csv'

def load_team_stats():
    """Load all team stats files and combine them with year labels"""
    all_stats = []
    
    for year, filename in TEAM_STATS_FILES.items():
        print(f"Loading {filename}...")
        # The CSV has duplicate 'rank' columns, pandas auto-renames to 'rank' and 'rank.1'
        df = pd.read_csv(filename)
        
        # Keep only the first 'rank' column and rename rank.1 to barthag_rank
        if 'rank.1' in df.columns:
            df.rename(columns={'rank.1': 'barthag_rank'}, inplace=True)
        
        df['stats_year'] = year
        all_stats.append(df)
        
        # Debug: print first few team names
        print(f"  Loaded {len(df)} teams. Sample teams: {list(df['team'].head(5))}")
    
    combined = pd.concat(all_stats, ignore_index=True)
    print(f"\nLoaded {len(combined)} team-season records")
    return combined

def parse_season_to_stats_year(season_str):
    """
    Convert season string to stats year
    '2021-2022' -> 2022 (use end year stats)
    '2024-2025' -> 2025
    """
    if '-' in season_str:
        return int(season_str.split('-')[1])
    return int(season_str)

def merge_team_stats(games_df, stats_df):
    """Merge team stats for both home and away teams"""
    
    # Parse season to get stats year
    games_df['stats_year'] = games_df['season'].apply(parse_season_to_stats_year)
    
    print(f"\nMerging stats for {len(games_df)} games...")
    
    # Merge home team stats
    merged = games_df.merge(
        stats_df,
        left_on=['team', 'stats_year'],
        right_on=['team', 'stats_year'],
        how='left',
        suffixes=('', '_home')
    )
    
    # Rename home team stat columns
    stat_columns = [col for col in stats_df.columns if col not in ['team', 'stats_year']]
    for col in stat_columns:
        if col in merged.columns:
            merged.rename(columns={col: f'home_{col}'}, inplace=True)
    
    # Merge away team stats
    merged = merged.merge(
        stats_df,
        left_on=['opp', 'stats_year'],
        right_on=['team', 'stats_year'],
        how='left',
        suffixes=('', '_away')
    )
    
    # Rename away team stat columns
    for col in stat_columns:
        if col in merged.columns and not col.startswith('home_'):
            merged.rename(columns={col: f'away_{col}'}, inplace=True)
    
    # Clean up duplicate team columns
    if 'team_away' in merged.columns:
        merged.drop(columns=['team_away'], inplace=True)
    
    return merged

def create_features(df):
    """Create differential and other engineered features"""
    
    print("\nCreating features...")
    
    # Key differential features
    if 'home_adjoe' in df.columns and 'away_adjoe' in df.columns:
        df['adjoe_diff'] = df['home_adjoe'] - df['away_adjoe']
    
    if 'home_adjde' in df.columns and 'away_adjde' in df.columns:
        df['adjde_diff'] = df['home_adjde'] - df['away_adjde']
        # Note: Lower defensive rating is better, so positive diff means home has worse defense
    
    if 'home_adjt' in df.columns and 'away_adjt' in df.columns:
        df['adjt_diff'] = df['home_adjt'] - df['away_adjt']
    
    if 'home_barthag' in df.columns and 'away_barthag' in df.columns:
        df['barthag_diff'] = df['home_barthag'] - df['away_barthag']
    
    # Rank differentials (lower rank is better, so flip the sign)
    if 'home_rank' in df.columns and 'away_rank' in df.columns:
        df['rank_diff'] = df['away_rank'] - df['home_rank']
    
    # SOS differentials
    if 'home_sos' in df.columns and 'away_sos' in df.columns:
        df['sos_diff'] = df['home_sos'] - df['away_sos']
    
    return df

def create_target_variable(df):
    """Create target: did home team cover the spread?"""
    
    print("\nCreating target variable...")
    
    # Parse score (format appears to be like "105-59")
    if 'score' in df.columns:
        scores = df['score'].str.split('-', expand=True)
        df['home_score'] = pd.to_numeric(scores[0], errors='coerce')
        df['away_score'] = pd.to_numeric(scores[1], errors='coerce')
        
        # Margin = home_score - away_score
        df['margin'] = df['home_score'] - df['away_score']
        
        # Home covers if: margin + spread > 0
        # (If home is favored by -7, they need to win by more than 7)
        df['home_covered'] = (df['margin'] + df['spread'] > 0).astype(int)
        
        # Add ATS margin (how much they covered/missed by)
        df['ats_margin'] = df['margin'] + df['spread']
    
    return df

def main():
    print("=" * 60)
    print("BASKETBALL SPREAD PREDICTION - DATA MERGE")
    print("=" * 60)
    
    # Load team stats
    stats_df = load_team_stats()
    
    # Process training data
    print("\n" + "=" * 60)
    print("PROCESSING TRAINING DATA (2021-2024)")
    print("=" * 60)
    
    train_games = pd.read_excel(TRAINING_GAMES)
    print(f"Loaded {len(train_games)} training games")
    
    # Normalize team names
    print("Normalizing team names...")
    train_games['team'] = train_games['team'].apply(normalize_team_name)
    train_games['opp'] = train_games['opp'].apply(normalize_team_name)
    
    # Create additional mappings from close matches
    print("Finding additional team name matches...")
    additional_mappings = create_full_mapping_from_data(train_games, stats_df)
    
    # Apply additional mappings
    if additional_mappings:
        for old_name, new_name in additional_mappings.items():
            train_games.loc[train_games['team'] == old_name, 'team'] = new_name
            train_games.loc[train_games['opp'] == old_name, 'opp'] = new_name
    
    train_merged = merge_team_stats(train_games, stats_df)
    train_merged = create_features(train_merged)
    train_merged = create_target_variable(train_merged)
    
    # Check for missing stats
    missing_train = train_merged['home_adjoe'].isna().sum()
    print(f"\nGames with missing home team stats: {missing_train}")
    missing_train_away = train_merged['away_adjoe'].isna().sum()
    print(f"Games with missing away team stats: {missing_train_away}")
    
    # Process testing data
    print("\n" + "=" * 60)
    print("PROCESSING TESTING DATA (2024-2025)")
    print("=" * 60)
    
    test_games = pd.read_excel(TESTING_GAMES)
    print(f"Loaded {len(test_games)} testing games")
    
    # Normalize team names
    print("Normalizing team names...")
    test_games['team'] = test_games['team'].apply(normalize_team_name)
    test_games['opp'] = test_games['opp'].apply(normalize_team_name)
    
    # Apply same additional mappings
    if additional_mappings:
        for old_name, new_name in additional_mappings.items():
            test_games.loc[test_games['team'] == old_name, 'team'] = new_name
            test_games.loc[test_games['opp'] == old_name, 'opp'] = new_name
    
    test_merged = merge_team_stats(test_games, stats_df)
    test_merged = create_features(test_merged)
    test_merged = create_target_variable(test_merged)
    
    # Check for missing stats
    missing_test = test_merged['home_adjoe'].isna().sum()
    print(f"\nGames with missing home team stats: {missing_test}")
    missing_test_away = test_merged['away_adjoe'].isna().sum()
    print(f"Games with missing away team stats: {missing_test_away}")
    
    # Save merged datasets
    print("\n" + "=" * 60)
    print("SAVING DATASETS")
    print("=" * 60)
    
    train_merged.to_csv(OUTPUT_TRAINING, index=False)
    print(f"Saved {OUTPUT_TRAINING}: {len(train_merged)} games, {len(train_merged.columns)} columns")
    
    test_merged.to_csv(OUTPUT_TESTING, index=False)
    print(f"Saved {OUTPUT_TESTING}: {len(test_merged)} games, {len(test_merged.columns)} columns")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nTraining Data:")
    print(f"  Total games: {len(train_merged)}")
    print(f"  Games with complete data: {len(train_merged.dropna(subset=['home_adjoe', 'away_adjoe']))}")
    print(f"  Home team covered: {train_merged['home_covered'].sum()} ({train_merged['home_covered'].mean()*100:.1f}%)")
    
    print("\nTesting Data:")
    print(f"  Total games: {len(test_merged)}")
    print(f"  Games with complete data: {len(test_merged.dropna(subset=['home_adjoe', 'away_adjoe']))}")
    print(f"  Home team covered: {test_merged['home_covered'].sum()} ({test_merged['home_covered'].mean()*100:.1f}%)")
    
    print("\nKey features created:")
    feature_cols = [col for col in train_merged.columns if '_diff' in col or col in ['home_covered', 'ats_margin']]
    for col in feature_cols:
        print(f"  - {col}")
    
    print("\n" + "=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
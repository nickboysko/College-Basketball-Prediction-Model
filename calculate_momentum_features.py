import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("CALCULATE MOMENTUM & DIFFERENTIAL FEATURES")
print("=" * 80)

def load_game_history(file_path='current_season_games.xlsx'):
    """Load and clean game history"""
    
    print(f"\n[LOAD] Loading {file_path}...")
    df = pd.read_excel(file_path)
    
    print(f"[OK] Loaded {len(df)} games")
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values(['Team', 'Date']).reset_index(drop=True)
    
    print(f"[INFO] Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"[INFO] Unique teams: {df['Team'].nunique()}")
    
    return df

def calculate_team_momentum(df, team_name, as_of_date=None):
    """
    Calculate momentum features for a specific team as of a specific date
    
    Returns dict with:
    - ats_last_5: ATS record in last 5 games
    - ats_last_10: ATS record in last 10 games
    - ats_streak: Current covering streak
    - games_played: Total games played
    - rest_days: Days since last game
    """
    
    # Filter to this team's games
    team_games = df[df['Team'] == team_name].copy()
    
    # If as_of_date provided, only use games before that date
    if as_of_date:
        team_games = team_games[team_games['Date'] < pd.to_datetime(as_of_date)]
    
    if len(team_games) == 0:
        # No history - return neutral values
        return {
            'ats_last_5': 0.5,
            'ats_last_10': 0.5,
            'ats_streak': 0,
            'games_played': 0,
            'rest_days': 5,  # Assume 5 days rest if no history
            'recent_margin': 0,
            'recent_opp_quality': 0
        }
    
    # Calculate ATS results (1 if covered, 0 if not)
    # ATS Margin > 0 means covered
    team_games['covered'] = (team_games['ATS Margin'] > 0).astype(int)
    
    # Last 5 games ATS
    last_5 = team_games.tail(5)['covered'].mean() if len(team_games) >= 5 else team_games['covered'].mean()
    
    # Last 10 games ATS
    last_10 = team_games.tail(10)['covered'].mean() if len(team_games) >= 10 else team_games['covered'].mean()
    
    # Calculate streak
    streak = 0
    if len(team_games) > 0:
        recent_results = team_games.tail(20)['covered'].values
        for result in reversed(recent_results):
            if streak == 0:
                streak = 1 if result == 1 else -1
            elif (streak > 0 and result == 1) or (streak < 0 and result == 0):
                streak += 1 if result == 1 else -1
            else:
                break
    
    # Games played
    games_played = len(team_games)
    
    # Rest days (days since last game)
    if as_of_date and len(team_games) > 0:
        last_game_date = team_games['Date'].iloc[-1]
        rest_days = (pd.to_datetime(as_of_date) - last_game_date).days
        rest_days = min(rest_days, 30)  # Cap at 30
    else:
        rest_days = 5  # Default
    
    # Recent margin trend (average margin in last 5 games)
    recent_margin = team_games.tail(5)['MOV'].mean() if len(team_games) >= 5 else 0
    
    # Recent opponent quality (placeholder - would need opponent barthag data)
    recent_opp_quality = 0
    
    return {
        'ats_last_5': last_5,
        'ats_last_10': last_10,
        'ats_streak': streak,
        'games_played': games_played,
        'rest_days': rest_days,
        'recent_margin': recent_margin,
        'recent_opp_quality': recent_opp_quality
    }

def calculate_all_differentials(df):
    """
    Calculate ALL differential features that the model expects
    """
    print("\n[CALC] Calculating differential features...")
    print("-" * 80)
    
    differentials_calculated = []
    
    # Core efficiency stats
    if 'adjoe_team' in df.columns and 'adjoe_opp' in df.columns:
        df['adjoe_diff'] = df['adjoe_team'] - df['adjoe_opp']
        differentials_calculated.append('adjoe_diff')
    
    if 'adjde_team' in df.columns and 'adjde_opp' in df.columns:
        df['adjde_diff'] = df['adjde_team'] - df['adjde_opp']
        differentials_calculated.append('adjde_diff')
    
    if 'barthag_team' in df.columns and 'barthag_opp' in df.columns:
        df['barthag_diff'] = df['barthag_team'] - df['barthag_opp']
        differentials_calculated.append('barthag_diff')
    
    if 'adjt_team' in df.columns and 'adjt_opp' in df.columns:
        df['adjt_diff'] = df['adjt_team'] - df['adjt_opp']
        differentials_calculated.append('adjt_diff')
    
    # Rankings and SOS
    if 'rank_team' in df.columns and 'rank_opp' in df.columns:
        df['rank_diff'] = df['rank_team'] - df['rank_opp']
        differentials_calculated.append('rank_diff')
    
    if 'sos_team' in df.columns and 'sos_opp' in df.columns:
        df['sos_diff'] = df['sos_team'] - df['sos_opp']
        differentials_calculated.append('sos_diff')
    
    if 'ncsos_team' in df.columns and 'ncsos_opp' in df.columns:
        df['ncsos_diff'] = df['ncsos_team'] - df['ncsos_opp']
        differentials_calculated.append('ncsos_diff')
    
    # Four Factors - Shooting Efficiency
    if 'eFG%_team' in df.columns and 'eFG%_opp' in df.columns:
        df['efg_pct_diff'] = df['eFG%_team'] - df['eFG%_opp']
        differentials_calculated.append('efg_pct_diff')
    
    if 'eFG% Def_team' in df.columns and 'eFG% Def_opp' in df.columns:
        df['efgd_pct_diff'] = df['eFG% Def_team'] - df['eFG% Def_opp']
        differentials_calculated.append('efgd_pct_diff')
    
    # Turnovers
    if 'TO%_team' in df.columns and 'TO%_opp' in df.columns:
        df['tor_diff'] = df['TO%_team'] - df['TO%_opp']
        differentials_calculated.append('tor_diff')
    
    if 'TO% Def._team' in df.columns and 'TO% Def._opp' in df.columns:
        df['tord_diff'] = df['TO% Def._team'] - df['TO% Def._opp']
        differentials_calculated.append('tord_diff')
    
    # Rebounding
    if 'OR%_team' in df.columns and 'OR%_opp' in df.columns:
        df['orb_diff'] = df['OR%_team'] - df['OR%_opp']
        differentials_calculated.append('orb_diff')
    
    if 'DR%_team' in df.columns and 'DR%_opp' in df.columns:
        df['drb_diff'] = df['DR%_team'] - df['DR%_opp']
        differentials_calculated.append('drb_diff')
    
    # Free Throws
    if 'FTR_team' in df.columns and 'FTR_opp' in df.columns:
        df['ftr_diff'] = df['FTR_team'] - df['FTR_opp']
        differentials_calculated.append('ftr_diff')
    
    if 'FTR Def_team' in df.columns and 'FTR Def_opp' in df.columns:
        df['ftrd_diff'] = df['FTR Def_team'] - df['FTR Def_opp']
        differentials_calculated.append('ftrd_diff')
    
    # 2-Point Shooting
    if '2p%_team' in df.columns and '2p%_opp' in df.columns:
        df['twop_pct_diff'] = df['2p%_team'] - df['2p%_opp']
        differentials_calculated.append('twop_pct_diff')
    
    if '2p%D_team' in df.columns and '2p%D_opp' in df.columns:
        df['twopd_pct_diff'] = df['2p%D_team'] - df['2p%D_opp']
        differentials_calculated.append('twopd_pct_diff')
    
    # 3-Point Shooting
    if '3P%_team' in df.columns and '3P%_opp' in df.columns:
        df['threep_pct_diff'] = df['3P%_team'] - df['3P%_opp']
        differentials_calculated.append('threep_pct_diff')
    
    if '3pD%_team' in df.columns and '3pD%_opp' in df.columns:
        df['threepd_pct_diff'] = df['3pD%_team'] - df['3pD%_opp']
        differentials_calculated.append('threepd_pct_diff')
    
    # Matchup features (if tempo data available)
    if 'adjt_team' in df.columns and 'adjt_opp' in df.columns:
        df['pace_mismatch'] = abs(df['adjt_team'] - df['adjt_opp'])
        df['combined_pace'] = (df['adjt_team'] + df['adjt_opp']) / 2
        differentials_calculated.extend(['pace_mismatch', 'combined_pace'])
    
    # Three point matchup
    if '3P rate_team' in df.columns and '3P rate_opp' in df.columns:
        df['three_point_matchup'] = df['3P rate_team'] - df['3P rate_opp']
        differentials_calculated.append('three_point_matchup')
    
    # Turnover matchup
    if 'TO%_team' in df.columns and 'TO% Def._opp' in df.columns:
        df['turnover_matchup'] = df['TO%_team'] - df['TO% Def._opp']
        differentials_calculated.append('turnover_matchup')
    
    # Rebounding matchup
    if 'OR%_team' in df.columns and 'DR%_opp' in df.columns:
        df['rebounding_matchup'] = df['OR%_team'] - df['DR%_opp']
        differentials_calculated.append('rebounding_matchup')
    
    print(f"[OK] Calculated {len(differentials_calculated)} differential features:")
    for feat in differentials_calculated:
        print(f"  ✓ {feat}")
    
    return df

def standardize_column_names(df):
    """
    Standardize column names to match what the model expects
    This is critical - model looks for specific column names!
    """
    print("\n[STANDARDIZE] Renaming columns to match model expectations...")
    
    column_mapping = {
        # Four Factors - Offense
        'eFG%_team': 'efg_pct_team',
        'TO%_team': 'tor_team',
        'OR%_team': 'orb_team',
        'DR%_team': 'drb_team',
        'FTR_team': 'ftr_team',
        '2p%_team': 'twop_pct_team',
        '3P%_team': 'threep_pct_team',
        '3P rate_team': 'threep_rate_team',
        
        # Four Factors - Defense
        'eFG% Def_team': 'efgd_pct_team',
        'TO% Def._team': 'tord_team',
        'FTR Def_team': 'ftrd_team',
        '2p%D_team': 'twopd_pct_team',
        '3pD%_team': 'threepd_pct_team',
        '3P rate D_team': 'threed_rate_team',
        
        # Same for opponent
        'eFG%_opp': 'efg_pct_opp',
        'TO%_opp': 'tor_opp',
        'OR%_opp': 'orb_opp',
        'DR%_opp': 'drb_opp',
        'FTR_opp': 'ftr_opp',
        '2p%_opp': 'twop_pct_opp',
        '3P%_opp': 'threep_pct_opp',
        '3P rate_opp': 'threep_rate_opp',
        'eFG% Def_opp': 'efgd_pct_opp',
        'TO% Def._opp': 'tord_opp',
        'FTR Def_opp': 'ftrd_opp',
        '2p%D_opp': 'twopd_pct_opp',
        '3pD%_opp': 'threepd_pct_opp',
        '3P rate D_opp': 'threed_rate_opp',
    }
    
    # Only rename columns that exist
    cols_to_rename = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=cols_to_rename)
    
    print(f"[OK] Renamed {len(cols_to_rename)} columns")
    
    return df

def add_momentum_to_predictions(predictions_file, game_history_file='current_season_games.xlsx', 
                                output_file='todays_games_with_momentum.csv'):
    """
    Add momentum features to prediction file
    """
    
    print(f"\n[LOAD] Loading predictions from {predictions_file}...")
    
    # Try to determine if it's Excel or CSV
    if predictions_file.endswith('.xlsx'):
        predictions = pd.read_excel(predictions_file)
    else:
        predictions = pd.read_csv(predictions_file)
    
    print(f"[OK] Loaded {len(predictions)} games to predict")
    
    # First, calculate ALL differential features
    predictions = calculate_all_differentials(predictions)
    
    # Load game history
    game_history = load_game_history(game_history_file)
    
    print(f"\n[CALC] Calculating momentum features...")
    print("-" * 80)
    
    # For each prediction, calculate momentum for both teams
    momentum_features = []
    
    for idx, game in predictions.iterrows():
        home_team = game['team']
        away_team = game['opp']
        game_date = game.get('game_date', datetime.now().strftime('%Y-%m-%d'))
        
        # Calculate for home team
        home_momentum = calculate_team_momentum(game_history, home_team, game_date)
        
        # Calculate for away team
        away_momentum = calculate_team_momentum(game_history, away_team, game_date)
        
        # Combine into one row
        momentum_row = {
            'home_ats_last_5': home_momentum['ats_last_5'],
            'away_ats_last_5': away_momentum['ats_last_5'],
            'ats_diff_last_5': home_momentum['ats_last_5'] - away_momentum['ats_last_5'],
            'home_ats_last_10': home_momentum['ats_last_10'],
            'away_ats_last_10': away_momentum['ats_last_10'],
            'ats_diff_last_10': home_momentum['ats_last_10'] - away_momentum['ats_last_10'],
            'home_ats_streak': home_momentum['ats_streak'],
            'away_ats_streak': away_momentum['ats_streak'],
            'home_rest_days': home_momentum['rest_days'],
            'away_rest_days': away_momentum['rest_days'],
            'rest_advantage': home_momentum['rest_days'] - away_momentum['rest_days'],
            'home_games_played': home_momentum['games_played'],
            'away_games_played': away_momentum['games_played'],
            'margin_trend': home_momentum['recent_margin'] - away_momentum['recent_margin'],
            'recent_opp_strength': 0,  # Placeholder
            'early_season': 1 if home_momentum['games_played'] < 10 or away_momentum['games_played'] < 10 else 0
        }
        
        momentum_features.append(momentum_row)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(predictions)} games...")
    
    # Convert to dataframe
    momentum_df = pd.DataFrame(momentum_features)
    
    # Merge with predictions (this keeps ALL original columns plus adds momentum)
    predictions_enhanced = pd.concat([predictions.reset_index(drop=True), momentum_df.reset_index(drop=True)], axis=1)
    
    # Copy raw team stats as separate columns (model expects these)
    if 'adjoe_team' in predictions_enhanced.columns:
        predictions_enhanced['home_adjoe'] = predictions_enhanced['adjoe_team']
        predictions_enhanced['away_adjoe'] = predictions_enhanced['adjoe_opp']
    
    if 'adjde_team' in predictions_enhanced.columns:
        predictions_enhanced['home_adjde'] = predictions_enhanced['adjde_team']
        predictions_enhanced['away_adjde'] = predictions_enhanced['adjde_opp']
    
    if 'barthag_team' in predictions_enhanced.columns:
        predictions_enhanced['home_barthag'] = predictions_enhanced['barthag_team']
        predictions_enhanced['away_barthag'] = predictions_enhanced['barthag_opp']
    
    # CRITICAL: Standardize column names to match model
    predictions_enhanced = standardize_column_names(predictions_enhanced)
    
    # Save
    if output_file.endswith('.xlsx'):
        predictions_enhanced.to_excel(output_file, index=False)
    else:
        predictions_enhanced.to_csv(output_file, index=False)
    
    print(f"\n[SAVED] {output_file}")
    print(f"[STATS] {len(predictions_enhanced)} games with {len(predictions_enhanced.columns)} columns")
    
    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print(f"\nAll features ready for model!")
    
    return predictions_enhanced

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 80)
    print("ADDING MOMENTUM & DIFFERENTIAL FEATURES")
    print("=" * 80)
    
    # Check if required files exist
    import os
    
    if not os.path.exists('current_season_games.xlsx'):
        print("\n[ERROR] current_season_games.xlsx not found!")
        print("[INFO] This file contains your game history for momentum calculations")
        sys.exit(1)
    
    if not os.path.exists('todays_games_merged.csv'):
        print("\n[ERROR] todays_games_merged.csv not found!")
        print("[INFO] Run merge_today.py first to create this file")
        sys.exit(1)
    
    # Add momentum features
    print("\nProcessing...")
    predictions_enhanced = add_momentum_to_predictions(
        predictions_file='todays_games_merged.csv',
        game_history_file='current_season_games.xlsx',
        output_file='todays_games_with_momentum.csv'
    )
    
    print("\n" + "=" * 80)
    print("READY FOR PREDICTIONS!")
    print("=" * 80)
    print("\nNext step: python predict_today.py")
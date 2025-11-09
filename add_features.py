import pandas as pd
import numpy as np

print("=" * 80)
print("ADDING ADVANCED FEATURES")
print("=" * 80)

# Load the clean merged data
print("\nLoading data...")
train = pd.read_csv('training_data.csv')
test = pd.read_csv('testing_data.csv')

# Combine for feature engineering, then split again
all_data = pd.concat([train, test], ignore_index=True)
all_data['game_date'] = pd.to_datetime(all_data['game_date'])
all_data = all_data.sort_values(['team', 'game_date']).reset_index(drop=True)

print(f"Total games: {len(all_data)}")

def add_rolling_features(df, team_col='team', date_col='game_date'):
    """Add rolling average features for each team"""
    
    print("\nAdding rolling average features...")
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Features to calculate rolling averages for
    team_features = ['home_adjoe', 'home_adjde', 'home_barthag']
    
    # For each team, calculate rolling stats
    for window in [5, 10]:
        print(f"  Calculating {window}-game rolling averages...")
        
        for feature in team_features:
            col_name = f'{feature}_L{window}'
            
            # Calculate rolling mean for home games (when team is 'home')
            df[col_name] = (df.groupby(team_col)[feature]
                           .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1)))
    
    return df

def add_win_streak_features(df):
    """Add winning/covering streak features"""
    
    print("\nAdding streak features...")
    
    df = df.copy()
    
    # Calculate if team covered in each game (from team's perspective)
    # For home games: use home_covered
    # For away games: inverse of home_covered
    df['team_covered'] = np.where(
        df['loc'] == 'Home',
        df['home_covered'],
        1 - df['home_covered']
    )
    
    # Calculate streaks
    def calculate_streak(series):
        """Calculate current winning/losing streak"""
        streak = []
        current_streak = 0
        
        for val in series:
            if pd.isna(val):
                streak.append(0)
            else:
                if val == 1:  # Win/Cover
                    current_streak = current_streak + 1 if current_streak > 0 else 1
                else:  # Loss
                    current_streak = current_streak - 1 if current_streak < 0 else -1
                streak.append(current_streak)
        
        return pd.Series(streak).shift(1).fillna(0)  # Shift to avoid leakage
    
    df['home_streak'] = df.groupby('team')['team_covered'].transform(calculate_streak)
    df['away_streak'] = df.groupby('opp')['team_covered'].transform(calculate_streak)
    
    return df

def add_rest_days(df):
    """Add days of rest since last game"""
    
    print("\nAdding rest days...")
    
    df = df.copy()
    
    # Calculate days since last game for each team
    df['home_rest_days'] = df.groupby('team')['game_date'].diff().dt.days
    df['away_rest_days'] = df.groupby('opp')['game_date'].diff().dt.days
    
    # Fill first games with median rest
    df['home_rest_days'] = df['home_rest_days'].fillna(df['home_rest_days'].median())
    df['away_rest_days'] = df['away_rest_days'].fillna(df['away_rest_days'].median())
    
    # Cap at reasonable maximum (30 days)
    df['home_rest_days'] = df['home_rest_days'].clip(0, 30)
    df['away_rest_days'] = df['away_rest_days'].clip(0, 30)
    
    # Create differential
    df['rest_diff'] = df['home_rest_days'] - df['away_rest_days']
    
    return df

def add_home_away_splits(df):
    """Add team's home/away performance differentials"""
    
    print("\nAdding home/away splits...")
    
    df = df.copy()
    
    # Calculate home vs away performance for each team
    # (This is more complex - simplified version here)
    
    # Home team's home record (ATS)
    home_home_record = (df[df['loc'] == 'Home']
                       .groupby('team')['home_covered']
                       .transform(lambda x: x.expanding().mean().shift(1)))
    
    # Away team's away record (ATS) 
    away_away_record = (df[df['loc'] == 'Away']
                       .groupby('opp')['home_covered']
                       .transform(lambda x: (1-x).expanding().mean().shift(1)))
    
    df['home_home_record'] = home_home_record
    df['away_away_record'] = away_away_record
    
    return df

# Add all features
print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

all_data_enhanced = all_data.copy()

# Add features
all_data_enhanced = add_rest_days(all_data_enhanced)
all_data_enhanced = add_win_streak_features(all_data_enhanced)
all_data_enhanced = add_rolling_features(all_data_enhanced)
all_data_enhanced = add_home_away_splits(all_data_enhanced)

# Split back into train/test
cutoff_date = pd.Timestamp('2024-11-01')
train_enhanced = all_data_enhanced[all_data_enhanced['game_date'] < cutoff_date].copy()
test_enhanced = all_data_enhanced[all_data_enhanced['game_date'] >= cutoff_date].copy()

print(f"\n✅ Feature engineering complete!")
print(f"   Training set: {len(train_enhanced)} games")
print(f"   Testing set: {len(test_enhanced)} games")
print(f"   Total features: {len(train_enhanced.columns)} columns")

# Show new features
new_features = [col for col in train_enhanced.columns if col not in train.columns]
print(f"\n📊 New features added ({len(new_features)}):")
for feature in sorted(new_features):
    print(f"   - {feature}")

# Save enhanced datasets
train_enhanced.to_csv('training_data_enhanced.csv', index=False)
test_enhanced.to_csv('testing_data_enhanced.csv', index=False)

print("\n" + "=" * 80)
print("SAVED ENHANCED DATASETS")
print("=" * 80)
print("   training_data_enhanced.csv")
print("   testing_data_enhanced.csv")
print("\nNext step: Run train_models_enhanced.py to test the new features!")

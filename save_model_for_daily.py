import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("SAVING LOGISTIC REGRESSION MODEL FOR DAILY PREDICTIONS")
print("=" * 80)

# Load enhanced training data
print("\n[LOAD] Loading training_data_enhanced.csv...")
train = pd.read_csv('training_data_enhanced.csv')
print(f"[OK] Loaded {len(train)} training games")

# Feature columns (same as train_models.py)
feature_columns = [
    'spread',
    'adjoe_diff', 'adjde_diff', 'adjt_diff', 'barthag_diff', 'rank_diff',
    'sos_diff', 'ncsos_diff',
    'efg_pct_diff', 'efgd_pct_diff',
    'tor_diff', 'tord_diff',
    'orb_diff', 'drb_diff',
    'ftr_diff', 'ftrd_diff',
    'twop_pct_diff', 'twopd_pct_diff',
    'threep_pct_diff', 'threepd_pct_diff',
    'home_adjoe', 'away_adjoe', 'home_adjde', 'away_adjde', 'home_barthag', 'away_barthag',
    'home_ats_last_5', 'away_ats_last_5', 'ats_diff_last_5',
    'home_ats_last_10', 'away_ats_last_10', 'ats_diff_last_10',
    'home_ats_streak', 'away_ats_streak',
    'rest_advantage', 'home_rest_days', 'away_rest_days',
    'margin_trend', 'recent_opp_strength', 'early_season',
    'pace_mismatch', 'combined_pace',
    'three_point_matchup', 'turnover_matchup', 'rebounding_matchup',
    'home_games_played', 'away_games_played',
]

target = 'home_covered'

# Handle missing values (same as train_models.py)
print("\n[CLEAN] Filling missing values...")

# For momentum features
momentum_features = [f for f in feature_columns if any(x in f for x in ['ats', 'streak', 'rest', 'trend', 'opp_strength', 'games_played'])]
for feat in momentum_features:
    if feat in train.columns:
        if 'ats' in feat and 'diff' not in feat and 'streak' not in feat:
            train[feat] = train[feat].fillna(0.5)
        elif 'rest' in feat:
            median_val = train[feat].median()
            train[feat] = train[feat].fillna(median_val)
        elif 'games_played' in feat:
            train[feat] = train[feat].fillna(0)
        else:
            train[feat] = train[feat].fillna(0)

# For matchup features
matchup_features = [f for f in feature_columns if any(x in f for x in ['matchup', 'mismatch', 'pace'])]
for feat in matchup_features:
    if feat in train.columns:
        train[feat] = train[feat].fillna(0)

# For all other features
for feat in feature_columns:
    if feat in train.columns and train[feat].isna().any():
        median_val = train[feat].median()
        train[feat] = train[feat].fillna(median_val)

# Drop rows with missing target
train_clean = train.dropna(subset=[target])

# Prepare features
X = train_clean[feature_columns]
y = train_clean[target]

print(f"[OK] Training on {len(X)} games with {len(feature_columns)} features")

# Scale features
print("\n[SCALE] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
print("\n[TRAIN] Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

accuracy = model.score(X_scaled, y)
print(f"[OK] Training accuracy: {accuracy*100:.2f}%")

# Save everything
print("\n[SAVE] Saving model files...")

with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("[OK] Saved: logistic_regression_model.pkl")

with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("[OK] Saved: feature_scaler.pkl")

with open('feature_list.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("[OK] Saved: feature_list.pkl")

print("\n" + "=" * 80)
print("SUCCESS! MODEL SAVED")
print("=" * 80)
print("\n[MODEL] Logistic Regression with 47 features")
print(f"[ACCURACY] {accuracy*100:.2f}%")
print(f"[FEATURES] {len(feature_columns)} total")
print("\n[FILES CREATED]")
print("  - logistic_regression_model.pkl")
print("  - feature_scaler.pkl") 
print("  - feature_list.pkl")
print("\n[NEXT STEP]")
print("  Run: python scrape_today_odds_api.py")
print("  (Make sure you have odds_api_key.txt with your API key)")
print("\n" + "=" * 80)

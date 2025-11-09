import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

print("="*80)
print("SAVE LOGISTIC REGRESSION MODEL - FIXED")
print("="*80)

# Load your training data
print("\n[LOAD] Loading training_data_enhanced.csv...")
train = pd.read_csv('training_data_enhanced.csv')
print(f"[OK] Loaded {len(train)} training games")

# Use the features that actually exist and have data
# These are the same features your train_models.py used
feature_cols = [
    'spread',
    'barthag_diff', 
    'adjoe_diff',
    'adjde_diff',
    'rank_diff',
    'sos_diff',
    'adjt_diff',
    'home_adjoe',  # Using home_ prefix
    'away_adjoe',  # Using away_ prefix
    'home_adjde',
    'away_adjde',
    'home_barthag',
    'away_barthag'
]

print(f"\n[INFO] Using {len(feature_cols)} features")

# Remove rows with missing values
print("\n[CLEAN] Removing rows with missing values in features...")
train_clean = train[feature_cols + ['team_covered']].dropna()
print(f"[OK] Clean data: {len(train_clean):,} games ({len(train_clean)/len(train)*100:.1f}%)")

X = train_clean[feature_cols]
y = train_clean['team_covered']

print(f"\n[TRAIN] Training Logistic Regression on {len(X):,} games...")
print(f"[INFO] Cover rate: {y.mean():.2%}")

# Scale features (your train_models.py does this)
print("\n[SCALE] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
print("[TRAIN] Training model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

accuracy = model.score(X_scaled, y)
print(f"[OK] Training accuracy: {accuracy:.2%}")

# Calculate metrics
wins = (model.predict(X_scaled) == y).sum()
total = len(y)
win_pct = wins / total

print(f"\n[RESULTS] Win rate: {win_pct:.2%}")
print(f"[RESULTS] Break-even: 52.38%")

if win_pct > 0.5238:
    edge = (win_pct - 0.5238) * 100
    print(f"[SUCCESS] Beats break-even! Edge: +{edge:.2f}%")
else:
    print(f"[WARN] Below break-even threshold")

# Save everything
print("\n[SAVE] Saving model files...")

# 1. Save the trained model
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("[OK] Saved: logistic_regression_model.pkl")

# 2. Save the scaler (critical for predictions!)
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("[OK] Saved: feature_scaler.pkl")

# 3. Save feature list
with open('feature_list.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("[OK] Saved: feature_list.pkl")

print("\n" + "="*80)
print("SUCCESS! Model files saved")
print("="*80)
print("\n[MODEL] Logistic Regression")
print(f"[ACCURACY] {accuracy:.2%}")
print(f"[TRAINED ON] {len(X):,} games")
print("\n[FILES] Created:")
print("  - logistic_regression_model.pkl")
print("  - feature_scaler.pkl")
print("  - feature_list.pkl")
print("\n[NEXT] Run: python run_daily_simple_odds_api.py")
print("="*80)
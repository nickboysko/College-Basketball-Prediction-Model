import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import sys
import os
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("TRAIN LOGISTIC REGRESSION MODEL")
print("="*80)

# Look for historical data files
HISTORICAL_FILES = [
    'historical_games_with_stats.csv',
    'cleaned_merged_data.csv',
    'training_data.csv',
    'all_games_merged.csv'
]

historical_file = None
for file in HISTORICAL_FILES:
    if os.path.exists(file):
        historical_file = file
        break

if not historical_file:
    print("\n[ERROR] No historical data file found!")
    print("[INFO] Looking for one of these files:")
    for f in HISTORICAL_FILES:
        print(f"  - {f}")
    print("\n[HELP] You need historical game data with:")
    print("  - Team stats: adjoe_team, adjde_team, barthag_team")
    print("  - Opponent stats: adjoe_opp, adjde_opp, barthag_opp")
    print("  - Outcome: covered (1 if team covered spread, 0 if not)")
    print("\n[HELP] If you have this data in a different file, update this script")
    sys.exit(1)

print(f"\n[OK] Found historical data: {historical_file}")
print(f"[LOAD] Loading data...")

data = pd.read_csv(historical_file)
print(f"[OK] Loaded {len(data)} historical games")

# Calculate differential features if not present
if 'adjoe_diff' not in data.columns:
    print("\n[CALC] Calculating differential features...")
    data['adjoe_diff'] = data['adjoe_team'] - data['adjoe_opp']
    data['adjde_diff'] = data['adjde_team'] - data['adjde_opp']
    data['adjthag_diff'] = data['barthag_team'] - data['barthag_opp']
    print("[OK] Features calculated")

# Check for required columns
REQUIRED_FEATURES = ['adjoe_diff', 'adjde_diff', 'barthag_diff']
REQUIRED_TARGET = 'covered'

missing = [f for f in REQUIRED_FEATURES if f not in data.columns]
if missing:
    print(f"\n[ERROR] Missing required feature columns: {missing}")
    print(f"[INFO] Available columns: {list(data.columns)}")
    sys.exit(1)

if REQUIRED_TARGET not in data.columns:
    print(f"\n[ERROR] Missing target column: {REQUIRED_TARGET}")
    print("[HELP] You need a 'covered' column (1 if team covered, 0 if not)")
    print("[INFO] Calculate it as: covered = (team_score + spread > opp_score)")
    sys.exit(1)

# Remove any rows with missing values
original_len = len(data)
data = data.dropna(subset=REQUIRED_FEATURES + [REQUIRED_TARGET])
print(f"\n[CLEAN] Removed {original_len - len(data)} rows with missing values")
print(f"[OK] Training data: {len(data)} games")

# Prepare features and target
X = data[REQUIRED_FEATURES]
y = data[REQUIRED_TARGET]

print(f"\n[INFO] Features: {REQUIRED_FEATURES}")
print(f"[INFO] Target: {REQUIRED_TARGET}")
print(f"[INFO] Cover rate: {y.mean():.2%}")

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[SPLIT] Training set: {len(X_train)} games")
print(f"[SPLIT] Test set: {len(X_test)} games")

# Train model
print(f"\n[TRAIN] Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("[OK] Model trained!")

# Evaluate
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"\n[EVAL] Training accuracy: {train_accuracy:.2%}")
print(f"[EVAL] Test accuracy: {test_accuracy:.2%}")

if test_accuracy > 0.5238:
    print(f"[SUCCESS] Model beats break-even threshold (52.38%)!")
    edge = (test_accuracy - 0.5238) * 100
    print(f"[SUCCESS] Edge: +{edge:.2f}%")
else:
    print(f"[WARN] Model below break-even threshold (52.38%)")

# Save model
MODEL_FILE = 'logistic_regression_model.pkl'
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

print(f"\n[SAVE] Model saved to {MODEL_FILE}")

# Show feature importance (coefficients)
print(f"\n[INFO] Feature coefficients:")
for feature, coef in zip(REQUIRED_FEATURES, model.coef_[0]):
    print(f"  {feature:20s}: {coef:+.6f}")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE")
print("="*80)
print(f"\n[OUTPUT] Model saved to: {MODEL_FILE}")
print(f"[ACCURACY] Test accuracy: {test_accuracy:.2%}")
print(f"[NEXT] Run: python predict_today.py")
print("="*80)

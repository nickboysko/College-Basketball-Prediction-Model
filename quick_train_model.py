import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import sys
import os

print("="*80)
print("QUICK MODEL TRAINER")
print("="*80)

# Try to find training data
training_files = [
    'training_data_enhanced.xlsx',
    'training_data_enhanced.csv',
    'training_data.xlsx',
    'training_data.csv',
]

data_file = None
for file in training_files:
    if os.path.exists(file):
        data_file = file
        print(f"\n[FOUND] {file}")
        break

if not data_file:
    print("\n[ERROR] No training data found!")
    print("Looking for: training_data.csv or training_data_enhanced.csv")
    sys.exit(1)

# Load data
print(f"[LOAD] Loading {data_file}...")
if data_file.endswith('.csv'):
    data = pd.read_csv(data_file)
else:
    data = pd.read_excel(data_file)

print(f"[OK] Loaded {len(data)} games")

# Check for required columns
required_cols = ['adjoe_diff', 'adjde_diff', 'barthag_diff', 'covered']
missing = [col for col in required_cols if col not in data.columns]

if missing:
    print(f"\n[ERROR] Missing columns: {missing}")
    print(f"[INFO] Available columns: {list(data.columns)[:20]}")
    sys.exit(1)

# Prepare data
X = data[['adjoe_diff', 'adjde_diff', 'barthag_diff']]
y = data['covered']

# Remove NaN
X = X.dropna()
y = y[X.index]

print(f"\n[TRAIN] Training on {len(X)} games...")
print(f"[INFO] Cover rate: {y.mean():.2%}")

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

accuracy = model.score(X, y)
print(f"[OK] Model accuracy: {accuracy:.2%}")

# Save model
model_file = 'logistic_regression_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"\n[SAVE] Model saved to {model_file}")
print("\n" + "="*80)
print("SUCCESS! Model is ready!")
print("="*80)
print("\n[NEXT] Run: python run_daily_simple_odds_api.py")

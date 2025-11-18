import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle
from datetime import datetime

print("=" * 80)
print("MODEL COMPARISON: LOGISTIC REGRESSION vs XGBOOST")
print("=" * 80)
print(f"\nTraining Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

train = pd.read_csv('training_data_with_momentum.csv')
test = pd.read_csv('testing_data_with_momentum.csv')

print(f"\n[OK] Training: {len(train)} games (2021-2024)")
print(f"[OK] Testing:  {len(test)} games (2024-2025)")

# ============================================================================
# DEFINE FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

# Core Barttorvik features
barttorvik_features = [
    'adjoe_diff', 'adjde_diff', 'barthag_diff', 
    'rank_diff', 'sos_diff', 'ncsos_diff',
    'efg_pct_diff', 'efgd_pct_diff',
    'tor_diff', 'tord_diff',
    'orb_diff', 'drb_diff',
    'ftr_diff', 'ftrd_diff',
    'twop_pct_diff', 'twopd_pct_diff',
    'threep_pct_diff', 'threepd_pct_diff'
]

# Momentum features
momentum_features = [
    'home_ats_last_5', 'away_ats_last_5', 'ats_diff_last_5',
    'home_ats_last_10', 'away_ats_last_10', 'ats_diff_last_10',
    'home_ats_streak', 'away_ats_streak',
    'home_rest_days', 'away_rest_days',
    'margin_trend', 'recent_opp_strength', 'team_ats_expanding',
    'home_games_played', 'away_games_played', 'early_season'
]

# Style matchup features
style_features = [
    'pace_mismatch', 'combined_pace',
    'three_point_matchup', 'turnover_matchup', 'rebounding_matchup'
]

# Combine all features
all_features = barttorvik_features + momentum_features + style_features

# Check which features exist
available_features = [f for f in all_features if f in train.columns]
missing_features = [f for f in all_features if f not in train.columns]

print(f"\n[INFO] Total features: {len(all_features)}")
print(f"[OK] Available: {len(available_features)}/{len(all_features)} features")

if missing_features:
    print(f"\n[WARN] Missing {len(missing_features)} features (will use available only)")

feature_columns = available_features

# ============================================================================
# PREPARE DATA
# ============================================================================

print("\n" + "=" * 80)
print("PREPARING DATA")
print("=" * 80)

train_clean = train.dropna(subset=feature_columns + ['home_covered'])
test_clean = test.dropna(subset=feature_columns + ['home_covered'])

print(f"\n[CLEAN] Training: {len(train_clean)}/{len(train)} games")
print(f"[CLEAN] Testing:  {len(test_clean)}/{len(test)} games")

X_train = train_clean[feature_columns]
y_train = train_clean['home_covered']

X_test = test_clean[feature_columns]
y_test = test_clean['home_covered']

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 80)

# Features that should NOT be normalized
no_normalize = [
    'home_ats_last_5', 'away_ats_last_5', 'home_ats_last_10', 'away_ats_last_10',
    'recent_opp_strength', 'team_ats_expanding', 'early_season', 'rank_diff'
]

normalize_features = [f for f in feature_columns if f not in no_normalize]

print(f"\n[INFO] Normalizing {len(normalize_features)} features")
print(f"[INFO] Leaving {len(no_normalize)} features as-is")

# Fit scaler
scaler = StandardScaler()
X_train_lr = X_train.copy()
X_test_lr = X_test.copy()

if len(normalize_features) > 0:
    X_train_lr[normalize_features] = scaler.fit_transform(X_train[normalize_features])
    X_test_lr[normalize_features] = scaler.transform(X_test[normalize_features])

# Train model
print("\n[TRAIN] Training Logistic Regression...")
lr_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_lr, y_train)
print("[OK] Logistic Regression trained")

# Predictions
lr_train_pred = lr_model.predict(X_train_lr)
lr_train_proba = lr_model.predict_proba(X_train_lr)[:, 1]

lr_test_pred = lr_model.predict(X_test_lr)
lr_test_proba = lr_model.predict_proba(X_test_lr)[:, 1]

# Calculate metrics
lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)

train_clean['lr_pred'] = lr_train_pred
train_clean['lr_bet'] = np.where(lr_train_pred == train_clean['home_covered'], 100, -110)
lr_train_roi = (train_clean['lr_bet'].sum() / (110 * len(train_clean))) * 100

test_clean['lr_pred'] = lr_test_pred
test_clean['lr_bet'] = np.where(lr_test_pred == test_clean['home_covered'], 100, -110)
lr_test_roi = (test_clean['lr_bet'].sum() / (110 * len(test_clean))) * 100

print(f"\n[RESULTS] Logistic Regression:")
print(f"  Training Accuracy: {lr_train_acc*100:.2f}%")
print(f"  Testing Accuracy:  {lr_test_acc*100:.2f}%")
print(f"  Training ROI:      {lr_train_roi:.2f}%")
print(f"  Testing ROI:       {lr_test_roi:.2f}%")

# ============================================================================
# MODEL 2: XGBOOST
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 2: XGBOOST")
print("=" * 80)

print("\n[TRAIN] Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train, verbose=False)
print("[OK] XGBoost trained")

# Predictions
xgb_train_pred = xgb_model.predict(X_train)
xgb_train_proba = xgb_model.predict_proba(X_train)[:, 1]

xgb_test_pred = xgb_model.predict(X_test)
xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]

# Calculate metrics
xgb_train_acc = accuracy_score(y_train, xgb_train_pred)
xgb_test_acc = accuracy_score(y_test, xgb_test_pred)

train_clean['xgb_pred'] = xgb_train_pred
train_clean['xgb_bet'] = np.where(xgb_train_pred == train_clean['home_covered'], 100, -110)
xgb_train_roi = (train_clean['xgb_bet'].sum() / (110 * len(train_clean))) * 100

test_clean['xgb_pred'] = xgb_test_pred
test_clean['xgb_bet'] = np.where(xgb_test_pred == test_clean['home_covered'], 100, -110)
xgb_test_roi = (test_clean['xgb_bet'].sum() / (110 * len(test_clean))) * 100

print(f"\n[RESULTS] XGBoost:")
print(f"  Training Accuracy: {xgb_train_acc*100:.2f}%")
print(f"  Testing Accuracy:  {xgb_test_acc*100:.2f}%")
print(f"  Training ROI:      {xgb_train_roi:.2f}%")
print(f"  Testing ROI:       {xgb_test_roi:.2f}%")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("HEAD-TO-HEAD COMPARISON")
print("=" * 80)

print(f"\n{'Metric':<25} {'Logistic Reg':<15} {'XGBoost':<15} {'Winner':<10}")
print("-" * 70)
print(f"{'Training Accuracy':<25} {lr_train_acc*100:>6.2f}%         {xgb_train_acc*100:>6.2f}%         {'XGB' if xgb_train_acc > lr_train_acc else 'LR'}")
print(f"{'Testing Accuracy':<25} {lr_test_acc*100:>6.2f}%         {xgb_test_acc*100:>6.2f}%         {'XGB' if xgb_test_acc > lr_test_acc else 'LR'}")
print(f"{'Training ROI':<25} {lr_train_roi:>6.2f}%         {xgb_train_roi:>6.2f}%         {'XGB' if xgb_train_roi > lr_train_roi else 'LR'}")
print(f"{'Testing ROI':<25} {lr_test_roi:>6.2f}%         {xgb_test_roi:>6.2f}%         {'XGB' if xgb_test_roi > lr_test_roi else 'LR'}")

# Determine winner
test_scores = {'LR': lr_test_acc, 'XGB': xgb_test_acc}
winner = 'XGBoost' if xgb_test_acc > lr_test_acc else 'Logistic Regression'
winner_acc = max(xgb_test_acc, lr_test_acc)
winner_roi = xgb_test_roi if xgb_test_acc > lr_test_acc else lr_test_roi

print("\n" + "=" * 80)
print("WINNER")
print("=" * 80)
print(f"\n🏆 {winner}")
print(f"   Test Accuracy: {winner_acc*100:.2f}%")
print(f"   Test ROI: {winner_roi:.2f}%")

if winner_acc > 0.5238:
    print(f"\n✅ PROFITABLE! {(winner_acc - 0.5238)*100:.2f}% above break-even")
else:
    print(f"\n⚠️  Below break-even by {(0.5238 - winner_acc)*100:.2f}%")

# ============================================================================
# XGBOOST FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("XGBOOST FEATURE IMPORTANCE (TOP 20)")
print("=" * 80)

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n")
for i, row in feature_importance.head(20).iterrows():
    print(f"{row['feature']:30s} {row['importance']:.4f}")

# ============================================================================
# SAVE BEST MODEL
# ============================================================================

print("\n" + "=" * 80)
print("SAVING BEST MODEL")
print("=" * 80)

if xgb_test_acc > lr_test_acc:
    best_model = xgb_model
    best_model_name = 'XGBoost'
    model_data = {
        'model': best_model,
        'model_type': 'xgboost',
        'feature_columns': feature_columns,
        'test_accuracy': xgb_test_acc,
        'test_roi': xgb_test_roi
    }
else:
    best_model = lr_model
    best_model_name = 'Logistic Regression'
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'model_type': 'logistic_regression',
        'feature_columns': feature_columns,
        'normalize_features': normalize_features,
        'no_normalize': no_normalize,
        'test_accuracy': lr_test_acc,
        'test_roi': lr_test_roi
    }

with open('spread_model_best.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n[SAVED] spread_model_best.pkl ({best_model_name})")

# Also save both models for comparison
with open('spread_model_lr.pkl', 'wb') as f:
    pickle.dump({
        'model': lr_model,
        'scaler': scaler,
        'model_type': 'logistic_regression',
        'feature_columns': feature_columns,
        'normalize_features': normalize_features,
        'no_normalize': no_normalize,
        'test_accuracy': lr_test_acc,
        'test_roi': lr_test_roi
    }, f)

with open('spread_model_xgb.pkl', 'wb') as f:
    pickle.dump({
        'model': xgb_model,
        'model_type': 'xgboost',
        'feature_columns': feature_columns,
        'test_accuracy': xgb_test_acc,
        'test_roi': xgb_test_roi
    }, f)

print(f"[SAVED] spread_model_lr.pkl (Logistic Regression)")
print(f"[SAVED] spread_model_xgb.pkl (XGBoost)")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)

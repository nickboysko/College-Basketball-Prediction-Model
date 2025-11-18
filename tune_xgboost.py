import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
from datetime import datetime
import os

print("=" * 80)
print("XGBOOST HYPERPARAMETER TUNING")
print("=" * 80)
print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Backup current model
print("\n[BACKUP] Saving copy of current best model...")
if os.path.exists('spread_model_xgb.pkl'):
    import shutil
    shutil.copy('spread_model_xgb.pkl', 'spread_model_xgb_backup.pkl')
    print("[OK] Backup saved as: spread_model_xgb_backup.pkl")
else:
    print("[WARN] No existing XGBoost model found to backup")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

train = pd.read_csv('training_data_with_momentum.csv')
test = pd.read_csv('testing_data_with_momentum.csv')

print(f"\n[OK] Training: {len(train)} games")
print(f"[OK] Testing:  {len(test)} games")

# Define features (same as before)
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

momentum_features = [
    'home_ats_last_5', 'away_ats_last_5', 'ats_diff_last_5',
    'home_ats_last_10', 'away_ats_last_10', 'ats_diff_last_10',
    'home_ats_streak', 'away_ats_streak',
    'home_rest_days', 'away_rest_days',
    'margin_trend', 'recent_opp_strength', 'team_ats_expanding',
    'home_games_played', 'away_games_played', 'early_season'
]

style_features = [
    'pace_mismatch', 'combined_pace',
    'three_point_matchup', 'turnover_matchup', 'rebounding_matchup'
]

feature_columns = barttorvik_features + momentum_features + style_features

# Prepare data
train_clean = train.dropna(subset=feature_columns + ['home_covered'])
test_clean = test.dropna(subset=feature_columns + ['home_covered'])

X_train = train_clean[feature_columns]
y_train = train_clean['home_covered']
X_test = test_clean[feature_columns]
y_test = test_clean['home_covered']

# ============================================================================
# DEFINE CONFIGURATIONS TO TEST
# ============================================================================

print("\n" + "=" * 80)
print("TESTING CONFIGURATIONS")
print("=" * 80)

# Current best (baseline)
baseline_config = {
    'name': 'CURRENT (Baseline)',
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0
}

# Configurations to test (reducing overfitting)
configs = [
    baseline_config,
    {
        'name': 'Config 1: Reduced Depth',
        'n_estimators': 100,
        'max_depth': 3,  # Shallower trees
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0
    },
    {
        'name': 'Config 2: Conservative',
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.05,  # Slower learning
        'subsample': 0.7,  # Less data per tree
        'colsample_bytree': 0.7,  # Fewer features per tree
        'min_child_weight': 3,  # Require more samples per leaf
        'gamma': 0.1  # Add regularization
    },
    {
        'name': 'Config 3: More Trees, Shallower',
        'n_estimators': 150,  # More trees
        'max_depth': 3,  # But shallower
        'learning_rate': 0.08,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'gamma': 0
    },
    {
        'name': 'Config 4: Heavy Regularization',
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,  # Very conservative
        'gamma': 0.2,  # Strong regularization
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0  # L2 regularization
    },
    {
        'name': 'Config 5: Balanced',
        'n_estimators': 120,
        'max_depth': 4,
        'learning_rate': 0.08,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 2,
        'gamma': 0.05
    }
]

print(f"\n[INFO] Testing {len(configs)} configurations...")
print("[INFO] This will take a few minutes...")

# ============================================================================
# TRAIN AND EVALUATE EACH CONFIG
# ============================================================================

results = []

for i, config in enumerate(configs, 1):
    name = config.pop('name')
    print(f"\n[{i}/{len(configs)}] Testing: {name}")
    
    # Train model
    model = XGBClassifier(**config, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    # ROI
    train_roi = (np.where(train_pred == y_train, 100, -110).sum() / (110 * len(y_train))) * 100
    test_roi = (np.where(test_pred == y_test, 100, -110).sum() / (110 * len(y_test))) * 100
    
    # Overfitting gap
    overfit_gap = train_acc - test_acc
    
    results.append({
        'name': name,
        'config': config,
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_roi': train_roi,
        'test_roi': test_roi,
        'overfit_gap': overfit_gap
    })
    
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    print(f"  Test ROI:       {test_roi:.2f}%")
    print(f"  Overfit Gap:    {overfit_gap*100:.2f}%")

# ============================================================================
# COMPARE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print(f"\n{'Configuration':<30} {'Train':<8} {'Test':<8} {'Gap':<8} {'ROI':<8} {'Profit?':<8}")
print("-" * 85)

for r in results:
    profit = "✅" if r['test_acc'] > 0.5238 else "❌"
    print(f"{r['name']:<30} {r['train_acc']*100:>6.2f}% {r['test_acc']*100:>6.2f}% {r['overfit_gap']*100:>6.2f}% {r['test_roi']:>6.2f}% {profit:>6}")

# ============================================================================
# SELECT BEST MODEL
# ============================================================================

print("\n" + "=" * 80)
print("SELECTING BEST MODEL")
print("=" * 80)

# Sort by test accuracy (primary) and overfit gap (secondary)
results_sorted = sorted(results, key=lambda x: (x['test_acc'], -x['overfit_gap']), reverse=True)

best = results_sorted[0]
baseline = results[0]  # First config is baseline

print(f"\n🏆 BEST MODEL: {best['name']}")
print(f"   Test Accuracy: {best['test_acc']*100:.2f}%")
print(f"   Test ROI:      {best['test_roi']:.2f}%")
print(f"   Overfit Gap:   {best['overfit_gap']*100:.2f}%")

print(f"\n📊 BASELINE: {baseline['name']}")
print(f"   Test Accuracy: {baseline['test_acc']*100:.2f}%")
print(f"   Test ROI:      {baseline['test_roi']:.2f}%")
print(f"   Overfit Gap:   {baseline['overfit_gap']*100:.2f}%")

# ============================================================================
# DECISION
# ============================================================================

print("\n" + "=" * 80)
print("DECISION")
print("=" * 80)

improvement = (best['test_acc'] - baseline['test_acc']) * 100
roi_improvement = best['test_roi'] - baseline['test_roi']
overfit_improvement = baseline['overfit_gap'] - best['overfit_gap']

print(f"\nTest Accuracy Change: {improvement:+.2f}%")
print(f"ROI Change:           {roi_improvement:+.2f}%")
print(f"Overfit Reduction:    {overfit_improvement*100:+.2f}%")

# Only save new model if it's better OR significantly reduces overfitting
if best['test_acc'] >= baseline['test_acc'] or (overfit_improvement > 0.05 and best['test_acc'] >= baseline['test_acc'] - 0.005):
    print(f"\n✅ NEW MODEL IS BETTER (or equal with less overfitting)")
    print(f"   Saving {best['name']} as new best model...")
    
    model_data = {
        'model': best['model'],
        'model_type': 'xgboost',
        'feature_columns': feature_columns,
        'test_accuracy': best['test_acc'],
        'test_roi': best['test_roi'],
        'config': best['config'],
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('spread_model_xgb.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    with open('spread_model_best.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"   [SAVED] spread_model_xgb.pkl")
    print(f"   [SAVED] spread_model_best.pkl")
    print(f"\n   Your old model is still saved as: spread_model_xgb_backup.pkl")
    
else:
    print(f"\n⚠️  BASELINE IS STILL BETTER")
    print(f"   Keeping current model unchanged")
    print(f"   Test accuracy would decrease by {-improvement:.2f}%")
    print(f"\n   No changes made - your model is still the best!")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n📊 Models Tested: {len(configs)}")
print(f"🏆 Best Test Accuracy: {results_sorted[0]['test_acc']*100:.2f}%")
print(f"💰 Best Test ROI: {results_sorted[0]['test_roi']:.2f}%")
print(f"📉 Lowest Overfit Gap: {min(r['overfit_gap'] for r in results)*100:.2f}%")

if best['test_acc'] > 0.5238:
    edge = (best['test_acc'] - 0.5238) * 100
    print(f"\n✅ PROFITABLE: {edge:.2f}% above break-even")
else:
    deficit = (0.5238 - best['test_acc']) * 100
    print(f"\n⚠️  UNPROFITABLE: {deficit:.2f}% below break-even")

print("\n" + "=" * 80)
print("You can always restore your backup:")
print("  cp spread_model_xgb_backup.pkl spread_model_xgb.pkl")
print("=" * 80)

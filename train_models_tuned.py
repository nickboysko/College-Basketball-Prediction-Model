import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TUNED MODELS - PREVENTING OVERFITTING")
print("=" * 80)

# Load enhanced data
print("\nLoading enhanced data...")
train = pd.read_csv('training_data_enhanced.csv')
test = pd.read_csv('testing_data_enhanced.csv')

# Original features
original_features = [
    'spread', 'adjoe_diff', 'adjde_diff', 'adjt_diff', 'barthag_diff',
    'rank_diff', 'sos_diff', 'home_adjoe', 'away_adjoe', 'home_adjde',
    'away_adjde', 'home_barthag', 'away_barthag'
]

# New features (if they exist)
new_features = [
    'rest_diff', 'home_rest_days', 'away_rest_days',
    'home_streak', 'away_streak',
    'home_adjoe_L5', 'home_adjde_L5', 'home_barthag_L5',
    'home_adjoe_L10', 'home_adjde_L10', 'home_barthag_L10',
    'home_home_record', 'away_away_record'
]

# Use all available features
feature_columns = original_features + [f for f in new_features if f in train.columns]
target = 'home_covered'

print(f"\nFeatures to use: {len(feature_columns)}")
print(f"Checking for missing values...")

# Check which features have too many missing values
for feature in feature_columns[:]:
    train_missing = train[feature].isna().sum()
    test_missing = test[feature].isna().sum()
    train_pct = train_missing / len(train) * 100
    test_pct = test_missing / len(test) * 100
    
    if train_pct > 50 or test_pct > 50:
        print(f"  Removing {feature}: {train_pct:.1f}% train missing, {test_pct:.1f}% test missing")
        feature_columns.remove(feature)

# Clean data - fill NaN with median instead of dropping
train_clean = train.copy()
test_clean = test.copy()

# Drop rows where target is missing
train_clean = train_clean.dropna(subset=[target])
test_clean = test_clean.dropna(subset=[target])

print(f"\nAfter removing target NaN:")
print(f"  Training: {len(train_clean)} games")
print(f"  Testing: {len(test_clean)} games")

# For features, fill NaN with median
for feature in feature_columns:
    if train_clean[feature].isna().any():
        median_val = train_clean[feature].median()
        train_clean[feature] = train_clean[feature].fillna(median_val)
        test_clean[feature] = test_clean[feature].fillna(median_val)

print(f"\nFinal dataset sizes:")
print(f"  Training: {len(train_clean)} games")
print(f"  Testing: {len(test_clean)} games")
print(f"  Features: {len(feature_columns)}")

X_train = train_clean[feature_columns]
y_train = train_clean[target]
X_test = test_clean[feature_columns]
y_test = test_clean[target]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 80)
print("MODELS WITH REGULARIZATION (PREVENT OVERFITTING)")
print("=" * 80)

# Define models with STRONGER regularization
models = {
    'Logistic Regression': LogisticRegression(
        C=0.1,  # Stronger regularization
        max_iter=1000, 
        random_state=42
    ),
    
    'Random Forest (Regularized)': RandomForestClassifier(
        n_estimators=100,
        max_depth=5,  # Shallower trees (was 10)
        min_samples_split=20,  # More samples needed to split
        min_samples_leaf=10,  # More samples needed in leaf
        max_features='sqrt',  # Fewer features per tree
        random_state=42
    ),
    
    'Gradient Boosting (Regularized)': GradientBoostingClassifier(
        n_estimators=50,  # Fewer trees
        max_depth=3,  # Shallower trees (was 5)
        learning_rate=0.05,  # Slower learning (was 0.1)
        subsample=0.8,  # Sample 80% of data
        random_state=42
    ),
    
    'XGBoost (Regularized)': xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,  # Sample 80% of features
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'=' * 80}")
    print(f"TRAINING: {name}")
    print('=' * 80)
    
    # Use scaled data for Logistic Regression, raw for trees
    if 'Logistic' in name:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # Train
    model.fit(X_train_use, y_train)
    
    # Predict
    train_pred = model.predict(X_train_use)
    test_pred = model.predict(X_test_use)
    test_proba = model.predict_proba(X_test_use)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Overfitting check
    overfit_gap = train_acc - test_acc
    
    print(f"\nAccuracy:")
    print(f"  Training: {train_acc*100:.2f}%")
    print(f"  Testing:  {test_acc*100:.2f}%")
    print(f"  Gap:      {overfit_gap*100:.2f}% {'✅' if overfit_gap < 0.10 else '⚠️  OVERFITTING'}")
    
    # Betting simulation
    total_wins = (test_pred == y_test).sum()
    total_losses = len(test_pred) - total_wins
    profit = (total_wins * 100) - (total_losses * 110)
    roi = (profit / (len(test_pred) * 110)) * 100
    
    print(f"\nBetting Performance:")
    print(f"  Total bets: {len(test_pred)}")
    print(f"  Wins: {total_wins} ({total_wins/len(test_pred)*100:.1f}%)")
    print(f"  Profit: ${profit:,.2f}")
    print(f"  ROI: {roi:.2f}%")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Features:")
        for idx, row in importances.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    results[name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'overfit_gap': overfit_gap,
        'roi': roi,
        'predictions': test_pred,
        'probabilities': test_proba
    }

# Comparison
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Acc': [r['train_acc']*100 for r in results.values()],
    'Test Acc': [r['test_acc']*100 for r in results.values()],
    'Gap': [r['overfit_gap']*100 for r in results.values()],
    'ROI %': [r['roi'] for r in results.values()]
})

print("\n" + comparison.to_string(index=False))

# Best model
best_model = max(results.keys(), key=lambda x: results[x]['test_acc'])
print(f"\n🏆 Best Test Accuracy: {best_model}")
print(f"   Test Accuracy: {results[best_model]['test_acc']*100:.2f}%")
print(f"   Overfit Gap: {results[best_model]['overfit_gap']*100:.2f}%")
print(f"   ROI: {results[best_model]['roi']:.2f}%")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n✓ Models with gap <10% are well-regularized")
print("✓ Higher test accuracy matters more than training accuracy")
print("✓ Compare ROI to your baseline: 10.94% (Logistic Regression)")

# Save best predictions
best_pred_df = test_clean.copy()
best_pred_df['predicted'] = results[best_model]['predictions']
best_pred_df['probability'] = results[best_model]['probabilities']
best_pred_df.to_csv('predictions_tuned_model.csv', index=False)

print(f"\n💾 Saved predictions: predictions_tuned_model.csv")

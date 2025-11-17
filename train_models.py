import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COLLEGE BASKETBALL SPREAD PREDICTION - MODEL TRAINING")
print("=" * 80)

# Load data
print("\nLoading data...")
train = pd.read_csv('training_data_enhanced.csv')
test = pd.read_csv('testing_data_enhanced.csv')

print(f"Training data: {len(train)} games")
print(f"Testing data: {len(test)} games")

# Select feature columns to use in the model
feature_columns = [
    'spread',
    # Core efficiency
    'adjoe_diff',
    'adjde_diff', 
    'adjt_diff',
    'barthag_diff',
    'rank_diff',
    # Strength of schedule
    'sos_diff',
    'ncsos_diff',
    # Shooting efficiency
    'efg_pct_diff',
    'efgd_pct_diff',
    # Turnovers
    'tor_diff',
    'tord_diff',
    # Rebounding
    'orb_diff',
    'drb_diff',
    # Free throws
    'ftr_diff',
    'ftrd_diff',
    # 2-point shooting
    'twop_pct_diff',
    'twopd_pct_diff',
    # 3-point shooting
    'threep_pct_diff',
    'threepd_pct_diff',
    # Raw stats
    'home_adjoe',
    'away_adjoe',
    'home_adjde',
    'away_adjde',
    'home_barthag',
    'away_barthag',
    # NEW MOMENTUM FEATURES
    'home_ats_last_5',
    'away_ats_last_5',
    'ats_diff_last_5',
    'home_ats_last_10',
    'away_ats_last_10',
    'ats_diff_last_10',
    'home_ats_streak',
    'away_ats_streak',
    'rest_advantage',
    'home_rest_days',
    'away_rest_days',
    'margin_trend',
    'recent_opp_strength',
    'early_season',
    'pace_mismatch',
    'combined_pace',
    'three_point_matchup',
    'turnover_matchup',
    'rebounding_matchup',
    'home_games_played',
    'away_games_played',
]

target = 'home_covered'

# Handle missing data intelligently
print("\nHandling missing data...")

# Check what features have the most missing values
print("\nChecking missing values in features:")
missing_counts = train[feature_columns].isna().sum()
high_missing = missing_counts[missing_counts > len(train) * 0.5]
if len(high_missing) > 0:
    print(f"Features with >50% missing values:")
    for feat, count in high_missing.items():
        print(f"  {feat}: {count}/{len(train)} ({count/len(train)*100:.1f}%)")

# Instead of dropping, FILL missing values with reasonable defaults
print("\nFilling missing values with sensible defaults...")

# For momentum features, fill with neutral values
momentum_features = [f for f in feature_columns if any(x in f for x in ['ats', 'streak', 'rest', 'trend', 'opp_strength', 'games_played'])]
for feat in momentum_features:
    if feat in train.columns:
        if 'ats' in feat and 'diff' not in feat and 'streak' not in feat:
            # ATS rates: fill with 0.5 (50% - neutral)
            train[feat] = train[feat].fillna(0.5)
            test[feat] = test[feat].fillna(0.5)
        elif 'rest' in feat:
            # Rest days: fill with median
            median_val = train[feat].median()
            train[feat] = train[feat].fillna(median_val)
            test[feat] = test[feat].fillna(median_val)
        elif 'games_played' in feat:
            # Games played: fill with 0 (start of season)
            train[feat] = train[feat].fillna(0)
            test[feat] = test[feat].fillna(0)
        else:
            # Everything else: fill with 0 (neutral differential)
            train[feat] = train[feat].fillna(0)
            test[feat] = test[feat].fillna(0)

# For matchup features, fill with 0 (no advantage)
matchup_features = [f for f in feature_columns if any(x in f for x in ['matchup', 'mismatch', 'pace'])]
for feat in matchup_features:
    if feat in train.columns:
        train[feat] = train[feat].fillna(0)
        test[feat] = test[feat].fillna(0)

# For all other features, fill with median
for feat in feature_columns:
    if feat in train.columns and train[feat].isna().any():
        median_val = train[feat].median()
        train[feat] = train[feat].fillna(median_val)
        test[feat] = test[feat].fillna(median_val)

# NOW drop any remaining rows with missing target
train_clean = train.dropna(subset=[target])
test_clean = test.dropna(subset=[target])

print(f"\nTraining data after cleaning: {len(train_clean)} games ({len(train_clean)/len(train)*100:.1f}%)")
print(f"Testing data after cleaning: {len(test_clean)} games ({len(test_clean)/len(test)*100:.1f}%)")

# Prepare data
X_train = train_clean[feature_columns]
y_train = train_clean[target]
X_test = test_clean[feature_columns]
y_test = test_clean[target]

print(f"\nFeatures used: {len(feature_columns)}")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Home covered in training: {y_train.mean()*100:.1f}%")
print(f"Home covered in test: {y_test.mean()*100:.1f}%")

# Scale features for models that benefit from it
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    print("\n" + "=" * 80)
    print(f"TRAINING: {name}")
    print("=" * 80)
    
    # Use scaled data for Logistic Regression, raw data for tree models
    if name == 'Logistic Regression':
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # Train
    print("Training model...")
    model.fit(X_train_use, y_train)
    
    # Predict
    print("Making predictions...")
    train_pred = model.predict(X_train_use)
    test_pred = model.predict(X_test_use)
    
    # Get prediction probabilities
    train_proba = model.predict_proba(X_train_use)[:, 1]
    test_proba = model.predict_proba(X_test_use)[:, 1]
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\nAccuracy:")
    print(f"  Training: {train_acc*100:.2f}%")
    print(f"  Testing:  {test_acc*100:.2f}%")
    
    # Calculate betting metrics
    total_bets = len(test_pred)
    total_wins = (test_pred == y_test).sum()
    total_losses = total_bets - total_wins
    
    # At -110 odds: win $100 for a win, lose $110 for a loss
    profit = (total_wins * 100) - (total_losses * 110)
    roi = (profit / (total_bets * 110)) * 100
    
    print(f"\nBetting Simulation (betting every game at -110 odds):")
    print(f"  Total bets: {total_bets}")
    print(f"  Wins: {total_wins} ({total_wins/total_bets*100:.1f}%)")
    print(f"  Losses: {total_losses} ({total_losses/total_bets*100:.1f}%)")
    print(f"  Profit: ${profit:,.2f}")
    print(f"  ROI: {roi:.2f}%")
    print(f"  Break-even accuracy: 52.38%")
    
    # Feature importance (for tree models)
    if hasattr(model, 'feature_importances_'):
        print(f"\nTop 10 Most Important Features:")
        importances = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        for idx, row in importances.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Store results
    results[name] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'total_bets': total_bets,
        'wins': total_wins,
        'losses': total_losses,
        'profit': profit,
        'roi': roi,
        'predictions': test_pred,
        'probabilities': test_proba
    }

# Summary comparison
print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)

comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Acc': [r['train_accuracy']*100 for r in results.values()],
    'Test Acc': [r['test_accuracy']*100 for r in results.values()],
    'Wins': [r['wins'] for r in results.values()],
    'Win %': [r['wins']/r['total_bets']*100 for r in results.values()],
    'Profit': [r['profit'] for r in results.values()],
    'ROI %': [r['roi'] for r in results.values()]
})

print("\n" + comparison.to_string(index=False))

# Find best model
best_model = max(results.keys(), key=lambda x: results[x]['roi'])
print(f"\n🏆 Best Model (by ROI): {best_model}")
print(f"   Test Accuracy: {results[best_model]['test_accuracy']*100:.2f}%")
print(f"   ROI: {results[best_model]['roi']:.2f}%")

# Save predictions from best model
print("\n" + "=" * 80)
print("SAVING PREDICTIONS")
print("=" * 80)

best_pred_df = test_clean.copy()
best_pred_df['predicted_home_covered'] = results[best_model]['predictions']
best_pred_df['prediction_probability'] = results[best_model]['probabilities']
best_pred_df['prediction_correct'] = (best_pred_df['predicted_home_covered'] == best_pred_df['home_covered'])

output_file = 'predictions_with_best_model.csv'
best_pred_df.to_csv(output_file, index=False)
print(f"Saved predictions to: {output_file}")

# Analysis by confidence level
print("\n" + "=" * 80)
print("PREDICTION CONFIDENCE ANALYSIS")
print("=" * 80)

proba = results[best_model]['probabilities']
pred = results[best_model]['predictions']
actual = y_test.values

# High confidence predictions (>70% or <30% probability)
high_conf_mask = (proba > 0.7) | (proba < 0.3)
high_conf_correct = (pred[high_conf_mask] == actual[high_conf_mask]).sum()
high_conf_total = high_conf_mask.sum()

if high_conf_total > 0:
    print(f"\nHigh Confidence Predictions (>70% or <30%):")
    print(f"  Total: {high_conf_total}")
    print(f"  Correct: {high_conf_correct} ({high_conf_correct/high_conf_total*100:.1f}%)")
    
    # Calculate ROI for high confidence bets only
    high_conf_wins = (pred[high_conf_mask] == actual[high_conf_mask]).sum()
    high_conf_losses = high_conf_total - high_conf_wins
    high_conf_profit = (high_conf_wins * 100) - (high_conf_losses * 110)
    high_conf_roi = (high_conf_profit / (high_conf_total * 110)) * 100
    print(f"  Profit (if betting only high confidence): ${high_conf_profit:,.2f}")
    print(f"  ROI: {high_conf_roi:.2f}%")

# Medium confidence
med_conf_mask = ((proba >= 0.55) & (proba <= 0.7)) | ((proba >= 0.3) & (proba <= 0.45))
med_conf_correct = (pred[med_conf_mask] == actual[med_conf_mask]).sum()
med_conf_total = med_conf_mask.sum()

if med_conf_total > 0:
    print(f"\nMedium Confidence Predictions (55-70% or 30-45%):")
    print(f"  Total: {med_conf_total}")
    print(f"  Correct: {med_conf_correct} ({med_conf_correct/med_conf_total*100:.1f}%)")

# Low confidence  
low_conf_mask = (proba > 0.45) & (proba < 0.55)
low_conf_correct = (pred[low_conf_mask] == actual[low_conf_mask]).sum()
low_conf_total = low_conf_mask.sum()

if low_conf_total > 0:
    print(f"\nLow Confidence Predictions (45-55% - near coin flip):")
    print(f"  Total: {low_conf_total}")
    print(f"  Correct: {low_conf_correct} ({low_conf_correct/low_conf_total*100:.1f}%)")
    print(f"  → Recommendation: Skip these bets")

print("\n" + "=" * 80)
print("MODELING COMPLETE!")
print("=" * 80)
print("\nKey Takeaways:")
print(f"1. Best model achieves {results[best_model]['test_accuracy']*100:.2f}% accuracy on unseen data")
print(f"2. With -110 odds, ROI is {results[best_model]['roi']:.2f}%")
print(f"3. Need 52.38% win rate to break even - model achieves {results[best_model]['wins']/results[best_model]['total_bets']*100:.1f}%")
print(f"4. High confidence bets may show better performance")
print(f"\nNext steps:")
print(f"- Review predictions_with_best_model.csv to see individual game predictions")
print(f"- Consider using only high-confidence predictions for actual betting")
print(f"- Can tune model hyperparameters to improve performance")
print(f"- Can add more features (rolling averages, recent form, etc.)")
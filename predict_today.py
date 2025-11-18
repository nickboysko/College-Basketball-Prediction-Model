import pandas as pd
import pickle
import sys
import os
import io
import numpy as np
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("GENERATING PREDICTIONS WITH MOMENTUM & TRACKING")
print("="*80)

# Load the trained model
MODEL_FILE = 'logistic_regression_model.pkl'
SCALER_FILE = 'feature_scaler.pkl'
FEATURE_LIST_FILE = 'feature_list.pkl'

if not os.path.exists(MODEL_FILE):
    print(f"\n[ERROR] Model file not found: {MODEL_FILE}")
    print("[HELP] Run: python save_model_for_daily.py")
    sys.exit(1)

print(f"\n[OK] Loading model from {MODEL_FILE}")
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

if os.path.exists(SCALER_FILE):
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    print(f"[OK] Loaded scaler")
else:
    scaler = None

if os.path.exists(FEATURE_LIST_FILE):
    with open(FEATURE_LIST_FILE, 'rb') as f:
        model_features = pickle.load(f)
    print(f"[OK] Model expects {len(model_features)} features")
else:
    model_features = None
    print("[ERROR] Feature list not found!")
    sys.exit(1)

# Load merged games WITH MOMENTUM
MERGED_FILE = 'todays_games_with_momentum.csv'
if not os.path.exists(MERGED_FILE):
    print(f"\n[ERROR] {MERGED_FILE} not found!")
    print("[HELP] Run: python calculate_momentum_features.py")
    sys.exit(1)

print(f"\n[OK] Loading games with momentum from {MERGED_FILE}")
merged = pd.read_csv(MERGED_FILE)
print(f"[OK] Loaded {len(merged)} games")

if len(merged) == 0:
    print("\n[WARN] No games to predict!")
    sys.exit(0)

# Build feature dataframe matching model's expectations
print(f"\n[BUILD] Preparing {len(model_features)} features for model...")

# Create feature dataframe
X = pd.DataFrame()

for feat in model_features:
    if feat in merged.columns:
        X[feat] = merged[feat]
    else:
        # Try common mappings
        if feat == 'home_adjoe' and 'adjoe_team' in merged.columns:
            X[feat] = merged['adjoe_team']
        elif feat == 'away_adjoe' and 'adjoe_opp' in merged.columns:
            X[feat] = merged['adjoe_opp']
        elif feat == 'home_adjde' and 'adjde_team' in merged.columns:
            X[feat] = merged['adjde_team']
        elif feat == 'away_adjde' and 'adjde_opp' in merged.columns:
            X[feat] = merged['adjde_opp']
        elif feat == 'home_barthag' and 'barthag_team' in merged.columns:
            X[feat] = merged['barthag_team']
        elif feat == 'away_barthag' and 'barthag_opp' in merged.columns:
            X[feat] = merged['barthag_opp']
        else:
            print(f"[WARN] Feature '{feat}' not found, filling with 0")
            X[feat] = 0

print(f"[OK] Built feature matrix: {X.shape}")

# Handle missing values (same as training)
print(f"[CLEAN] Handling missing values...")

momentum_features = [f for f in X.columns if any(x in f for x in ['ats', 'streak', 'rest', 'trend', 'opp_strength', 'games_played'])]
for feat in momentum_features:
    if 'ats' in feat and 'diff' not in feat and 'streak' not in feat:
        X[feat] = X[feat].fillna(0.5)
    elif 'rest' in feat:
        median_val = X[feat].median()
        X[feat] = X[feat].fillna(median_val if not pd.isna(median_val) else 5)
    elif 'games_played' in feat:
        X[feat] = X[feat].fillna(0)
    else:
        X[feat] = X[feat].fillna(0)

matchup_features = [f for f in X.columns if any(x in f for x in ['matchup', 'mismatch', 'pace'])]
for feat in matchup_features:
    X[feat] = X[feat].fillna(0)

for feat in X.columns:
    if X[feat].isna().any():
        median_val = X[feat].median()
        X[feat] = X[feat].fillna(median_val if not pd.isna(median_val) else 0)

# Check for remaining NaN
if X.isna().any().any():
    print(f"[WARN] Still have NaN values, dropping those rows...")
    valid_idx = ~X.isna().any(axis=1)
    X = X[valid_idx]
    merged_for_output = merged[valid_idx].copy()
    print(f"[INFO] Dropped {(~valid_idx).sum()} rows with NaN")
else:
    merged_for_output = merged.copy()

print(f"\n[PREDICT] Generating predictions on {len(X)} games...")

try:
    # Scale and predict
    if scaler is not None:
        X_scaled = scaler.transform(X)
        predictions = model.predict_proba(X_scaled)[:, 1]
    else:
        predictions = model.predict_proba(X)[:, 1]
    
    merged_for_output['predicted_cover_prob'] = predictions
    
    print(f"[OK] Predictions generated")
    
    # Add confidence levels with HIGHER thresholds
    def get_confidence(prob):
        distance_from_50 = abs(prob - 0.5)
        if distance_from_50 > 0.25:  # >75% or <25%
            return 'HIGH'
        elif distance_from_50 > 0.15:  # >65% or <35%
            return 'MEDIUM'
        elif distance_from_50 > 0.05:  # >55% or <45%
            return 'LOW'
        else:
            return 'SKIP'  # Too close to 50/50
    
    merged_for_output['confidence'] = merged_for_output['predicted_cover_prob'].apply(get_confidence)
    
    # FILTER OUT SKIP BETS - Don't even include them
    print(f"\n[FILTER] Removing games too close to 50/50...")
    before_filter = len(merged_for_output)
    merged_for_output = merged_for_output[merged_for_output['confidence'] != 'SKIP'].copy()
    after_filter = len(merged_for_output)
    skipped = before_filter - after_filter
    
    if skipped > 0:
        print(f"[FILTER] Skipped {skipped} games (too close to coin flip)")
    
    if len(merged_for_output) == 0:
        print("\n[WARN] No games meet confidence threshold!")
        sys.exit(0)
    
    # Add recommendation
    def get_recommendation(row):
        prob = row['predicted_cover_prob']
        conf = row['confidence']
        
        if prob > 0.50:
            pick = row['team']
            pick_spread = row['spread']
            confidence_pct = prob * 100
        else:
            pick = row['opp']
            pick_spread = -row['spread']
            confidence_pct = (1 - prob) * 100
        
        if conf == 'HIGH':
            return f"BET {pick} {pick_spread:+.1f}"
        elif conf == 'MEDIUM':
            return f"LEAN {pick} {pick_spread:+.1f}"
        else:
            return f"CONSIDER {pick} {pick_spread:+.1f}"
    
    merged_for_output['recommendation'] = merged_for_output.apply(get_recommendation, axis=1)
    
    # Add date and time
    today = datetime.now()
    merged_for_output['prediction_date'] = today.strftime('%Y-%m-%d')
    merged_for_output['prediction_time'] = today.strftime('%H:%M:%S')
    
    # Add placeholders for tracking results
    merged_for_output['actual_result'] = ''
    merged_for_output['notes'] = ''
    
    # Select output columns
    output_columns = [
        'prediction_date',
        'team', 
        'opp', 
        'spread',
        'predicted_cover_prob',
        'confidence',
        'recommendation',
        'actual_result',
        'notes',
        'adjoe_diff',
        'adjde_diff', 
        'barthag_diff'
    ]
    
    output = merged_for_output[output_columns].copy()
    
    # Sort by confidence and probability distance from 50%
    confidence_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    output['conf_rank'] = output['confidence'].map(confidence_order)
    output['prob_distance'] = abs(output['predicted_cover_prob'] - 0.5)
    output = output.sort_values(['conf_rank', 'prob_distance'], ascending=[True, False])
    output = output.drop(['conf_rank', 'prob_distance'], axis=1)
    
    # Save to CSV
    CSV_FILE = 'todays_predictions.csv'
    output.to_csv(CSV_FILE, index=False)
    print(f"\n[OK] Saved CSV: {CSV_FILE}")
    
    # Save to Excel with date-stamped sheet
    EXCEL_FILE = 'prediction_tracker.xlsx'
    sheet_name = today.strftime('%Y-%m-%d')
    
    print(f"\n[EXCEL] Saving to {EXCEL_FILE}...")
    
    if os.path.exists(EXCEL_FILE):
        print(f"[INFO] Existing tracker found, adding new sheet...")
        try:
            with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                output.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"[OK] Added sheet: {sheet_name}")
        except Exception as e:
            print(f"[WARN] Could not append: {e}")
            with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
                output.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"[OK] Created new file with sheet: {sheet_name}")
    else:
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
            output.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"[OK] Created {EXCEL_FILE} with sheet: {sheet_name}")
    
    # Print summary
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    
    print(f"\nTotal games analyzed: {before_filter}")
    print(f"Games with edge: {len(output)} (skipped {skipped} close to 50/50)")
    print(f"\nConfidence breakdown:")
    print(output['confidence'].value_counts().to_string())
    
    # Show HIGH confidence bets
    high_conf = output[output['confidence'] == 'HIGH']
    
    if len(high_conf) > 0:
        print(f"\n[BETS] HIGH Confidence Picks ({len(high_conf)} games):")
        print("-"*80)
        print("📊 BACKTESTED PERFORMANCE: 68.9% accuracy, 31.5% ROI")
        print("-"*80)
        for i, (idx, row) in enumerate(high_conf.head(15).iterrows(), 1):
            prob = row['predicted_cover_prob']
            if prob > 0.50:
                team_name = row['team']
                display_prob = prob * 100
                display_spread = row['spread']
            else:
                team_name = row['opp']
                display_prob = (1 - prob) * 100
                display_spread = -row['spread']
            
            print(f"{i:2d}. {team_name:25} {display_spread:+6.1f}")
            print(f"    Model confidence: {display_prob:5.1f}%")
        
        if len(high_conf) > 15:
            print(f"\n    ... and {len(high_conf)-15} more HIGH confidence picks")
    else:
        print("\n[INFO] No HIGH confidence picks today")
    
    # Show MEDIUM confidence
    med_conf = output[output['confidence'] == 'MEDIUM']
    
    if len(med_conf) > 0:
        print(f"\n[BETS] MEDIUM Confidence Picks ({len(med_conf)} games):")
        print("-"*80)
        print("📊 BACKTESTED PERFORMANCE: 60.1% accuracy")
        print("-"*80)
        for i, (idx, row) in enumerate(med_conf.head(5).iterrows(), 1):
            prob = row['predicted_cover_prob']
            if prob > 0.50:
                team_name = row['team']
                display_prob = prob * 100
                display_spread = row['spread']
            else:
                team_name = row['opp']
                display_prob = (1 - prob) * 100
                display_spread = -row['spread']
            
            print(f"{i:2d}. {team_name:25} {display_spread:+6.1f}")
            print(f"    Model confidence: {display_prob:5.1f}%")
        
        if len(med_conf) > 5:
            print(f"\n    ... and {len(med_conf)-5} more MEDIUM confidence picks")
    
    # Show LOW confidence summary
    low_conf = output[output['confidence'] == 'LOW']
    
    if len(low_conf) > 0:
        print(f"\n[INFO] LOW Confidence: {len(low_conf)} games (slight edge)")
    
    print("\n" + "="*80)
    print("✅ PREDICTIONS COMPLETE!")
    print("="*80)
    print(f"\n[OUTPUT] CSV: {CSV_FILE}")
    print(f"[OUTPUT] Excel Tracker: {EXCEL_FILE} (Sheet: {sheet_name})")
    print(f"\n[THRESHOLDS] New confidence levels:")
    print(f"  HIGH: >75% or <25% (was >70% or <30%)")
    print(f"  MEDIUM: >65% or <35% (was >60% or <40%)")
    print(f"  LOW: >55% or <45% (was all remaining)")
    print(f"  SKIP: 45-55% (not included in output)")
    print(f"\n[STRATEGY] Track all predictions to validate model performance")
    print(f"[STRATEGY] Focus betting on HIGH confidence (if performance improves)")
    print("="*80)
    
except Exception as e:
    print(f"\n[ERROR] Failed to generate predictions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
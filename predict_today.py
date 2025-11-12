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
print("GENERATING PREDICTIONS WITH TRACKING")
print("="*80)

# Load the trained model
MODEL_FILE = 'logistic_regression_model.pkl'
SCALER_FILE = 'feature_scaler.pkl'
FEATURE_LIST_FILE = 'feature_list.pkl'

if not os.path.exists(MODEL_FILE):
    print(f"\n[ERROR] Model file not found: {MODEL_FILE}")
    print("[HELP] Run: python save_model_fixed.py")
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

# Load merged games
MERGED_FILE = 'todays_games_merged.csv'
if not os.path.exists(MERGED_FILE):
    print(f"\n[ERROR] Merged games file not found: {MERGED_FILE}")
    sys.exit(1)

print(f"\n[OK] Loading merged games from {MERGED_FILE}")
merged = pd.read_csv(MERGED_FILE)
print(f"[OK] Loaded {len(merged)} games")

if len(merged) == 0:
    print("\n[WARN] No games to predict!")
    sys.exit(0)

# Calculate missing differential features if needed
print(f"\n[CALC] Calculating additional features...")

# Rank diff (if columns exist)
if 'rank_team' in merged.columns and 'rank_opp' in merged.columns:
    merged['rank_diff'] = merged['rank_team'] - merged['rank_opp']
elif 'rank.1_team' in merged.columns and 'rank.1_opp' in merged.columns:
    merged['rank_diff'] = merged['rank.1_team'] - merged['rank.1_opp']
else:
    merged['rank_diff'] = 0

# SOS diff
if 'sos_team' in merged.columns and 'sos_opp' in merged.columns:
    merged['sos_diff'] = merged['sos_team'] - merged['sos_opp']
else:
    merged['sos_diff'] = 0

# Tempo diff
if 'adjt_team' in merged.columns and 'adjt_opp' in merged.columns:
    merged['adjt_diff'] = merged['adjt_team'] - merged['adjt_opp']
else:
    merged['adjt_diff'] = 0

print(f"[OK] Features calculated")

# Build feature dataframe matching model's expectations
print(f"\n[BUILD] Preparing features for model...")

feature_data = []
feature_names = []

if model_features:
    for feat in model_features:
        if feat == 'spread' and 'spread' in merged.columns:
            feature_data.append(merged['spread'])
            feature_names.append('spread')
        elif feat == 'barthag_diff' and 'barthag_diff' in merged.columns:
            feature_data.append(merged['barthag_diff'])
            feature_names.append('barthag_diff')
        elif feat == 'adjoe_diff' and 'adjoe_diff' in merged.columns:
            feature_data.append(merged['adjoe_diff'])
            feature_names.append('adjoe_diff')
        elif feat == 'adjde_diff' and 'adjde_diff' in merged.columns:
            feature_data.append(merged['adjde_diff'])
            feature_names.append('adjde_diff')
        elif feat == 'rank_diff':
            feature_data.append(merged['rank_diff'])
            feature_names.append('rank_diff')
        elif feat == 'sos_diff':
            feature_data.append(merged['sos_diff'])
            feature_names.append('sos_diff')
        elif feat == 'adjt_diff':
            feature_data.append(merged['adjt_diff'])
            feature_names.append('adjt_diff')
        elif feat == 'home_adjoe' and 'adjoe_team' in merged.columns:
            feature_data.append(merged['adjoe_team'])
            feature_names.append('home_adjoe')
        elif feat == 'away_adjoe' and 'adjoe_opp' in merged.columns:
            feature_data.append(merged['adjoe_opp'])
            feature_names.append('away_adjoe')
        elif feat == 'home_adjde' and 'adjde_team' in merged.columns:
            feature_data.append(merged['adjde_team'])
            feature_names.append('home_adjde')
        elif feat == 'away_adjde' and 'adjde_opp' in merged.columns:
            feature_data.append(merged['adjde_opp'])
            feature_names.append('away_adjde')
        elif feat == 'home_barthag' and 'barthag_team' in merged.columns:
            feature_data.append(merged['barthag_team'])
            feature_names.append('home_barthag')
        elif feat == 'away_barthag' and 'barthag_opp' in merged.columns:
            feature_data.append(merged['barthag_opp'])
            feature_names.append('away_barthag')
        else:
            print(f"[WARN] Cannot map feature: {feat}")
            feature_data.append(pd.Series([0] * len(merged)))
            feature_names.append(feat)

X = pd.DataFrame(dict(zip(feature_names, feature_data)))
print(f"[OK] Built feature matrix: {X.shape}")

# Remove NaN
if X.isna().any().any():
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
    
    # Add confidence levels
    def get_confidence(prob):
        distance_from_50 = abs(prob - 0.5)
        if distance_from_50 > 0.20:
            return 'HIGH'
        elif distance_from_50 > 0.10:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    merged_for_output['confidence'] = merged_for_output['predicted_cover_prob'].apply(get_confidence)
    
    # Add recommendation
    def get_recommendation(row):
        prob = row['predicted_cover_prob']
        conf = row['confidence']
        
        if prob > 0.50:
            favorite = row['team']
            confidence_pct = prob * 100
        else:
            favorite = row['opp']
            confidence_pct = (1 - prob) * 100
        
        if conf == 'HIGH':
            return f"BET {favorite} to COVER"
        elif conf == 'MEDIUM':
            return f"LEAN {favorite} (Medium confidence)"
        else:
            return "SKIP - Low confidence"
    
    merged_for_output['recommendation'] = merged_for_output.apply(get_recommendation, axis=1)
    
    # Add date and time
    today = datetime.now()
    merged_for_output['prediction_date'] = today.strftime('%Y-%m-%d')
    merged_for_output['prediction_time'] = today.strftime('%H:%M:%S')
    
    # Add placeholders for tracking results (to be filled in later)
    merged_for_output['actual_result'] = ''  # Will be: 'WIN', 'LOSS', or blank
    merged_for_output['notes'] = ''
    
    # Select output columns in logical order
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
    
    # Save to CSV (for backward compatibility)
    CSV_FILE = 'todays_predictions.csv'
    output.to_csv(CSV_FILE, index=False)
    print(f"\n[OK] Saved CSV: {CSV_FILE}")
    
    # Save to Excel with date-stamped sheet
    EXCEL_FILE = 'prediction_tracker.xlsx'
    sheet_name = today.strftime('%Y-%m-%d')
    
    print(f"\n[EXCEL] Saving to {EXCEL_FILE}...")
    
    if os.path.exists(EXCEL_FILE):
        # Load existing workbook
        print(f"[INFO] Existing tracker found, adding new sheet...")
        
        try:
            # Try to load with openpyxl
            with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                output.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"[OK] Added sheet: {sheet_name}")
        except Exception as e:
            print(f"[WARN] Could not append to existing file: {e}")
            print(f"[INFO] Creating new file...")
            with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
                output.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"[OK] Created new file with sheet: {sheet_name}")
    else:
        # Create new workbook
        print(f"[INFO] Creating new tracker file...")
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
            output.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"[OK] Created {EXCEL_FILE} with sheet: {sheet_name}")
    
    # Print summary
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    
    print(f"\nTotal games: {len(output)}")
    print(f"\nConfidence breakdown:")
    print(output['confidence'].value_counts().to_string())
    
    # Show top HIGH confidence bets
    high_conf = output[output['confidence'] == 'HIGH']
    
    if len(high_conf) > 0:
        print(f"\n[BETS] HIGH Confidence Recommendations ({len(high_conf)} games):")
        print("-"*80)
        for i, (idx, row) in enumerate(high_conf.head(10).iterrows(), 1):
            prob = row['predicted_cover_prob']
            if prob > 0.50:
                team_name = row['team']
                display_prob = prob * 100
            else:
                team_name = row['opp']
                display_prob = (1 - prob) * 100
            
            print(f"{i:2d}. {team_name:25} (Spread: {row['spread']:+6.1f})")
            print(f"    Confidence: {display_prob:5.1f}% | {row['recommendation']}")
        
        if len(high_conf) > 10:
            print(f"\n    ... and {len(high_conf)-10} more HIGH confidence bets")
    else:
        print("\n[INFO] No HIGH confidence bets today")
    
    # Show MEDIUM confidence bets
    med_conf = output[output['confidence'] == 'MEDIUM']
    
    if len(med_conf) > 0:
        print(f"\n[BETS] MEDIUM Confidence Bets ({len(med_conf)} games):")
        print("-"*80)
        for i, (idx, row) in enumerate(med_conf.head(5).iterrows(), 1):
            prob = row['predicted_cover_prob']
            if prob > 0.50:
                team_name = row['team']
                display_prob = prob * 100
            else:
                team_name = row['opp']
                display_prob = (1 - prob) * 100
            
            print(f"{i:2d}. {team_name:25} (Spread: {row['spread']:+6.1f})")
            print(f"    Confidence: {display_prob:5.1f}% | {row['recommendation']}")
        
        if len(med_conf) > 5:
            print(f"\n    ... and {len(med_conf)-5} more MEDIUM confidence bets")
    
    print("\n" + "="*80)
    print("PREDICTIONS COMPLETE")
    print("="*80)
    print(f"\n[OUTPUT] CSV: {CSV_FILE}")
    print(f"[OUTPUT] Excel Tracker: {EXCEL_FILE} (Sheet: {sheet_name})")
    print("\n[TRACKING] To track results:")
    print(f"  1. Open {EXCEL_FILE}")
    print(f"  2. After games complete, fill in 'actual_result' column:")
    print(f"     - 'WIN' if your bet covered")
    print(f"     - 'LOSS' if your bet didn't cover")
    print(f"  3. Use Excel formulas to calculate accuracy:")
    print(f"     =COUNTIF(H:H,\"WIN\")/COUNTA(H:H)")
    print("\n[INFO] Focus on HIGH confidence bets for best ROI (71% win rate)")
    print("[INFO] MEDIUM confidence bets are OK but less reliable (60-70% range)")
    print("[INFO] SKIP low confidence bets (too close to coin flip)")
    print("="*80)
    
except Exception as e:
    print(f"\n[ERROR] Failed to generate predictions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
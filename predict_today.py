import pandas as pd
import pickle
import sys
import os
import io
import numpy as np

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("GENERATING PREDICTIONS")
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
    merged['rank_diff'] = 0  # Default if not available

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

# Map model features to available data
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
        if prob > 0.70 or prob < 0.30:
            return 'HIGH'
        elif prob > 0.60 or prob < 0.40:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    merged_for_output['confidence'] = merged_for_output['predicted_cover_prob'].apply(get_confidence)
    
    # Add recommendation
    def get_recommendation(row):
        prob = row['predicted_cover_prob']
        conf = row['confidence']
        
        if conf == 'HIGH':
            if prob > 0.70:
                return f"BET {row['team']} to COVER"
            else:
                return f"BET {row['opp']} to COVER"
        elif conf == 'MEDIUM':
            return "CAUTION - Medium confidence"
        else:
            return "SKIP - Low confidence"
    
    merged_for_output['recommendation'] = merged_for_output.apply(get_recommendation, axis=1)
    
    # Select output columns
    output_columns = ['team', 'opp', 'spread', 'predicted_cover_prob', 'confidence', 'recommendation']
    
    # Add differential features if they exist
    for col in ['adjoe_diff', 'adjde_diff', 'barthag_diff']:
        if col in merged_for_output.columns:
            output_columns.append(col)
    
    output = merged_for_output[output_columns].copy()
    
    # Sort by confidence and probability
    confidence_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    output['conf_rank'] = output['confidence'].map(confidence_order)
    output = output.sort_values(['conf_rank', 'predicted_cover_prob'], ascending=[True, False])
    output = output.drop('conf_rank', axis=1)
    
    # Save predictions
    OUTPUT_FILE = 'todays_predictions.csv'
    output.to_csv(OUTPUT_FILE, index=False)
    
    print(f"[OK] Saved predictions to {OUTPUT_FILE}")
    
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
            prob_pct = row['predicted_cover_prob'] * 100
            print(f"{i:2d}. {row['team']:25} vs {row['opp']:25}")
            print(f"    Spread: {row['spread']:+6.1f} | Prob: {prob_pct:5.1f}% | {row['recommendation']}")
        
        if len(high_conf) > 10:
            print(f"\n    ... and {len(high_conf)-10} more HIGH confidence bets")
    else:
        print("\n[INFO] No HIGH confidence bets today")
    
    print("\n" + "="*80)
    print("PREDICTIONS COMPLETE")
    print("="*80)
    print(f"\n[OUTPUT] Full predictions: {OUTPUT_FILE}")
    print("[INFO] Focus on HIGH confidence bets for best ROI")
    print("="*80)
    
except Exception as e:
    print(f"\n[ERROR] Failed to generate predictions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
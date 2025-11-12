import pandas as pd
import os
import sys

print("="*80)
print("PREDICTION TRACKER - ACCURACY CALCULATOR")
print("="*80)

EXCEL_FILE = 'prediction_tracker.xlsx'

if not os.path.exists(EXCEL_FILE):
    print(f"\n[ERROR] {EXCEL_FILE} not found!")
    print("[INFO] Run predictions first to create the tracker file.")
    sys.exit(1)

print(f"\n[LOAD] Reading {EXCEL_FILE}...")

# Load all sheets
excel_file = pd.ExcelFile(EXCEL_FILE)
sheet_names = excel_file.sheet_names

print(f"[OK] Found {len(sheet_names)} prediction days")

all_predictions = []
stats_by_date = []
stats_by_confidence = {'HIGH': [], 'MEDIUM': [], 'LOW': []}

for sheet in sheet_names:
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet)
    
    # Count predictions with results
    total = len(df)
    tracked = df['actual_result'].notna() & (df['actual_result'] != '')
    num_tracked = tracked.sum()
    
    if num_tracked > 0:
        wins = (df['actual_result'] == 'WIN').sum()
        losses = (df['actual_result'] == 'LOSS').sum()
        accuracy = wins / num_tracked * 100 if num_tracked > 0 else 0
        
        stats_by_date.append({
            'date': sheet,
            'total_games': total,
            'tracked': num_tracked,
            'wins': wins,
            'losses': losses,
            'accuracy': accuracy
        })
        
        # Break down by confidence level
        for conf_level in ['HIGH', 'MEDIUM', 'LOW']:
            conf_mask = (df['confidence'] == conf_level) & tracked
            conf_tracked = conf_mask.sum()
            
            if conf_tracked > 0:
                conf_wins = ((df['confidence'] == conf_level) & (df['actual_result'] == 'WIN')).sum()
                conf_acc = conf_wins / conf_tracked * 100
                
                stats_by_confidence[conf_level].append({
                    'date': sheet,
                    'tracked': conf_tracked,
                    'wins': conf_wins,
                    'accuracy': conf_acc
                })
    
    all_predictions.append(df)

# Combine all predictions
all_df = pd.concat(all_predictions, ignore_index=True)

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)

total_predictions = len(all_df)
total_tracked = (all_df['actual_result'].notna() & (all_df['actual_result'] != '')).sum()
total_wins = (all_df['actual_result'] == 'WIN').sum()
total_losses = (all_df['actual_result'] == 'LOSS').sum()

print(f"\nTotal predictions made: {total_predictions}")
print(f"Results tracked: {total_tracked}")
print(f"Not yet tracked: {total_predictions - total_tracked}")

if total_tracked > 0:
    overall_accuracy = total_wins / total_tracked * 100
    print(f"\n{'='*40}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"{'='*40}")
    print(f"Wins: {total_wins}")
    print(f"Losses: {total_losses}")
    print(f"\nBreak-even threshold: 52.38%")
    if overall_accuracy > 52.38:
        edge = overall_accuracy - 52.38
        print(f"✅ PROFITABLE! Edge: +{edge:.2f}%")
    else:
        print(f"⚠️  Below break-even")

print("\n" + "="*80)
print("ACCURACY BY CONFIDENCE LEVEL")
print("="*80)

for conf_level in ['HIGH', 'MEDIUM', 'LOW']:
    conf_mask = (all_df['confidence'] == conf_level) & (all_df['actual_result'].notna()) & (all_df['actual_result'] != '')
    conf_total = conf_mask.sum()
    
    if conf_total > 0:
        conf_wins = ((all_df['confidence'] == conf_level) & (all_df['actual_result'] == 'WIN')).sum()
        conf_accuracy = conf_wins / conf_total * 100
        
        print(f"\n{conf_level} Confidence:")
        print(f"  Tracked: {conf_total}")
        print(f"  Wins: {conf_wins}")
        print(f"  Accuracy: {conf_accuracy:.2f}%")
        
        # Expected accuracy based on historical data
        expected = {'HIGH': 70.6, 'MEDIUM': 65.0, 'LOW': 52.0}
        if conf_accuracy >= expected[conf_level]:
            print(f"  ✅ Meets expectations ({expected[conf_level]}%)")
        else:
            print(f"  ⚠️  Below expectations ({expected[conf_level]}%)")

print("\n" + "="*80)
print("DAILY BREAKDOWN")
print("="*80)

if stats_by_date:
    daily_df = pd.DataFrame(stats_by_date)
    print("\n" + daily_df.to_string(index=False))
else:
    print("\n[INFO] No tracked results yet. Fill in 'actual_result' column in Excel.")

print("\n" + "="*80)
print("TRACKING INSTRUCTIONS")
print("="*80)
print(f"\n1. Open {EXCEL_FILE}")
print(f"2. Navigate to today's sheet")
print(f"3. After games finish, fill in the 'actual_result' column:")
print(f"   - Type 'WIN' if the recommended bet covered")
print(f"   - Type 'LOSS' if the recommended bet did not cover")
print(f"   - Leave blank if you didn't bet on it")
print(f"4. Run this script again to see updated accuracy")
print(f"\nCommand: python calculate_accuracy.py")
print("="*80)

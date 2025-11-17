import pandas as pd
import os
from datetime import datetime

print("=" * 80)
print("PREDICTION ACCURACY TRACKER")
print("=" * 80)

# Configuration
TRACKER_FILE = 'prediction_tracker.xlsx'
EXCLUDE_SHEETS = ['2025-11-09', '2025-11-10', '2025-11-11']  # Old model without momentum

if not os.path.exists(TRACKER_FILE):
    print(f"\n[ERROR] {TRACKER_FILE} not found!")
    exit(1)

# Load all sheets
xl = pd.ExcelFile(TRACKER_FILE)

print(f"\n[INFO] Found {len(xl.sheet_names)} sheets in tracker")
print(f"\n[EXCLUDE] Ignoring sheets (old model): {EXCLUDE_SHEETS}")

# Collect results from valid sheets
all_results = []
sheet_summaries = []

for sheet in xl.sheet_names:
    if sheet in EXCLUDE_SHEETS:
        print(f"\n[SKIP] {sheet} - excluded")
        continue
    
    print(f"\n[PROCESS] {sheet}")
    
    df = pd.read_excel(TRACKER_FILE, sheet_name=sheet)
    
    # Check if we have results
    if 'actual_result' not in df.columns:
        print(f"  [WARN] No 'actual_result' column found")
        continue
    
    # Filter to rows with results
    has_result = df['actual_result'].notna() & (df['actual_result'] != '')
    results = df[has_result].copy()
    
    if len(results) == 0:
        print(f"  [INFO] No results recorded yet")
        continue
    
    # Calculate stats for this sheet
    total = len(results)
    wins = (results['actual_result'] == 'WIN').sum()
    losses = (results['actual_result'] == 'LOSS').sum()
    accuracy = wins / total * 100 if total > 0 else 0
    
    # Calculate by confidence level
    if 'confidence' in results.columns:
        high_conf = results[results['confidence'] == 'HIGH']
        med_conf = results[results['confidence'] == 'MEDIUM']
        low_conf = results[results['confidence'] == 'LOW']
        
        high_acc = (high_conf['actual_result'] == 'WIN').sum() / len(high_conf) * 100 if len(high_conf) > 0 else 0
        med_acc = (med_conf['actual_result'] == 'WIN').sum() / len(med_conf) * 100 if len(med_conf) > 0 else 0
        low_acc = (low_conf['actual_result'] == 'WIN').sum() / len(low_conf) * 100 if len(low_conf) > 0 else 0
    else:
        high_acc = med_acc = low_acc = 0
        high_conf = med_conf = low_conf = pd.DataFrame()
    
    print(f"  Total: {wins}-{losses} ({accuracy:.1f}%)")
    if len(high_conf) > 0:
        print(f"  HIGH:  {(high_conf['actual_result'] == 'WIN').sum()}-{(high_conf['actual_result'] == 'LOSS').sum()} ({high_acc:.1f}%)")
    if len(med_conf) > 0:
        print(f"  MED:   {(med_conf['actual_result'] == 'WIN').sum()}-{(med_conf['actual_result'] == 'LOSS').sum()} ({med_acc:.1f}%)")
    if len(low_conf) > 0:
        print(f"  LOW:   {(low_conf['actual_result'] == 'WIN').sum()}-{(low_conf['actual_result'] == 'LOSS').sum()} ({low_acc:.1f}%)")
    
    # Store for overall calculation
    all_results.append(results)
    
    sheet_summaries.append({
        'date': sheet,
        'total': total,
        'wins': wins,
        'losses': losses,
        'accuracy': accuracy,
        'high_total': len(high_conf),
        'high_wins': (high_conf['actual_result'] == 'WIN').sum() if len(high_conf) > 0 else 0,
        'high_acc': high_acc,
        'med_total': len(med_conf),
        'med_wins': (med_conf['actual_result'] == 'WIN').sum() if len(med_conf) > 0 else 0,
        'med_acc': med_acc,
    })

if len(all_results) == 0:
    print("\n[INFO] No results to analyze yet")
    exit(0)

# Combine all results
combined = pd.concat(all_results, ignore_index=True)

print("\n" + "=" * 80)
print("OVERALL PERFORMANCE (Momentum Model Only)")
print("=" * 80)

total_bets = len(combined)
total_wins = (combined['actual_result'] == 'WIN').sum()
total_losses = (combined['actual_result'] == 'LOSS').sum()
overall_acc = total_wins / total_bets * 100

print(f"\nTotal Predictions: {total_bets}")
print(f"Record: {total_wins}-{total_losses}")
print(f"Accuracy: {overall_acc:.2f}%")
print(f"Break-even: 52.38%")
print(f"Model Backtest: 58.31%")

# ROI calculation (assuming -110 odds)
profit = (total_wins * 100) - (total_losses * 110)
roi = (profit / (total_bets * 110)) * 100

print(f"\nProfit/Loss: ${profit:,.2f}")
print(f"ROI: {roi:.2f}%")

# By confidence level
print("\n" + "=" * 80)
print("PERFORMANCE BY CONFIDENCE LEVEL")
print("=" * 80)

if 'confidence' in combined.columns:
    for conf_level in ['HIGH', 'MEDIUM', 'LOW']:
        conf_data = combined[combined['confidence'] == conf_level]
        
        if len(conf_data) == 0:
            continue
        
        conf_total = len(conf_data)
        conf_wins = (conf_data['actual_result'] == 'WIN').sum()
        conf_losses = (conf_data['actual_result'] == 'LOSS').sum()
        conf_acc = conf_wins / conf_total * 100
        
        conf_profit = (conf_wins * 100) - (conf_losses * 110)
        conf_roi = (conf_profit / (conf_total * 110)) * 100
        
        # Backtest comparison
        if conf_level == 'HIGH':
            backtest_acc = 68.9
            backtest_roi = 31.5
        elif conf_level == 'MEDIUM':
            backtest_acc = 60.1
            backtest_roi = None
        else:
            backtest_acc = 53.3
            backtest_roi = None
        
        print(f"\n{conf_level} Confidence:")
        print(f"  Record: {conf_wins}-{conf_losses} ({conf_acc:.1f}%)")
        print(f"  Backtest: {backtest_acc:.1f}%")
        print(f"  Difference: {conf_acc - backtest_acc:+.1f}%")
        print(f"  ROI: {conf_roi:.2f}%")
        if backtest_roi:
            print(f"  Backtest ROI: {backtest_roi:.1f}%")

# Daily breakdown
print("\n" + "=" * 80)
print("DAILY BREAKDOWN")
print("=" * 80)

summary_df = pd.DataFrame(sheet_summaries)
print("\n" + summary_df[['date', 'wins', 'losses', 'accuracy', 'high_wins', 'high_total', 'high_acc']].to_string(index=False))

print("\n" + "=" * 80)
print("TRACKING COMPLETE")
print("=" * 80)
print(f"\nExcluded sheets: {len(EXCLUDE_SHEETS)}")
print(f"Analyzed sheets: {len(sheet_summaries)}")
print(f"\nTo exclude different sheets, edit EXCLUDE_SHEETS list at top of script")
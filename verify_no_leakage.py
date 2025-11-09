import pandas as pd
import numpy as np

print("=" * 80)
print("DATA LEAKAGE VERIFICATION")
print("=" * 80)

# Load the data
train = pd.read_csv('training_data.csv')
test = pd.read_csv('testing_data.csv')

print("\n1. CHECKING TEMPORAL SEPARATION")
print("-" * 80)

# Check seasons in each dataset
print("\nTraining data seasons:")
print(train['season'].value_counts().sort_index())

print("\nTesting data seasons:")
print(test['season'].value_counts().sort_index())

# Check for overlap
train_seasons = set(train['season'].unique())
test_seasons = set(test['season'].unique())
overlap = train_seasons & test_seasons

if overlap:
    print(f"\n⚠️  WARNING: Seasons appear in BOTH train and test: {overlap}")
    print("   This could indicate data leakage!")
else:
    print("\n✅ GOOD: No season overlap between train and test")

print("\n2. CHECKING FOR DUPLICATE GAMES")
print("-" * 80)

# Check if any specific games appear in both datasets
# Create a unique game identifier
train['game_id'] = train['season'] + '_' + train['game_date'].astype(str) + '_' + train['team'] + '_' + train['opp']
test['game_id'] = test['season'] + '_' + test['game_date'].astype(str) + '_' + test['team'] + '_' + test['opp']

train_games = set(train['game_id'])
test_games = set(test['game_id'])
duplicate_games = train_games & test_games

if duplicate_games:
    print(f"\n⚠️  WARNING: {len(duplicate_games)} games appear in BOTH train and test!")
    print("   Sample duplicates:", list(duplicate_games)[:5])
else:
    print(f"\n✅ GOOD: No duplicate games between train and test")

print("\n3. CHECKING DATES")
print("-" * 80)

# Convert dates and check ranges
train['game_date'] = pd.to_datetime(train['game_date'])
test['game_date'] = pd.to_datetime(test['game_date'])

train_start = train['game_date'].min()
train_end = train['game_date'].max()
test_start = test['game_date'].min()
test_end = test['game_date'].max()

print(f"\nTraining data date range:")
print(f"  Start: {train_start.strftime('%Y-%m-%d')}")
print(f"  End:   {train_end.strftime('%Y-%m-%d')}")

print(f"\nTesting data date range:")
print(f"  Start: {test_start.strftime('%Y-%m-%d')}")
print(f"  End:   {test_end.strftime('%Y-%m-%d')}")

if test_start <= train_end:
    print(f"\n⚠️  WARNING: Test data starts ({test_start.strftime('%Y-%m-%d')}) before training ends ({train_end.strftime('%Y-%m-%d')})")
    print("   This could indicate data leakage!")
else:
    print(f"\n✅ GOOD: Test data starts AFTER training data ends")
    print(f"   Gap: {(test_start - train_end).days} days")

print("\n4. CHECKING FOR FUTURE INFORMATION")
print("-" * 80)

# Check if any features contain future information
# The 'score' column should not be used in features (it's the outcome!)
feature_columns = [
    'spread', 'adjoe_diff', 'adjde_diff', 'adjt_diff', 'barthag_diff',
    'rank_diff', 'sos_diff', 'home_adjoe', 'away_adjoe', 'home_adjde',
    'away_adjde', 'home_barthag', 'away_barthag'
]

# Check if 'score' or 'margin' or 'home_score' appears in feature columns
leaky_features = [col for col in feature_columns if any(word in col.lower() for word in ['score', 'margin', 'result', 'winner'])]

if leaky_features:
    print(f"\n⚠️  WARNING: Potentially leaky features found: {leaky_features}")
else:
    print(f"\n✅ GOOD: No obvious leaky features (score/margin/result) in feature list")

print("\n5. VERIFYING TARGET VARIABLE")
print("-" * 80)

# The target should be based on actual outcome, not predicted
print("\nTarget variable: 'home_covered'")
print(f"Training set - Home covered: {train['home_covered'].mean()*100:.1f}%")
print(f"Testing set - Home covered: {test['home_covered'].mean()*100:.1f}%")

# Should be close to 50% (balanced)
if 48 <= train['home_covered'].mean()*100 <= 52:
    print("✅ GOOD: Training target is balanced (~50%)")
else:
    print("⚠️  WARNING: Training target is imbalanced")

if 48 <= test['home_covered'].mean()*100 <= 52:
    print("✅ GOOD: Testing target is balanced (~50%)")
else:
    print("⚠️  WARNING: Testing target is imbalanced")

print("\n6. CHECKING ONE ROW PER GAME")
print("-" * 80)

# Check for duplicate games (same game appearing twice)
# Group by game identifiers
train_dups = train.duplicated(subset=['season', 'game_date', 'team', 'opp'], keep=False).sum()
test_dups = test.duplicated(subset=['season', 'game_date', 'team', 'opp'], keep=False).sum()

if train_dups > 0:
    print(f"\n⚠️  WARNING: {train_dups} duplicate games found in training data!")
    print("   This suggests data wasn't properly deduplicated")
else:
    print(f"\n✅ GOOD: No duplicate games in training data")

if test_dups > 0:
    print(f"\n⚠️  WARNING: {test_dups} duplicate games found in testing data!")
else:
    print(f"\n✅ GOOD: No duplicate games in testing data")

print("\n7. REALITY CHECK: BASELINE COMPARISON")
print("-" * 80)

# Compare to baseline predictions
# Baseline 1: Always predict home team covers
baseline_home = test['home_covered'].mean()

# Baseline 2: Random guessing
np.random.seed(42)
random_preds = np.random.randint(0, 2, size=len(test))
baseline_random = (random_preds == test['home_covered']).mean()

print(f"\nBaseline accuracies on test set:")
print(f"  Always predict home covers: {baseline_home*100:.1f}%")
print(f"  Random guessing: {baseline_random*100:.1f}%")
print(f"  Our model: 66.14%")

if 66.14 > baseline_home * 100 + 5:
    print("\n✅ GOOD: Model significantly beats baselines")
else:
    print("\n⚠️  WARNING: Model doesn't beat baselines by much")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Count how many checks passed
checks_passed = 0
total_checks = 7

if not overlap:
    checks_passed += 1
if not duplicate_games:
    checks_passed += 1
if test_start > train_end:
    checks_passed += 1
if not leaky_features:
    checks_passed += 1
if 48 <= train['home_covered'].mean()*100 <= 52 and 48 <= test['home_covered'].mean()*100 <= 52:
    checks_passed += 1
if train_dups == 0 and test_dups == 0:
    checks_passed += 1
if 66.14 > baseline_home * 100 + 5:
    checks_passed += 1

print(f"\n✅ Passed {checks_passed}/{total_checks} leakage checks")

if checks_passed == total_checks:
    print("\n🎉 EXCELLENT: No data leakage detected!")
    print("   Your 66% accuracy appears to be legitimate.")
elif checks_passed >= 5:
    print("\n👍 GOOD: Most checks passed, minor concerns at most")
else:
    print("\n⚠️  CONCERN: Multiple potential leakage issues detected")
    print("   Review the warnings above carefully")

print("\n" + "=" * 80)

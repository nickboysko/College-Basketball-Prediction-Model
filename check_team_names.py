import pandas as pd

# Load one of your game files
games = pd.read_excel('teamrankings_2021-2022_to_2023-2024.csv.xlsx')

# Load one of the stat files
stats = pd.read_csv('historical_stats/20220115_stats.csv')

# Rename stat columns
col_names = ['rank', 'team', 'conf', 'record', 'adjoe', 'oe_rank',
            'adjde', 'de_rank', 'barthag', 'barthag_rank']
while len(col_names) < len(stats.columns):
    col_names.append(f'col_{len(col_names)}')
stats.columns = col_names

print("=== SAMPLE TEAM NAMES FROM YOUR GAMES FILE ===")
print(games['team'].head(20).tolist())

print("\n=== SAMPLE TEAM NAMES FROM BARTTORVIK ===")
print(stats['team'].head(20).tolist())

print("\n=== CHECKING MATCHES ===")
game_teams = set(games['team'].unique())
stat_teams = set(stats['team'].unique())
matches = game_teams & stat_teams

print(f"\nMatching teams: {len(matches)} out of {len(game_teams)}")

# Show non-matching examples
non_matching = sorted(list(game_teams - stat_teams))[:20]
if non_matching:
    print(f"\nSample teams that DON'T match:")
    for team in non_matching:
        print(f"  '{team}'")
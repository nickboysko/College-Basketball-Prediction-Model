import requests
import pandas as pd
from datetime import datetime
import sys
import os
import io
from team_name_mapping import map_team_name

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("SCRAPING TODAY'S GAMES FROM THE ODDS API")
print("="*80)

# Configuration
API_KEY_FILE = 'odds_api_key.txt'

# Check for API key
if not os.path.exists(API_KEY_FILE):
    print("\n[ERROR] API key file not found!")
    print("\nTo get started:")
    print("1. Go to: https://the-odds-api.com/")
    print("2. Click 'Get API Key' and sign up for FREE tier (500 requests/month)")
    print("3. Save your API key in a file named 'odds_api_key.txt'")
    print("\nThe free tier gives you 500 requests per month - plenty for daily predictions!")
    sys.exit(1)

# Read API key
with open(API_KEY_FILE, 'r') as f:
    API_KEY = f.read().strip()

if not API_KEY:
    print("[ERROR] API key file is empty!")
    print("Please add your API key to odds_api_key.txt")
    sys.exit(1)

print(f"\n[OK] API key loaded from {API_KEY_FILE}")

# Get today's date
today = datetime.now()
date_str = today.strftime('%Y-%m-%d')
season_str = f"{today.year}-{today.year+1}"

print(f"\nDate: {date_str}")
print(f"Season: {season_str}")

# The Odds API endpoint for NCAA basketball
sport = 'basketball_ncaab'
regions = 'us'  # US bookmakers
markets = 'spreads'  # We only need spreads
oddsFormat = 'american'

url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'

params = {
    'apiKey': API_KEY,
    'regions': regions,
    'markets': markets,
    'oddsFormat': oddsFormat
}

print(f"\n[FETCH] Fetching NCAA basketball odds from The Odds API...")
print(f"[INFO] Sport: {sport}")
print(f"[INFO] Markets: {markets}")

try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    
    # Check remaining requests
    remaining = response.headers.get('x-requests-remaining', 'unknown')
    used = response.headers.get('x-requests-used', 'unknown')
    
    print(f"\n[API] Requests used: {used}")
    print(f"[API] Requests remaining: {remaining}")
    
    data = response.json()
    
    if len(data) == 0:
        print("\n[WARN] No games found for today")
        print("[INFO] This could mean:")
        print("  - No games scheduled for today")
        print("  - Games not yet listed by bookmakers")
        print("  - Season hasn't started yet")
        sys.exit(0)
    
    print(f"\n[OK] Found {len(data)} games")
    
    # Parse the data
    games = []
    
    for game in data:
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = game['commence_time']
        
        # Get spreads from bookmakers
        if 'bookmakers' in game and len(game['bookmakers']) > 0:
            # Use the first bookmaker with spreads
            for bookmaker in game['bookmakers']:
                if 'markets' in bookmaker:
                    for market in bookmaker['markets']:
                        if market['key'] == 'spreads':
                            outcomes = market['outcomes']
                            
                            # Find home team spread
                            home_spread = None
                            
                            for outcome in outcomes:
                                if outcome['name'] == home_team:
                                    home_spread = outcome['point']
                                    break
                            
                            # Only add ONE entry per game (home team perspective)
                            # This matches your training data format where each game appears once
                            if home_spread is not None:
                                # Map team names from Odds API format to Barttorvik format
                                home_team_mapped = map_team_name(home_team)
                                away_team_mapped = map_team_name(away_team)
                                
                                games.append({
                                    'team': home_team_mapped,  # Home team
                                    'opp': away_team_mapped,   # Away team
                                    'spread': home_spread,
                                    'game_date': date_str,
                                    'season': season_str
                                })
                            
                            break  # Use first bookmaker with spreads
                    break
    
    if len(games) == 0:
        print("\n[WARN] No spreads found in game data")
        print("[INFO] Games may not have betting lines available yet")
        sys.exit(0)
    
    # Save to CSV
    df = pd.DataFrame(games)
    
    # No need to remove duplicates now since we only create one entry per game
    output_file = 'todays_games_raw.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n[OK] Saved {len(df)} games to {output_file}")
    
    print("\n" + "="*80)
    print("SCRAPING COMPLETE")
    print("="*80)
    print("\nGames found:")
    for i, game in enumerate(games, 1):
        spread_sign = '+' if game['spread'] > 0 else ''
        print(f"  {i}. {game['team']} ({spread_sign}{game['spread']}) vs {game['opp']}")
    
    print(f"\n[NEXT] Run: python merge_today.py")
    
except requests.exceptions.RequestException as e:
    print(f"\n[ERROR] Error fetching from The Odds API: {e}")
    print("\n[HELP] Troubleshooting:")
    print("  1. Check your internet connection")
    print("  2. Verify your API key is correct")
    print("  3. Check if you have remaining requests (500/month on free tier)")
    print("  4. Visit: https://the-odds-api.com/ for help")
    sys.exit(1)

except Exception as e:
    print(f"\n[ERROR] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
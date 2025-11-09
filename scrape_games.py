import requests
import pandas as pd
from datetime import datetime
import json
import sys

# Safe print function for Windows console
def safe_print(message, alt_message=None):
    """Print message with Unicode emoji fallback for Windows"""
    try:
        print(message)
    except UnicodeEncodeError:
        if alt_message:
            print(alt_message)
        else:
            # Remove emojis and print
            safe_msg = message.encode('ascii', 'ignore').decode('ascii')
            print(safe_msg)

print("=" * 80)
print("SCRAPING TODAY'S GAMES FROM TEAMRANKINGS")
print("=" * 80)

def scrape_todays_games():
    """
    Scrape today's college basketball games with spreads from TeamRankings
    """
    
    # TeamRankings uses an API endpoint for their odds data
    # This is the same approach we used before
    
    print("\nFetching today's games from TeamRankings...")
    
    # Base URL for TeamRankings API
    # They expose data through XHR calls that we can replicate
    url = "https://api.teamrankings.com/v1/public/odds/spreads"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    params = {
        'sport': 'ncaa-basketball',
        'league': 'ncaa-basketball',
        'date': datetime.now().strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            games = []
            for game in data.get('games', []):
                # Extract game info
                home_team = game.get('home_team')
                away_team = game.get('away_team')
                spread = game.get('spread')  # Negative means home favored
                game_time = game.get('game_time')
                
                if home_team and away_team and spread is not None:
                    games.append({
                        'team': home_team,
                        'opp': away_team,
                        'spread': float(spread),
                        'game_date': datetime.now().strftime('%Y-%m-%d'),
                        'game_time': game_time,
                        'season': '2025-2026'
                    })
            
            if games:
                df = pd.DataFrame(games)
                safe_print(f"✅ Found {len(df)} games", 
                          f"[OK] Found {len(df)} games")
                return df
            else:
                safe_print("⚠️  No games found for today", 
                          "[WARNING] No games found for today")
                return None
                
        else:
            safe_print(f"❌ API request failed with status {response.status_code}",
                      f"[ERROR] API request failed with status {response.status_code}")
            return None
            
    except Exception as e:
        error_msg = str(e)
        safe_print(f"❌ Error fetching from API: {error_msg}", 
                   f"[ERROR] Error fetching from API: {error_msg}")
        print("\nTrying alternative scraping method...")
        return scrape_teamrankings_html()

def scrape_teamrankings_html():
    """
    Fallback: Scrape the HTML page directly
    """
    
    url = "https://www.teamrankings.com/ncaa-basketball/odds/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Use pandas to read HTML tables
            tables = pd.read_html(response.text)
            
            # Find the odds table (usually first or second table)
            for table in tables:
                if 'Spread' in table.columns or 'Line' in table.columns:
                    # Process the table to extract games
                    games = []
                    
                    for idx, row in table.iterrows():
                        # Extract team names and spreads
                        # Format varies, adapt based on actual structure
                        try:
                            if 'Team' in table.columns and 'Spread' in table.columns:
                                matchup = row.get('Matchup', '')
                                spread = row.get('Spread', None)
                                
                                # Parse matchup (format: "Home vs Away" or similar)
                                if ' vs ' in matchup or ' @ ' in matchup:
                                    parts = matchup.replace(' @ ', ' vs ').split(' vs ')
                                    if len(parts) == 2:
                                        home_team = parts[0].strip()
                                        away_team = parts[1].strip()
                                        
                                        games.append({
                                            'team': home_team,
                                            'opp': away_team,
                                            'spread': float(spread) if spread else None,
                                            'game_date': datetime.now().strftime('%Y-%m-%d'),
                                            'season': '2025-2026'
                                        })
                        except Exception as e:
                            continue
                    
                    if games:
                        df = pd.DataFrame(games)
                        safe_print(f"✅ Scraped {len(df)} games from HTML",
                                  f"[OK] Scraped {len(df)} games from HTML")
                        return df
            
            safe_print("⚠️  Could not find odds table in page",
                      "[WARNING] Could not find odds table in page")
            return None
            
    except Exception as e:
        error_msg = str(e)
        safe_print(f"❌ HTML scraping failed: {error_msg}",
                  f"[ERROR] HTML scraping failed: {error_msg}")
        return None

def manual_entry_prompt():
    """
    Prompt user to manually enter games if scraping fails
    """
    print("\n" + "=" * 80)
    print("MANUAL ENTRY REQUIRED")
    print("=" * 80)
    
    print("\nAutomatic scraping failed. Please create 'todays_games_raw.csv' manually.")
    print("\nGo to: https://www.teamrankings.com/ncaa-basketball/odds/")
    print("\nFormat your CSV as:")
    print("team,opp,spread,game_date,season")
    print("Duke,North Carolina,-5.5,2024-11-07,2025-2026")
    print("\nAfter creating the file, run: python merge_today.py")

# Main execution
if __name__ == "__main__":
    games_df = scrape_todays_games()
    
    if games_df is not None and len(games_df) > 0:
        # Save to file
        output_file = 'todays_games_raw.csv'
        games_df.to_csv(output_file, index=False)
        
        safe_print(f"\n✅ Saved {len(games_df)} games to: {output_file}",
                  f"\n[OK] Saved {len(games_df)} games to: {output_file}")
        
        print("\nGames found:")
        for idx, row in games_df.iterrows():
            print(f"  {row['team']} vs {row['opp']} (Spread: {row['spread']:+.1f})")
        
        print("\n" + "=" * 80)
        print("NEXT STEP")
        print("=" * 80)
        print("\nRun: python merge_today.py")
        
    else:
        manual_entry_prompt()

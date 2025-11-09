import subprocess
import sys
import os
from datetime import datetime

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
print("DAILY BETTING PREDICTIONS - SIMPLIFIED WORKFLOW")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'=' * 80}")
    print(f"STEP: {description}")
    print('=' * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Print output safely (may contain emojis)
        if result.stdout:
            try:
                print(result.stdout, end='')
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Fallback: print with ASCII-safe encoding
                try:
                    safe_stdout = result.stdout.encode('ascii', 'replace').decode('ascii')
                    print(safe_stdout, end='')
                except:
                    print("[OUTPUT] Could not display output (encoding issue)")
        
        if result.returncode != 0:
            safe_print(f"\n❌ Error running {script_name}",
                      f"\n[ERROR] Error running {script_name}")
            if result.stderr:
                # Print stderr safely - handle Unicode errors
                try:
                    print(result.stderr, end='')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    # Try to decode and print with safe encoding
                    try:
                        safe_stderr = result.stderr.encode('ascii', 'replace').decode('ascii')
                        print(safe_stderr, end='')
                    except:
                        print("[ERROR] Could not display error message (encoding issue)")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        safe_print(f"❌ {script_name} timed out (>60 seconds)",
                  f"[ERROR] {script_name} timed out (>60 seconds)")
        return False
    except Exception as e:
        safe_print(f"❌ Error: {str(e)}",
                  f"[ERROR] Error: {str(e)}")
        return False

def check_file_exists(filename):
    """Check if a required file exists"""
    if os.path.exists(filename):
        safe_print(f"✅ Found: {filename}",
                  f"[OK] Found: {filename}")
        return True
    else:
        safe_print(f"❌ Missing: {filename}",
                  f"[ERROR] Missing: {filename}")
        return False

# Workflow
safe_print("\n🤖 Starting daily prediction workflow...",
          "\n[START] Starting daily prediction workflow...")

# Step 1: Check for current season stats (manual download)
current_year = datetime.now().year
next_year = current_year + 1

# Basketball season spans two years, check both
stats_file = None
if os.path.exists(f'{current_year}_team_results.csv'):
    stats_file = f'{current_year}_team_results.csv'
elif os.path.exists(f'{next_year}_team_results.csv'):
    stats_file = f'{next_year}_team_results.csv'

safe_print(f"\n📊 Checking for current season stats...",
          f"\n[CHECK] Checking for current season stats...")

if stats_file is None:
    safe_print(f"\n⚠️  Current season stats not found!",
              f"\n[WARNING] Current season stats not found!")
    print(f"\nPlease download manually (takes 5 seconds):")
    print(f"1. Go to: https://barttorvik.com/{next_year}.php")
    print(f"2. The CSV will auto-download")
    print(f"3. Save it as: {next_year}_team_results.csv")
    print(f"4. Put it in this folder")
    print(f"\nNote: If 'record' column shows dates, that's OK - we don't use it!")
    print(f"\nAfter downloading, run this script again: python run_daily_simple.py")
    sys.exit(1)
else:
    safe_print(f"✅ Using {stats_file}",
              f"[OK] Using {stats_file}")

# Step 2: Scrape today's games (automated)
safe_print(f"\n🏀 Getting today's games with spreads...",
          f"\n[GET] Getting today's games with spreads...")

if not run_script('scrape_games.py', 'Scrape Today\'s Games from TeamRankings'):
    safe_print("\n⚠️  Automatic game scraping failed.",
              "\n[WARNING] Automatic game scraping failed.")
    
    # Check if file exists from manual entry
    if not check_file_exists('todays_games_raw.csv'):
        safe_print("\n📝 Manual entry required:",
                  "\n[INFO] Manual entry required:")
        print("1. Go to: https://www.teamrankings.com/ncaa-basketball/odds/")
        print("2. Create 'todays_games_raw.csv' with format:")
        print("   team,opp,spread,game_date,season")
        print("   Duke,North Carolina,-5.5,2024-11-07,2025-2026")
        print("\n3. After creating file, run: python merge_today.py")
        print("   Then run: python predict_today.py")
        sys.exit(1)

# Step 3: Merge games with team stats
safe_print(f"\n🔄 Merging games with team stats...",
          f"\n[MERGE] Merging games with team stats...")

if not run_script('merge_today.py', 'Merge Games with Team Stats'):
    safe_print("\n❌ Merge failed. Check error messages above.",
              "\n[ERROR] Merge failed. Check error messages above.")
    sys.exit(1)

# Check merged file exists
if not check_file_exists('todays_games.csv'):
    safe_print("\n❌ Merged games file not created",
              "\n[ERROR] Merged games file not created")
    sys.exit(1)

# Step 4: Make predictions
safe_print(f"\n🎯 Generating predictions...",
          f"\n[PREDICT] Generating predictions...")

if not run_script('predict_today.py', 'Make Predictions'):
    safe_print("\n❌ Prediction failed. Check error messages above.",
              "\n[ERROR] Prediction failed. Check error messages above.")
    sys.exit(1)

# Success!
print("\n" + "=" * 80)
safe_print("✅ WORKFLOW COMPLETE!",
          "[SUCCESS] WORKFLOW COMPLETE!")
print("=" * 80)

safe_print("\n📁 Files Created:",
          "\n[FILES] Files Created:")
print("  - todays_games_raw.csv (today's games with spreads)")
print("  - todays_games.csv (merged with team stats)")
print(f"  - predictions_{datetime.now().strftime('%Y%m%d')}.csv (predictions)")

safe_print("\n💡 Next Steps:",
          "\n[NEXT] Next Steps:")
print("  1. Review the HIGH confidence picks above")
print("  2. Track your bets")
print("  3. Run daily: python run_daily_simple.py")

safe_print("\n📈 Expected Performance:",
          "\n[STATS] Expected Performance:")
print("  - Overall: 58% win rate, 11% ROI")
print("  - High Confidence: 71% win rate, 35% ROI")

print("\n" + "=" * 80)
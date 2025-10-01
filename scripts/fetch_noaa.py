import requests
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

# NOAA 3-day forecast URL
NOAA_URL = "https://services.swpc.noaa.gov/text/3-day-forecast.txt"

# File paths
DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "noaa_raw.txt"

def fetch_noaa():
    """Download the latest NOAA 3-day forecast and save it to noaa_raw.txt"""
    print("📡 Fetching NOAA 3-day forecast...")
    r = requests.get(NOAA_URL, timeout=30)
    r.raise_for_status()
    text = r.text

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_text(text, encoding="utf-8")
    print(f"✅ Saved latest forecast to {RAW_FILE}")

def run_pipeline():
    """Run the pipeline scripts using the same Python interpreter (venv)"""
    print("⚙️ Running pipeline scripts...")
    python = sys.executable  # ensures venv python is used
    subprocess.run([python, "scripts/parser_all_blocks.py", str(RAW_FILE)], check=True)
    subprocess.run([python, "scripts/add_daily_probs.py"], check=True)
    subprocess.run([python, "scripts/keep_latest_block.py"], check=True)
    print("✅ Pipeline complete. Data refreshed.")

if __name__ == "__main__":
    try:
        fetch_noaa()
        run_pipeline()
        print(f"🎉 Update finished at {datetime.now(timezone.utc).isoformat()}")
    except Exception as e:
        print("❌ Error:", e)

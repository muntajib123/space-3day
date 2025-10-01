# scripts/keep_latest_block.py
from pathlib import Path

RAW = Path("data/noaa_raw.txt")
if not RAW.exists():
    print("noaa_raw.txt not found")
    raise SystemExit(1)

text = RAW.read_text(encoding="utf-8", errors="ignore")
parts = text.split(":Product: 3-Day Forecast")
# keep last non-empty block (strip leading/trailing whitespace)
if len(parts) <= 1:
    print("No ':Product: 3-Day Forecast' blocks found; leaving file unchanged.")
else:
    last = parts[-1].strip()
    new_text = ":Product: 3-Day Forecast\n" + last + "\n"
    RAW.write_text(new_text, encoding="utf-8")
    print("Kept latest block and rewrote", RAW)

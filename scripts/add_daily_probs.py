# scripts/add_daily_probs.py
import re
from pathlib import Path
import pandas as pd
from dateutil import parser as dateparser
from datetime import timedelta

RAW = Path("data/noaa_raw.txt")
PARSED = Path("data/noaa_all_parsed.csv")
OUT = Path("data/noaa_all_parsed_enriched.csv")

text = RAW.read_text()

# split raw into blocks (same as earlier)
blocks = text.split(":Product: 3-Day Forecast")
blocks = [b for b in blocks if "NOAA Kp index breakdown" in b]

# build list of per-block metadata: start_date and three daily pct lists
meta = []
for b in blocks:
    # find block start date (e.g. "NOAA Kp index breakdown Dec 17-Dec 19 2022")
    m = re.search(r'NOAA Kp index breakdown\s+([A-Za-z]{3,9}\s+\d{1,2})', b)
    if m:
        start = dateparser.parse(m.group(1))
    else:
        start = None
    # find first 6 percentages in this block (solar then radio)
    perc = re.findall(r'(\d{1,3})\s*%', b)
    # Heuristic: assume first 3 -> solar, next 3 -> radio (if present)
    solar = [int(x) for x in perc[:3]] if len(perc) >= 3 else [None, None, None]
    radio = [int(x) for x in perc[3:6]] if len(perc) >= 6 else [None, None, None]
    meta.append({"start": start, "solar": solar, "radio": radio})

# load parsed kp CSV
df = pd.read_csv(PARSED, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# create columns and fill with NaN
df["solar_radiation"] = pd.NA
df["radio_blackout"] = pd.NA

# iterate blocks and assign daily probs to the corresponding 24 rows (8 per day)
# For matching, we'll group file by consecutive 24-row blocks by order of appearance.
# Build index groups of 24 rows sequentially (assuming parser produced blocks in same order)
n = len(df)
if n % 24 != 0:
    # if not perfectly divisible, we'll still process in chunks of 24 starting from 0
    chunks = [(i, min(i+24, n)) for i in range(0, n, 24)]
else:
    chunks = [(i, i+24) for i in range(0, n, 24)]

for idx, (start_idx, end_idx) in enumerate(chunks):
    if idx < len(meta):
        solar = meta[idx]["solar"]
        radio = meta[idx]["radio"]
        # assign each day (8 rows) the daily value
        for day in range(3):
            s_val = solar[day] if day < len(solar) else None
            r_val = radio[day] if day < len(radio) else None
            row_start = start_idx + day*8
            row_end = min(row_start + 8, end_idx)
            if row_start < n:
                df.loc[row_start:row_end-1, "solar_radiation"] = s_val
                df.loc[row_start:row_end-1, "radio_blackout"] = r_val

# save
df.to_csv(OUT, index=False)
print("Wrote:", OUT, "rows:", len(df))

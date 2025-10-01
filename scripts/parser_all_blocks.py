# scripts/parser_all_blocks.py
import re
import sys
import pandas as pd
from pathlib import Path
from dateutil import parser as dateparser
from datetime import timedelta

def parse_block(text):
    # get base date (like "Dec 17 2022")
    m = re.search(r'NOAA Kp index breakdown\s+([A-Za-z]{3}\s+\d{1,2})-([A-Za-z]{3}\s+\d{1,2})\s+(\d{4})', text)
    if not m:
        return pd.DataFrame()
    start_text, end_text, year = m.groups()
    start_date = dateparser.parse(start_text + " " + year)

    # extract numbers (Kp values in table)
    kp_lines = re.findall(r'^\s*\d{2}-\d{2}UT\s+(.+)$', text, flags=re.MULTILINE)
    rows = []
    for i, line in enumerate(kp_lines):
        values = re.findall(r'\d+\.\d+', line)
        if len(values) == 3:  # 3 days
            for j, v in enumerate(values):
                dt = start_date + timedelta(days=j, hours=i*3)
                rows.append({"datetime": dt, "kp": float(v)})
    return pd.DataFrame(rows)

def parse_file(path):
    text = Path(path).read_text()
    blocks = text.split(":Product: 3-Day Forecast")
    dfs = []
    for b in blocks:
        if "NOAA Kp index breakdown" in b:
            df = parse_block(b)
            if not df.empty:
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/parser_all_blocks.py data/noaa_raw.txt")
        sys.exit(1)
    df = parse_file(sys.argv[1])
    out = Path("data/noaa_all_parsed.csv")
    df.to_csv(out, index=False)
    print(f"✅ Parsed {len(df)} rows -> {out}")
    print(df.head())

# scripts/prep_dataset.py
"""
Prepare dataset for LSTM training.
- Reads data/noaa_all_parsed_enriched.csv
- Scales features (kp, solar_radiation, radio_blackout)
- Builds sliding windows: past 24 steps -> next 24 steps
- Saves dataset to data/dataset.npz
- Saves scaler to models/scaler.pkl
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler

# Input/Output paths
DATA_CSV = Path("data/noaa_all_parsed_enriched.csv")
OUT_NPZ = Path("data/dataset.npz")
SCALER_PATH = Path("models/scaler.pkl")

def make_sequences(values, in_steps=24, out_steps=24, stride=1):
    """Build sliding windows: past in_steps -> next out_steps"""
    X, Y = [], []
    max_start = len(values) - in_steps - out_steps + 1
    for start in range(0, max_start, stride):
        x = values[start: start + in_steps]
        y = values[start + in_steps: start + in_steps + out_steps]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

def main(in_steps=24, out_steps=24, stride=1):
    # Load CSV
    df = pd.read_csv(DATA_CSV, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Features to use
    features = ["kp", "solar_radiation", "radio_blackout"]

    # Convert to numpy
    arr = df[features].astype(float).values

    # Scale features 0–1
    scaler = MinMaxScaler()
    arr_scaled = scaler.fit_transform(arr)

    # Save scaler
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Saved scaler -> {SCALER_PATH}")

    # Make sequences
    X, Y = make_sequences(arr_scaled, in_steps=in_steps, out_steps=out_steps, stride=stride)
    print("✅ Sequences shapes:", X.shape, Y.shape)

    # Save dataset
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, X=X, Y=Y, in_steps=in_steps, out_steps=out_steps)
    print(f"✅ Saved dataset -> {OUT_NPZ}")

if __name__ == "__main__":
    main()

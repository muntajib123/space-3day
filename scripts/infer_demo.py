# scripts/infer_demo.py
"""
Produce a 72-hour forecast using the trained seq2seq LSTM.
Start time = midnight UTC of the day AFTER the latest NOAA datetime
found in data/noaa_all_parsed_enriched.csv (fallback: next UTC midnight).
Saves output to data/predictions.csv (timezone-aware UTC).
"""
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime, timezone, timedelta

MODEL_PATH = Path("models/seq2seq_lstm.h5")
SCALER_PATH = Path("models/scaler.pkl")
DATA_NPZ = Path("data/dataset.npz")
CSV_PATH = Path("data/noaa_all_parsed_enriched.csv")
OUT_PATH = Path("data/predictions.csv")


def compute_start_from_noaa(csv_path: Path) -> pd.Timestamp:
    """Return midnight UTC of the day AFTER latest datetime in csv_path.
    If csv missing/unreadable/empty, return next UTC midnight from now."""
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    fallback = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc) + timedelta(days=1)

    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["datetime"], dayfirst=False)
            if not df.empty:
                latest = pd.to_datetime(df["datetime"].max())
                # if naive, treat as UTC
                if latest.tzinfo is None:
                    latest = latest.tz_localize("UTC")
                latest_date = latest.date()
                next_midnight = datetime(latest_date.year, latest_date.month, latest_date.day, tzinfo=timezone.utc) + timedelta(days=1)
                return pd.Timestamp(next_midnight)
    except Exception as e:
        print("Warning: couldn't read NOAA CSV to compute start date:", e)

    return pd.Timestamp(fallback)


def main():
    # checks
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler: {SCALER_PATH}")
    if not DATA_NPZ.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_NPZ}")

    # load artifacts
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    data = np.load(DATA_NPZ)

    X = data["X"]
    out_steps = int(data["out_steps"])

    # predict
    sample = X[-1:].astype(np.float32)
    pred_scaled = model.predict(sample)
    pred_scaled = pred_scaled.reshape(out_steps, -1)
    pred = scaler.inverse_transform(pred_scaled)

    # compute start (day AFTER NOAA latest date)
    start_time = compute_start_from_noaa(CSV_PATH)
    print("Using start_time (UTC):", start_time.strftime("%Y-%m-%d %H:%M:%S%z"))

    # build future times (3-hour cadence)
    future_times = pd.date_range(start=start_time, periods=out_steps, freq="3h", tz="UTC")

    # dataframe and save
    df = pd.DataFrame(pred, columns=["kp", "solar_radiation", "radio_blackout"])
    df.insert(0, "datetime", future_times)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S%z")
    print(f"✅ Predictions saved -> {OUT_PATH}")
    print(df.head(12).to_string(index=False))  # show first ~36h


if __name__ == "__main__":
    main()

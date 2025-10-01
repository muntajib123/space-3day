# backend/app.py
"""
FastAPI server with:
- model loading & /api/predictions/3day endpoint
- daily scheduler to regenerate predictions
- saves predictions to CSV (and optionally to Mongo)
- healthcheck endpoint

Behavior:
Predictions start at midnight UTC of the day AFTER the last NOAA datetime found in
data/noaa_all_parsed_enriched.csv (so NOAA Sep 28–30 -> predictions Oct 1 onward).
"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone, date, time as dt_time
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("space-3day")

# ---------- load environment ----------
load_dotenv()

MODEL_PATH = Path("models/seq2seq_lstm.h5")
SCALER_PATH = Path("models/scaler.pkl")
DATASET_NPZ = Path("data/dataset.npz")
PRED_CSV = Path("data/predictions.csv")
PARSED_ENRICHED_CSV = Path("data/noaa_all_parsed_enriched.csv")
RAW_NOAA_TXT = Path("data/noaa_raw.txt")
# optional file where we persist the latest NOAA bulletin (if you want)
NOAA_PRESENT_SAVE = Path("data/noaa_present.txt")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "spaceweather")
PRED_COLLECTION = os.getenv("PRED_COLLECTION", "predictions")
RAW_COLLECTION = os.getenv("RAW_COLLECTION", "raw_forecasts")

SCHEDULE_CRON_HOUR = os.getenv("SCHEDULE_CRON_HOUR", "06")

FRONTEND_ORIGINS = [
    os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
    "http://127.0.0.1:3000"
]

# ---------- FastAPI app ----------
app = FastAPI(title="3-day Space Weather Forecast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------- globals ----------
_model = None
_scaler = None
_scheduler: Optional[BackgroundScheduler] = None
_latest_prediction: Optional[Dict[str, Any]] = None
_mongo_client = None

# ---------- utility helpers ----------
UTC = timezone.utc
CANDIDATE_DATE_COL_KEYS = ["datetime", "valid_time", "valid", "timestamp", "time", "date", "issued"]


def _parse_datetime_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Attempt to parse a dataframe column into timezone-aware UTC datetimes."""
    ser = pd.to_datetime(df[col], errors="coerce", utc=False)  # don't force UTC yet
    # if series tz is None, localize to UTC
    if ser.dt.tz is None:
        ser = ser.dt.tz_localize("UTC")
    else:
        ser = ser.dt.tz_convert("UTC")
    return ser


def detect_latest_noaa_timestamp(csv_path: Path) -> Optional[datetime]:
    """
    Robustly detect the latest NOAA timestamp from the given CSV.
    - Prefer column named 'datetime'
    - Else pick the best candidate column that parses into datetimes
    - Returns a timezone-aware datetime (UTC) or None
    """
    if not csv_path.exists():
        logger.debug("detect_latest_noaa_timestamp: CSV not found: %s", csv_path)
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        logger.warning("Failed to read CSV %s: %s", csv_path, e)
        return None

    if df.empty:
        logger.debug("detect_latest_noaa_timestamp: CSV is empty")
        return None

    # If 'datetime' column exists, prefer it
    cols = list(df.columns)
    chosen_col = None
    if "datetime" in [c.lower() for c in cols]:
        # get real column name with matching case
        for c in cols:
            if c.lower() == "datetime":
                chosen_col = c
                break

    # If not found, search candidate names
    if chosen_col is None:
        for key in CANDIDATE_DATE_COL_KEYS:
            for c in cols:
                if key in c.lower():
                    chosen_col = c
                    break
            if chosen_col:
                break

    # If still not found, heuristically pick a column with many parseable values
    if chosen_col is None:
        best = None
        best_score = 0.0
        for c in cols:
            parsed = pd.to_datetime(df[c], errors="coerce")
            score = parsed.notna().sum() / max(1, len(parsed))
            if score > best_score:
                best_score = score
                best = c
        if best_score >= 0.4:
            chosen_col = best
            logger.debug("detect_latest_noaa_timestamp: heuristically chose column %s (score=%.2f)", chosen_col, best_score)

    if chosen_col is None:
        logger.debug("detect_latest_noaa_timestamp: could not choose a suitable datetime column")
        return None

    # parse chosen column into timezone-aware UTC datetimes
    try:
        ser = _parse_datetime_series(df, chosen_col)
    except Exception:
        # fallback to pandas parse with coercion and manual UTC localize
        ser = pd.to_datetime(df[chosen_col], errors="coerce")
        if ser.dt.tz is None:
            ser = ser.dt.tz_localize("UTC")
        else:
            ser = ser.dt.tz_convert("UTC")

    ser = ser.dropna()
    if ser.empty:
        logger.debug("detect_latest_noaa_timestamp: chosen column %s had no parseable datetimes", chosen_col)
        return None

    latest = ser.max().to_pydatetime()
    # ensure tz aware UTC
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=UTC)
    else:
        latest = latest.astimezone(UTC)

    logger.info("detect_latest_noaa_timestamp: chosen_col=%s latest=%s", chosen_col, latest.isoformat())
    return latest


def _next_utc_midnight_after(dt_utc: datetime) -> datetime:
    """Given a timezone-aware UTC datetime, return next day's midnight UTC."""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC)
    latest_date = dt_utc.date()
    next_day = latest_date + timedelta(days=1)
    return datetime(next_day.year, next_day.month, next_day.day, 0, 0, tzinfo=UTC)


# ---------- model helpers ----------
def load_artifacts():
    global _model, _scaler
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model or scaler not found. Run training first.")
    _model = load_model(str(MODEL_PATH), compile=False)
    _scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded from disk.")


def generate_prediction() -> pd.DataFrame:
    """
    Run inference. Predictions start at midnight UTC of the day AFTER
    the latest NOAA datetime (as read from data/noaa_all_parsed_enriched.csv).
    Returns a dataframe with timezone-aware UTC datetimes in a 'datetime' column.
    """
    if not DATASET_NPZ.exists():
        raise FileNotFoundError("Dataset not available. Run prep_dataset.py")
    if _model is None or _scaler is None:
        raise RuntimeError("Model or scaler not loaded.")

    # load model inputs
    data = np.load(DATASET_NPZ)
    X = data["X"]
    out_steps = int(data["out_steps"])
    sample = X[-1:].astype(np.float32)
    pred_scaled = _model.predict(sample)
    pred_scaled = pred_scaled.reshape(out_steps, -1)
    pred = _scaler.inverse_transform(pred_scaled)

    # determine next day midnight UTC after latest NOAA timestamp (if available)
    latest_noaa_ts = detect_latest_noaa_timestamp(PARSED_ENRICHED_CSV)
    if latest_noaa_ts is None:
        # fallback to file raw or to "today"
        logger.warning("No latest NOAA timestamp detected, trying fallback to raw file or 'today'.")
        # try raw NOAA text (if present) by reading and looking for date tokens (light fallback)
        if RAW_NOAA_TXT.exists():
            try:
                text = RAW_NOAA_TXT.read_text(encoding="utf-8", errors="ignore")
                # simple heuristic: find last 3-letter month + day tokens, attempt parse
                import re
                tokens = re.findall(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{4}", text)
                if tokens:
                    last_tok = tokens[-1]
                    latest_noaa_ts = datetime.strptime(last_tok, "%b %d %Y").replace(tzinfo=UTC)
                else:
                    latest_noaa_ts = datetime.utcnow().replace(tzinfo=UTC)
            except Exception:
                latest_noaa_ts = datetime.utcnow().replace(tzinfo=UTC)
        else:
            latest_noaa_ts = datetime.utcnow().replace(tzinfo=UTC)

    start_midnight = _next_utc_midnight_after(latest_noaa_ts)

    # build datetime index at 3-hour cadence
    future_times = pd.date_range(start=pd.Timestamp(start_midnight), periods=out_steps, freq="3h", tz="UTC")

    # build DF with predictions
    df_pred = pd.DataFrame(pred, columns=["kp", "solar_radiation", "radio_blackout"])
    df_pred.insert(0, "datetime", future_times)

    # Save metadata about used latest NOAA date and start
    df_pred.attrs["latest_noaa_timestamp_utc"] = latest_noaa_ts.isoformat()
    df_pred.attrs["start_date_utc"] = start_midnight.isoformat()

    return df_pred


def save_prediction(df: pd.DataFrame) -> Dict[str, Any]:
    """Save prediction to CSV (and optionally Mongo). Add meta fields to CSV as top-of-file JSON comment."""
    # convert datetimes to ISO (UTC)
    if "datetime" in df.columns:
        # ensure tz-aware and in UTC
        try:
            if df["datetime"].dt.tz is None:
                df["datetime"] = df["datetime"].dt.tz_localize("UTC")
            else:
                df["datetime"] = df["datetime"].dt.tz_convert("UTC")
        except Exception:
            pass

    # Save the CSV
    df.to_csv(PRED_CSV, index=False)

    meta = {
        "generated_at": datetime.utcnow().replace(tzinfo=UTC).isoformat(),
        "rows": len(df),
        "csv": str(PRED_CSV),
        "latest_noaa_timestamp_utc": df.attrs.get("latest_noaa_timestamp_utc"),
        "start_date_utc": df.attrs.get("start_date_utc")
    }
    return meta


def run_job_and_update_memory():
    """Scheduler: generate prediction, save and update latest."""
    global _latest_prediction
    try:
        df_pred = generate_prediction()
        meta = save_prediction(df_pred)
        _latest_prediction = {
            "meta": meta,
            "predictions": json.loads(df_pred.to_json(orient="records", date_format="iso"))
        }
        logger.info("Prediction generated, rows=%s, start=%s", len(df_pred), meta.get("start_date_utc"))
    except Exception as e:
        logger.exception("Prediction job failed: %s", e)


# ---------- NOAA present forecast saving helper ----------
def save_noaa_present_forecast(raw_text: str):
    """
    Simple helper: save the raw NOAA bulletin text to NOA_PRESENT_SAVE and RAW_NOAA_TXT.
    Optionally, you can call your parsing pipeline to update data/noaa_all_parsed_enriched.csv
    after saving — here we only save the raw text and return the parseable latest date if found.
    """
    try:
        NOAA_PRESENT_SAVE.write_text(raw_text, encoding="utf-8")
        RAW_NOAA_TXT.write_text(raw_text, encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save NOAA present text: %s", e)

    # Light parse for last calendar date using simple patterns
    try:
        import re
        # pattern: "Sep 29-Oct 01 2025" etc
        m = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s*[-–]\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s*(\d{4})?", raw_text, flags=re.IGNORECASE)
        if m:
            # get the last month/day token and year
            tokens = re.findall(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})", raw_text, flags=re.IGNORECASE)
            if tokens:
                last_mon, last_day = tokens[-1]
                # find year if present
                y = re.search(r"\b(\d{4})\b", raw_text)
                year = int(y.group(1)) if y else datetime.utcnow().year
                latest_date = datetime.strptime(f"{last_mon} {last_day} {year}", "%b %d %Y").replace(tzinfo=UTC)
                return latest_date
    except Exception:
        pass

    return None


# ---------- startup / scheduler ----------
@app.on_event("startup")
def startup_event():
    try:
        load_artifacts()
    except Exception as e:
        logger.error("Failed to load artifacts: %s", e)

    try:
        if _model is not None:
            run_job_and_update_memory()
    except Exception as e:
        logger.error("Initial prediction failed: %s", e)

    global _scheduler
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(lambda: run_job_and_update_memory(),
                       CronTrigger(hour=SCHEDULE_CRON_HOUR, minute=10))
    _scheduler.start()
    logger.info("Scheduler started: daily job at hour=%s UTC", SCHEDULE_CRON_HOUR)


@app.on_event("shutdown")
def shutdown_event():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)


# ---------- API endpoints ----------
class PredictionResponse(BaseModel):
    meta: Dict[str, Any]
    predictions: List[Dict[str, Any]]


# Healthcheck
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


# Endpoint to receive and save NOAA present bulletin (optional)
@app.post("/api/noaa/present/save")
async def api_save_noaa_present(request: Request):
    """
    POST a plain text NOAA bulletin in the request body (text/plain or JSON {"text": "..."}).
    This will save the raw bulletin locally and return the best-effort parsed latest timestamp.
    Useful when you have a process that fetches NOAA and then calls this endpoint to persist.
    """
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            raw_text = body.get("text", "")
        else:
            raw_text = await request.body()
            if isinstance(raw_text, (bytes, bytearray)):
                raw_text = raw_text.decode("utf-8", errors="ignore")
        latest_dt = save_noaa_present_forecast(raw_text)
        if latest_dt:
            return {"saved": True, "latest_noaa_timestamp_utc": latest_dt.isoformat()}
        return {"saved": True, "latest_noaa_timestamp_utc": None}
    except Exception as e:
        logger.exception("Failed saving NOAA present bulletin")
        raise HTTPException(status_code=500, detail=str(e))


# Predictions endpoint
@app.get("/api/predictions/3day", response_model=PredictionResponse)
def get_predictions():
    try:
        if PRED_CSV.exists():
            df = pd.read_csv(PRED_CSV, parse_dates=["datetime"])
            if not df.empty:
                # timezone handling
                try:
                    if df["datetime"].dt.tz is None:
                        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
                    else:
                        df["datetime"] = df["datetime"].dt.tz_convert("UTC")
                except Exception:
                    pass
            records = json.loads(df.to_json(orient="records", date_format="iso"))
            # derive metadata: try to read attrs if present by regenerating detection
            latest_noaa_ts = detect_latest_noaa_timestamp(PARSED_ENRICHED_CSV)
            start_midnight = _next_utc_midnight_after(latest_noaa_ts) if latest_noaa_ts else None
            mtime_epoch = os.path.getmtime(PRED_CSV)
            mtime_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime_epoch))
            meta = {
                "source": "csv",
                "file_mtime": mtime_iso,
                "rows": len(records),
                "latest_noaa_timestamp_utc": latest_noaa_ts.isoformat() if latest_noaa_ts else None,
                "start_date_utc": start_midnight.isoformat() if start_midnight else None
            }
            return JSONResponse(content={"meta": meta, "predictions": records}, headers={"Cache-Control": "no-store"})

        if _latest_prediction is not None:
            # make sure meta includes latest_noaa and start_date
            meta = {**_latest_prediction.get("meta", {})}
            # if missing, try to calculate
            if "latest_noaa_timestamp_utc" not in meta:
                latest_noaa_ts = detect_latest_noaa_timestamp(PARSED_ENRICHED_CSV)
                meta["latest_noaa_timestamp_utc"] = latest_noaa_ts.isoformat() if latest_noaa_ts else None
            if "start_date_utc" not in meta and meta.get("latest_noaa_timestamp_utc"):
                dt = datetime.fromisoformat(meta["latest_noaa_timestamp_utc"])
                meta["start_date_utc"] = _next_utc_midnight_after(dt).isoformat()
            return JSONResponse(content={"meta": meta, "predictions": _latest_prediction.get("predictions", [])},
                                headers={"Cache-Control": "no-store"})

        raise HTTPException(status_code=503, detail="Predictions not ready yet.")
    except Exception as e:
        logger.exception("Error serving predictions")
        raise HTTPException(status_code=500, detail=str(e))

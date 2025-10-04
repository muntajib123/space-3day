# backend/app.py
"""
FastAPI server with:
- model loading & /api/predictions/3day endpoint
- daily scheduler to regenerate predictions
- automatic NOAA 3-day bulletin fetch (startup + every 3h)
- saves predictions to CSV
- ✅ saves each forecast run to MongoDB (predictions_runs)
- ✅ accepts ground-truth observations into MongoDB (observations)
- ✅ computes & stores MSE/MAE metrics by joining predictions↔observations (metrics)
- secure cron endpoint to trigger a run (for external schedulers)
- healthcheck endpoint
- PAST history endpoint parsing NOAA 3-day text bulletins (since 2022)

Behavior (effective with START_OFFSET_DAYS):
Predictions start at midnight UTC of the day AFTER the last NOAA datetime found in
data/noaa_all_parsed_enriched.csv, plus an extra configurable day offset.
Set START_OFFSET_DAYS=1 to align model to show Days 7–9 when NOAA shows Days 4–6.
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests  # for NOAA live fetch
from fastapi import FastAPI, HTTPException, Request, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# --- MongoDB ---
from pymongo import MongoClient, ASCENDING, ReturnDocument

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("space-3day")

# ---------- load environment ----------
load_dotenv()

# ---------- paths / constants ----------
MODEL_PATH = Path("models/seq2seq_lstm.h5")
SCALER_PATH = Path("models/scaler.pkl")
DATASET_NPZ = Path("data/dataset.npz")
PRED_CSV = Path("data/predictions.csv")
PARSED_ENRICHED_CSV = Path("data/noaa_all_parsed_enriched.csv")
RAW_NOAA_TXT = Path("data/noaa_raw.txt")
NOAA_PRESENT_SAVE = Path("data/noaa_present.txt")
NOAA_FORECAST_URL = "https://services.swpc.noaa.gov/text/3-day-forecast.txt"

# ✅ Your historical NOAA bulletins text file (since 2022)
HISTORY_TXT = Path("data/noaa_history.txt")

SCHEDULE_CRON_HOUR = os.getenv("SCHEDULE_CRON_HOUR", "06")

FRONTEND_ORIGINS = [
    os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
    "http://127.0.0.1:3000",
]

# START OFFSET: 1 → start at Day 7 when NOAA shows 4–6 (recommended)
START_OFFSET_DAYS = int(os.getenv("START_OFFSET_DAYS", "1"))

# --- Mongo/env config ---
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "space_weather")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "predictions_runs")
MONGODB_OBS_COLLECTION = os.getenv("MONGODB_OBS_COLLECTION", "observations")
MONGODB_METRICS_COLLECTION = os.getenv("MONGODB_METRICS_COLLECTION", "metrics")
CRON_TOKEN = os.getenv("CRON_TOKEN")

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

# --- Mongo globals ---
_mongo_client: Optional[MongoClient] = None
_runs = None
_obs = None
_metrics = None

# ---------- utility helpers ----------
UTC = timezone.utc
CANDIDATE_DATE_COL_KEYS = ["datetime", "valid_time", "valid", "timestamp", "time", "date", "issued"]


def _parse_datetime_series(df: pd.DataFrame, col: str) -> pd.Series:
    ser = pd.to_datetime(df[col], errors="coerce", utc=False)
    if ser.dt.tz is None:
        ser = ser.dt.tz_localize("UTC")
    else:
        ser = ser.dt.tz_convert("UTC")
    return ser


def detect_latest_noaa_timestamp(csv_path: Path) -> Optional[datetime]:
    if not csv_path.exists():
        logger.debug("detect_latest_noaa_timestamp: CSV not found: %s", csv_path)
        return None
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        logger.warning("Failed to read CSV %s: %s", csv_path, e)
        return None
    if df.empty:
        return None

    cols = list(df.columns)
    chosen_col = None
    if "datetime" in [c.lower() for c in cols]:
        for c in cols:
            if c.lower() == "datetime":
                chosen_col = c
                break
    if chosen_col is None:
        for key in CANDIDATE_DATE_COL_KEYS:
            for c in cols:
                if key in c.lower():
                    chosen_col = c
                    break
            if chosen_col:
                break
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

    if chosen_col is None:
        return None

    try:
        ser = _parse_datetime_series(df, chosen_col)
    except Exception:
        ser = pd.to_datetime(df[chosen_col], errors="coerce")
        if ser.dt.tz is None:
            ser = ser.dt.tz_localize("UTC")
        else:
            ser = ser.dt.tz_convert("UTC")
    ser = ser.dropna()
    if ser.empty:
        return None

    latest = ser.max().to_pydatetime()
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=UTC)
    else:
        latest = latest.astimezone(UTC)
    logger.info("detect_latest_noaa_timestamp: chosen_col=%s latest=%s", chosen_col, latest.isoformat())
    return latest


def _next_utc_midnight_after(dt_utc: datetime) -> datetime:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC)
    d = dt_utc.date()
    next_day = d + timedelta(days=1)
    return datetime(next_day.year, next_day.month, next_day.day, 0, 0, tzinfo=UTC)


def _start_after_noaa(latest_noaa_ts: datetime) -> datetime:
    return _next_utc_midnight_after(latest_noaa_ts + timedelta(days=START_OFFSET_DAYS))

# ---------- NOAA live fetch helper ----------
def fetch_and_save_noaa_present():
    try:
        resp = requests.get(NOAA_FORECAST_URL, timeout=15)
        resp.raise_for_status()
        text = resp.text or ""
        NOAA_PRESENT_SAVE.write_text(text, encoding="utf-8")
        RAW_NOAA_TXT.write_text(text, encoding="utf-8")
        logger.info("NOAA 3-day forecast updated automatically.")
        return text
    except Exception as e:
        logger.error(f"Failed to fetch NOAA forecast: {e}")
        return None


# ---------- Mongo helpers ----------
def _init_mongo():
    global _mongo_client, _runs, _obs, _metrics
    if not MONGODB_URI:
        logger.warning("MONGODB_URI not set; Mongo features disabled.")
        return
    _mongo_client = MongoClient(MONGODB_URI)
    db = _mongo_client[MONGODB_DB]
    _runs = db[MONGODB_COLLECTION]
    _obs = db[MONGODB_OBS_COLLECTION]
    _metrics = db[MONGODB_METRICS_COLLECTION]

    _runs.create_index([("run_id", ASCENDING)], unique=True)
    _runs.create_index([("start_date_utc", ASCENDING)])
    _runs.create_index([("generated_at", ASCENDING)])
    _obs.create_index([("datetime", ASCENDING)], unique=True)
    _metrics.create_index([("run_id", ASCENDING)], unique=True)

def _save_run_to_mongo(meta: Dict[str, Any], records: List[Dict[str, Any]]):
    if _runs is None:
        return
    run_id = meta.get("generated_at") or datetime.utcnow().replace(tzinfo=UTC).isoformat()
    doc = {
        "run_id": run_id,
        "generated_at": meta.get("generated_at"),
        "start_date_utc": meta.get("start_date_utc"),
        "latest_noaa_timestamp_utc": meta.get("latest_noaa_timestamp_utc"),
        "start_offset_days": meta.get("start_offset_days"),
        "rows": meta.get("rows"),
        "predictions": records,
        "first_datetime": records[0]["datetime"] if records else None,
        "last_datetime": records[-1]["datetime"] if records else None,
        "source": "model",
        "csv_path": meta.get("csv"),
    }
    _runs.find_one_and_replace({"run_id": run_id}, doc, upsert=True, return_document=ReturnDocument.AFTER)
    logger.info("Saved run to MongoDB: %s (%s rows)", run_id, len(records))

def _compute_metrics_for_run(run_id: str) -> Dict[str, Any]:
    """
    Join predictions with observations on 'datetime' and compute MSE/MAE for
    kp, solar_radiation, radio_blackout. Stores metrics doc and returns it.
    """
    if _runs is None or _obs is None or _metrics is None:
        raise RuntimeError("Mongo not initialized")

    run = _runs.find_one({"run_id": run_id})
    if not run:
        raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")

    preds = pd.DataFrame(run.get("predictions", []))
    if preds.empty or "datetime" not in preds.columns:
        raise HTTPException(status_code=400, detail="No predictions found for this run")

    # fetch observations for this horizon
    dt_min = preds["datetime"].min()
    dt_max = preds["datetime"].max()
    obs_cursor = _obs.find({"datetime": {"$gte": dt_min, "$lte": dt_max}})
    obs_df = pd.DataFrame(list(obs_cursor))
    if obs_df.empty:
        raise HTTPException(status_code=400, detail="No observations found for this run's horizon")

    # keep only needed columns
    obs_df = obs_df[["datetime", "kp", "solar_radiation", "radio_blackout"]]
    # merge on datetime
    df = preds.merge(obs_df, on="datetime", suffixes=("_pred", "_obs"))
    if df.empty:
        raise HTTPException(status_code=400, detail="No matching datetimes between predictions and observations")

    def mse(a, b):
        s = np.array(a, dtype=float)
        t = np.array(b, dtype=float)
        mask = np.isfinite(s) & np.isfinite(t)
        if not mask.any():
            return None
        d = s[mask] - t[mask]
        return float(np.mean(d * d))

    def mae(a, b):
        s = np.array(a, dtype=float)
        t = np.array(b, dtype=float)
        mask = np.isfinite(s) & np.isfinite(t)
        if not mask.any():
            return None
        d = np.abs(s[mask] - t[mask])
        return float(np.mean(d))

    metrics = {
        "run_id": run_id,
        "generated_at": run.get("generated_at"),
        "start_date_utc": run.get("start_date_utc"),
        "n_matched": int(len(df)),
        "per_metric": {
            "kp": {"mse": mse(df["kp_pred"], df["kp_obs"]), "mae": mae(df["kp_pred"], df["kp_obs"])},
            "solar_radiation": {
                "mse": mse(df["solar_radiation_pred"], df["solar_radiation_obs"]),
                "mae": mae(df["solar_radiation_pred"], df["solar_radiation_obs"]),
            },
            "radio_blackout": {
                "mse": mse(df["radio_blackout_pred"], df["radio_blackout_obs"]),
                "mae": mae(df["radio_blackout_pred"], df["radio_blackout_obs"]),
            },
        },
    }

    _metrics.find_one_and_replace({"run_id": run_id}, metrics, upsert=True, return_document=ReturnDocument.AFTER)
    logger.info("Metrics computed for run %s (matched %s rows)", run_id, metrics["n_matched"])
    return metrics


# ---------- model helpers ----------
def load_artifacts():
    global _model, _scaler
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model or scaler not found. Run training first.")
    _model = load_model(str(MODEL_PATH), compile=False)
    _scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded from disk.")


def generate_prediction() -> pd.DataFrame:
    if not DATASET_NPZ.exists():
        raise FileNotFoundError("Dataset not available. Run prep_dataset.py")
    if _model is None or _scaler is None:
        raise RuntimeError("Model or scaler not loaded.")

    data = np.load(DATASET_NPZ)
    X = data["X"]
    out_steps = int(data["out_steps"])
    sample = X[-1:].astype(np.float32)
    pred_scaled = _model.predict(sample)
    pred_scaled = pred_scaled.reshape(out_steps, -1)
    pred = _scaler.inverse_transform(pred_scaled)

    latest_noaa_ts = detect_latest_noaa_timestamp(PARSED_ENRICHED_CSV)
    if latest_noaa_ts is None:
        logger.warning("No latest NOAA timestamp detected; falling back to 'today'.")
        latest_noaa_ts = datetime.utcnow().replace(tzinfo=UTC)

    start_midnight = _start_after_noaa(latest_noaa_ts)

    future_times = pd.date_range(start=pd.Timestamp(start_midnight), periods=out_steps, freq="3h", tz="UTC")

    df_pred = pd.DataFrame(pred, columns=["kp", "solar_radiation", "radio_blackout"])
    df_pred.insert(0, "datetime", future_times)

    df_pred.attrs["latest_noaa_timestamp_utc"] = latest_noaa_ts.isoformat()
    df_pred.attrs["start_date_utc"] = start_midnight.isoformat()
    df_pred.attrs["start_offset_days"] = START_OFFSET_DAYS

    return df_pred


def save_prediction(df: pd.DataFrame) -> Dict[str, Any]:
    if "datetime" in df.columns:
        try:
            if df["datetime"].dt.tz is None:
                df["datetime"] = df["datetime"].dt.tz_localize("UTC")
            else:
                df["datetime"] = df["datetime"].dt.tz_convert("UTC")
        except Exception:
            pass

    df.to_csv(PRED_CSV, index=False)

    meta = {
        "generated_at": datetime.utcnow().replace(tzinfo=UTC).isoformat(),
        "rows": len(df),
        "csv": str(PRED_CSV),
        "latest_noaa_timestamp_utc": df.attrs.get("latest_noaa_timestamp_utc"),
        "start_date_utc": df.attrs.get("start_date_utc"),
        "start_offset_days": df.attrs.get("start_offset_days"),
    }
    return meta


def run_job_and_update_memory():
    global _latest_prediction
    try:
        df_pred = generate_prediction()
        meta = save_prediction(df_pred)
        records = json.loads(df_pred.to_json(orient="records", date_format="iso"))

        _latest_prediction = {"meta": meta, "predictions": records}

        # NEW: save the run into Mongo
        _save_run_to_mongo(meta, records)

        logger.info(
            "Prediction generated, rows=%s, start=%s, offset_days=%s",
            len(df_pred),
            meta.get("start_date_utc"),
            meta.get("start_offset_days"),
        )
    except Exception as e:
        logger.exception("Prediction job failed: %s", e)


# ---------- startup / scheduler ----------
@app.on_event("startup")
def startup_event():
    global _scheduler
    _init_mongo()
    try:
        load_artifacts()
    except Exception as e:
        logger.error("Failed to load artifacts: %s", e)

    try:
        if _model is not None:
            run_job_and_update_memory()
    except Exception as e:
        logger.error("Initial prediction failed: %s", e)

    _scheduler = BackgroundScheduler()

    # Daily prediction job
    _scheduler.add_job(lambda: run_job_and_update_memory(),
                       CronTrigger(hour=SCHEDULE_CRON_HOUR, minute=10))

    # NOAA bulletin job every 3 hours (HH:05)
    _scheduler.add_job(fetch_and_save_noaa_present,
                       CronTrigger(minute=5, hour="*/3"))

    # Fetch NOAA once at startup so file is never empty
    fetch_and_save_noaa_present()

    _scheduler.start()
    logger.info("Scheduler started: predictions daily + NOAA bulletin every 3h")


@app.on_event("shutdown")
def shutdown_event():
    global _scheduler, _mongo_client
    if _scheduler:
        _scheduler.shutdown(wait=False)
    if _mongo_client:
        _mongo_client.close()


# ---------- API endpoints ----------
class PredictionResponse(BaseModel):
    meta: Dict[str, Any]
    predictions: List[Dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "mongo": bool(_runs is not None)}


# Manual save endpoint (kept)
@app.post("/api/noaa/present/save")
async def api_save_noaa_present(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            raw_text = body.get("text", "")
        else:
            raw_text = await request.body()
            if isinstance(raw_text, (bytes, bytearray)):
                raw_text = raw_text.decode("utf-8", errors="ignore")
        NOAA_PRESENT_SAVE.write_text(raw_text, encoding="utf-8")
        RAW_NOAA_TXT.write_text(raw_text, encoding="utf-8")
        return {"saved": True}
    except Exception as e:
        logger.exception("Failed saving NOAA present bulletin")
        raise HTTPException(status_code=500, detail=str(e))


# Present: serve latest NOAA bulletin, auto-fetch if missing/stale
@app.get("/api/noaa/present/text")
def get_noaa_present_text():
    try:
        if NOAA_PRESENT_SAVE.exists():
            mtime = datetime.utcfromtimestamp(NOAA_PRESENT_SAVE.stat().st_mtime).replace(tzinfo=UTC)
            age_hours = (datetime.now(tz=UTC) - mtime).total_seconds() / 3600.0
            if age_hours < 3:
                return {
                    "text": NOAA_PRESENT_SAVE.read_text(encoding="utf-8", errors="ignore"),
                    "fetched_at": mtime.isoformat(),
                    "source": "cached",
                }
        text = fetch_and_save_noaa_present()
        if not text:
            raise HTTPException(status_code=500, detail="Failed to fetch NOAA forecast.")
        return {"text": text, "fetched_at": datetime.now(tz=UTC).isoformat(), "source": "live"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Predictions endpoint
@app.get("/api/predictions/3day", response_model=PredictionResponse)
def get_predictions():
    try:
        if PRED_CSV.exists():
            df = pd.read_csv(PRED_CSV, parse_dates=["datetime"])
            if not df.empty:
                try:
                    if df["datetime"].dt.tz is None:
                        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
                    else:
                        df["datetime"] = df["datetime"].dt.tz_convert("UTC")
                except Exception:
                    pass
            records = json.loads(df.to_json(orient="records", date_format="iso"))

            latest_noaa_ts = detect_latest_noaa_timestamp(PARSED_ENRICHED_CSV)
            start_midnight = _start_after_noaa(latest_noaa_ts) if latest_noaa_ts else None

            mtime_epoch = os.path.getmtime(PRED_CSV)
            mtime_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime_epoch))
            meta = {
                "source": "csv",
                "file_mtime": mtime_iso,
                "rows": len(records),
                "latest_noaa_timestamp_utc": latest_noaa_ts.isoformat() if latest_noaa_ts else None,
                "start_date_utc": start_midnight.isoformat() if start_midnight else None,
                "start_offset_days": START_OFFSET_DAYS,
            }
            return JSONResponse(content={"meta": meta, "predictions": records}, headers={"Cache-Control": "no-store"})

        if _latest_prediction is not None:
            meta = {**_latest_prediction.get("meta", {})}
            if "latest_noaa_timestamp_utc" not in meta:
                latest_noaa_ts = detect_latest_noaa_timestamp(PARSED_ENRICHED_CSV)
                meta["latest_noaa_timestamp_utc"] = latest_noaa_ts.isoformat() if latest_noaa_ts else None
            if "start_date_utc" not in meta and meta.get("latest_noaa_timestamp_utc"):
                dt = datetime.fromisoformat(meta["latest_noaa_timestamp_utc"])
                meta["start_date_utc"] = _start_after_noaa(dt).isoformat()
            if "start_offset_days" not in meta:
                meta["start_offset_days"] = START_OFFSET_DAYS
            return JSONResponse(content={"meta": meta, "predictions": _latest_prediction.get("predictions", [])},
                                headers={"Cache-Control": "no-store"})

        raise HTTPException(status_code=503, detail="Predictions not ready yet.")
    except Exception as e:
        logger.exception("Error serving predictions")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- CRON endpoint (external trigger, e.g., GitHub Actions) ----------
@app.post("/cron/daily")
def cron_daily(request: Request):
    token = request.headers.get("x-cron-token") or request.query_params.get("token")
    if not token or token != CRON_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    run_job_and_update_memory()
    return {"ok": True, "ran_at": datetime.utcnow().replace(tzinfo=UTC).isoformat()}


# ---------- OBSERVATIONS (ground truth) ----------
# Upsert observations: accepts array of items with datetime (ISO) and fields
# Example body:
# { "rows": [ {"datetime":"2025-10-07T00:00:00Z", "kp":3.2, "solar_radiation":10, "radio_blackout":60}, ... ] }
@app.post("/api/observations/upsert")
def upsert_observations(payload: Dict[str, Any] = Body(...)):
    if _obs is None:
        raise HTTPException(status_code=500, detail="Mongo not initialized")
    rows = payload.get("rows") or []
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="rows must be a non-empty list")
    count = 0
    for r in rows:
        dt = r.get("datetime")
        if not dt:
            continue
        doc = {
            "datetime": dt,
            "kp": r.get("kp"),
            "solar_radiation": r.get("solar_radiation"),
            "radio_blackout": r.get("radio_blackout"),
        }
        _obs.find_one_and_replace({"datetime": dt}, doc, upsert=True, return_document=ReturnDocument.AFTER)
        count += 1
    return {"ok": True, "upserted": count}


# ---------- METRICS ----------
@app.post("/api/metrics/compute")
def compute_metrics(run_id: str = Query(...)):
    metrics = _compute_metrics_for_run(run_id)
    return {"ok": True, "metrics": metrics}

@app.get("/api/metrics/latest")
def get_latest_metrics():
    if _metrics is None:
        raise HTTPException(status_code=500, detail="Mongo not initialized")
    doc = _metrics.find_one(sort=[("run_id", -1)])
    if not doc:
        return {"metrics": None}
    # cast numpy types to plain python if any
    return {"metrics": json.loads(json.dumps(doc, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o))}


# =========================
# PAST HISTORY (parse NOAA 3-day bulletins text file)
# =========================

# month name maps
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}
_MONTHS_FULL = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], start=1)}

def _month_to_num(name: str) -> Optional[int]:
    if not name:
        return None
    n = name.strip().lower()
    return _MONTHS.get(n, _MONTHS_FULL.get(n))

_num_re = re.compile(r"(\d+(?:\.\d+)?)")  # e.g., "6.67 (G3)" -> 6.67

def _num_or_none(cell: str) -> Optional[float]:
    if not cell:
        return None
    m = _num_re.search(cell)
    return float(m.group(1)) if m else None

def _parse_kp_table(lines: List[str]) -> Dict[str, List[Optional[float]]]:
    slots: Dict[str, List[Optional[float]]] = {}
    for ln in lines:
        if not ln or "UT" not in ln:
            continue
        parts = re.split(r"\s{2,}", ln.strip())
        if not parts:
            continue
        label = parts[0]
        cols = parts[1:] if len(parts) > 1 else []
        while len(cols) < 3:
            cols.append("")
        vals = [_num_or_none(c) for c in cols[:3]]
        if re.match(r"^\d{2}-\d{2}UT$", label):
            slots[label] = vals
    return slots

def _parse_percentage_row(line: Optional[str]) -> List[Optional[float]]:
    if not line:
        return [None, None, None]
    nums = re.findall(r"(\d+)\s*%", line)
    return [float(x) for x in nums[:3]] if nums else [None, None, None]

def _extract_dates_triplet(header_line: str) -> List[datetime]:
    m = re.search(
        r"breakdown\s+([A-Za-z]+)\s+(\d{1,2})\s*-\s*(?:(\w+)\s*)?(\d{1,2})\s+(\d{4})",
        header_line
    )
    if not m:
        raise ValueError(f"Cannot parse dates in header: {header_line}")
    mon1, d1_s, mon2_opt, d3_s, year_s = m.groups()
    y = int(year_s)
    m1 = _month_to_num(mon1)
    if m1 is None:
        raise ValueError("bad month in header")
    d1 = int(d1_s)
    if mon2_opt:
        m2 = _month_to_num(mon2_opt)
        if m2 is None:
            raise ValueError("bad month2 in header")
        d3 = int(d3_s)
        day1 = datetime(y, m1, d1, tzinfo=UTC)
        day2 = day1 + timedelta(days=1)
        day3 = datetime(y, m2, d3, tzinfo=UTC)
        return [day1, day2, day3]
    else:
        d3 = int(d3_s)
        day1 = datetime(y, m1, d1, tzinfo=UTC)
        day2 = datetime(y, m1, d1 + 1, tzinfo=UTC)
        day3 = datetime(y, m1, d3, tzinfo=UTC)
        return [day1, day2, day3]

def _rows_from_bulletin(kp_dates: List[datetime],
                        kp_table_lines: List[str],
                        s_row_line: Optional[str],
                        r_row_line: Optional[str]) -> List[Dict[str, Any]]:
    order = ["00-03UT","03-06UT","06-09UT","09-12UT","12-15UT","15-18UT","18-21UT","21-00UT"]
    kp_slots = _parse_kp_table(kp_table_lines)
    s_daily = _parse_percentage_row(s_row_line)
    r_daily = _parse_percentage_row(r_row_line)

    out: List[Dict[str, Any]] = []
    for day_idx, base_date in enumerate(kp_dates):
        for slot in order:
            vals = kp_slots.get(slot, [None, None, None])
            kp_val = vals[day_idx] if day_idx < len(vals) else None
            m = re.match(r"(\d{2})-(\d{2})UT", slot)
            hh = int(m.group(1)) if m else 0
            dt = base_date.replace(hour=hh, minute=0, second=0, microsecond=0)
            out.append({
                "datetime": dt.isoformat().replace("+00:00", "Z"),
                "kp": kp_val,
                "solar_radiation": s_daily[day_idx] if day_idx < len(s_daily) else None,
                "radio_blackout": r_daily[day_idx] if day_idx < len(r_daily) else None,
            })
    return out

def _parse_noaa_history_text(all_text: str) -> List[Dict[str, Any]]:
    lines = all_text.splitlines()
    rows: List[Dict[str, Any]] = []
    i = 0
    n = len(lines)

    while i < n:
        ln = lines[i]

        if "NOAA Kp index breakdown" in ln:
            try:
                kp_dates = _extract_dates_triplet(ln)
            except Exception as e:
                logger.debug("skip header parse error: %s", e)
                i += 1
                continue

            kp_table: List[str] = []
            j = i + 1
            while j < n and len(kp_table) < 12:
                txt = lines[j].rstrip()
                if re.search(r"^\s*\d{2}-\d{2}UT", txt):
                    kp_table.append(txt)
                if "Rationale:" in txt:
                    break
                j += 1

            s_row = None
            r_row = None
            k = i
            while k < min(i + 180, n):
                t = lines[k]
                if "Solar Radiation Storm Forecast" in t:
                    for k2 in range(k, min(k + 20, n)):
                        if re.search(r"S1\s+or\s+greater", lines[k2], re.I):
                            s_row = lines[k2]
                            break
                if "Radio Blackout Forecast" in t:
                    for k2 in range(k, min(k + 20, n)):
                        if re.search(r"^\s*R1-R2", lines[k2]):
                            r_row = lines[k2]
                            break
                if s_row and r_row:
                    break
                k += 1

            rows.extend(_rows_from_bulletin(kp_dates, kp_table, s_row, r_row))
            i = j
        else:
            i += 1

    dedup: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        dedup[r["datetime"]] = r
    out = list(dedup.values())
    out.sort(key=lambda r: r["datetime"])
    return out

@app.get("/api/obs/history")
def get_obs_history(year: int = Query(...), month: int = Query(...)):
    if not HISTORY_TXT.exists():
        raise HTTPException(status_code=404, detail=f"History file not found: {HISTORY_TXT}")

    try:
        text = HISTORY_TXT.read_text(encoding="utf-8", errors="ignore")
        all_rows = _parse_noaa_history_text(text)
        ym = f"{year:04d}-{month:02d}-"
        month_rows = [r for r in all_rows if r["datetime"].startswith(ym)]
        return {"data": month_rows}
    except Exception as e:
        logger.exception("Failed to parse history")
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")

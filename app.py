#!/usr/bin/env python3
"""
app.py

FastAPI service for the digital twin. Wraps run_live_pipeline.py's logic
(simulate an hour -> ingest -> build features -> score with
xgboost_rul_model.joblib -> join asset metadata) into a web API instead of
a CLI loop.

On startup, a background thread runs the exact same simulate -> ingest ->
score cycle as run_live_pipeline.py, forever, ticking every TICK_SECONDS
(wall-clock seconds) with each tick advancing the simulated clock by one
hour. Every asset's latest prediction is kept in memory (refreshed from
SQLite after each tick) so GET requests are instant and never touch the
DB on the request path.

ENDPOINTS
  GET  /predictions              -> list of latest prediction per asset (all 36)
  GET  /predictions/{asset_id}   -> single asset's latest prediction
  GET  /health                   -> service + simulation status
  POST /tick                     -> force one extra simulated hour immediately (debugging)

ENV VARS (all optional, sane defaults for Render)
  TICK_SECONDS   how many real seconds between simulated hours (default 10)
  DB_PATH        SQLite file path (default ./bharati.db)
  MODEL_PATH     path to the .joblib model (default ./xgboost_rul_model.joblib)
  OUT_PATH       predictions.jsonl append path (default ./predictions.jsonl)
  SEED           RNG seed for the simulator (default 42)
  START_TIME     ISO8601 UTC start for the simulated clock (default: now)
"""

import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from asset_fleet import build_fleet
from bharati_sensor_simulator import simulate_stream
from db_pipeline import init_db, seed_assets, ingest_readings_batch, load_model, score_pending

TICK_SECONDS = float(os.environ.get("TICK_SECONDS", "10"))
DB_PATH = os.environ.get("DB_PATH", "bharati.db")
MODEL_PATH = os.environ.get("MODEL_PATH", "xgboost_rul_model.joblib")
OUT_PATH = os.environ.get("OUT_PATH", "predictions.jsonl")
SEED = int(os.environ.get("SEED", "42"))
START_TIME = os.environ.get("START_TIME")  # e.g. "2025-05-01T00:00:00Z"

# --- shared state, guarded by _cache_lock ---
_cache_lock = threading.Lock()
latest_predictions: dict[str, dict] = {}   # asset_id -> prediction dict
sim_status = {"running": False, "hours_simulated": 0, "last_tick_utc": None, "error": None}
_stop_event = threading.Event()


def refresh_cache(conn: sqlite3.Connection):
    """Pull the latest (max timestamp) prediction row per asset from SQLite
    into the in-memory cache that GET requests read from."""
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT p.asset_id, p.timestamp, p.predicted_rul_days, p.current_state,
               p.component_id, p.component_type, p.machine_id, p.machine_name, p.room_id
        FROM predictions p
        INNER JOIN (
            SELECT asset_id, MAX(timestamp) AS ts FROM predictions GROUP BY asset_id
        ) latest ON p.asset_id = latest.asset_id AND p.timestamp = latest.ts
    """).fetchall()
    with _cache_lock:
        for r in rows:
            (asset_id, timestamp, predicted_rul_days, current_state,
             component_id, component_type, machine_id, machine_name, room_id) = r
            latest_predictions[asset_id] = {
                "asset_id": asset_id,
                "timestamp": timestamp,
                "predicted_rul_days": predicted_rul_days,
                "current_state": current_state,
                "component_id": component_id,
                "component_type": component_type,
                "machine_id": machine_id,
                "machine_name": machine_name,
                "room_id": room_id,
            }


def live_loop():
    """Background thread: same simulate -> ingest -> score cycle as
    run_live_pipeline.py, but ticking every TICK_SECONDS wall-clock seconds
    (instead of sleeping a real hour) so the digital twin visibly updates."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        init_db(conn)
        seed_assets(conn)

        fleet = build_fleet()
        model = load_model(MODEL_PATH)

        start_dt = (datetime.strptime(START_TIME, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    if START_TIME else datetime.now(timezone.utc))

        builders = {}
        sim_status["running"] = True

        for timestamp, readings, _labels in simulate_stream(fleet, start_dt, hours=0, seed=SEED):
            if _stop_event.is_set():
                break
            builders = ingest_readings_batch(conn, fleet, readings, builders)
            score_pending(conn, model, OUT_PATH, append=True)
            refresh_cache(conn)

            sim_status["hours_simulated"] += 1
            sim_status["last_tick_utc"] = timestamp.isoformat()

            time.sleep(TICK_SECONDS)
    except Exception as e:  # keep the error visible via /health instead of silently dying
        sim_status["error"] = str(e)
        sim_status["running"] = False
        raise
    finally:
        sim_status["running"] = False
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=live_loop, daemon=True)
    thread.start()
    yield
    _stop_event.set()


app = FastAPI(title="Digital Twin RUL API", lifespan=lifespan)

# Wide-open CORS so a browser-based digital twin dashboard can call this
# directly. Tighten allow_origins to your dashboard's domain in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.responses import RedirectResponse


@app.get("/")
def root():
    return RedirectResponse(url="/predictions")


@app.get("/health")
def health():
    with _cache_lock:
        n_assets = len(latest_predictions)
    return {
        "status": "ok" if sim_status["running"] else "starting_or_stopped",
        "assets_tracked": n_assets,
        "hours_simulated": sim_status["hours_simulated"],
        "last_tick_utc": sim_status["last_tick_utc"],
        "tick_seconds": TICK_SECONDS,
        "error": sim_status["error"],
    }


@app.get("/predictions")
def get_predictions():
    """Latest prediction for every asset -- this is what your digital twin
    dashboard should poll (e.g. every TICK_SECONDS)."""
    with _cache_lock:
        return list(latest_predictions.values())


@app.get("/predictions/{asset_id}")
def get_prediction(asset_id: str):
    with _cache_lock:
        row = latest_predictions.get(asset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No prediction yet for asset_id={asset_id}")
    return row


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)

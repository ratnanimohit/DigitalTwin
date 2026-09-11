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

PUBLIC ENDPOINTS
  GET  /predictions              -> list of latest prediction per asset (all 36)
  GET  /predictions/{asset_id}   -> single asset's latest prediction
  GET  /health                   -> service + simulation status

ADMIN ENDPOINTS (require ?token=ADMIN_TOKEN -- see ENV VARS below)
  GET  /admin/download-db        -> download the live bharati.db SQLite file
  GET  /admin/query              -> peek at rows in a table as JSON (no download needed)
  POST /admin/inject             -> manually push a sensor reading for one asset and
                                     immediately re-score it, for live demos

ENV VARS (all optional, sane defaults for Render)
  TICK_SECONDS   how many real seconds between simulated hours (default 10)
  DB_PATH        SQLite file path (default ./bharati.db)
  MODEL_PATH     path to the .joblib model (default ./xgboost_rul_model.joblib)
  OUT_PATH       predictions.jsonl append path (default ./predictions.jsonl)
  SEED           RNG seed for the simulator (default 42)
  START_TIME     ISO8601 UTC start for the simulated clock (default: now)
  ADMIN_TOKEN    secret string required by every /admin/* endpoint. If unset,
                 /admin/* routes are disabled (return 503) rather than left open.
"""

import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from asset_fleet import build_fleet
from bharati_sensor_simulator import simulate_stream
from db_pipeline import init_db, seed_assets, ingest_readings_batch, load_model, score_pending

TICK_SECONDS = float(os.environ.get("TICK_SECONDS", "10"))
DB_PATH = os.environ.get("DB_PATH", "bharati.db")
MODEL_PATH = os.environ.get("MODEL_PATH", "xgboost_rul_model.joblib")
OUT_PATH = os.environ.get("OUT_PATH", "predictions.jsonl")
SEED = int(os.environ.get("SEED", "42"))
START_TIME = os.environ.get("START_TIME")  # e.g. "2025-05-01T00:00:00Z"
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")  # set this in Render's env vars -- keep it secret

# --- shared state, guarded by _cache_lock (predictions cache) ---
_cache_lock = threading.Lock()
latest_predictions: dict[str, dict] = {}   # asset_id -> prediction dict
sim_status = {"running": False, "hours_simulated": 0, "last_tick_utc": None, "error": None}
_stop_event = threading.Event()

# --- shared pipeline state, guarded by _pipeline_lock (conn/fleet/model/builders) ---
# Moved to module scope (instead of local to live_loop) so /admin/* endpoints can
# safely reuse the SAME connection and the SAME per-asset feature builders --
# a manually injected reading needs to flow through the identical ingest ->
# feature -> score path the live simulator uses, or the rolling z-score
# history would go out of sync between the two.
_pipeline_lock = threading.Lock()
_pipeline = {"conn": None, "fleet": None, "model": None, "builders": {}}


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


def init_pipeline():
    """Open the one shared DB connection / fleet / model that both the live
    background loop and the admin endpoints operate on."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    init_db(conn)
    seed_assets(conn)
    _pipeline["conn"] = conn
    _pipeline["fleet"] = build_fleet()
    _pipeline["model"] = load_model(MODEL_PATH)
    _pipeline["builders"] = {}


def live_loop():
    """Background thread: same simulate -> ingest -> score cycle as
    run_live_pipeline.py, but ticking every TICK_SECONDS wall-clock seconds
    (instead of sleeping a real hour) so the digital twin visibly updates."""
    try:
        fleet = _pipeline["fleet"]
        start_dt = (datetime.strptime(START_TIME, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    if START_TIME else datetime.now(timezone.utc))

        sim_status["running"] = True

        for timestamp, readings, _labels in simulate_stream(fleet, start_dt, hours=0, seed=SEED):
            if _stop_event.is_set():
                break

            with _pipeline_lock:
                _pipeline["builders"] = ingest_readings_batch(
                    _pipeline["conn"], _pipeline["fleet"], readings, _pipeline["builders"]
                )
                score_pending(_pipeline["conn"], _pipeline["model"], OUT_PATH, append=True)
                refresh_cache(_pipeline["conn"])

            sim_status["hours_simulated"] += 1
            sim_status["last_tick_utc"] = timestamp.isoformat()

            time.sleep(TICK_SECONDS)
    except Exception as e:  # keep the error visible via /health instead of silently dying
        sim_status["error"] = str(e)
        sim_status["running"] = False
        raise
    finally:
        sim_status["running"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pipeline()
    thread = threading.Thread(target=live_loop, daemon=True)
    thread.start()
    yield
    _stop_event.set()
    if _pipeline["conn"] is not None:
        _pipeline["conn"].close()


app = FastAPI(title="Digital Twin RUL API", lifespan=lifespan)

# Wide-open CORS so a browser-based digital twin dashboard can call this
# directly. Tighten allow_origins to your dashboard's domain in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_admin(token: Optional[str]):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN is not set on the server, admin routes are disabled")
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing admin token")


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


# ---------------------------------------------------------------------------
# ADMIN: private database access + manual reading injection for demos
# All of these require ?token=<ADMIN_TOKEN>. Set ADMIN_TOKEN as a secret env
# var on Render (Dashboard -> your service -> Environment) and only share it
# with yourself.
# ---------------------------------------------------------------------------

_ADMIN_TABLES = {"assets", "raw_readings", "model_features", "predictions"}


@app.get("/admin/download-db")
def admin_download_db(token: Optional[str] = Query(default=None)):
    """Download the live SQLite file as-is. Open it locally with any free
    SQLite viewer (e.g. 'DB Browser for SQLite', or the sqlite3 CLI) to
    browse every table with full query power, on your own machine."""
    _check_admin(token)
    with _pipeline_lock:
        _pipeline["conn"].commit()  # make sure everything is flushed to disk first
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="DB file not found on disk")
    return FileResponse(DB_PATH, filename="bharati.db", media_type="application/octet-stream")


@app.get("/admin/query")
def admin_query(
    token: Optional[str] = Query(default=None),
    table: str = Query(default="predictions"),
    limit: int = Query(default=50, le=500),
):
    """Quick JSON peek at a table's most recent rows, no download needed.
    table is one of: assets, raw_readings, model_features, predictions."""
    _check_admin(token)
    if table not in _ADMIN_TABLES:
        raise HTTPException(status_code=400, detail=f"table must be one of {sorted(_ADMIN_TABLES)}")
    with _pipeline_lock:
        cur = _pipeline["conn"].cursor()
        columns = [row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()]
        order_by = "timestamp DESC" if "timestamp" in columns else "1"
        rows = cur.execute(f"SELECT * FROM {table} ORDER BY {order_by} LIMIT ?", (limit,)).fetchall()
    return {"table": table, "columns": columns, "row_count": len(rows), "rows": rows}


class InjectReading(BaseModel):
    asset_id: str
    # sensor_id -> value, e.g. {"S0011": 480.0} to spike Exhaust Gas Temp on A001.
    # Any sensor on the asset you don't mention here keeps its normal baseline
    # value, so you only need to specify the sensor(s) you want to change.
    overrides: dict[str, float]
    # Optional custom label for this reading; defaults to a MANUAL-<UTC time> tag
    # so it's obviously distinguishable from the simulator's own timestamps.
    label: Optional[str] = None


@app.post("/admin/inject")
def admin_inject(payload: InjectReading, token: Optional[str] = Query(default=None)):
    """Manually push one hourly reading for a single asset -- for demoing
    the automation live. Runs it through the SAME ingest -> feature build ->
    score pipeline the background simulator uses, then updates the public
    /predictions cache immediately, so you can call this and then GET
    /predictions/{asset_id} right after to show the RUL/state change."""
    _check_admin(token)

    with _pipeline_lock:
        fleet_by_id = {a["asset_id"]: a for a in _pipeline["fleet"]}
        asset = fleet_by_id.get(payload.asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail=f"unknown asset_id={payload.asset_id}")

        unknown_sensors = set(payload.overrides) - {s["sensor_id"] for s in asset["sensors"]}
        if unknown_sensors:
            raise HTTPException(
                status_code=400,
                detail=f"asset {payload.asset_id} has no sensor(s) {sorted(unknown_sensors)}. "
                       f"Valid sensor_ids: {[s['sensor_id'] for s in asset['sensors']]}",
            )

        ts = payload.label or f"MANUAL-{datetime.now(timezone.utc).isoformat()}"
        sensors = [
            {
                "sensor_id": s["sensor_id"],
                "sensor_type": s["sensor_type"],
                "measurement": s["measurement"],
                "unit": s["unit"],
                "value": payload.overrides.get(s["sensor_id"], s["baseline_mean"]),
            }
            for s in asset["sensors"]
        ]
        reading = {"asset_id": payload.asset_id, "timestamp": ts, "sensors": sensors}

        _pipeline["builders"] = ingest_readings_batch(
            _pipeline["conn"], _pipeline["fleet"], [reading], _pipeline["builders"]
        )
        score_pending(_pipeline["conn"], _pipeline["model"], OUT_PATH, append=True)
        refresh_cache(_pipeline["conn"])

    with _cache_lock:
        result = latest_predictions.get(payload.asset_id)
    return {"injected_reading": reading, "new_prediction": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)

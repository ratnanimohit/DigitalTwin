# Digital Twin RUL API

Serves live remaining-useful-life predictions for all 36 assets, for a
digital twin dashboard to poll.

## What's in here

| File | Role |
|---|---|
| `app.py` | **New.** FastAPI service — the thing you deploy. |
| `asset_fleet.py` | Your 36-asset fleet definition + `MODEL_FEATURE_ORDER`. |
| `score_stream.py` | Rolling z-score feature builder + RUL→state binning. |
| `db_pipeline.py` | SQLite ingest/feature/score/join pipeline. |
| `bharati_sensor_simulator.py` | Generates realistic hourly sensor readings per asset. |
| `run_live_pipeline.py` | Original CLI loop (kept for offline/manual runs — the API doesn't call this file, it re-implements the same loop as a background thread so it can serve HTTP requests concurrently). |
| `xgboost_rul_model.joblib` | Your trained model. |
| `requirements.txt`, `render.yaml`, `Procfile` | Deployment config. |

## How it works

On startup, a background thread repeats the same cycle as
`run_live_pipeline.py` — simulate one more hour → ingest → build features →
`model.predict()` → join asset metadata — but ticks every `TICK_SECONDS`
**real** seconds (default 10) instead of a real hour, so your digital twin
visibly updates. Each tick's results are cached in memory, so `GET`
requests are instant and never touch SQLite on the request path.

## Endpoints

- `GET /predictions` — latest prediction for **all** assets (array), e.g.:
  ```json
  [
    {
      "asset_id": "A002",
      "timestamp": "2025-05-01T12:00:00Z",
      "predicted_rul_days": 23.74,
      "current_state": "DEGRADING",
      "component_id": "C002",
      "component_type": "Bearing",
      "machine_id": "M002",
      "machine_name": "CHP Unit 1",
      "room_id": "R001"
    },
    ...
  ]
  ```
- `GET /predictions/{asset_id}` — single asset, e.g. `/predictions/A002`. `404` until that asset's first tick completes.
- `GET /health` — `{status, assets_tracked, hours_simulated, last_tick_utc, tick_seconds, error}`. Point your dashboard's connectivity check and Render's health check here.

## Deploy to Render

1. Push this folder to a GitHub repo.
2. In Render: **New → Blueprint**, point it at the repo — `render.yaml` configures everything automatically. (Or **New → Web Service** manually: build command `pip install -r requirements.txt`, start command `uvicorn app:app --host 0.0.0.0 --port $PORT`.)
3. Once live, poll `https://<your-service>.onrender.com/predictions` from your digital twin frontend every `TICK_SECONDS` (or a bit slower) for the live feed.

**Free-tier note:** Render's free web services spin down after 15 min idle and cold-start on the next request (losing in-memory state and restarting the simulated clock from `START_TIME`/now). SQLite history in `bharati.db` also lives on ephemeral disk — it resets on redeploy/restart. For a persistent 24/7 twin, use a paid instance (no spin-down) and/or swap SQLite for a managed Postgres add-on — `db_pipeline.py`'s comments note this is just a connection-string change.

## Tuning

- `TICK_SECONDS` — real seconds per simulated hour. Lower = faster-moving twin, higher CPU/DB churn.
- `SEED` — simulator RNG seed, for reproducible demo runs.
- `START_TIME` — e.g. `2025-05-01T00:00:00Z` to pin the simulated clock instead of starting from "now".

## Local test

```bash
pip install -r requirements.txt
TICK_SECONDS=5 uvicorn app:app --reload
curl http://localhost:8000/predictions
```

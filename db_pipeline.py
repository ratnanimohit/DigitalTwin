#!/usr/bin/env python3
"""
db_pipeline.py

Full pipeline: raw sensor readings -> SQLite database -> preprocessed model
features -> model.predict() -> RUL + state label -> joined with asset/
component/machine/room metadata -> your output schema.

FLOW (matches what you described):

  sensor_batches.jsonl                          (raw readings, your input schema)
        |
        v
  [1] ingest_readings()      --> SQLite: raw_readings table
        |
        v
  [2] build_features()       --> SQLite: model_features table
        (z-scores, rolling stats, one-hots -- see score_stream.py for the
         exact math, reused here via AssetFeatureBuilder)
        |
        v
  [3] load_and_preprocess()  <-- SQLite: model_features table
        (reads back out, casts one-hot/boolean columns to int, reindexes
         to MODEL_FEATURE_ORDER so column order exactly matches training)
        |
        v
  [4] model.predict(X)       --> predicted_rul_days
        (model = joblib.load(...) -- variable is literally named `model`,
         as requested, so you can swap the loading line for your own code)
        |
        v
  [5] rul_to_state()         --> current_state
        |
        v
  [6] join with SQLite: assets table (component_id/component_type/
        machine_id/machine_name/room_id, seeded from asset_fleet.py)
        |
        v
  predictions.jsonl + SQLite: predictions table   (your output schema)

WHY SQLite: zero setup (stdlib, one file on disk), but every table here is a
straightforward relational table -- swapping in Postgres/MySQL later is just
changing the connection string, the SQL stays ~the same.

PREPROCESSING NOTE: I don't have your actual training script, so step [3]'s
preprocessing is my best-effort reconstruction: one-hot/boolean columns cast
to int (0/1), numeric z-features and t_idx left as float/int, columns
reindexed to the exact 35-column order extracted from the model file, no
scaling applied (I didn't find a scaler artifact embedded in the .joblib).
If your training script did anything more (fillna strategy, a fitted scaler,
dtype choices), paste it or upload it and I'll line this up exactly instead
of guessing.

USAGE
  # one-shot: ingest a batch file, score everything not yet scored, write predictions
  python3 db_pipeline.py --sensor-file output/sensor_batches.jsonl \
      --db bharati.db --model xgboost_rul_model.joblib --out predictions.jsonl

  # just re-score what's already in the DB (e.g. after a model update), no new ingest
  python3 db_pipeline.py --db bharati.db --model xgboost_rul_model.joblib \
      --out predictions.jsonl --skip-ingest
"""

import argparse
import json
import sqlite3
import sys

from asset_fleet import build_fleet, MODEL_FEATURE_ORDER
from score_stream import AssetFeatureBuilder, DEFAULT_STATE_BINS, rul_to_state

# SQLite column names can't contain spaces -- "component_type_Power Supply"
# becomes "component_type_Power_Supply" in the DB, mapped back to the exact
# MODEL_FEATURE_ORDER name right before scoring.
def db_col(name: str) -> str:
    return name.replace(" ", "_")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            asset_id TEXT PRIMARY KEY,
            component_id TEXT,
            component_type TEXT,
            machine_id TEXT,
            machine_name TEXT,
            room_id TEXT,
            criticality TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_readings (
            asset_id TEXT,
            timestamp TEXT,
            sensor_id TEXT,
            sensor_type TEXT,
            measurement TEXT,
            unit TEXT,
            value REAL,
            PRIMARY KEY (asset_id, timestamp, sensor_id)
        )
    """)

    feature_cols_sql = ",\n            ".join(
        f'"{db_col(c)}" INTEGER' if c.startswith(("machine_id_", "component_type_", "criticality_"))
        else f'"{db_col(c)}" REAL'
        for c in MODEL_FEATURE_ORDER
    )
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS model_features (
            asset_id TEXT,
            timestamp TEXT,
            {feature_cols_sql},
            scored INTEGER DEFAULT 0,
            PRIMARY KEY (asset_id, timestamp)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            asset_id TEXT,
            timestamp TEXT,
            predicted_rul_days REAL,
            current_state TEXT,
            component_id TEXT,
            component_type TEXT,
            machine_id TEXT,
            machine_name TEXT,
            room_id TEXT,
            PRIMARY KEY (asset_id, timestamp)
        )
    """)
    conn.commit()


def seed_assets(conn: sqlite3.Connection):
    """Populate the assets table from asset_fleet.py (idempotent)."""
    cur = conn.cursor()
    for a in build_fleet():
        cur.execute("""
            INSERT OR REPLACE INTO assets
                (asset_id, component_id, component_type, machine_id, machine_name, room_id, criticality)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (a["asset_id"], a["component_id"], a["component_type"], a["machine_id"],
              a["machine_name"], a["room_id"], a["criticality"]))
    conn.commit()


# ---------------------------------------------------------------------------
# [1] + [2] ingest raw readings, build model features, store both
# ---------------------------------------------------------------------------

def ingest_readings_batch(conn: sqlite3.Connection, fleet, readings, builders=None):
    """In-memory version of ingestion: takes an already-parsed list of
    reading payloads for ONE hour (as produced by simulate_stream()) instead
    of reading a whole file. `builders` is a dict of {asset_id:
    AssetFeatureBuilder} that the CALLER keeps alive across hours (so rolling
    features carry over correctly) -- pass {} on the first call and reuse
    the same dict on every subsequent call. Returns the (possibly updated)
    builders dict."""
    if builders is None:
        builders = {}
    cur = conn.cursor()
    fleet_by_id = {a["asset_id"]: a for a in fleet}

    for payload in readings:
        asset_id, timestamp = payload["asset_id"], payload["timestamp"]
        asset = fleet_by_id.get(asset_id)
        if asset is None:
            print(f"WARNING: unknown asset_id {asset_id}, skipping", file=sys.stderr)
            continue

        for s in payload["sensors"]:
            cur.execute("""
                INSERT OR REPLACE INTO raw_readings
                    (asset_id, timestamp, sensor_id, sensor_type, measurement, unit, value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (asset_id, timestamp, s["sensor_id"], s["sensor_type"],
                  s["measurement"], s["unit"], s["value"]))

        if asset_id not in builders:
            builders[asset_id] = AssetFeatureBuilder(asset)
        feature_row = builders[asset_id].add(payload["sensors"])
        if feature_row is None:
            continue

        cols = ", ".join(f'"{db_col(c)}"' for c in MODEL_FEATURE_ORDER)
        placeholders = ", ".join("?" for _ in MODEL_FEATURE_ORDER)
        cur.execute(f"""
            INSERT OR REPLACE INTO model_features (asset_id, timestamp, {cols})
            VALUES (?, ?, {placeholders})
        """, (asset_id, timestamp, *feature_row))

    conn.commit()
    return builders


def ingest_and_build_features(conn: sqlite3.Connection, sensor_file: str):
    """File-based ingestion (one-shot CLI use) -- reads a whole .jsonl file
    and delegates each line to the same logic as ingest_readings_batch()."""
    fleet = build_fleet()
    builders = {}
    n_raw, n_features = 0, 0

    with open(sensor_file, encoding="utf-8") as f:
        # group consecutive same-timestamp lines isn't required -- we can
        # just feed readings one at a time, since ingest_readings_batch
        # accepts a list, wrap each line in a one-element list.
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            n_raw += len(payload.get("sensors", []))
            builders = ingest_readings_batch(conn, fleet, [payload], builders)
            n_features += 1

    print(f"Ingested {n_raw} raw sensor readings, built {n_features} feature rows.")


# ---------------------------------------------------------------------------
# [3] read features back out of the DB, preprocess for the model
# ---------------------------------------------------------------------------

def load_and_preprocess(conn: sqlite3.Connection):
    """Returns (X, meta) where X is a preprocessed DataFrame with columns in
    MODEL_FEATURE_ORDER, and meta is a list of (asset_id, timestamp) for each row."""
    import pandas as pd

    db_cols = [db_col(c) for c in MODEL_FEATURE_ORDER]
    select_cols = ", ".join(f'"{c}"' for c in db_cols)
    df = pd.read_sql_query(
        f'SELECT asset_id, timestamp, {select_cols} FROM model_features WHERE scored = 0',
        conn,
    )
    if df.empty:
        return None, []

    meta = list(zip(df["asset_id"], df["timestamp"]))

    # rename DB columns (spaces -> underscores) back to the model's exact names
    rename_map = {db_col(c): c for c in MODEL_FEATURE_ORDER}
    X = df.rename(columns=rename_map)[MODEL_FEATURE_ORDER].copy()

    # --- basic preprocessing, as requested ---
    # one-hot / boolean columns -> int
    onehot_cols = [c for c in MODEL_FEATURE_ORDER
                   if c.startswith(("machine_id_", "component_type_", "criticality_"))]
    X[onehot_cols] = X[onehot_cols].fillna(0).astype(int)
    # numeric feature columns -> float (t_idx -> int, matches your training CSV dtype)
    numeric_cols = ["z_mean", "z_std", "z_max", "z_min",
                     "z_mean_roll5", "z_mean_slope5", "z_max_roll5", "z_max_slope5"]
    X[numeric_cols] = X[numeric_cols].astype(float)
    X["t_idx"] = X["t_idx"].astype(int)

    return X, meta


# ---------------------------------------------------------------------------
# [4] + [5] + [6] load model, predict, label, join asset metadata, write out
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Loads the model once -- variable name is `model`, as requested, so
    you can swap this line for your own loading code if you'd rather load
    it yourself. Exits with a clear message on failure (e.g. xgboost not
    installed) rather than failing deep inside a scoring loop."""
    try:
        import joblib
    except ImportError:
        print("ERROR: joblib not installed. pip install joblib --break-system-packages", file=sys.stderr)
        sys.exit(1)
    try:
        model = joblib.load(model_path)
    except ModuleNotFoundError as e:
        print(f"ERROR: {e}. This model needs xgboost installed: "
              f"pip install xgboost --break-system-packages", file=sys.stderr)
        sys.exit(1)
    return model


def score_pending(conn: sqlite3.Connection, model, out_path: str, append: bool = False) -> int:
    """Scores every unscored row currently in model_features, writes results
    to out_path (append=True for the live orchestrator, False for one-shot
    CLI use) and to the SQLite predictions table. Returns the number of rows
    scored. Safe to call repeatedly -- each call only touches rows with
    scored = 0, so it's the natural unit of work for a live loop."""
    X, meta = load_and_preprocess(conn)
    if X is None:
        return 0

    predicted_rul_days = model.predict(X)  # <-- the actual model call

    cur = conn.cursor()
    mode = "a" if append else "w"
    written = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for (asset_id, timestamp), pred in zip(meta, predicted_rul_days):
            pred = float(max(0.0, pred))
            state = rul_to_state(pred, DEFAULT_STATE_BINS)  # [5]

            row = cur.execute(
                "SELECT component_id, component_type, machine_id, machine_name, room_id "
                "FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if row is None:
                print(f"WARNING: no asset metadata for {asset_id}, skipping", file=sys.stderr)
                continue
            component_id, component_type, machine_id, machine_name, room_id = row

            out = {
                "asset_id": asset_id,
                "timestamp": timestamp,
                "predicted_rul_days": round(pred, 2),
                "current_state": state,
                "component_id": component_id,
                "component_type": component_type,
                "machine_id": machine_id,
                "machine_name": machine_name,
                "room_id": room_id,
            }
            f.write(json.dumps(out) + "\n")

            cur.execute("""
                INSERT OR REPLACE INTO predictions
                    (asset_id, timestamp, predicted_rul_days, current_state,
                     component_id, component_type, machine_id, machine_name, room_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (asset_id, timestamp, out["predicted_rul_days"], state,
                  component_id, component_type, machine_id, machine_name, room_id))
            cur.execute(
                "UPDATE model_features SET scored = 1 WHERE asset_id = ? AND timestamp = ?",
                (asset_id, timestamp),
            )
            written += 1

    conn.commit()
    return written


def score_and_write(conn: sqlite3.Connection, model_path: str, out_path: str):
    """One-shot CLI entry point: load the model fresh, score everything
    pending, overwrite out_path."""
    model = load_model(model_path)
    written = score_pending(conn, model, out_path, append=False)
    if written == 0:
        print("Nothing new to score (model_features has no unscored rows).")
    else:
        print(f"Scored {written} rows -> {out_path} (and SQLite table 'predictions')")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor-file", default=None, help="Path to sensor_batches.jsonl to ingest.")
    parser.add_argument("--db", default="bharati.db", help="SQLite database file.")
    parser.add_argument("--model", default="xgboost_rul_model.joblib")
    parser.add_argument("--out", default="predictions.jsonl")
    parser.add_argument("--skip-ingest", action="store_true",
                         help="Don't ingest a new sensor file, just score whatever is already in the DB.")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    init_db(conn)
    seed_assets(conn)

    if not args.skip_ingest:
        if not args.sensor_file:
            print("ERROR: --sensor-file is required unless --skip-ingest is set.", file=sys.stderr)
            sys.exit(1)
        ingest_and_build_features(conn, args.sensor_file)

    score_and_write(conn, args.model, args.out)
    conn.close()


if __name__ == "__main__":
    main()

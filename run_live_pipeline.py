#!/usr/bin/env python3
"""
run_live_pipeline.py

Wires bharati_sensor_simulator.py and db_pipeline.py together into one
continuously-running loop: generate an hour of data -> ingest into SQLite ->
build model features -> score with the model -> append to predictions.jsonl
-> repeat.

This is what actually makes "run both files and it feeds itself" true --
bharati_sensor_simulator.py and db_pipeline.py are each still fine to run
standalone (e.g. to regenerate a batch file, or to re-score a DB after
swapping models), but neither auto-starts the other on its own. This script
is the connector.

STOPPING (both are graceful -- nothing is lost either way, since every
hour's DB writes are committed before the loop checks for a stop signal):
  - Ctrl+C at any time
  - or create a file at --stop-file (default ./STOP) from anywhere, e.g.:
      touch STOP
    it's picked up and deleted at the end of the hour in progress.

USAGE
  # run until you stop it, fast (no sleeping between simulated hours)
  python3 run_live_pipeline.py --model xgboost_rul_model.joblib

  # run exactly 72 simulated hours then exit on its own
  python3 run_live_pipeline.py --model xgboost_rul_model.joblib --hours 72

  # actually wait ~1 real hour between batches (true live-feed simulation)
  python3 run_live_pipeline.py --model xgboost_rul_model.joblib --real-time
"""

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

from asset_fleet import build_fleet
from bharati_sensor_simulator import simulate_stream
from db_pipeline import init_db, seed_assets, ingest_readings_batch, load_model, score_pending


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="bharati.db")
    parser.add_argument("--model", default="xgboost_rul_model.joblib")
    parser.add_argument("--out", default="predictions.jsonl")
    parser.add_argument("--hours", type=int, default=0, help="0 = run until stopped (default).")
    parser.add_argument("--start", default=None, help="ISO8601 UTC, e.g. 2025-05-01T00:00:00Z")
    parser.add_argument("--assets", default=None, help="Comma-separated asset_ids. Default: all 36.")
    parser.add_argument("--real-time", action="store_true", help="Actually sleep between hours.")
    parser.add_argument("--sleep-seconds", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-file", default="./STOP")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    init_db(conn)
    seed_assets(conn)

    fleet = build_fleet()
    if args.assets:
        wanted = set(a.strip() for a in args.assets.split(","))
        fleet = [a for a in fleet if a["asset_id"] in wanted]
        if not fleet:
            print(f"No matching assets for --assets {args.assets}", file=sys.stderr)
            sys.exit(1)

    model = load_model(args.model)  # exits cleanly with a helpful message if this fails

    start_dt = (datetime.strptime(args.start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if args.start else datetime.now(timezone.utc))

    if os.path.exists(args.stop_file):
        os.remove(args.stop_file)  # clear a stale stop file from a previous run

    builders = {}  # carried across hours so rolling z-features stay correct
    hour_count = 0
    mode_desc = "until stopped" if args.hours == 0 else f"for {args.hours} hours"
    print(f"Running {mode_desc}. Ctrl+C or `touch {args.stop_file}` to stop cleanly.")

    try:
        for timestamp, readings, _labels in simulate_stream(fleet, start_dt, args.hours, args.seed):
            builders = ingest_readings_batch(conn, fleet, readings, builders)
            n = score_pending(conn, model, args.out, append=True)
            hour_count += 1
            print(f"  hour {hour_count}: {timestamp.isoformat()}  scored {n} readings")

            if os.path.exists(args.stop_file):
                print(f"Stop file detected -- stopping cleanly after hour {hour_count}.")
                os.remove(args.stop_file)
                break

            if args.real_time and (args.hours == 0 or hour_count < args.hours):
                time.sleep(args.sleep_seconds)

    except KeyboardInterrupt:
        print(f"\nStopped by Ctrl+C after {hour_count} hours. Everything through the last "
              f"completed hour is already committed in {args.db} and {args.out} -- nothing lost.")
    finally:
        conn.close()

    if hour_count and not os.path.exists(args.stop_file):
        print(f"Done. Ran {hour_count} hours -> {args.db} / {args.out}")


if __name__ == "__main__":
    main()

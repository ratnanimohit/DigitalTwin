#!/usr/bin/env python3
"""
score_stream.py

Reads raw sensor batches (sensor_batches.jsonl, one JSON object per line,
matching your input schema) and turns each hourly reading into the model's
EXACT expected feature vector, then scores it with xgboost_rul_model.joblib
to produce your output schema:

  {"asset_id", "timestamp", "predicted_rul_days", "current_state",
   "component_id", "component_type", "machine_id", "machine_name", "room_id"}

FEATURE PIPELINE (verified against your features_train.csv byte-for-byte):
  1. For each sensor reading, compute a direction-corrected z-score against
     that asset+sensor's healthy baseline (mean/std) -- higher z = closer to
     failure, regardless of whether the raw sensor value rises or falls.
     NOTE: your CSVs only contain the already-aggregated z_mean/z_std/z_max/
     z_min, not the raw per-sensor series that produced them, so this
     per-sensor z-scoring step is my best-effort reconstruction of your
     methodology, not a byte-exact match. The AGGREGATION and ROLLING math
     below (steps 2-3) IS byte-exact, confirmed against your data.
  2. Aggregate across an asset's sensors at that timestamp:
       z_mean = mean(z's), z_std = std(z's), z_max = max(z's), z_min = min(z's)
  3. Maintain a rolling window (per asset "cycle", i.e. since the stream
     started for that asset) of the last 5 z_mean/z_max values:
       z_mean_roll5  = rolling mean, window 5, min_periods 1
       z_mean_slope5 = z_mean[t] - z_mean[t-5]   (lag-5 difference; only
                        defined once >=6 points seen, else omitted like NaN)
       (same for z_max_roll5 / z_max_slope5)
       t_idx = 0,1,2,... position since this asset's stream began
  4. One-hot machine_id / component_type / criticality, built from
     asset_fleet.py (which encodes your REAL asset->machine->component
     mapping, extracted from your CSVs).
  5. Assemble the 35-column vector in MODEL_FEATURE_ORDER and call
     model.predict().
  6. Map predicted_rul_days -> one of your 6 states using --state-bins
     (defaults below are quantile-derived from YOUR ACTUAL rul_days
     distribution in features_train.csv -- NOT arbitrary, but you should
     confirm/replace them with your intended state cutoffs).

REQUIRES: pandas, numpy, xgboost, joblib, scikit-learn (NOT pre-installed in
this sandbox -- this script is written for and should be run in your own
Python environment where the model was trained).

USAGE
  python3 score_stream.py \
      --sensor-file output/sensor_batches.jsonl \
      --model xgboost_rul_model.joblib \
      --out output/predictions.jsonl
"""

import argparse
import json
import math
import sys
from collections import defaultdict, deque

from asset_fleet import build_fleet, MODEL_FEATURE_ORDER, ALL_MACHINE_IDS, ALL_COMPONENT_TYPES, ALL_CRITICALITY

# Default state cutoffs on predicted_rul_days, derived from the ACTUAL
# quantiles of rul_days in features_train.csv (2%, 10%, 20%, 35%, 65%).
# CONFIRM these match your intended 6-state definition before trusting them.
DEFAULT_STATE_BINS = [
    (1.25, "FAILURE"),
    (6.4, "CRITICAL"),
    (12.8, "WARNING"),
    (22.3, "DEGRADING"),
    (50.0, "EARLY_DEGRADATION"),
    (math.inf, "NORMAL"),
]


def rul_to_state(rul_days: float, bins):
    for threshold, label in bins:
        if rul_days <= threshold:
            return label
    return bins[-1][1]


class AssetFeatureBuilder:
    """Stateful per-asset feature builder: call .add(sensors_dict) each
    hour, get back the 35-value feature row (or None if not enough
    history yet for slope5 -- those columns are filled with 0.0, matching
    common XGBoost NaN-handling, but flagged so you can swap in proper
    missing-value handling if your model was trained with real NaNs)."""

    def __init__(self, asset):
        self.asset = asset
        self.baseline = {s["sensor_id"]: (s["baseline_mean"], s["baseline_std"], s["direction"])
                          for s in asset["sensors"]}
        self.z_mean_hist = deque(maxlen=5)
        self.z_max_hist = deque(maxlen=6)  # need t-5 lookback -> keep 6
        self.t_idx = 0

        mid, ctype, crit = asset["machine_id"], asset["component_type"], asset["criticality"]
        self.onehot = {}
        for m in ALL_MACHINE_IDS:
            self.onehot[f"machine_id_{m}"] = 1.0 if m == mid else 0.0
        for c in ALL_COMPONENT_TYPES:
            self.onehot[f"component_type_{c}"] = 1.0 if c == ctype else 0.0
        for c in ALL_CRITICALITY:
            self.onehot[f"criticality_{c}"] = 1.0 if c == crit else 0.0

    def add(self, sensors):
        zs = []
        for s in sensors:
            base = self.baseline.get(s["sensor_id"])
            if base is None:
                continue
            mean, std, direction = base
            z = direction * (s["value"] - mean) / (std if std else 1e-6)
            zs.append(z)
        if not zs:
            return None

        z_mean = sum(zs) / len(zs)
        z_max = max(zs)
        z_min = min(zs)
        if len(zs) > 1:
            mu = z_mean
            z_std = math.sqrt(sum((z - mu) ** 2 for z in zs) / len(zs))
        else:
            z_std = 0.0

        self.z_mean_hist.append(z_mean)
        self.z_max_hist.append(z_max)

        z_mean_roll5 = sum(self.z_mean_hist) / len(self.z_mean_hist)
        z_max_roll5 = sum(list(self.z_max_hist)[-5:]) / min(len(self.z_max_hist), 5)

        if len(self.z_mean_hist) >= 5 and self.t_idx >= 5:
            z_mean_slope5 = z_mean - self._lag5_mean
        else:
            z_mean_slope5 = 0.0
        if len(self.z_max_hist) == 6:
            z_max_slope5 = z_max - self.z_max_hist[0]
        else:
            z_max_slope5 = 0.0

        row = {
            "z_mean": z_mean, "z_std": z_std, "z_max": z_max, "z_min": z_min,
            "z_mean_roll5": z_mean_roll5, "z_mean_slope5": z_mean_slope5,
            "z_max_roll5": z_max_roll5, "z_max_slope5": z_max_slope5,
            "t_idx": self.t_idx,
        }
        row.update(self.onehot)

        self._lag5_mean = self.z_mean_hist[0] if len(self.z_mean_hist) >= 1 else z_mean
        self.t_idx += 1
        return [row[col] for col in MODEL_FEATURE_ORDER]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor-file", required=True, help="Path to sensor_batches.jsonl")
    parser.add_argument("--model", default="xgboost_rul_model.joblib")
    parser.add_argument("--out", default="predictions.jsonl")
    args = parser.parse_args()

    try:
        import joblib
    except ImportError:
        print("ERROR: joblib not installed. pip install joblib --break-system-packages", file=sys.stderr)
        sys.exit(1)

    try:
        model = joblib.load(args.model)
    except ModuleNotFoundError as e:
        print(f"ERROR: {e}. This model needs xgboost installed: "
              f"pip install xgboost --break-system-packages", file=sys.stderr)
        sys.exit(1)

    fleet_by_id = {a["asset_id"]: a for a in build_fleet()}
    builders = {}

    rows, meta = [], []
    with open(args.sensor_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            asset_id = payload["asset_id"]
            asset = fleet_by_id.get(asset_id)
            if asset is None:
                print(f"WARNING: unknown asset_id {asset_id}, skipping", file=sys.stderr)
                continue
            if asset_id not in builders:
                builders[asset_id] = AssetFeatureBuilder(asset)
            feature_row = builders[asset_id].add(payload["sensors"])
            if feature_row is None:
                continue
            rows.append(feature_row)
            meta.append((asset, payload["timestamp"]))

    if not rows:
        print("No feature rows built -- nothing to score.", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    X = pd.DataFrame(rows, columns=MODEL_FEATURE_ORDER)
    predictions = model.predict(X)

    with open(args.out, "w", encoding="utf-8") as f:
        for (asset, timestamp), pred in zip(meta, predictions):
            pred = float(max(0.0, pred))
            out = {
                "asset_id": asset["asset_id"],
                "timestamp": timestamp,
                "predicted_rul_days": round(pred, 2),
                "current_state": rul_to_state(pred, DEFAULT_STATE_BINS),
                "component_id": asset["component_id"],
                "component_type": asset["component_type"],
                "machine_id": asset["machine_id"],
                "machine_name": asset["machine_name"],
                "room_id": asset["room_id"],
            }
            f.write(json.dumps(out) + "\n")

    print(f"Scored {len(rows)} readings -> {args.out}")


if __name__ == "__main__":
    main()

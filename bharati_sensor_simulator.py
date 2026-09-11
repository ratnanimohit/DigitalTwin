#!/usr/bin/env python3
"""
bharati_sensor_simulator_v2.py

Simulates hourly raw sensor batches for your REAL 36-asset fleet (see
asset_fleet.py -- extracted from your features_train.csv / features_test.csv,
not guessed). Output matches your input schema exactly:

  {"asset_id": ..., "timestamp": ..., "sensors": [{sensor_id, sensor_type,
   measurement, unit, value}, ...]}

On top of each asset's own degradation trend (health 0->1), sensor values now
vary the way a real station's plant actually would, grounded in published
data on Bharati Station (Larsemann Hills, East Antarctica, 69.4 S):

  - SEASONAL OUTDOOR TEMPERATURE: monthly mean ~0 degC in Dec/Jan (austral
    summer) down to ~-19 degC in May-Sep (austral winter) -- from a published
    meteorological study of the station (see sources in chat). Modeled as a
    smooth annual sinusoid, not a step function.
  - SEASONAL OCCUPANCY: the station houses ~23-25 people in winter and up to
    ~46-47 in summer (NCPOR figures). Water/galley/wastewater load scales
    with this, not with a fixed baseline.
  - HEATING DEMAND: the 3 CHP (combined heat & power) units -- confirmed as
    Bharati's actual diesel-fired CHP plant in the same study -- run harder
    as outdoor temperature drops, independent of any fault.
  - DIURNAL CYCLE: a 24h activity rhythm (meals, work hours) modulates
    water/galley/pump load on top of the seasonal trend; the server room
    stays close to flat.

None of this is measured sensor data (I don't have that) -- it's a
physically-reasoned layer on top of real climate/occupancy figures, so the
variation you see is directionally realistic rather than arbitrary noise.
Adjust outdoor_temp_c() / load_factor() if you have real station logs.

USAGE
  python3 bharati_sensor_simulator_v2.py --hours 72 --out ./output
  python3 bharati_sensor_simulator_v2.py --hours 72 --assets A002,A008,A014 --out ./output
  python3 bharati_sensor_simulator_v2.py --hours 24 --real-time --endpoint http://localhost:8000/ingest

Also writes ground_truth_labels.jsonl (true_health/true_state/true_rul_days)
per asset per hour, for validating your model's predictions against reality.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

from asset_fleet import build_fleet

try:
    import requests
except ImportError:
    requests = None


STATE_THRESHOLDS = [
    (0.00, "NORMAL"),
    (0.15, "EARLY_DEGRADATION"),
    (0.35, "DEGRADING"),
    (0.60, "WARNING"),
    (0.85, "CRITICAL"),
    (1.00, "FAILURE"),
]


def outdoor_temp_c(dt: datetime) -> float:
    """Smooth annual cycle: ~0 degC around mid-Jan (summer), ~-19 degC around
    mid-Jul (winter) -- matches the published monthly means for Bharati
    Station. Southern hemisphere, so trough is in the Jun-Sep window."""
    doy = dt.timetuple().tm_yday
    mean_c, amplitude = -9.5, 9.5
    phase = 2 * math.pi * (doy - 196) / 365.25  # trough centered ~mid-July
    return mean_c - amplitude * math.cos(phase)


def station_occupancy(dt: datetime) -> float:
    """Smooth annual cycle: ~23-25 crew in winter, ~46-47 in summer
    (NCPOR figures), same phase as outdoor_temp_c (peak occupancy = summer)."""
    doy = dt.timetuple().tm_yday
    mid, amplitude = 35.0, 12.0
    phase = 2 * math.pi * (doy - 196) / 365.25
    return mid - amplitude * math.cos(phase + math.pi)  # peaks opposite the temp trough


def diurnal_factor(dt: datetime, amplitude: float = 0.08, peak_hour: float = 13.0) -> float:
    """24h activity rhythm -- meals and work hours -- peaking mid-afternoon."""
    hour = dt.hour + dt.minute / 60.0
    return 1.0 + amplitude * math.sin(2 * math.pi * (hour - (peak_hour - 6)) / 24.0)


def load_factor(asset: dict, dt: datetime) -> float:
    """Operational load multiplier, independent of degradation -- this is
    what makes 'normal' readings vary hour to hour and season to season
    instead of hovering at a fixed baseline."""
    machine = asset["machine_id"]
    if machine in ("CHP01", "CHP02", "CHP03"):
        outdoor = outdoor_temp_c(dt)
        heating_demand = max(0.0, (-5.0 - outdoor)) / 25.0  # ramps up as it gets colder than -5C
        return (0.95 + heating_demand) * diurnal_factor(dt, amplitude=0.03)
    if machine in ("PUMP01", "RO01", "WW01", "DISH01"):
        occupancy_factor = station_occupancy(dt) / 35.0  # normalized to the annual mean crew size
        return occupancy_factor * diurnal_factor(dt, amplitude=0.12)
    if machine == "SERVER01":
        return diurnal_factor(dt, amplitude=0.02)
    return diurnal_factor(dt)


def health_to_state(health: float) -> str:
    state = "NORMAL"
    for threshold, label in STATE_THRESHOLDS:
        if health >= threshold:
            state = label
    return state


def health_to_rul_days(health: float, horizon_days: float) -> float:
    return round(max(0.0, 1.0 - health) * horizon_days, 2)


class OUNoise:
    """Ornstein-Uhlenbeck mean-reverting noise: real sensors drift smoothly
    from one reading to the next rather than jumping independently each
    hour. theta controls how fast it snaps back to 0, sigma controls the
    per-step kick -- both ideas taken from the reference generator's
    ou_noise() (see generator_core.py in the uploaded synthetic dataset)."""
    def __init__(self, sigma, theta=0.25, seed=None):
        self.theta = theta
        self.sigma = sigma
        self.x = 0.0
        self.rng = random.Random(seed)

    def step(self):
        self.x += self.theta * (0.0 - self.x) + self.sigma * self.rng.gauss(0, 1)
        return self.x


class AssetTrajectory:
    def __init__(self, horizon_days, start_health=0.0, noise_scale=0.004, seed=None):
        self.horizon_days = horizon_days
        self.health = start_health
        self.noise_scale = noise_scale
        self.rng = random.Random(seed)
        # one OU noise process per sensor, seeded off the same rng
        self.ou = {}

    def step(self, hours=1.0):
        base_step = hours / (self.horizon_days * 24.0)
        accel = 1.0 + max(0.0, self.rng.gauss(0, 0.4))
        jitter = self.rng.gauss(0, self.noise_scale)
        self.health = min(1.0, max(0.0, self.health + base_step * accel + jitter))
        return self.health

    def ou_for(self, sensor):
        sid = sensor["sensor_id"]
        if sid not in self.ou:
            self.ou[sid] = OUNoise(sigma=sensor["noise_std"], theta=0.25,
                                    seed=self.rng.randint(0, 10_000))
        return self.ou[sid].step()


def sensor_value(sensor, health, ou_value, lf, outdoor):
    base = sensor["baseline_mean"]

    if sensor["measurement"] in ("Load", "Flow", "Current"):
        base = base * lf
    elif sensor["measurement"] == "Vibration":
        base = base * (0.85 + 0.15 * lf)
    elif sensor["measurement"] == "Temperature":
        cooling_term = (outdoor - (-9.5)) * 0.2
        load_heat_term = (lf - 1.0) * base * 0.15
        base = base + cooling_term + load_heat_term

    drift = sensor["direction"] * sensor["degradation_gain"] * health
    value = base + drift + ou_value   # ou_value replaces plain white noise
    if sensor["measurement"] in ("Vibration", "Speed", "Flow", "Level", "Current"):
        value = max(0.0, value)
    return round(value, 3)


def build_reading_payload(asset, timestamp, health, traj):
    lf = load_factor(asset, timestamp)
    outdoor = outdoor_temp_c(timestamp)
    return {
        "asset_id": asset["asset_id"],
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sensors": [
            {
                "sensor_id": s["sensor_id"],
                "sensor_type": s["sensor_type"],
                "measurement": s["measurement"],
                "unit": s["unit"],
                "value": sensor_value(s, health, traj.ou_for(s), lf, outdoor),
            }
            for s in asset["sensors"]
        ],
    }


def build_label_payload(asset, timestamp, health, horizon_days):
    return {
        "asset_id": asset["asset_id"],
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "true_health": round(health, 4),
        "true_state": health_to_state(health),
        "true_rul_days": health_to_rul_days(health, horizon_days),
        "component_id": asset["component_id"],
        "component_type": asset["component_type"],
        "machine_id": asset["machine_id"],
        "machine_name": asset["machine_name"],
        "room_id": asset["room_id"],
    }


def send_batch(readings, endpoint, out_dir):
    if endpoint:
        if requests is None:
            print("ERROR: --endpoint given but 'requests' isn't installed. "
                  "pip install requests --break-system-packages", file=sys.stderr)
            sys.exit(1)
        for payload in readings:
            try:
                resp = requests.post(endpoint, json=payload, timeout=10)
                print(f"POST {payload['asset_id']} @ {payload['timestamp']} -> {resp.status_code}")
            except Exception as e:
                print(f"POST FAILED for {payload['asset_id']}: {e}", file=sys.stderr)
    else:
        path = os.path.join(out_dir, "sensor_batches.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for payload in readings:
                f.write(json.dumps(payload) + "\n")


def write_labels(labels, out_dir):
    path = os.path.join(out_dir, "ground_truth_labels.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for payload in labels:
            f.write(json.dumps(payload) + "\n")


def simulate_stream(fleet, start_dt, hours, seed=42):
    """Generator: yields (timestamp, readings_list, labels_list) one hour at
    a time. hours=0 means run forever (caller stops it, e.g. via Ctrl+C or
    a stop file). This is the reusable core -- both the standalone CLI below
    and run_live_pipeline.py's orchestrator loop use this same function, so
    a live-fed run and a batch-file run generate identically."""
    rng = random.Random(seed)
    trajectories = {
        a["asset_id"]: AssetTrajectory(
            horizon_days=rng.uniform(10, 90),
            start_health=rng.uniform(0.0, 0.08),
            noise_scale=0.004,
            seed=rng.randint(0, 10_000),
        )
        for a in fleet
    }

    i = 0
    while hours == 0 or i < hours:
        timestamp = start_dt + timedelta(hours=i)
        readings, labels = [], []
        for asset in fleet:
            traj = trajectories[asset["asset_id"]]
            health = traj.step(1.0)
            readings.append(build_reading_payload(asset, timestamp, health, traj))
            labels.append(build_label_payload(asset, timestamp, health, traj.horizon_days))
        yield timestamp, readings, labels
        i += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--start", type=str, default=None, help="ISO8601 UTC, e.g. 2025-05-01T00:00:00Z")
    parser.add_argument("--out", type=str, default="./output")
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--assets", type=str, default=None,
                         help="Comma-separated asset_ids to simulate, e.g. A002,A008. Default: all 36.")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fleet = build_fleet()
    if args.assets:
        wanted = set(a.strip() for a in args.assets.split(","))
        fleet = [a for a in fleet if a["asset_id"] in wanted]
        if not fleet:
            print(f"No matching assets for --assets {args.assets}", file=sys.stderr)
            sys.exit(1)

    start_dt = (datetime.strptime(args.start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if args.start else datetime.now(timezone.utc))

    print(f"Simulating {args.hours} hourly batches for {len(fleet)} assets, "
          f"starting {start_dt.isoformat()}. Output -> {args.out if not args.endpoint else args.endpoint}")

    for i, (timestamp, readings, labels) in enumerate(simulate_stream(fleet, start_dt, args.hours, args.seed)):
        send_batch(readings, args.endpoint, args.out)
        write_labels(labels, args.out)

        if i % 24 == 0:
            sample = ", ".join(f"{l['asset_id']}:{l['true_state']}" for l in labels[:6])
            print(f"  hour {i}: {timestamp.isoformat()}  [{sample}, ...]")

        if args.real_time and i < args.hours - 1:
            time.sleep(args.sleep_seconds)

    print("Done." if args.endpoint else
          f"Done. Wrote {args.out}/sensor_batches.jsonl and {args.out}/ground_truth_labels.jsonl")


if __name__ == "__main__":
    main()

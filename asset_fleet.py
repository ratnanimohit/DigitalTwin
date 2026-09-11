"""
asset_fleet.py

Real fleet definition, extracted directly from features_train.csv /
features_test.csv (asset_id -> machine_id -> component_type -> criticality ->
failure_mode). This is NOT guessed -- it's the exact 36-asset fleet your
model was trained on.

machine_name / room_id are NOT present in your CSVs, so those are my
best-effort labels consistent with a research-station power/utilities plant
(3x CHP unit, pump station, RO water plant, wastewater plant, galley
dishwasher, server room). Rename freely in MACHINE_META below to match your
actual station layout.

Per-component-type SENSOR_TEMPLATES are also my best-effort reconstruction:
your CSVs only contain the already-aggregated z_mean/z_std/z_max/z_min
features, not the underlying raw multi-sensor readings, so I don't have the
exact sensors/units/baselines that produced them. These are physically
sensible defaults for each component type -- swap in your real sensor specs
if you have them (e.g. from a P&ID or sensor manifest).
"""

# asset_id -> (machine_id, component_type, criticality, failure_mode)
# Extracted verbatim from features_train.csv
ASSET_LOOKUP = {
    "A001": ("CHP01", "Engine", "Critical", "Engine overheating and degradation"),
    "A002": ("CHP01", "Bearing", "Critical", "Bearing wear"),
    "A003": ("CHP01", "Lubrication", "Critical", "Low oil pressure"),
    "A004": ("CHP01", "Cooling", "Critical", "Cooling degradation"),
    "A005": ("CHP01", "Generator", "Critical", "Electrical degradation"),
    "A006": ("CHP01", "Fuel", "Critical", "Fuel flow degradation"),
    "A007": ("CHP02", "Engine", "Critical", "Engine overheating and degradation"),
    "A008": ("CHP02", "Bearing", "Critical", "Bearing wear"),
    "A009": ("CHP02", "Lubrication", "Critical", "Low oil pressure"),
    "A010": ("CHP02", "Cooling", "Critical", "Cooling degradation"),
    "A011": ("CHP02", "Generator", "Critical", "Electrical degradation"),
    "A012": ("CHP02", "Fuel", "Critical", "Fuel flow degradation"),
    "A013": ("CHP03", "Engine", "Critical", "Engine overheating and degradation"),
    "A014": ("CHP03", "Bearing", "Critical", "Bearing wear"),
    "A015": ("CHP03", "Lubrication", "Critical", "Low oil pressure"),
    "A016": ("CHP03", "Cooling", "Critical", "Cooling degradation"),
    "A017": ("CHP03", "Generator", "Critical", "Electrical degradation"),
    "A018": ("CHP03", "Fuel", "Critical", "Fuel flow degradation"),
    "A019": ("PUMP01", "Pump", "Critical", "Pump performance degradation"),
    "A020": ("PUMP01", "Motor", "Critical", "Motor degradation"),
    "A021": ("PUMP01", "Bearing", "High", "Bearing wear"),
    "A022": ("RO01", "Pump", "Critical", "Pump performance degradation"),
    "A023": ("RO01", "Membrane", "Critical", "Membrane fouling"),
    "A024": ("RO01", "Filter", "High", "Filter clogging"),
    "A025": ("RO01", "Valve", "High", "Valve degradation and leakage"),
    "A026": ("WW01", "Pump", "Critical", "Pump performance degradation"),
    "A027": ("WW01", "Filter", "High", "Filter clogging"),
    "A028": ("WW01", "Valve", "High", "Valve degradation and leakage"),
    "A029": ("DISH01", "Drive", "Critical", "Mechanical drive degradation"),
    "A030": ("DISH01", "Drive", "Critical", "Mechanical drive degradation"),
    "A031": ("DISH01", "Gearbox", "High", "Gear wear"),
    "A032": ("DISH01", "Motor", "High", "Motor degradation"),
    "A033": ("SERVER01", "Fan", "High", "Fan degradation"),
    "A034": ("SERVER01", "Storage", "Critical", "Storage degradation"),
    "A035": ("SERVER01", "Power Supply", "Critical", "Power supply degradation"),
    "A036": ("SERVER01", "Cooling", "High", "Thermal management degradation"),
}

# machine_id -> (machine_name, room_id) -- EDIT to match your real station layout
MACHINE_META = {
    "CHP01":    ("CHP Unit 1", "R001"),
    "CHP02":    ("CHP Unit 2", "R001"),
    "CHP03":    ("CHP Unit 3", "R001"),
    "PUMP01":   ("Utility Pump Station 1", "R002"),
    "RO01":     ("Reverse Osmosis Plant", "R003"),
    "WW01":     ("Wastewater Treatment Plant", "R004"),
    "DISH01":   ("Galley Dishwasher Unit", "R005"),
    "SERVER01": ("Server Room Rack", "R006"),
}

# component_type -> list of sensor templates, derived from equipment_specs.md
# (realistic industry-standard specs for the equipment classes Bharati-type
# stations use: containerized diesel CHP gensets, seawater RO desal, tracking
# satellite dish, rack servers -- NOT confirmed Bharati hardware, but grounded
# in real operating ranges rather than arbitrary numbers).
# Each: (sensor_type, measurement, unit, baseline_mean, baseline_std,
#        direction [+1 = rises with degradation, -1 = falls], degradation_gain, noise_std)
# baseline_mean = midpoint of the spec's "normal range"; degradation_gain is
# sized so health=1.0 pushes the value just past the spec's "critical" band.
SENSOR_TEMPLATES = {
    # --- CHP diesel genset subsystems (Engine/Bearing/Lubrication/Cooling/
    # Generator/Fuel components map 1:1 onto equipment_specs.md section 1) ---
    "Engine": [
        ("Exhaust Gas Temperature", "Temperature", "degC", 415.0, 20.0, 1, 140.0, 4.0),
        ("Engine Speed", "Speed", "rpm", 1500.0, 5.0, -1, 260.0, 4.0),
        ("Crankcase Blow-by Pressure", "Pressure", "mbar", 6.0, 2.0, 1, 22.0, 0.8),
    ],
    "Bearing": [
        ("Main Bearing Temperature", "Temperature", "degC", 80.0, 3.0, 1, 30.0, 0.8),
        ("Bearing Vibration RMS", "Vibration", "mm/s", 2.2, 0.5, 1, 6.0, 0.15),
    ],
    "Lubrication": [
        ("Oil Pressure", "Pressure", "bar", 5.25, 0.4, -1, 3.5, 0.12),
        ("Oil Temperature", "Temperature", "degC", 95.0, 3.0, 1, 28.0, 0.8),
    ],
    "Cooling": [
        ("Coolant Outlet Temperature", "Temperature", "degC", 86.0, 2.0, 1, 17.0, 0.6),
        ("Coolant Flow Rate", "Flow", "m3/h", 50.0, 2.0, -1, 18.0, 0.7),
    ],
    "Generator": [
        ("Stator Winding Temperature", "Temperature", "degC", 115.0, 5.0, 1, 45.0, 1.0),
        ("Generator Bearing Temperature", "Temperature", "degC", 72.0, 4.0, 1, 32.0, 0.9),
    ],
    "Fuel": [
        ("Fuel Inlet Pressure", "Pressure", "bar", 4.0, 0.3, -1, 2.7, 0.1),
        ("Fuel Consumption Rate", "Flow", "L/h", 85.0, 4.0, 1, 25.0, 1.2),
    ],
    # --- RO / freshwater / wastewater plant (equipment_specs.md section 2) ---
    "Pump": [
        ("Discharge Pressure", "Pressure", "bar", 60.0, 2.0, 1, 15.0, 0.6),
        ("Motor Current", "Current", "A", 40.0, 1.0, 1, 10.0, 0.4),
        ("Pump RPM", "Speed", "rpm", 1450.0, 10.0, -1, 260.0, 4.0),
    ],
    "Motor": [
        ("Motor Current Draw", "Current", "A", 40.0, 1.2, 1, 9.0, 0.4),
        ("Winding Temperature", "Temperature", "degC", 70.0, 3.0, 1, 25.0, 0.8),
    ],
    "Membrane": [
        ("Membrane Differential Pressure", "Pressure", "bar", 1.5, 0.15, 1, 1.8, 0.06),
        ("Permeate Flow", "Flow", "L/h", 625.0, 10.0, -1, 220.0, 4.0),
        ("Permeate Conductivity", "Conductivity", "uS/cm", 300.0, 20.0, 1, 250.0, 8.0),
    ],
    "Filter": [
        ("Pre-filter Differential Pressure", "Pressure", "bar", 0.55, 0.05, 1, 0.9, 0.02),
        ("Flow Rate", "Flow", "L/h", 600.0, 15.0, -1, 200.0, 5.0),
    ],
    "Valve": [
        ("Distribution Valve Pressure", "Pressure", "bar", 4.0, 0.3, -1, 3.0, 0.1),
        ("Seat Leakage Rate", "Flow", "L/min", 0.05, 0.02, 1, 2.0, 0.03),
    ],
    # --- Satellite tracking dish (equipment_specs.md section 3) ---
    "Drive": [
        ("Azimuth Motor Current", "Current", "A", 2.5, 0.4, 1, 3.5, 0.15),
        ("Tracking Position Error", "Angle", "deg", 0.005, 0.002, 1, 0.12, 0.001),
    ],
    "Gearbox": [
        ("Gearbox Grease Temperature", "Temperature", "degC", 20.0, 5.0, 1, 40.0, 1.2),
        ("Az/El Bearing Vibration RMS", "Vibration", "mm/s", 1.5, 0.4, 1, 3.5, 0.12),
    ],
    # ("Motor" reused for dish elevation drive below, since component_type is shared)
    # --- Server / IT infrastructure (equipment_specs.md section 4) ---
    "Fan": [
        ("CPU Fan RPM", "Speed", "rpm", 5000.0, 500.0, 1, 4500.0, 150.0),
        ("Chassis Fan RPM", "Speed", "rpm", 4000.0, 400.0, 1, 3500.0, 120.0),
    ],
    "Storage": [
        ("SSD/NVMe Temperature", "Temperature", "degC", 40.0, 5.0, 1, 35.0, 1.2),
        ("HDD Temperature", "Temperature", "degC", 35.0, 4.0, 1, 30.0, 1.0),
    ],
    "Power Supply": [
        ("PSU Temperature", "Temperature", "degC", 42.0, 4.0, 1, 33.0, 1.0),
        ("12V Rail Voltage", "Voltage", "V", 12.0, 0.05, -1, 0.65, 0.02),
    ],
}
# Dish elevation-drive assets (A032, component_type "Motor") reuse a
# dish-appropriate motor profile instead of the RO/pump motor one above --
# handled via ASSET-level override, not a second "Motor" key (dict keys
# must be unique). See DISH_MOTOR_OVERRIDE below, applied in build_fleet().
DISH_MOTOR_OVERRIDE = [
    ("Elevation Motor Current", "Current", "A", 1.8, 0.3, 1, 3.0, 0.12),
    ("Motor Winding Temperature", "Temperature", "degC", 60.0, 6.0, 1, 55.0, 1.5),
]


def build_fleet():
    """Returns a list of asset dicts, ready for the simulator."""
    fleet = []
    for asset_id, (machine_id, component_type, criticality, failure_mode) in sorted(ASSET_LOOKUP.items()):
        machine_name, room_id = MACHINE_META[machine_id]
        component_id = "C" + asset_id[1:]  # A002 -> C002, matches your example
        # A032 is the satellite dish's elevation-drive Motor -- use the
        # dish-appropriate profile instead of the RO/pump Motor profile
        # (same component_type label, different physical equipment).
        template = DISH_MOTOR_OVERRIDE if (asset_id == "A032") else SENSOR_TEMPLATES[component_type]
        sensors = []
        for i, (sensor_type, measurement, unit, mean, std, direction, gain, noise) in enumerate(
                template, start=1):
            sensors.append(dict(
                sensor_id=f"S{asset_id[1:]}{i}",   # e.g. A002 sensor 1 -> S0021
                sensor_type=sensor_type,
                measurement=measurement,
                unit=unit,
                baseline_mean=mean,
                baseline_std=std,
                direction=direction,
                degradation_gain=gain,
                noise_std=noise,
            ))
        fleet.append(dict(
            asset_id=asset_id,
            component_id=component_id,
            component_type=component_type,
            machine_id=machine_id,
            machine_name=machine_name,
            room_id=room_id,
            criticality=criticality,
            failure_mode=failure_mode,
            sensors=sensors,
        ))
    return fleet


# Exact 35-column feature order the model expects, extracted byte-for-byte
# from xgboost_rul_model.joblib's embedded feature_names list. DO NOT reorder.
MODEL_FEATURE_ORDER = [
    "z_mean", "z_std", "z_max", "z_min",
    "z_mean_roll5", "z_mean_slope5", "z_max_roll5", "z_max_slope5", "t_idx",
    "machine_id_CHP01", "machine_id_CHP02", "machine_id_CHP03", "machine_id_DISH01",
    "machine_id_PUMP01", "machine_id_RO01", "machine_id_SERVER01", "machine_id_WW01",
    "component_type_Bearing", "component_type_Cooling", "component_type_Drive",
    "component_type_Engine", "component_type_Fan", "component_type_Filter",
    "component_type_Fuel", "component_type_Gearbox", "component_type_Generator",
    "component_type_Lubrication", "component_type_Membrane", "component_type_Motor",
    "component_type_Power Supply", "component_type_Pump", "component_type_Storage",
    "component_type_Valve",
    "criticality_Critical", "criticality_High",
]

ALL_MACHINE_IDS = ["CHP01", "CHP02", "CHP03", "DISH01", "PUMP01", "RO01", "SERVER01", "WW01"]
ALL_COMPONENT_TYPES = [
    "Bearing", "Cooling", "Drive", "Engine", "Fan", "Filter", "Fuel", "Gearbox",
    "Generator", "Lubrication", "Membrane", "Motor", "Power Supply", "Pump",
    "Storage", "Valve",
]
ALL_CRITICALITY = ["Critical", "High"]  # third category (if any) is the dropped reference level

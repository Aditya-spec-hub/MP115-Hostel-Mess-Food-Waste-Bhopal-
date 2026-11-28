import pandas as pd

from src.mp115.cleaning import (drop_duplicates, fix_stuck_sensor,
                                fix_target_typos, fix_unit_mixup)

P = {
    "unit_column": "plate_size_cm", "unit_group_column": "hostel",
    "unit_group_value": "Boys Hostel 1", "unit_factor": 10,
    "target": "waste_kg", "typo_threshold": 100, "typo_factor": 10,
    "stuck_column": "attendance_pct",
    "stuck_start": "2026-03-02", "stuck_end": "2026-03-18",
}


def test_drop_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2]})
    assert len(drop_duplicates(df)) == 2


def test_fix_unit_mixup_only_changes_one_hostel():
    df = pd.DataFrame({"hostel": ["Boys Hostel 1", "Girls Hostel"],
                       "plate_size_cm": [270.0, 27.0]})
    out = fix_unit_mixup(df, P)
    assert out["plate_size_cm"].tolist() == [27.0, 27.0]


def test_fix_target_typos():
    df = pd.DataFrame({"waste_kg": [40.0, 400.0]})
    assert fix_target_typos(df, P)["waste_kg"].tolist() == [40.0, 40.0]


def test_fix_stuck_sensor_replaces_frozen_rows():
    df = pd.DataFrame({"date": ["2026-02-01", "2026-03-05", "2026-04-01"],
                       "attendance_pct": [60, 88, 80]})
    out = fix_stuck_sensor(df, P)
    assert out.loc[1, "attendance_pct"] == 70.0

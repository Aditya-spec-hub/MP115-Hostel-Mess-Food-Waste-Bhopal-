"""Cleaning steps for MP115. Every setting comes from params.yaml."""
import pandas as pd
import yaml


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def drop_duplicates(df):
    """Remove exact duplicate rows (keep the first copy)."""
    return df.drop_duplicates().reset_index(drop=True)


def fix_unit_mixup(df, p):
    """Divide the wrongly recorded column by the factor for one hostel."""
    df = df.copy()
    mask = df[p["unit_group_column"]] == p["unit_group_value"]
    df.loc[mask, p["unit_column"]] = df.loc[mask, p["unit_column"]] / p["unit_factor"]
    return df


def fix_target_typos(df, p):
    """Targets above the threshold were typed 10 times too big."""
    df = df.copy()
    mask = df[p["target"]] > p["typo_threshold"]
    df.loc[mask, p["target"]] = df.loc[mask, p["target"]] / p["typo_factor"]
    return df


def fix_stuck_sensor(df, p):
    """Replace the frozen readings with the median of the normal rows."""
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    stuck = (dates >= p["stuck_start"]) & (dates <= p["stuck_end"])
    col = p["stuck_column"]
    df[col] = df[col].astype(float)
    df.loc[stuck, col] = df.loc[~stuck, col].median()
    return df


def clean(df, p):
    df = drop_duplicates(df)
    df = fix_unit_mixup(df, p)
    df = fix_target_typos(df, p)
    df = fix_stuck_sensor(df, p)
    return df

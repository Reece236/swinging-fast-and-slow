from __future__ import annotations

import math
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_baseballsavant_csv(csv_path: str) -> pd.DataFrame:
    """Load Baseball Savant pitch-level data.

    Expects the file used in the R pipeline (data/baseballsavant.csv).
    """
    df = pd.read_csv(csv_path)
    return df


def add_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper fields mirroring the R scripts.

    - batter_side_id: paste0(batter_id, bat_side)
    - plate_x_ref: reflect plate_x for LHB so inside/outside align
    - is_contact: foul or fair-in-play
    - is_fair: fair-in-play
    """
    out = df.copy()
    if "batter_id" in out.columns and "bat_side" in out.columns:
        out["batter_side_id"] = out["batter_id"].astype(str) + out["bat_side"].astype(str)
    else:
        raise ValueError("Data must include columns `batter_id` and `bat_side`.")

    if "plate_x" not in out.columns:
        raise ValueError("Data must include column `plate_x`.")
    out["plate_x_ref"] = np.where(out["bat_side"] == "R", out["plate_x"], -out["plate_x"])

    if "description" not in out.columns:
        raise ValueError("Data must include column `description`.")
    out["is_contact"] = out["description"].isin(["foul", "hit_into_play"]).astype(bool)
    out["is_fair"] = (out["description"] == "hit_into_play").astype(bool)
    return out


def recreate_squared_up(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate MLB's squared-up metric following the R logic.

    Requires columns: ax, ay, az, bx, by, bz, cy, bat_speed, launch_speed, description
    If any are missing, the function warns and returns a copy with `squared_up=False`.
    """
    required = {"ax", "ay", "az", "bx", "by", "bz", "cy", "bat_speed", "launch_speed", "description"}
    missing = [c for c in required if c not in df.columns]
    out = df.copy()
    if missing:
        warnings.warn(
            f"Missing columns for squared_up computation: {missing}. Setting squared_up=False.",
            RuntimeWarning,
        )
        out["squared_up"] = False
        return out

    # plate_y = 17/12; plate_time solves y(t) = plate_y given quadratic motion params (ay, by, cy)
    plate_y = 17.0 / 12.0
    # Protect the discriminant; if invalid, mark squared_up False for that row
    by = out["by"].to_numpy()
    ay = out["ay"].to_numpy()
    cy = out["cy"].to_numpy()
    ax = out["ax"].to_numpy()
    bx = out["bx"].to_numpy()
    az = out["az"].to_numpy()
    bz = out["bz"].to_numpy()

    # Quadratic is y(t) = (ay/2) t^2 + by t + cy; solve for t crossing plate_y
    a = ay / 2.0
    b = by
    c = cy - plate_y
    disc = b * b - 4.0 * a * c
    valid = disc >= 0
    t = np.full(len(out), np.nan, dtype=float)
    t[valid] = (-b[valid] - np.sqrt(disc[valid])) / (2.0 * a[valid])

    # 0.6818182 factor from R to convert ft/s to mph
    plate_speed = 0.6818182 * np.sqrt((ax * t + bx) ** 2 + (ay * t + by) ** 2 + (az * t + bz) ** 2)

    launch_speed = out["launch_speed"].to_numpy()
    bat_speed = out["bat_speed"].to_numpy()
    is_fair = out["description"].eq("hit_into_play").to_numpy()

    theoretical_max = 1.23 * bat_speed + 0.23 * plate_speed
    ratio = np.divide(launch_speed, theoretical_max, out=np.zeros_like(launch_speed, dtype=float), where=(theoretical_max > 0))
    squared_up = np.where(is_fair & ~np.isnan(launch_speed), ratio > 0.8, False)
    out["squared_up"] = squared_up.astype(bool)
    return out


def remove_partial_swings(df: pd.DataFrame) -> pd.DataFrame:
    """Remove bunt attempts and checked swings following the R threshold (bat_speed > 50)."""
    out = df.copy()
    des = out.get("des", pd.Series(["" for _ in range(len(out))]))
    description = out["description"].astype(str)
    is_bunt = description.str.contains("bunt", case=False) | ((description == "hit_into_play") & des.str.contains(" bunt", case=False))
    is_checked = out.get("bat_speed").fillna(0) <= 50
    mask = (~is_bunt) & (~is_checked)
    return out.loc[mask].copy()


def get_primary_fastballs(df: pd.DataFrame) -> pd.DataFrame:
    """Identify each pitcher's primary fastball among FF/SI/FC."""
    if "pitch_type" not in df.columns:
        raise ValueError("Data must include column `pitch_type`.")
    fast = df[df["pitch_type"].isin(["FF", "SI", "FC"])].copy()
    counts = fast.groupby(["pitcher_id", "pitch_type"]).size().rename("n").reset_index()
    idx = counts.sort_values(["pitcher_id", "n"], ascending=[True, False]).drop_duplicates("pitcher_id")
    idx = idx[["pitcher_id", "pitch_type"]].rename(columns={"pitch_type": "primary_fastball"})
    return idx


def build_intention_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the R pipeline to construct the intention modeling dataset.

    Steps:
    - filter balls < 4 and strikes < 3
    - add basic fields
    - recreate squared_up
    - filter to primary fastballs and squared_up
    - remove partial swings
    """
    data = df.copy()
    data = data[(data["balls"] < 4) & (data["strikes"] < 3)].copy()
    data = add_basic_fields(data)
    data = recreate_squared_up(data)
    data = remove_partial_swings(data)

    prim = get_primary_fastballs(data)
    data = data.merge(prim, on="pitcher_id", how="left")
    data["is_primary"] = data["pitch_type"].eq(data["primary_fastball"]) & data["pitch_type"].isin(["FF", "SI", "FC"]) 
    intent = data[(data["is_primary"]) & (data["squared_up"])].copy()
    return intent


def ensure_required_columns(df: pd.DataFrame, cols: Tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

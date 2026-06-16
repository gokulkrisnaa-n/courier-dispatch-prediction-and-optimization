"""Load and clean the raw LaDe pickup CSV into a tidy training frame.

Responsibilities (offline only — never imported by the serving path):
  * parse the year-less ``MM-DD HH:MM:SS`` timestamps using the configured year
  * keep only physically plausible Shanghai order coordinates
  * flag/null out missing-or-invalid courier accept GPS (``accept_loc_missing``)
  * build the regression target ``pickup_duration_min`` and trim it

The output is one row per pickup order with parsed datetimes, clean coordinates,
the missing-GPS flag, and the target. Feature engineering lives in features.py so
that the exact same transform can run at serve time.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config, load_config

logger = logging.getLogger(__name__)

# Raw datetime columns share the year-less "MM-DD HH:MM:SS" format.
_DATETIME_COLS = [
    "accept_time",
    "time_window_start",
    "time_window_end",
    "pickup_time",
    "accept_gps_time",
    "pickup_gps_time",
]

# Order location is mandatory; courier accept GPS is optional (often missing).
_ORDER_COORDS = ("lng", "lat")
_ACCEPT_COORDS = ("accept_gps_lng", "accept_gps_lat")


def _parse_dt(series: pd.Series, year: int) -> pd.Series:
    """Parse 'MM-DD HH:MM:SS' by prefixing the assumed year. Blanks -> NaT."""
    s = series.astype("string").str.strip()
    s = s.where(s.str.len() > 0, other=pd.NA)
    return pd.to_datetime(f"{year}-" + s, format="%Y-%m-%d %H:%M:%S", errors="coerce")


def load_raw(cfg: Config | None = None, *, nrows: int | None = None) -> pd.DataFrame:
    """Read the raw CSV with sane dtypes. No cleaning yet."""
    cfg = cfg or load_config()
    # IDs are categorical codes, not numbers we do arithmetic on -> read as string.
    dtype = {
        "order_id": "string",
        "region_id": "string",
        "city": "string",
        "courier_id": "string",
        "aoi_id": "string",
        "aoi_type": "string",
        "ds": "string",
    }
    logger.info("Reading %s", cfg.raw_csv)
    df = pd.read_csv(cfg.raw_csv, dtype=dtype, nrows=nrows)
    logger.info("Read %d raw rows", len(df))
    return df


def clean(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Parse datetimes, drop bad order coords, flag missing GPS, build target."""
    cfg = cfg or load_config()
    df = df.copy()
    n0 = len(df)

    # --- 1. parse datetimes ---
    for col in _DATETIME_COLS:
        if col in df.columns:
            df[col] = _parse_dt(df[col], cfg.assumed_year)

    # --- 2. order coordinates must be valid and inside Shanghai ---
    lng, lat = df["lng"].astype(float), df["lat"].astype(float)
    bbox = cfg.bbox
    order_ok = (
        ~((lng == 0) & (lat == 0))
        & lng.between(bbox.lng_min, bbox.lng_max)
        & lat.between(bbox.lat_min, bbox.lat_max)
    )
    df = df.loc[order_ok].copy()
    logger.info("Dropped %d rows with invalid order coords", n0 - len(df))

    # --- 3. courier accept GPS: flag missing/invalid, null the coords ---
    a_lng = pd.to_numeric(df["accept_gps_lng"], errors="coerce")
    a_lat = pd.to_numeric(df["accept_gps_lat"], errors="coerce")
    accept_valid = (
        a_lng.notna() & a_lat.notna()
        & ~((a_lng == 0) & (a_lat == 0))
        & a_lng.between(bbox.lng_min, bbox.lng_max)
        & a_lat.between(bbox.lat_min, bbox.lat_max)
    )
    df["accept_loc_missing"] = (~accept_valid).astype("int8")
    # Keep only validated accept coords; the rest become NaN (feature handles it).
    df["accept_gps_lng"] = a_lng.where(accept_valid)
    df["accept_gps_lat"] = a_lat.where(accept_valid)
    logger.info("accept_loc_missing share: %.1f%%", 100 * df["accept_loc_missing"].mean())

    # --- 4. target: pickup_duration in minutes ---
    df = _build_target(df, cfg)

    # --- 5. order by accept_time (point-in-time ordering for history features) ---
    df = df.sort_values("accept_time", kind="mergesort").reset_index(drop=True)
    logger.info("Clean frame: %d rows (%.1f%% of raw)", len(df), 100 * len(df) / n0)
    return df


def _build_target(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """pickup_duration_min = pickup_time - accept_time, trimmed per config."""
    dur = (df["pickup_time"] - df["accept_time"]).dt.total_seconds() / 60.0
    df[cfg.target_name] = dur

    df = df.loc[df[cfg.target_name].notna()].copy()
    if cfg.drop_nonpositive:
        df = df.loc[df[cfg.target_name] > 0].copy()
    # Drop the long upper tail (data-entry outliers, orders left for hours).
    hi = df[cfg.target_name].quantile(cfg.upper_pct_clip)
    df = df.loc[df[cfg.target_name] <= hi].copy()
    logger.info(
        "Target trimmed at p%.0f = %.1f min; %d rows remain",
        100 * cfg.upper_pct_clip, hi, len(df),
    )
    return df


def load_clean(
    cfg: Config | None = None, *, sample: bool = False
) -> pd.DataFrame:
    """End-to-end: read raw CSV -> clean frame. `sample` honors dev.sample_rows."""
    cfg = cfg or load_config()
    df = load_raw(cfg)
    if sample and cfg.sample_rows:
        df = df.sample(
            n=min(cfg.sample_rows, len(df)), random_state=cfg.sample_seed
        ).reset_index(drop=True)
        logger.info("Sampled %d rows for dev iteration", len(df))
    return clean(df, cfg)

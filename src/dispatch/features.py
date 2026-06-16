"""THE shared feature module — identical transform for training and serving.

Two layers, kept separate on purpose:

  * ``add_courier_history`` is an OFFLINE, point-in-time-safe transform that
    derives each courier's rolling-average pickup duration and active workload
    from the *past only*. It is used during training. At serve time the
    dispatcher maintains these aggregates in its own state and supplies them.

  * ``build_features`` is a PURE, row-wise transform with no cross-row state.
    Given a frame that already contains the raw inputs plus the two courier
    history columns, it produces the exact model-ready feature matrix. Training
    calls it after ``add_courier_history``; the API calls it per (courier, order)
    pair. Because it is stateless it cannot leak — that is what the
    point-in-time test pins.

The reference time for all time-derived features is ``accept_time``. At serve
time the caller sets ``accept_time`` to the snapshot's ``as_of_time``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load_config

EARTH_RADIUS_KM = 6371.0088

# Columns build_features expects to already exist on the input frame.
RAW_INPUT_COLS = [
    "accept_time",
    "lng",
    "lat",
    "accept_gps_lng",
    "accept_gps_lat",
    "accept_loc_missing",
    "time_window_start",
    "time_window_end",
    "region_id",
    "aoi_id",
    "aoi_type",
    "city",
    "courier_rolling_avg_min",
    "courier_active_load",
]


def haversine_km(
    lng1: pd.Series | np.ndarray,
    lat1: pd.Series | np.ndarray,
    lng2: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km. NaN in any input propagates to NaN out."""
    lng1, lat1, lng2, lat2 = (np.asarray(x, dtype="float64") for x in (lng1, lat1, lng2, lat2))
    rlng1, rlat1, rlng2, rlat2 = map(np.radians, (lng1, lat1, lng2, lat2))
    dlng = rlng2 - rlng1
    dlat = rlat2 - rlat1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _cyclical(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Encode an integer-ish cycle as (sin, cos) so 23:00 sits next to 00:00."""
    radians = 2 * np.pi * values / period
    return np.sin(radians), np.cos(radians)


# --------------------------------------------------------------------------- #
# Offline: courier history (point-in-time safe)
# --------------------------------------------------------------------------- #
def add_courier_history(
    df: pd.DataFrame, cfg: Config | None = None, *, global_avg: float | None = None
) -> tuple[pd.DataFrame, float]:
    """Add ``courier_rolling_avg_min`` and ``courier_active_load`` using past only.

    Must be called on a frame sorted by ``accept_time``. Returns the frame and the
    global average pickup duration used to fill cold-start couriers (persist it so
    the dispatcher can apply the same default for unseen couriers).
    """
    cfg = cfg or load_config()
    df = df.sort_values("accept_time", kind="mergesort").reset_index(drop=True)
    target = cfg.target_name

    if global_avg is None:
        global_avg = float(df[target].mean())

    # Rolling mean of the courier's OWN prior pickups (shift(1) drops the current
    # row -> strictly past). Cold rows (no prior) fall back to the global mean.
    grp = df.groupby("courier_id", sort=False)[target]
    rolling = grp.transform(
        lambda s: s.shift(1).rolling(cfg.rolling_window, min_periods=cfg.rolling_min_periods).mean()
    )
    df["courier_rolling_avg_min"] = rolling.fillna(global_avg)

    # Active load: orders this courier already accepted but not yet picked up at
    # the moment they accept the current order. Per courier, count via searchsorted
    # on sorted accept/pickup times -> O(n log n), point-in-time exact.
    df["courier_active_load"] = _active_load(df)
    return df, global_avg


def _active_load(df: pd.DataFrame) -> pd.Series:
    """For each row: # of the courier's orders with accept < t and pickup >= t."""
    out = np.zeros(len(df), dtype="int32")
    for _, idx in df.groupby("courier_id", sort=False).groups.items():
        rows = df.loc[idx]
        accepts = np.sort(rows["accept_time"].values)
        pickups = np.sort(rows["pickup_time"].values)
        t = rows["accept_time"].values
        # accepted strictly before t, minus already picked up before t.
        accepted_before = np.searchsorted(accepts, t, side="left")
        picked_before = np.searchsorted(pickups, t, side="left")
        out[df.index.get_indexer(idx)] = (accepted_before - picked_before).astype("int32")
    return pd.Series(out, index=df.index, name="courier_active_load")


# --------------------------------------------------------------------------- #
# Shared: pure row-wise feature builder
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Pure transform: raw inputs (+ courier history cols) -> model feature matrix.

    Returns a DataFrame with exactly ``cfg.feature_columns``; categorical columns
    carry pandas ``category`` dtype for XGBoost's native categorical support.
    """
    cfg = cfg or load_config()
    accept = pd.to_datetime(df["accept_time"])
    win_start = pd.to_datetime(df["time_window_start"])
    win_end = pd.to_datetime(df["time_window_end"])

    out = pd.DataFrame(index=df.index)

    # --- location ---
    out["lng"] = df["lng"].astype("float64")
    out["lat"] = df["lat"].astype("float64")
    out["accept_loc_missing"] = df["accept_loc_missing"].astype("int8")
    out["distance_to_pickup_km"] = haversine_km(
        df["accept_gps_lng"], df["accept_gps_lat"], df["lng"], df["lat"]
    )

    # --- cyclical time of accept ---
    hour = accept.dt.hour + accept.dt.minute / 60.0
    out["hour_sin"], out["hour_cos"] = _cyclical(hour, 24)
    dow = accept.dt.dayofweek
    out["dow_sin"], out["dow_cos"] = _cyclical(dow, 7)
    out["is_weekend"] = (dow >= 5).astype("int8")
    out["is_peak_hour"] = accept.dt.hour.isin(cfg.peak_hours).astype("int8")
    out["minutes_since_midnight"] = (accept.dt.hour * 60 + accept.dt.minute).astype("int32")

    # --- pickup window ---
    out["window_duration_min"] = (win_end - win_start).dt.total_seconds() / 60.0
    out["slack_min"] = (win_end - accept).dt.total_seconds() / 60.0

    # --- courier history (supplied; computed offline or by the dispatcher) ---
    out["courier_rolling_avg_min"] = df["courier_rolling_avg_min"].astype("float64")
    out["courier_active_load"] = df["courier_active_load"].astype("float64")

    # --- categorical codes ---
    for col in cfg.categorical_features:
        out[col] = df[col].astype("string").astype("category")

    return out[cfg.feature_columns]

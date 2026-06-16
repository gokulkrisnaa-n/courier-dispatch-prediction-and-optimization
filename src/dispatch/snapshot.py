"""Domain types and point-in-time reconstruction of the dispatch state.

A ``Snapshot`` is the input to the assignment layer: the free couriers, the open
orders, and the wall-clock ``as_of_time`` that plays the role of ``accept_time``
for feature building.

The live streaming dispatcher maintains this state incrementally from Kafka
events. ``reconstruct_at`` does the same thing offline from the cleaned historic
frame — used to seed the simulation and, crucially, by the tests so the snapshot
logic can be checked without standing up a broker. Both produce the same types.

Courier history aggregates (rolling-average pickup duration, active load) and the
courier's last-known location are computed strictly from the past (events with
timestamps < t), so a snapshot never peeks at the future.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, load_config


@dataclass
class Order:
    order_id: str
    lng: float
    lat: float
    region_id: str
    aoi_id: str
    aoi_type: str
    city: str
    window_start: datetime
    window_end: datetime
    available_time: datetime          # when the order entered the open set


@dataclass
class Courier:
    courier_id: str
    lng: float | None                 # last-known location; None if never reported
    lat: float | None
    rolling_avg_min: float            # running mean of recent pickup durations
    active_load: int                  # accepted-but-not-picked count
    loc_missing: bool = False

    def __post_init__(self) -> None:
        if self.lng is None or self.lat is None or (
            isinstance(self.lng, float) and math.isnan(self.lng)
        ):
            self.loc_missing = True
            self.lng = None
            self.lat = None


@dataclass
class Snapshot:
    as_of_time: datetime
    couriers: list[Courier] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)

    def is_assignable(self) -> bool:
        return bool(self.couriers) and bool(self.orders)


# --------------------------------------------------------------------------- #
# Offline reconstruction from the cleaned historic frame
# --------------------------------------------------------------------------- #
def _courier_state(
    rows: pd.DataFrame, t: datetime, cfg: Config, default_rolling_avg: float
) -> Courier:
    """Point-in-time state for one courier from their own order history."""
    cid = rows["courier_id"].iloc[0]
    accepts = rows["accept_time"]
    pickups = rows["pickup_time"]

    # Active load: accepted before t, not yet picked at t.
    active_load = int(((accepts < t) & (pickups >= t)).sum())

    # Rolling average over pickups completed before t (strictly past).
    past = rows.loc[pickups < t].sort_values("pickup_time")
    if len(past):
        recent = past[cfg.target_name].tail(cfg.rolling_window)
        rolling = float(recent.mean())
        last = past.iloc[-1]                       # most recent completed pickup
        lng, lat = float(last["lng"]), float(last["lat"])
    else:
        rolling = default_rolling_avg
        lng = lat = None

    return Courier(
        courier_id=str(cid), lng=lng, lat=lat,
        rolling_avg_min=rolling, active_load=active_load,
    )


def reconstruct_at(
    df: pd.DataFrame,
    t: datetime,
    cfg: Config | None = None,
    *,
    default_rolling_avg: float,
    region_id: str | None = None,
) -> Snapshot:
    """Reconstruct the open orders and free couriers at time ``t``.

    ``df`` is the cleaned frame from ``data.load_clean`` (parsed datetimes,
    ``courier_id``, coordinates, window columns, target). ``default_rolling_avg``
    is the cold-start fill (use ``DispatchModel.global_avg``).

    * Open order: available (``snapshot.available_from`` column <= t) and not yet
      picked up (``pickup_time`` > t).
    * Free courier: active within ``snapshot.recent_window_min`` of t and, if a
      ``courier_capacity`` is set, below it.
    """
    cfg = cfg or load_config()
    snap_cfg = cfg.snapshot
    available_from = snap_cfg.get("available_from", "time_window_start")
    recent_window = timedelta(minutes=float(snap_cfg.get("recent_window_min", 120)))
    capacity = snap_cfg.get("courier_capacity")

    if region_id is not None:
        df = df.loc[df["region_id"].astype("string") == str(region_id)]

    # --- open orders ---
    avail = df[available_from]
    open_mask = (avail <= t) & (df["pickup_time"] > t)
    orders = [
        Order(
            order_id=str(r.order_id), lng=float(r.lng), lat=float(r.lat),
            region_id=str(r.region_id), aoi_id=str(r.aoi_id),
            aoi_type=str(r.aoi_type), city=str(r.city),
            window_start=r.time_window_start, window_end=r.time_window_end,
            available_time=getattr(r, available_from),
        )
        for r in df.loc[open_mask].itertuples(index=False)
    ]

    # --- free couriers: recently active near t ---
    active = df.loc[
        (df["accept_time"] <= t)
        & (df["accept_time"] >= t - recent_window)
        | ((df["pickup_time"] <= t) & (df["pickup_time"] >= t - recent_window))
    ]
    couriers: list[Courier] = []
    for cid, rows in active.groupby("courier_id", sort=False):
        c = _courier_state(rows, t, cfg, default_rolling_avg)
        if capacity is not None and c.active_load >= int(capacity):
            continue
        couriers.append(c)

    return Snapshot(as_of_time=t, couriers=couriers, orders=orders)

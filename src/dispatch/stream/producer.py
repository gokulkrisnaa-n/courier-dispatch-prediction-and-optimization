"""Replay the historic LaDe pickups as a time-ordered event stream.

Each order in the cleaned frame expands into up to three events:

  * ``courier_location`` at accept_time — the courier reports where they were
    when they accepted (from accept_gps; skipped when that GPS is missing).
  * ``order_available`` at time_window_start — the order opens for pickup.
  * ``order_picked_up`` at pickup_time — carries the courier and the pickup
    location (the order's coordinates), closing the order.

All events across all orders are sorted by ``event_time`` and replayed to the bus
with time compression: a gap of N simulated seconds is slept for
``N / time_compression`` real seconds (set very high, or sleep=False, to fire as
fast as possible — used by tests and the local simulation).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .bus import Bus, Event

logger = logging.getLogger(__name__)


def _iso(ts: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(ts).isoformat()


def build_events(df: pd.DataFrame, cfg: Config | None = None) -> list[Event]:
    """Expand the cleaned frame into a flat, time-sorted list of event dicts."""
    cfg = cfg or load_config()
    events: list[Event] = []
    for r in df.itertuples(index=False):
        # courier_location (only when accept GPS was recorded)
        if pd.notna(r.accept_gps_lng) and pd.notna(r.accept_gps_lat):
            events.append({
                "event_type": "courier_location",
                "event_time": _iso(r.accept_time),
                "courier_id": str(r.courier_id),
                "lng": float(r.accept_gps_lng), "lat": float(r.accept_gps_lat),
            })
        # order_available
        events.append({
            "event_type": "order_available",
            "event_time": _iso(r.time_window_start),
            "order_id": str(r.order_id),
            "lng": float(r.lng), "lat": float(r.lat),
            "aoi_id": str(r.aoi_id), "aoi_type": str(r.aoi_type),
            "window_start": _iso(r.time_window_start),
            "window_end": _iso(r.time_window_end),
            "city": str(r.city), "region_id": str(r.region_id),
        })
        # order_picked_up (carries courier GPS == pickup location)
        events.append({
            "event_type": "order_picked_up",
            "event_time": _iso(r.pickup_time),
            "order_id": str(r.order_id),
            "courier_id": str(r.courier_id),
            "lng": float(r.lng), "lat": float(r.lat),
        })
    events.sort(key=lambda e: e["event_time"])
    logger.info("Built %d events from %d orders", len(events), len(df))
    return events


def replay(
    bus: Bus,
    events: list[Event],
    *,
    time_compression: float | None = None,
    sleep: bool = True,
    signal_done: bool = True,
    cfg: Config | None = None,
) -> int:
    """Publish events to the bus in order, optionally sleeping the compressed gaps."""
    cfg = cfg or load_config()
    comp = time_compression if time_compression is not None else float(
        cfg.stream.get("time_compression", 60.0)
    )
    prev: datetime | None = None
    for ev in events:
        t = datetime.fromisoformat(ev["event_time"])
        if sleep and prev is not None:
            gap = (t - prev).total_seconds() / max(comp, 1e-9)
            if gap > 0:
                time.sleep(min(gap, 5.0))      # cap any single sleep at 5s
        bus.publish(ev)
        prev = t
    if signal_done:
        bus.done()
    logger.info("Replayed %d events (compression=%.0f, sleep=%s)", len(events), comp, sleep)
    return len(events)


def main() -> None:
    """Service entrypoint: replay historic events onto the configured broker.

    Driven by env vars (set in docker-compose): KAFKA_BOOTSTRAP, DISPATCH_TOPIC,
    REPLAY_NROWS, REPLAY_REGION (optional), REPLAY_COMPRESSION.
    """
    import os

    from ..data import clean, load_raw
    from .bus import KafkaBus

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    bootstrap = os.getenv("KAFKA_BOOTSTRAP", cfg.stream.get("kafka_bootstrap", "localhost:9092"))
    topic = os.getenv("DISPATCH_TOPIC", cfg.stream.get("topic", "pickup_events"))
    nrows = int(os.getenv("REPLAY_NROWS", "120000"))
    region = os.getenv("REPLAY_REGION")
    comp = float(os.getenv("REPLAY_COMPRESSION", str(cfg.stream.get("time_compression", 60.0))))

    df = clean(load_raw(cfg, nrows=nrows), cfg)
    if region:
        df = df[df["region_id"].astype("string") == region]
    logger.info("Producer: %d orders -> broker %s topic %s", len(df), bootstrap, topic)

    bus = KafkaBus(bootstrap, topic)
    try:
        replay(bus, build_events(df, cfg), time_compression=comp, sleep=True, cfg=cfg)
    finally:
        bus.close()


if __name__ == "__main__":
    main()

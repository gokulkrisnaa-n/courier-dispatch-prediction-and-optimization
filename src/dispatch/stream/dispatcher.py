"""Streaming dispatcher: consume events, maintain state, tick the assignment.

The dispatcher is the runtime heart of the system. It consumes the event stream
and keeps the live picture the optimizer needs:

  * open orders (available, not yet picked/assigned),
  * free couriers with their last-known location, and
  * the per-courier running aggregates the model wants — recent average pickup
    duration and current workload — maintained here, in the dispatcher's state,
    exactly as the API contract expects them to arrive.

On each ``tick`` it snapshots that state, hands it to a *solver* (a local model or
the FastAPI ``/score`` endpoint), runs the Hungarian assignment, and books the
results. When a booked order is later picked up it joins predicted vs realized
duration into a performance log for the monitoring layer.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from ..assignment import AssignmentResult, ScoredPair, assign_from_pairs, solve
from ..config import Config, load_config
from ..model import DispatchModel
from ..snapshot import Courier, Order, Snapshot
from .bus import Bus, Event

logger = logging.getLogger(__name__)

# A solver turns a snapshot into an assignment (local model or remote API).
Solver = Callable[[Snapshot], AssignmentResult]


@dataclass
class CourierRuntime:
    courier_id: str
    lng: float | None = None
    lat: float | None = None
    durations: deque = field(default_factory=lambda: deque(maxlen=20))
    active_orders: set[str] = field(default_factory=set)
    last_seen: datetime | None = None

    def rolling_avg(self, default: float) -> float:
        return sum(self.durations) / len(self.durations) if self.durations else default


@dataclass
class PerfRecord:
    assign_time: datetime
    pickup_time: datetime
    courier_id: str
    order_id: str
    predicted_min: float
    realized_min: float


class Dispatcher:
    def __init__(
        self,
        solver: Solver,
        global_avg: float,
        cfg: Config | None = None,
        *,
        perf_log_path: str | Path | None = None,
        ops_metrics_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.solver = solver
        self.global_avg = global_avg
        snap_cfg = self.cfg.snapshot
        self._recent = timedelta(minutes=float(snap_cfg.get("recent_window_min", 120)))
        self._capacity = snap_cfg.get("courier_capacity")
        self._roll_window = self.cfg.rolling_window

        self.open_orders: dict[str, Order] = {}
        self.couriers: dict[str, CourierRuntime] = {}
        self.pending: dict[str, tuple[str, datetime, float]] = {}  # order -> (courier, t, pred)
        self.now: datetime | None = None

        self.perf_log: list[PerfRecord] = []
        self.tick_latencies_ms: list[float] = []
        self.n_ticks = 0
        self.n_assignments = 0
        self.n_late_pickups = 0          # picked up before we could assign

        # Live-write hooks: when set, perf records and ops metrics are persisted
        # incrementally (as they happen) rather than only at shutdown, so a
        # dashboard polling these paths sees a running stack update in real time.
        self.perf_log_path = Path(perf_log_path) if perf_log_path else None
        self.ops_metrics_path = Path(ops_metrics_path) if ops_metrics_path else None
        if self.perf_log_path is not None:
            self.perf_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.perf_log_path.write_text("")     # start this run's log clean
        self._flush_ops_metrics()

    # ------------------------------------------------------------------ #
    # event handling
    # ------------------------------------------------------------------ #
    def process(self, event: Event) -> None:
        t = datetime.fromisoformat(event["event_time"])
        self.now = t if self.now is None else max(self.now, t)
        kind = event["event_type"]
        if kind == "order_available":
            self._on_order_available(event)
        elif kind == "courier_location":
            self._on_courier_location(event, t)
        elif kind == "order_picked_up":
            self._on_order_picked_up(event, t)

    def _courier(self, courier_id: str) -> CourierRuntime:
        c = self.couriers.get(courier_id)
        if c is None:
            c = CourierRuntime(courier_id=courier_id, durations=deque(maxlen=self._roll_window))
            self.couriers[courier_id] = c
        return c

    def _on_order_available(self, e: Event) -> None:
        oid = e["order_id"]
        if oid in self.pending:           # already booked in a prior tick
            return
        ws = datetime.fromisoformat(e["window_start"])
        we = datetime.fromisoformat(e["window_end"])
        self.open_orders[oid] = Order(
            order_id=oid, lng=e["lng"], lat=e["lat"],
            region_id=e["region_id"], aoi_id=e["aoi_id"], aoi_type=e["aoi_type"],
            city=e["city"], window_start=ws, window_end=we, available_time=ws,
        )

    def _on_courier_location(self, e: Event, t: datetime) -> None:
        c = self._courier(e["courier_id"])
        c.lng, c.lat, c.last_seen = e["lng"], e["lat"], t

    def _flush_ops_metrics(self) -> None:
        if self.ops_metrics_path is not None:
            from ..monitoring import save_ops_metrics
            save_ops_metrics(self.ops_metrics(), self.ops_metrics_path)

    def _on_order_picked_up(self, e: Event, t: datetime) -> None:
        oid, cid = e["order_id"], e["courier_id"]
        self.open_orders.pop(oid, None)
        c = self._courier(cid)
        c.lng, c.lat, c.last_seen = e["lng"], e["lat"], t   # courier now at pickup loc

        booked = self.pending.pop(oid, None)
        if booked is None:
            self.n_late_pickups += 1
            self._flush_ops_metrics()        # late_pickups counter just moved — surface it live
            return
        b_cid, assign_time, predicted = booked
        realized = (t - assign_time).total_seconds() / 60.0
        bc = self._courier(b_cid)
        bc.active_orders.discard(oid)
        if realized > 0:
            record = PerfRecord(assign_time, t, b_cid, oid, predicted, realized)
            bc.durations.append(realized)
            self.perf_log.append(record)
            if self.perf_log_path is not None:
                from ..monitoring import save_perf_log
                save_perf_log([record.__dict__], self.perf_log_path)
            self._flush_ops_metrics()        # perf_records counter just moved — surface it live

    # ------------------------------------------------------------------ #
    # snapshot + tick
    # ------------------------------------------------------------------ #
    def snapshot(self, now: datetime | None = None) -> Snapshot:
        now = now or self.now
        couriers: list[Courier] = []
        for c in self.couriers.values():
            if c.last_seen is None or (now - c.last_seen) > self._recent:
                continue
            load = len(c.active_orders)
            if self._capacity is not None and load >= int(self._capacity):
                continue
            couriers.append(Courier(
                courier_id=c.courier_id, lng=c.lng, lat=c.lat,
                rolling_avg_min=c.rolling_avg(self.global_avg), active_load=load,
            ))
        return Snapshot(as_of_time=now, couriers=couriers, orders=list(self.open_orders.values()))

    def tick(self, now: datetime | None = None) -> AssignmentResult | None:
        snap = self.snapshot(now)
        if not snap.is_assignable():
            return None
        start = time.perf_counter()
        result = self.solver(snap)
        self.tick_latencies_ms.append((time.perf_counter() - start) * 1000.0)
        for cid, oid, pred in result.pairs:                 # book the assignments
            self.pending[oid] = (cid, snap.as_of_time, pred)
            self._courier(cid).active_orders.add(oid)
            self.open_orders.pop(oid, None)
        self.n_ticks += 1
        self.n_assignments += len(result.pairs)
        self._flush_ops_metrics()
        return result

    # ------------------------------------------------------------------ #
    # run loop + reporting
    # ------------------------------------------------------------------ #
    def run(
        self,
        bus: Bus,
        *,
        tick_every_sec: float = 300.0,
        poll_timeout: float = 1.0,
        max_idle_polls: int | None = None,
    ) -> None:
        """Consume the bus, ticking on simulated-time boundaries.

        Stops when the stream signals end-of-stream (``bus.is_done()`` — used by
        the in-memory replay) or, on a real broker, after ``max_idle_polls``
        consecutive empty polls (so a service container exits once the producer
        finishes). ``None`` means run until interrupted.
        """
        next_tick: datetime | None = None
        idle = 0
        while True:
            event = bus.poll(timeout=poll_timeout)
            if event is None:
                if bus.is_done():
                    break
                idle += 1
                if max_idle_polls is not None and idle >= max_idle_polls:
                    break
                continue
            idle = 0
            self.process(event)
            if next_tick is None:
                next_tick = self.now + timedelta(seconds=tick_every_sec)
            while self.now >= next_tick:
                self.tick(next_tick)
                next_tick += timedelta(seconds=tick_every_sec)
        self.tick()   # final sweep of anything still open
        logger.info("Dispatcher done: %d ticks, %d assignments, %d perf records",
                    self.n_ticks, self.n_assignments, len(self.perf_log))

    def performance_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.perf_log])

    def save_perf_log(self, path) -> None:
        """Persist the predicted-vs-realized join as JSONL for the dashboard."""
        from ..monitoring import save_perf_log
        save_perf_log([r.__dict__ for r in self.perf_log], path)

    def ops_metrics(self) -> dict[str, float]:
        lat = self.tick_latencies_ms
        return {
            "ticks": self.n_ticks,
            "assignments": self.n_assignments,
            "avg_assignments_per_tick": self.n_assignments / max(self.n_ticks, 1),
            "avg_tick_latency_ms": (sum(lat) / len(lat)) if lat else 0.0,
            "max_tick_latency_ms": max(lat) if lat else 0.0,
            "late_pickups": self.n_late_pickups,
            "perf_records": len(self.perf_log),
        }


# --------------------------------------------------------------------------- #
# solver factories
# --------------------------------------------------------------------------- #
def local_solver(model: DispatchModel, cfg: Config | None = None) -> Solver:
    cfg = cfg or load_config()
    return lambda snap: solve(snap, lambda frame: model.predict_from_raw(frame, cfg), cfg)


def api_solver(base_url: str, cfg: Config | None = None) -> Solver:
    """Solve via the FastAPI /score endpoint, reshaping pairs into a result."""
    import httpx

    cfg = cfg or load_config()

    def _solve(snap: Snapshot) -> AssignmentResult:
        courier_ids = [c.courier_id for c in snap.couriers]
        order_ids = [o.order_id for o in snap.orders]
        payload = {
            "as_of_time": snap.as_of_time.isoformat(),
            "couriers": [
                {"courier_id": c.courier_id, "lng": c.lng, "lat": c.lat,
                 "rolling_avg_min": c.rolling_avg_min, "active_load": c.active_load}
                for c in snap.couriers
            ],
            "orders": [
                {"order_id": o.order_id, "lng": o.lng, "lat": o.lat,
                 "region_id": o.region_id, "aoi_id": o.aoi_id, "aoi_type": o.aoi_type,
                 "city": o.city, "window_start": o.window_start.isoformat(),
                 "window_end": o.window_end.isoformat()}
                for o in snap.orders
            ],
        }
        resp = httpx.post(f"{base_url.rstrip('/')}/score", json=payload, timeout=30.0)
        resp.raise_for_status()
        scored = [
            ScoredPair(p["courier_id"], p["order_id"], p["predicted_minutes"])
            for p in resp.json()["pairs"]
        ]
        return assign_from_pairs(scored, courier_ids, order_ids)

    return _solve


# --------------------------------------------------------------------------- #
# self-contained in-memory simulation (producer -> bus -> dispatcher)
# --------------------------------------------------------------------------- #
def simulate(
    df: pd.DataFrame,
    model: DispatchModel,
    cfg: Config | None = None,
    *,
    tick_every_sec: float = 300.0,
) -> Dispatcher:
    """Wire producer -> InMemoryBus -> dispatcher and run the whole replay."""
    from .bus import InMemoryBus
    from .producer import build_events, replay

    cfg = cfg or load_config()
    bus = InMemoryBus()
    events = build_events(df, cfg)
    replay(bus, events, sleep=False, cfg=cfg)          # fire all events instantly
    disp = Dispatcher(local_solver(model, cfg), model.global_avg, cfg)
    disp.run(bus, tick_every_sec=tick_every_sec)
    return disp


def main() -> None:
    """Service entrypoint: consume the broker and dispatch via the API solver.

    Env vars (docker-compose): KAFKA_BOOTSTRAP, DISPATCH_TOPIC, API_URL,
    TICK_EVERY_SEC, MAX_IDLE_POLLS.
    """
    import os

    from ..model import DispatchModel
    from .bus import KafkaBus

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    bootstrap = os.getenv("KAFKA_BOOTSTRAP", cfg.stream.get("kafka_bootstrap", "localhost:9092"))
    topic = os.getenv("DISPATCH_TOPIC", cfg.stream.get("topic", "pickup_events"))
    api_url = os.getenv("API_URL", "http://localhost:8000")
    tick = float(os.getenv("TICK_EVERY_SEC", "300"))
    max_idle = int(os.getenv("MAX_IDLE_POLLS", "60"))

    monitoring_dir = cfg.artifacts_dir / "monitoring"
    perf_path = monitoring_dir / "perf_log.jsonl"
    ops_path = monitoring_dir / "ops_metrics.json"

    model = DispatchModel.load(cfg.model_path)         # only for global_avg (cold start)
    bus = KafkaBus(bootstrap, topic)
    disp = Dispatcher(
        api_solver(api_url, cfg), model.global_avg, cfg,
        perf_log_path=perf_path, ops_metrics_path=ops_path,
    )
    logger.info("Dispatcher consuming %s topic %s, scoring via %s", bootstrap, topic, api_url)
    logger.info("Live perf log -> %s · live ops metrics -> %s", perf_path, ops_path)
    try:
        disp.run(bus, tick_every_sec=tick, max_idle_polls=max_idle)
    finally:
        bus.close()

    logger.info("Final ops metrics: %s", disp.ops_metrics())


if __name__ == "__main__":
    main()

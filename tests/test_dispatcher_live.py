"""Dispatcher's live-write hooks: perf_log.jsonl / ops_metrics.json should update
incrementally as events are processed, not only at shutdown — this is what lets
the dashboard's "Live" mode reflect a running stack in real time."""
from __future__ import annotations

import json
from datetime import datetime

from dispatch.assignment import AssignmentResult
from dispatch.monitoring import load_ops_metrics, load_perf_log
from dispatch.snapshot import Snapshot
from dispatch.stream.dispatcher import Dispatcher

T0 = datetime(2022, 6, 1, 8, 0, 0)


def _one_pair_solver(snap: Snapshot) -> AssignmentResult:
    """Always books the first courier against the first order, if both exist."""
    if not snap.couriers or not snap.orders:
        return AssignmentResult(pairs=[], total_minutes=0.0,
                                 unassigned_orders=[o.order_id for o in snap.orders],
                                 unassigned_couriers=[c.courier_id for c in snap.couriers])
    cid, oid = snap.couriers[0].courier_id, snap.orders[0].order_id
    return AssignmentResult(pairs=[(cid, oid, 10.0)], total_minutes=10.0,
                             unassigned_orders=[], unassigned_couriers=[])


def test_ops_metrics_written_at_construction(tmp_path, cfg):
    ops_path = tmp_path / "ops_metrics.json"
    Dispatcher(_one_pair_solver, global_avg=30.0, cfg=cfg, ops_metrics_path=ops_path)
    snapshot = load_ops_metrics(ops_path)
    assert snapshot["ticks"] == 0
    assert snapshot["assignments"] == 0


def test_perf_log_and_ops_metrics_update_live(tmp_path, cfg):
    perf_path = tmp_path / "perf_log.jsonl"
    ops_path = tmp_path / "ops_metrics.json"
    disp = Dispatcher(_one_pair_solver, global_avg=30.0, cfg=cfg,
                       perf_log_path=perf_path, ops_metrics_path=ops_path)

    disp.process({"event_type": "courier_location", "event_time": T0.isoformat(),
                  "courier_id": "C1", "lng": 121.47, "lat": 31.23})
    disp.process({"event_type": "order_available", "event_time": T0.isoformat(),
                  "order_id": "O1", "lng": 121.48, "lat": 31.24,
                  "aoi_id": "1", "aoi_type": "10", "city": "Shanghai", "region_id": "0",
                  "window_start": T0.isoformat(), "window_end": T0.isoformat()})

    disp.tick(T0)
    # Booked the instant the tick ran — file reflects it before any pickup arrives.
    after_tick = load_ops_metrics(ops_path)
    assert after_tick["ticks"] == 1
    assert after_tick["assignments"] == 1
    assert load_perf_log(perf_path).empty   # nothing realized yet

    pickup_time = T0.replace(minute=12)
    disp.process({"event_type": "order_picked_up", "event_time": pickup_time.isoformat(),
                  "order_id": "O1", "courier_id": "C1", "lng": 121.48, "lat": 31.24})

    perf = load_perf_log(perf_path)
    assert len(perf) == 1
    row = perf.iloc[0]
    assert row["courier_id"] == "C1" and row["order_id"] == "O1"
    assert row["predicted_min"] == 10.0
    assert row["realized_min"] == 12.0

    final_ops = load_ops_metrics(ops_path)
    assert final_ops["perf_records"] == 1


def test_perf_log_path_truncated_on_fresh_construction(tmp_path, cfg):
    """A new Dispatcher instance starts this run's log clean, even if a stale
    file from a previous run is sitting at that path."""
    perf_path = tmp_path / "perf_log.jsonl"
    perf_path.write_text(json.dumps({"stale": "from a previous run"}) + "\n")
    Dispatcher(_one_pair_solver, global_avg=30.0, cfg=cfg, perf_log_path=perf_path)
    assert load_perf_log(perf_path).empty

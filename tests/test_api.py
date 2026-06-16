"""API tests — health, batched scoring, single-pair debug parity, and that a
known input yields a sane prediction. The tiny synthetic model is injected so the
tests never depend on a trained artifact on disk."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import dispatch.api.main as api_main

T = "2022-06-01T10:00:00"
COURIER = {"courier_id": "c0", "lng": 121.47, "lat": 31.23,
           "rolling_avg_min": 30.0, "active_load": 1}
ORDER = {"order_id": "o0", "lng": 121.50, "lat": 31.20,
         "region_id": "0", "aoi_id": "1", "aoi_type": "10", "city": "Shanghai",
         "window_start": "2022-06-01T10:20:00", "window_end": "2022-06-01T11:30:00"}
ORDER2 = {**ORDER, "order_id": "o1", "lng": 121.46, "lat": 31.24}


@pytest.fixture
def client(tiny_model, monkeypatch):
    # The app's lifespan calls load_model(cfg); hand it our synthetic model.
    monkeypatch.setattr(api_main, "load_model", lambda cfg=None: tiny_model)
    with TestClient(api_main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["baseline_test_mae"] == pytest.approx(9.9)


def test_score_known_input_is_sane(client):
    r = client.post("/score", json={"as_of_time": T, "couriers": [COURIER], "orders": [ORDER, ORDER2]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_couriers"] == 1 and body["n_orders"] == 2
    assert len(body["pairs"]) == 2
    for p in body["pairs"]:
        assert 0.0 < p["predicted_minutes"] < 600.0     # finite, positive, bounded
    assert "X-Process-Time-Ms" in r.headers


def test_score_empty_snapshot(client):
    r = client.post("/score", json={"as_of_time": T, "couriers": [], "orders": []})
    assert r.status_code == 200
    assert r.json()["pairs"] == []


def test_debug_matches_local_model(client, tiny_model, cfg):
    """Single-pair /debug must equal a direct local prediction (parity check)."""
    r = client.post("/debug", json={"as_of_time": T, "courier": COURIER, "order": ORDER})
    assert r.status_code == 200
    api_pred = r.json()["predicted_minutes"]

    raw = pd.DataFrame([{
        "accept_time": pd.Timestamp(T),
        "lng": ORDER["lng"], "lat": ORDER["lat"],
        "accept_gps_lng": COURIER["lng"], "accept_gps_lat": COURIER["lat"],
        "accept_loc_missing": 0,
        "time_window_start": pd.Timestamp(ORDER["window_start"]),
        "time_window_end": pd.Timestamp(ORDER["window_end"]),
        "region_id": ORDER["region_id"], "aoi_id": ORDER["aoi_id"],
        "aoi_type": ORDER["aoi_type"], "city": ORDER["city"],
        "courier_rolling_avg_min": COURIER["rolling_avg_min"],
        "courier_active_load": COURIER["active_load"],
    }])
    local = float(tiny_model.predict_from_raw(raw, cfg)[0])
    assert api_pred == pytest.approx(local, abs=1e-3)


def test_score_matches_local_batch(client, tiny_model, cfg):
    r = client.post("/score", json={"as_of_time": T, "couriers": [COURIER], "orders": [ORDER, ORDER2]})
    api_map = {(p["courier_id"], p["order_id"]): p["predicted_minutes"] for p in r.json()["pairs"]}
    from dispatch.api.main import _to_courier, _to_order
    from dispatch.assignment import score_pairs
    from dispatch.snapshot import Snapshot
    from dispatch.api.schemas import CourierIn, OrderIn

    snap = Snapshot(
        as_of_time=pd.Timestamp(T),
        couriers=[_to_courier(CourierIn(**COURIER))],
        orders=[_to_order(OrderIn(**ORDER)), _to_order(OrderIn(**ORDER2))],
    )
    local = score_pairs(snap, lambda f: tiny_model.predict_from_raw(f, cfg))
    for p in local:
        assert api_map[(p.courier_id, p.order_id)] == pytest.approx(p.predicted_minutes, abs=1e-3)

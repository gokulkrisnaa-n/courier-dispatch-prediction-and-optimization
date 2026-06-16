"""Assignment tests — the Hungarian optimization beats the greedy baseline, plus
matrix reshaping and rectangular / unfilled handling."""
from __future__ import annotations

import numpy as np

from dispatch.assignment import (
    ScoredPair,
    _UNFILLED,
    assign_from_pairs,
    cost_matrix_from_pairs,
    greedy_assign,
    hungarian_assign,
    score_pairs,
    solve,
)
from dispatch.snapshot import Courier, Order, Snapshot
from datetime import datetime


def _total(cost, pairs):
    return sum(cost[r, c] for r, c in pairs)


def test_hungarian_strictly_beats_greedy():
    # Greedy grabs the global min (0,0)=1 then is forced into (1,1)=4 -> 5.
    # Hungarian takes (0,1)+(1,0)=2+2=4. Optimal < greedy.
    cost = np.array([[1.0, 2.0], [2.0, 4.0]])
    h = _total(cost, hungarian_assign(cost))
    g = _total(cost, greedy_assign(cost))
    assert h == 4.0
    assert g == 5.0
    assert h < g


def test_hungarian_never_worse_than_greedy_random():
    rng = np.random.default_rng(0)
    for _ in range(50):
        cost = rng.random((5, 5)) * 100
        h = _total(cost, hungarian_assign(cost))
        g = _total(cost, greedy_assign(cost))
        assert h <= g + 1e-9


def test_rectangular_more_couriers_than_orders():
    cost = np.array([[5.0, 9.0], [1.0, 8.0], [7.0, 2.0]])  # 3 couriers, 2 orders
    pairs = hungarian_assign(cost)
    assert len(pairs) == 2                       # min(3, 2)
    rows = {r for r, _ in pairs}
    assert len(rows) == 2                        # one courier left unassigned


def test_cost_matrix_from_pairs_fills_and_defaults():
    scored = [
        ScoredPair("c0", "o0", 3.0),
        ScoredPair("c0", "o1", 7.0),
        ScoredPair("c1", "o0", 5.0),
        # (c1, o1) intentionally missing -> stays _UNFILLED
    ]
    cost = cost_matrix_from_pairs(scored, ["c0", "c1"], ["o0", "o1"])
    assert cost[0, 0] == 3.0 and cost[0, 1] == 7.0 and cost[1, 0] == 5.0
    assert cost[1, 1] == _UNFILLED


def test_assign_from_pairs_skips_unfilled():
    scored = [ScoredPair("c0", "o0", 3.0), ScoredPair("c1", "o0", 1.0)]
    res = assign_from_pairs(scored, ["c0", "c1"], ["o0", "o1"])
    # o1 was never scored -> no courier should be assigned to it.
    assert all(oid != "o1" for _, oid, _ in res.pairs)
    assert "o1" in res.unassigned_orders


def test_solve_end_to_end_with_dummy_scorer():
    t = datetime(2022, 6, 1, 10, 0, 0)
    couriers = [
        Courier("c0", 121.47, 31.23, rolling_avg_min=30, active_load=0),
        Courier("c1", 121.50, 31.20, rolling_avg_min=30, active_load=0),
    ]
    orders = [
        Order("o0", 121.47, 31.23, "0", "1", "10", "Shanghai", t, t, t),
        Order("o1", 121.50, 31.20, "0", "1", "10", "Shanghai", t, t, t),
    ]
    snap = Snapshot(as_of_time=t, couriers=couriers, orders=orders)

    # Scorer = distance proxy: each courier cheapest at its own coords.
    def scorer(frame):
        dlng = (frame["accept_gps_lng"] - frame["lng"]).abs()
        dlat = (frame["accept_gps_lat"] - frame["lat"]).abs()
        return (dlng + dlat).to_numpy() * 1000.0

    res = solve(snap, scorer)
    assigned = {cid: oid for cid, oid, _ in res.pairs}
    assert assigned == {"c0": "o0", "c1": "o1"}     # each to its co-located order
    assert res.total_minutes == 0.0


def test_score_pairs_batches_full_cross_product():
    t = datetime(2022, 6, 1, 10, 0, 0)
    snap = Snapshot(
        as_of_time=t,
        couriers=[Courier(f"c{i}", 121.47, 31.23, 30, 0) for i in range(3)],
        orders=[Order(f"o{j}", 121.47, 31.23, "0", "1", "10", "Shanghai", t, t, t)
                for j in range(4)],
    )
    calls = {"n": 0}

    def scorer(frame):
        calls["n"] += 1                          # must be ONE batched call
        return np.full(len(frame), 5.0)

    pairs = score_pairs(snap, scorer)
    assert len(pairs) == 12                       # 3 x 4 cross product
    assert calls["n"] == 1

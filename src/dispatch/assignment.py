"""Courier-to-order assignment: batched cost matrix + Hungarian optimization.

This is a LOCAL component, not a service. Given a ``Snapshot`` of free couriers
and open orders it:

  1. builds every (courier, order) feature row — the full cross product,
  2. scores them in ONE vectorized batch (either a local model or a scoring
     function injected by the dispatcher, e.g. an API client),
  3. shapes the scores into a cost matrix ``cost[i][j]`` = predicted minutes for
     courier *i* to pick up order *j*,
  4. solves the one-to-one assignment with ``scipy.linear_sum_assignment``
     (Hungarian) to minimize total predicted pickup time.

The cross-product / batch-scoring split mirrors the deployed path: the dispatcher
sends the raw snapshot to the API, the API returns flat scored pairs, and the
dispatcher reshapes them here with ``cost_matrix_from_pairs``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import Config, load_config
from .snapshot import Snapshot

# A scorer maps a raw pair-frame (RAW_INPUT_COLS) to a 1-D array of minutes.
Scorer = Callable[[pd.DataFrame], np.ndarray]

# Sentinel cost for pairs that were never scored (keeps Hungarian from using them).
_UNFILLED = 1e9


@dataclass(frozen=True)
class ScoredPair:
    courier_id: str
    order_id: str
    predicted_minutes: float


@dataclass
class AssignmentResult:
    pairs: list[tuple[str, str, float]]        # (courier_id, order_id, minutes)
    total_minutes: float
    unassigned_orders: list[str]
    unassigned_couriers: list[str]

    @property
    def n_assigned(self) -> int:
        return len(self.pairs)


def build_pair_frame(snapshot: Snapshot) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the full cross-product of raw feature rows for the snapshot.

    Returns the frame (one row per courier×order, in row-major courier order),
    the courier_id list, and the order_id list defining the matrix axes.
    """
    couriers, orders = snapshot.couriers, snapshot.orders
    t = snapshot.as_of_time
    records: list[dict] = []
    for c in couriers:
        for o in orders:
            records.append({
                "accept_time": t,
                "lng": o.lng, "lat": o.lat,
                "accept_gps_lng": c.lng, "accept_gps_lat": c.lat,
                "accept_loc_missing": int(c.loc_missing),
                "time_window_start": o.window_start, "time_window_end": o.window_end,
                "region_id": o.region_id, "aoi_id": o.aoi_id,
                "aoi_type": o.aoi_type, "city": o.city,
                "courier_rolling_avg_min": c.rolling_avg_min,
                "courier_active_load": c.active_load,
            })
    frame = pd.DataFrame.from_records(records)
    return frame, [c.courier_id for c in couriers], [o.order_id for o in orders]


def score_pairs(snapshot: Snapshot, scorer: Scorer) -> list[ScoredPair]:
    """Score every pair in one batched call via the injected scorer."""
    frame, courier_ids, order_ids = build_pair_frame(snapshot)
    if frame.empty:
        return []
    preds = np.asarray(scorer(frame), dtype="float64")
    pairs: list[ScoredPair] = []
    k = 0
    for cid in courier_ids:
        for oid in order_ids:
            pairs.append(ScoredPair(cid, oid, float(preds[k])))
            k += 1
    return pairs


def cost_matrix_from_pairs(
    scored: list[ScoredPair], courier_ids: list[str], order_ids: list[str]
) -> np.ndarray:
    """Reshape flat scored pairs into a (couriers × orders) cost matrix."""
    ci = {c: i for i, c in enumerate(courier_ids)}
    oj = {o: j for j, o in enumerate(order_ids)}
    cost = np.full((len(courier_ids), len(order_ids)), _UNFILLED, dtype="float64")
    for p in scored:
        if p.courier_id in ci and p.order_id in oj:
            cost[ci[p.courier_id], oj[p.order_id]] = p.predicted_minutes
    return cost


def hungarian_assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one min-cost assignment. Returns (row, col) index pairs."""
    if cost.size == 0:
        return []
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))


def greedy_assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """Greedy baseline: repeatedly take the lowest-cost free (row, col).

    Provided for comparison/tests — Hungarian's total cost is always <= greedy's.
    """
    if cost.size == 0:
        return []
    order = np.argsort(cost, axis=None)
    n_cols = cost.shape[1]
    used_rows, used_cols, pairs = set(), set(), []
    for flat in order:
        r, c = divmod(int(flat), n_cols)
        if r in used_rows or c in used_cols:
            continue
        if cost[r, c] >= _UNFILLED:
            continue
        used_rows.add(r); used_cols.add(c); pairs.append((r, c))
        if len(pairs) == min(cost.shape):
            break
    return pairs


def assign_from_pairs(
    scored: list[ScoredPair], courier_ids: list[str], order_ids: list[str]
) -> AssignmentResult:
    """Dispatcher path: flat scored pairs -> cost matrix -> Hungarian result."""
    cost = cost_matrix_from_pairs(scored, courier_ids, order_ids)
    idx_pairs = hungarian_assign(cost)
    pairs, assigned_r, assigned_c = [], set(), set()
    for r, c in idx_pairs:
        if cost[r, c] >= _UNFILLED:
            continue
        pairs.append((courier_ids[r], order_ids[c], float(cost[r, c])))
        assigned_r.add(r); assigned_c.add(c)
    return AssignmentResult(
        pairs=pairs,
        total_minutes=float(sum(p[2] for p in pairs)),
        unassigned_orders=[o for j, o in enumerate(order_ids) if j not in assigned_c],
        unassigned_couriers=[c for i, c in enumerate(courier_ids) if i not in assigned_r],
    )


def solve(snapshot: Snapshot, scorer: Scorer, cfg: Config | None = None) -> AssignmentResult:
    """End-to-end local solve: snapshot -> batched scoring -> Hungarian.

    ``scorer`` turns a raw pair-frame into predicted minutes. For a fully local
    run pass ``model.predict_from_raw``; the dispatcher instead posts to the API.
    """
    _ = cfg or load_config()
    _, courier_ids, order_ids = build_pair_frame(snapshot)
    scored = score_pairs(snapshot, scorer)
    return assign_from_pairs(scored, courier_ids, order_ids)

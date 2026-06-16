"""FastAPI scoring service for the dispatch system.

Endpoints:
  * POST /score  — batch: takes the raw snapshot {couriers, orders, as_of_time},
                   builds every (courier, order) pair internally, runs ONE
                   vectorized prediction, and returns flat scored pairs. The
                   dispatcher reshapes these into its cost matrix locally.
  * POST /debug  — single pair: one prediction plus the exact feature row, for
                   parity checks against the local model and hand-testing in /docs.
  * GET  /health — liveness + which model artifact is loaded.

The model is loaded once at startup. A middleware stamps each response with its
server-side latency (X-Process-Time-Ms) and logs it for the monitoring layer.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from ..assignment import build_pair_frame, score_pairs
from ..config import load_config
from ..model import DispatchModel, load_model
from ..snapshot import Courier, Order, Snapshot
from .schemas import (
    CourierIn,
    DebugRequest,
    DebugResponse,
    HealthResponse,
    OrderIn,
    ScoredPairOut,
    ScoreRequest,
    ScoreResponse,
)

logger = logging.getLogger("dispatch.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    try:
        app.state.model = load_model(cfg)
        app.state.cfg = cfg
        logger.info("Loaded model from %s", cfg.model_path)
    except FileNotFoundError:
        app.state.model = None
        app.state.cfg = cfg
        logger.warning("No model artifact at %s — train first.", cfg.model_path)
    yield


app = FastAPI(title="Courier Dispatch Scoring API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info("%s %s -> %d  %.2fms", request.method, request.url.path,
                response.status_code, elapsed_ms)
    return response


# --------------------------------------------------------------------------- #
# request -> domain conversion
# --------------------------------------------------------------------------- #
def _to_courier(c: CourierIn) -> Courier:
    return Courier(
        courier_id=c.courier_id, lng=c.lng, lat=c.lat,
        rolling_avg_min=c.rolling_avg_min, active_load=c.active_load,
    )


def _to_order(o: OrderIn) -> Order:
    return Order(
        order_id=o.order_id, lng=o.lng, lat=o.lat,
        region_id=o.region_id, aoi_id=o.aoi_id, aoi_type=o.aoi_type, city=o.city,
        window_start=o.window_start, window_end=o.window_end,
        available_time=o.window_start,            # unused by features; kept for type
    )


def _model(request: Request) -> DispatchModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — train first.")
    return model


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model = request.app.state.model
    if model is None:
        return HealthResponse(status="degraded", model_loaded=False)
    meta = model.metadata
    return HealthResponse(
        status="ok", model_loaded=True,
        trained_at=meta.get("trained_at"),
        baseline_test_mae=meta.get("test_metrics", {}).get("mae"),
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, request: Request) -> ScoreResponse:
    """Batch-score the full courier×order cross product in one prediction call."""
    model = _model(request)
    cfg = request.app.state.cfg
    snapshot = Snapshot(
        as_of_time=req.as_of_time,
        couriers=[_to_courier(c) for c in req.couriers],
        orders=[_to_order(o) for o in req.orders],
    )
    pairs = score_pairs(snapshot, lambda frame: model.predict_from_raw(frame, cfg))
    return ScoreResponse(
        as_of_time=req.as_of_time,
        n_couriers=len(req.couriers),
        n_orders=len(req.orders),
        pairs=[
            ScoredPairOut(courier_id=p.courier_id, order_id=p.order_id,
                          predicted_minutes=round(p.predicted_minutes, 3))
            for p in pairs
        ],
    )


@app.post("/debug", response_model=DebugResponse)
def debug(req: DebugRequest, request: Request) -> DebugResponse:
    """Single-pair prediction + the exact feature row used to produce it."""
    model = _model(request)
    cfg = request.app.state.cfg
    snapshot = Snapshot(
        as_of_time=req.as_of_time,
        couriers=[_to_courier(req.courier)],
        orders=[_to_order(req.order)],
    )
    frame, _, _ = build_pair_frame(snapshot)
    pred = float(model.predict_from_raw(frame, cfg)[0])
    # Echo the built feature row (json-safe) for hand-verification.
    from ..features import build_features
    feat_row = build_features(frame, cfg).iloc[0]
    features = {
        k: (None if _is_nan(v) else _json_safe(v)) for k, v in feat_row.items()
    }
    return DebugResponse(predicted_minutes=round(pred, 3), features=features)


def _is_nan(v) -> bool:
    try:
        return v != v  # NaN is the only value not equal to itself
    except Exception:
        return False


def _json_safe(v):
    if hasattr(v, "item"):
        return v.item()
    return str(v) if not isinstance(v, (int, float, str)) else v

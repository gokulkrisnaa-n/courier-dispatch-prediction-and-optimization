"""Pydantic request/response models for the scoring API.

The dispatcher sends the *raw snapshot* — free couriers, open orders, and the
as-of time — and the API forms the (courier, order) cross product internally.
Courier history aggregates (rolling-average pickup duration, current workload)
are maintained by the dispatcher and travel with each courier in the request.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CourierIn(BaseModel):
    courier_id: str
    # Last-known location. Null => location unknown (sets accept_loc_missing=1).
    lng: float | None = None
    lat: float | None = None
    # Running aggregates maintained by the dispatcher.
    rolling_avg_min: float = Field(..., description="recent mean pickup duration (min)")
    active_load: int = Field(0, ge=0, description="accepted-but-not-picked count")


class OrderIn(BaseModel):
    order_id: str
    lng: float
    lat: float
    region_id: str
    aoi_id: str
    aoi_type: str
    city: str = "Shanghai"
    window_start: datetime
    window_end: datetime


class ScoreRequest(BaseModel):
    as_of_time: datetime
    couriers: list[CourierIn] = Field(default_factory=list)
    orders: list[OrderIn] = Field(default_factory=list)


class ScoredPairOut(BaseModel):
    courier_id: str
    order_id: str
    predicted_minutes: float


class ScoreResponse(BaseModel):
    as_of_time: datetime
    n_couriers: int
    n_orders: int
    pairs: list[ScoredPairOut]


class DebugRequest(BaseModel):
    """Single (courier, order) pair — for parity checks and manual /docs testing."""
    as_of_time: datetime
    courier: CourierIn
    order: OrderIn


class DebugResponse(BaseModel):
    predicted_minutes: float
    features: dict[str, float | int | str | None]


class HealthResponse(BaseModel):
    # Allow the "model_loaded" field name without pydantic's protected-namespace warning.
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    trained_at: str | None = None
    baseline_test_mae: float | None = None

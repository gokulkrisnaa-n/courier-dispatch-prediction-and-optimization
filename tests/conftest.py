"""Shared test fixtures — fully synthetic, so tests need neither the 218 MB CSV
nor a pre-trained artifact. The synthetic frame mimics the schema produced by
``data.clean`` (parsed datetimes, clean coords, the missing-GPS flag, target)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from dispatch.config import load_config
from dispatch.features import add_courier_history, build_features
from dispatch.model import DispatchModel

# Shanghai-ish centre for plausible coordinates.
_CENTER = (121.47, 31.23)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_synthetic_clean(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Build a cleaned-style frame with a learnable target and several couriers."""
    rng = np.random.default_rng(seed)
    couriers = [f"C{i}" for i in range(8)]
    start = pd.Timestamp("2022-06-01 08:00:00")

    accept = start + pd.to_timedelta(np.sort(rng.integers(0, 60 * 24 * 20, n)), unit="m")
    lng = _CENTER[0] + rng.normal(0, 0.05, n)
    lat = _CENTER[1] + rng.normal(0, 0.05, n)
    # ~40% missing courier accept GPS, like the real data.
    miss = rng.random(n) < 0.4
    a_lng = np.where(miss, np.nan, _CENTER[0] + rng.normal(0, 0.05, n))
    a_lat = np.where(miss, np.nan, _CENTER[1] + rng.normal(0, 0.05, n))

    win_start = accept + pd.to_timedelta(rng.integers(10, 40, n), unit="m")
    win_end = win_start + pd.to_timedelta(rng.integers(60, 150, n), unit="m")
    # Target correlates with window slack + noise -> the model can learn signal.
    slack = (win_end - accept).total_seconds().to_numpy() / 60.0
    duration = np.clip(0.4 * slack + rng.normal(0, 10, n), 3, 200)
    pickup = accept + pd.to_timedelta(duration, unit="m")

    return pd.DataFrame({
        "order_id": [f"O{i}" for i in range(n)],
        "region_id": rng.choice(["0", "5", "6"], n),
        "city": "Shanghai",
        "courier_id": rng.choice(couriers, n),
        "accept_time": accept,
        "time_window_start": win_start,
        "time_window_end": win_end,
        "lng": lng, "lat": lat,
        "aoi_id": rng.choice(["1", "2", "3", "4"], n),
        "aoi_type": rng.choice(["10", "11", "12"], n),
        "pickup_time": pickup,
        "accept_gps_lng": a_lng, "accept_gps_lat": a_lat,
        "accept_loc_missing": miss.astype("int8"),
        "pickup_duration_min": duration,
    })


@pytest.fixture
def clean_df():
    return make_synthetic_clean()


@pytest.fixture(scope="session")
def tiny_model(cfg) -> DispatchModel:
    """A small, fast XGBoost model trained on synthetic data for serving tests."""
    df = make_synthetic_clean(500, seed=11)
    df, global_avg = add_courier_history(df, cfg)
    X = build_features(df, cfg)
    y = df[cfg.target_name]
    booster = XGBRegressor(
        n_estimators=40, max_depth=4, tree_method="hist",
        enable_categorical=True, random_state=0,
    )
    booster.fit(X, y)
    cat_dtypes = {c: X[c].dtype for c in cfg.categorical_features}
    return DispatchModel(
        booster=booster, feature_columns=cfg.feature_columns,
        categorical_dtypes=cat_dtypes, global_avg=float(global_avg),
        metadata={"trained_at": "2022-06-01T00:00:00+00:00",
                  "test_metrics": {"mae": 9.9}},
    )

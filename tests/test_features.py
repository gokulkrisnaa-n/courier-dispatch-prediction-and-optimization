"""Feature tests — contract, distance, cyclical encoding, and the central
point-in-time leakage guarantee."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dispatch.features import add_courier_history, build_features, haversine_km
from tests.conftest import make_synthetic_clean


def test_feature_contract(clean_df, cfg):
    df, _ = add_courier_history(clean_df, cfg)
    X = build_features(df, cfg)
    assert list(X.columns) == cfg.feature_columns
    for c in cfg.categorical_features:
        assert str(X[c].dtype) == "category"
    assert not np.isinf(X.select_dtypes("number").to_numpy()).any()


def test_haversine_known_distance():
    # ~1 deg of latitude ≈ 111 km; check a short, known east-west hop.
    d = haversine_km(np.array([121.0]), np.array([31.0]),
                     np.array([121.0]), np.array([31.1]))[0]
    assert d == pytest.approx(11.12, abs=0.2)


def test_distance_is_nan_when_courier_loc_missing(clean_df, cfg):
    df, _ = add_courier_history(clean_df, cfg)
    X = build_features(df, cfg)
    missing = df["accept_loc_missing"] == 1
    assert X.loc[missing, "distance_to_pickup_km"].isna().all()
    assert X.loc[~missing, "distance_to_pickup_km"].notna().all()


def test_cyclical_encoding_wraps():
    df = make_synthetic_clean(50)
    X = build_features(*_with_history(df))
    assert X["hour_sin"].between(-1, 1).all() and X["hour_cos"].between(-1, 1).all()
    # 23:59 should sit next to 00:00 in (sin, cos) space.
    base = pd.Timestamp("2022-06-01")
    near = _one_row_features(base + pd.Timedelta("23:59:00"))
    midnight = _one_row_features(base + pd.Timedelta("00:00:00"))
    dist = np.hypot(near["hour_sin"] - midnight["hour_sin"],
                    near["hour_cos"] - midnight["hour_cos"])
    assert dist < 0.05


def test_rolling_avg_uses_only_past(cfg):
    """Single courier, known durations -> rolling avg excludes the current row."""
    base = pd.Timestamp("2022-06-01 08:00:00")
    df = make_synthetic_clean(4, seed=1).iloc[:4].copy()
    df["courier_id"] = "C0"
    df["accept_time"] = [base + pd.Timedelta(hours=i) for i in range(4)]
    df["pickup_time"] = df["accept_time"] + pd.Timedelta(minutes=5)
    df["pickup_duration_min"] = [10.0, 20.0, 30.0, 40.0]
    out, global_avg = add_courier_history(df, cfg)
    roll = out.sort_values("accept_time")["courier_rolling_avg_min"].tolist()
    assert roll[0] == pytest.approx(global_avg)        # no prior -> global fill
    assert roll[1] == pytest.approx(10.0)              # mean of [10]
    assert roll[2] == pytest.approx(15.0)              # mean of [10, 20]
    assert roll[3] == pytest.approx(20.0)              # mean of [10, 20, 30]


def test_no_future_leakage(cfg):
    """Mutating row k's target may only change rows AFTER k — never k or earlier.

    This is the point-in-time guarantee: a row's features depend solely on the
    past, so the target it is trained to predict can never inform its own inputs.
    """
    df = make_synthetic_clean(120, seed=3)
    # global_avg is frozen at training time (lives in the artifact), so hold it
    # fixed across both runs — otherwise mutating one target shifts the cold-start
    # fill globally, which is a separate effect from point-in-time leakage.
    gavg = float(df["pickup_duration_min"].mean())
    feats0 = build_features(add_courier_history(df, cfg, global_avg=gavg)[0], cfg)

    k = 60
    mutated = df.copy().reset_index(drop=True)
    order = mutated.sort_values("accept_time").index
    target_row = order[k]
    mutated.loc[target_row, "pickup_duration_min"] += 999.0
    feats1 = build_features(add_courier_history(mutated, cfg, global_avg=gavg)[0], cfg)

    feats0 = feats0.reset_index(drop=True)
    feats1 = feats1.reset_index(drop=True)
    roll0 = feats0["courier_rolling_avg_min"].to_numpy()
    roll1 = feats1["courier_rolling_avg_min"].to_numpy()

    # rows up to and including k: identical (no self/future leak)
    np.testing.assert_allclose(roll0[: k + 1], roll1[: k + 1])
    # the change must surface in at least one later row of the same courier
    assert np.any(~np.isclose(roll0[k + 1:], roll1[k + 1:]))


def test_active_load_point_in_time(cfg):
    """Overlapping intervals for one courier -> exact accepted-but-not-picked count."""
    base = pd.Timestamp("2022-06-01 09:00:00")
    df = make_synthetic_clean(3, seed=2).iloc[:3].copy()
    df["courier_id"] = "C0"
    df["accept_time"] = [base, base + pd.Timedelta("10min"), base + pd.Timedelta("15min")]
    df["pickup_time"] = [base + pd.Timedelta("30min"),
                         base + pd.Timedelta("20min"),
                         base + pd.Timedelta("40min")]
    df["pickup_duration_min"] = [30.0, 10.0, 25.0]
    out, _ = add_courier_history(df, cfg)
    loads = out.sort_values("accept_time")["courier_active_load"].tolist()
    assert loads == [0, 1, 2]


# --- helpers ---
def _with_history(df, cfg=None):
    from dispatch.config import load_config
    cfg = cfg or load_config()
    out, _ = add_courier_history(df, cfg)
    return out, cfg


def _one_row_features(accept_time):
    from dispatch.config import load_config
    cfg = load_config()
    row = make_synthetic_clean(1, seed=5).iloc[:1].copy()
    row["accept_time"] = accept_time
    row["time_window_start"] = accept_time + pd.Timedelta("20min")
    row["time_window_end"] = accept_time + pd.Timedelta("90min")
    row["pickup_time"] = accept_time + pd.Timedelta("40min")
    row["courier_rolling_avg_min"] = 30.0
    row["courier_active_load"] = 0
    return build_features(row, cfg).iloc[0]

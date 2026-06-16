"""Offline training pipeline: raw LaDe -> clean -> features -> XGBoost -> artifacts.

Run:
    python -m dispatch.train            # dev: sampled rows (config dev.sample_rows)
    python -m dispatch.train --full     # full dataset

Writes two artifacts:
    artifacts/model.joblib            (DispatchModel: booster + contract)
    artifacts/reference_profile.json  (drift/perf baseline for monitoring)

Splitting is temporal (by accept_time month) and CV uses TimeSeriesSplit so no
future row ever informs a past prediction.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from .config import Config, load_config
from .data import load_clean
from .features import add_courier_history, build_features
from .model import DispatchModel
from .monitoring import build_reference_profile, regression_metrics

logger = logging.getLogger(__name__)


def _temporal_split(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Boolean test mask: rows whose accept_time month >= test_start_month."""
    return (df["accept_time"].dt.month >= cfg.test_start_month).to_numpy()


def _new_estimator(cfg: Config) -> XGBRegressor:
    params = cfg.xgb_params
    return XGBRegressor(objective="reg:squarederror", n_jobs=-1, **params)


def _cross_validate(X: pd.DataFrame, y: pd.Series, cfg: Config) -> dict[str, float]:
    """TimeSeriesSplit CV on the (time-ordered) training data."""
    tscv = TimeSeriesSplit(n_splits=cfg.cv_folds)
    maes, rmses = [], []
    for k, (tr, va) in enumerate(tscv.split(X), start=1):
        est = _new_estimator(cfg)
        est.fit(X.iloc[tr], y.iloc[tr])
        m = regression_metrics(y.iloc[va].to_numpy(), est.predict(X.iloc[va]))
        maes.append(m["mae"]); rmses.append(m["rmse"])
        logger.info("  fold %d/%d  MAE=%.2f  RMSE=%.2f", k, cfg.cv_folds, m["mae"], m["rmse"])
    return {"cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)),
            "cv_rmse_mean": float(np.mean(rmses))}


def train(cfg: Config | None = None, *, full: bool = False) -> DispatchModel:
    cfg = cfg or load_config()
    logger.info("Loading + cleaning data (full=%s)...", full)
    df = load_clean(cfg, sample=not full)

    logger.info("Engineering courier history + features...")
    df, global_avg = add_courier_history(df, cfg)
    X = build_features(df, cfg)
    y = df[cfg.target_name].astype("float64")

    test_mask = _temporal_split(df, cfg)
    X_tr, X_te = X.loc[~test_mask], X.loc[test_mask]
    y_tr, y_te = y.loc[~test_mask], y.loc[test_mask]
    logger.info("Split: %d train (months <%d) / %d test (months >=%d)",
                len(X_tr), cfg.test_start_month, len(X_te), cfg.test_start_month)
    if len(X_te) == 0:
        raise RuntimeError("Empty test set — check split.test_start_month vs data months.")

    logger.info("Cross-validating...")
    cv = _cross_validate(X_tr, y_tr, cfg)

    logger.info("Fitting final model on training split...")
    booster = _new_estimator(cfg)
    booster.fit(X_tr, y_tr)

    test_metrics = regression_metrics(y_te.to_numpy(), booster.predict(X_te))
    logger.info("TEST  MAE=%.2f  RMSE=%.2f  R2=%.3f",
                test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"])

    importance = dict(sorted(
        zip(cfg.feature_columns, booster.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True,
    ))
    logger.info("Top features: %s",
                ", ".join(f"{k}={v:.3f}" for k, v in list(importance.items())[:6]))

    # --- assemble + save the model artifact ---
    cat_dtypes = {c: X[c].dtype for c in cfg.categorical_features}
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "full_data": full, "target": cfg.target_name,
        "cv": cv, "test_metrics": test_metrics,
        "feature_importance": importance,
    }
    model = DispatchModel(
        booster=booster, feature_columns=cfg.feature_columns,
        categorical_dtypes=cat_dtypes, global_avg=global_avg, metadata=metadata,
    )
    model.save(cfg.model_path)
    logger.info("Saved model -> %s", cfg.model_path)

    # --- reference profile for monitoring (built from TRAIN distributions) ---
    profile = build_reference_profile(
        X_tr, y_tr,
        numeric=cfg.numeric_features, categorical=cfg.categorical_features,
        baseline_mae=test_metrics["mae"], created_at=metadata["trained_at"],
    )
    profile.to_json(cfg.reference_profile_path)
    logger.info("Saved reference profile -> %s", cfg.reference_profile_path)

    print(json.dumps({"cv": cv, "test": test_metrics}, indent=2))
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the dispatch pickup-duration model.")
    ap.add_argument("--full", action="store_true", help="use the full dataset")
    ap.add_argument("--config", default=None, help="path to config YAML")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config) if args.config else load_config()
    train(cfg, full=args.full)


if __name__ == "__main__":
    main()

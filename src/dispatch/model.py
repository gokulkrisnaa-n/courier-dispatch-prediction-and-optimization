"""The serializable model artifact and its load/predict interface.

``DispatchModel`` bundles everything the serving path needs to turn raw input
rows into a pickup-duration prediction: the fitted XGBoost booster, the exact
feature order, the categorical levels seen at training time (so unseen codes map
to NaN instead of crashing), and the global average used for cold-start couriers.

It is saved/loaded with joblib. The serving layer only ever touches this class —
it never imports train.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from .config import Config, load_config
from .features import build_features


@dataclass
class DispatchModel:
    booster: Any                                   # fitted xgboost.XGBRegressor
    feature_columns: list[str]
    categorical_dtypes: dict[str, CategoricalDtype]
    global_avg: float                              # cold-start courier_rolling_avg_min
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- align an already-built feature frame to the training contract --
    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, dtype in self.categorical_dtypes.items():
            # Re-apply the training categories; unseen levels -> NaN (XGBoost-safe).
            X[col] = X[col].astype("string").astype(dtype)
        return X[self.feature_columns]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict from a frame already produced by ``build_features``."""
        return self.booster.predict(self._align(X))

    def predict_from_raw(self, raw: pd.DataFrame, cfg: Config | None = None) -> np.ndarray:
        """Predict from raw input rows: build_features -> align -> predict.

        ``raw`` must carry the columns in ``features.RAW_INPUT_COLS`` (including the
        caller-supplied ``courier_rolling_avg_min`` / ``courier_active_load``).
        """
        feats = build_features(raw, cfg or load_config())
        return self.predict(feats)

    # -- persistence --
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DispatchModel":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} is not a DispatchModel artifact")
        return obj


_CACHED: DispatchModel | None = None


def load_model(cfg: Config | None = None) -> DispatchModel:
    """Process-cached model load for the serving path."""
    global _CACHED
    if _CACHED is None:
        cfg = cfg or load_config()
        _CACHED = DispatchModel.load(cfg.model_path)
    return _CACHED

"""Loads config/*.yaml into a typed, attribute-accessible object.

The whole system reads its paths, feature lists, split dates, and hyperparameters
from one YAML file so training and serving never drift apart. Import `load_config`
and pass the result around; nothing else should read the YAML directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/dispatch/config.py -> root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lade.yaml"


def _resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative config path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class BBox:
    lng_min: float
    lng_max: float
    lat_min: float
    lat_max: float

    def contains(self, lng: float, lat: float) -> bool:
        return (
            self.lng_min <= lng <= self.lng_max
            and self.lat_min <= lat <= self.lat_max
        )


@dataclass(frozen=True)
class Config:
    """Typed view over config/lade.yaml. `raw` keeps the untouched dict."""

    raw: dict[str, Any]
    root: Path = PROJECT_ROOT

    # ---- paths (resolved to absolute) ----
    @property
    def raw_csv(self) -> Path:
        return _resolve(self.raw["paths"]["raw_csv"])

    @property
    def processed_path(self) -> Path:
        return _resolve(self.raw["paths"]["processed"])

    @property
    def artifacts_dir(self) -> Path:
        return _resolve(self.raw["paths"]["artifacts_dir"])

    @property
    def model_path(self) -> Path:
        return _resolve(self.raw["paths"]["model"])

    @property
    def reference_profile_path(self) -> Path:
        return _resolve(self.raw["paths"]["reference_profile"])

    # ---- data cleaning ----
    @property
    def assumed_year(self) -> int:
        return int(self.raw["data"]["assumed_year"])

    @property
    def bbox(self) -> BBox:
        return BBox(**self.raw["data"]["bbox"])

    # ---- target ----
    @property
    def target_name(self) -> str:
        return self.raw["target"]["name"]

    @property
    def drop_nonpositive(self) -> bool:
        return bool(self.raw["target"]["drop_nonpositive"])

    @property
    def upper_pct_clip(self) -> float:
        return float(self.raw["target"]["upper_pct_clip"])

    # ---- split ----
    @property
    def test_start_month(self) -> int:
        return int(self.raw["split"]["test_start_month"])

    @property
    def cv_folds(self) -> int:
        return int(self.raw["split"]["cv_folds"])

    # ---- features ----
    @property
    def categorical_features(self) -> list[str]:
        return list(self.raw["features"]["categorical"])

    @property
    def numeric_features(self) -> list[str]:
        return list(self.raw["features"]["numeric"])

    @property
    def feature_columns(self) -> list[str]:
        """Full ordered feature list the model consumes (numeric + categorical)."""
        return self.numeric_features + self.categorical_features

    # ---- courier history ----
    @property
    def rolling_window(self) -> int:
        return int(self.raw["courier_history"]["rolling_window"])

    @property
    def rolling_min_periods(self) -> int:
        return int(self.raw["courier_history"]["min_periods"])

    @property
    def peak_hours(self) -> set[int]:
        return set(self.raw["peak_hours"])

    # ---- model ----
    @property
    def xgb_params(self) -> dict[str, Any]:
        return dict(self.raw["model"]["xgboost"])

    # ---- dev ----
    @property
    def sample_rows(self) -> int | None:
        return self.raw.get("dev", {}).get("sample_rows")

    @property
    def sample_seed(self) -> int:
        return int(self.raw.get("dev", {}).get("sample_seed", 42))

    # ---- snapshot ----
    @property
    def snapshot(self) -> dict[str, Any]:
        return dict(self.raw.get("snapshot", {}))

    # ---- stream ----
    @property
    def stream(self) -> dict[str, Any]:
        return dict(self.raw.get("stream", {}))


@lru_cache(maxsize=4)
def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """Load and cache the config. Pass an explicit path to override the default."""
    cfg_path = _resolve(path)
    with open(cfg_path, "r") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)

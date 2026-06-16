"""Monitoring primitives: reference profile + drift (PSI) + performance metrics.

The Streamlit dashboard (built later) sits on top of these pure functions. The
anchor is a *reference profile* saved at training time: per-feature distributions
(bin edges + proportions) and the baseline test MAE. At serve time we compare live
feature/target distributions against this profile via PSI, and compare realized
pickup durations against predictions to track performance drift.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_EPS = 1e-6
PSI_BINS = 10


@dataclass
class FeatureProfile:
    kind: str                       # "numeric" | "categorical"
    bin_edges: list[float] = field(default_factory=list)      # numeric only
    proportions: list[float] = field(default_factory=list)    # aligned to bins/levels
    levels: list[str] = field(default_factory=list)           # categorical only
    nan_share: float = 0.0


@dataclass
class ReferenceProfile:
    features: dict[str, FeatureProfile]
    target_bin_edges: list[float]
    target_proportions: list[float]
    baseline_mae: float
    n_train: int
    created_at: str

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "ReferenceProfile":
        raw = json.loads(Path(path).read_text())
        feats = {k: FeatureProfile(**v) for k, v in raw.pop("features").items()}
        return cls(features=feats, **raw)


def _numeric_profile(s: pd.Series, bins: int = PSI_BINS) -> FeatureProfile:
    vals = s.to_numpy(dtype="float64")
    nan_share = float(np.isnan(vals).mean())
    vals = vals[~np.isnan(vals)]
    # Quantile bin edges keep ~equal mass per bin; widen degenerate edges.
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([vals.min() - 1, vals.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(vals, bins=edges)
    props = counts / max(counts.sum(), 1)
    return FeatureProfile(
        kind="numeric", bin_edges=edges.tolist(), proportions=props.tolist(), nan_share=nan_share
    )


def _categorical_profile(s: pd.Series) -> FeatureProfile:
    s = s.astype("string")
    nan_share = float(s.isna().mean())
    counts = s.value_counts(normalize=True)
    return FeatureProfile(
        kind="categorical",
        levels=counts.index.tolist(),
        proportions=counts.to_numpy().tolist(),
        nan_share=nan_share,
    )


def build_reference_profile(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    numeric: list[str],
    categorical: list[str],
    baseline_mae: float,
    created_at: str,
) -> ReferenceProfile:
    """Snapshot training feature/target distributions for later drift comparison."""
    features: dict[str, FeatureProfile] = {}
    for col in numeric:
        features[col] = _numeric_profile(X[col])
    for col in categorical:
        features[col] = _categorical_profile(X[col])
    tgt = _numeric_profile(y)
    return ReferenceProfile(
        features=features,
        target_bin_edges=tgt.bin_edges,
        target_proportions=tgt.proportions,
        baseline_mae=float(baseline_mae),
        n_train=int(len(X)),
        created_at=created_at,
    )


# --------------------------------------------------------------------------- #
# Drift + performance metrics (used by the dashboard)
# --------------------------------------------------------------------------- #
def psi(reference_props: np.ndarray, actual_props: np.ndarray) -> float:
    """Population Stability Index between two proportion vectors."""
    ref = np.asarray(reference_props, dtype="float64") + _EPS
    act = np.asarray(actual_props, dtype="float64") + _EPS
    ref, act = ref / ref.sum(), act / act.sum()
    return float(np.sum((act - ref) * np.log(act / ref)))


def numeric_psi(profile: FeatureProfile, live: pd.Series) -> float:
    """PSI for a numeric feature using the profile's fixed bin edges."""
    edges = np.array(profile.bin_edges, dtype="float64")
    vals = live.to_numpy(dtype="float64")
    vals = vals[~np.isnan(vals)]
    counts, _ = np.histogram(vals, bins=edges)
    actual = counts / max(counts.sum(), 1)
    return psi(np.array(profile.proportions), actual)


def categorical_psi(profile: FeatureProfile, live: pd.Series) -> float:
    """PSI for a categorical feature, aligning live levels to the profile's."""
    live_props = live.astype("string").value_counts(normalize=True)
    actual = np.array([live_props.get(lvl, 0.0) for lvl in profile.levels])
    return psi(np.array(profile.proportions), actual)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / R^2 in a plain dict (no sklearn import needed at serve)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or _EPS
    return {"mae": mae, "rmse": rmse, "r2": 1.0 - ss_res / ss_tot}


# --------------------------------------------------------------------------- #
# Drift report + retraining trigger (closes the monitoring loop)
# --------------------------------------------------------------------------- #
# Conventional PSI bands.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
# Performance guardrail: live error this many x the baseline triggers retrain.
MAE_DEGRADE_RATIO = 1.30


def classify_psi(value: float) -> str:
    if value >= PSI_SIGNIFICANT:
        return "significant"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def feature_drift(
    profile: "ReferenceProfile", X: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> pd.DataFrame:
    """Per-feature PSI of a live feature frame vs the training reference profile."""
    rows = []
    for col in numeric:
        if col in profile.features and col in X:
            val = numeric_psi(profile.features[col], X[col])
            rows.append({"feature": col, "kind": "numeric", "psi": val})
    for col in categorical:
        if col in profile.features and col in X:
            val = categorical_psi(profile.features[col], X[col])
            rows.append({"feature": col, "kind": "categorical", "psi": val})
    df = pd.DataFrame(rows)
    if len(df):
        df["severity"] = df["psi"].map(classify_psi)
        df = df.sort_values("psi", ascending=False).reset_index(drop=True)
    return df


def target_drift(profile: "ReferenceProfile", y: pd.Series) -> float:
    """PSI of the live target distribution vs the training target distribution."""
    ref = FeatureProfile(
        kind="numeric", bin_edges=profile.target_bin_edges,
        proportions=profile.target_proportions,
    )
    return numeric_psi(ref, y)


def retraining_recommendation(
    drift: pd.DataFrame,
    target_psi: float,
    live_mae: float | None,
    baseline_mae: float,
) -> dict[str, Any]:
    """Decide whether to retrain, with human-readable reasons."""
    reasons: list[str] = []
    drifted = drift.loc[drift["psi"] >= PSI_SIGNIFICANT, "feature"].tolist() if len(drift) else []
    if drifted:
        reasons.append(f"significant feature drift: {', '.join(drifted)}")
    if target_psi >= PSI_SIGNIFICANT:
        reasons.append(f"target distribution drift (PSI={target_psi:.2f})")
    if live_mae is not None and live_mae > baseline_mae * MAE_DEGRADE_RATIO:
        reasons.append(
            f"live MAE {live_mae:.1f} > {MAE_DEGRADE_RATIO:g}x baseline {baseline_mae:.1f}"
        )
    return {"should_retrain": bool(reasons), "reasons": reasons}


# --------------------------------------------------------------------------- #
# Prediction log persistence (logger -> joiner already done by the dispatcher)
# --------------------------------------------------------------------------- #
def save_perf_log(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Append performance records (predicted vs realized) as JSON lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    return path


def load_perf_log(path: str | Path) -> pd.DataFrame:
    """Load the JSONL performance log into a frame (empty frame if absent)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def save_ops_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Overwrite the live ops-metrics snapshot, atomically.

    Written via a temp file + ``os.replace`` so a concurrent reader (the
    dashboard, polling on its own schedule) never sees a half-written file —
    ``os.replace`` is atomic on both POSIX and Windows.
    """
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(metrics, default=str))
    os.replace(tmp, path)
    return path


def load_ops_metrics(path: str | Path) -> dict[str, Any]:
    """Load the live ops-metrics snapshot ({} if the dispatcher hasn't written one yet)."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}   # caught mid-write despite the atomic replace (e.g. truncated read) — retry next poll

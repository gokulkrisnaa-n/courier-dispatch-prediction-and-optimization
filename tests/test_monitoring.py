"""monitoring.py: live ops-metrics persistence (atomic overwrite) used by the
dashboard's Live mode to poll a running dispatcher without ever reading a
half-written file."""
from __future__ import annotations

from dispatch.monitoring import load_ops_metrics, save_ops_metrics


def test_save_load_ops_metrics_roundtrip(tmp_path):
    path = tmp_path / "ops_metrics.json"
    metrics = {"ticks": 3, "assignments": 7, "avg_tick_latency_ms": 12.5}
    save_ops_metrics(metrics, path)
    assert load_ops_metrics(path) == metrics


def test_save_ops_metrics_overwrites_not_appends(tmp_path):
    path = tmp_path / "ops_metrics.json"
    save_ops_metrics({"ticks": 1}, path)
    save_ops_metrics({"ticks": 2}, path)
    assert load_ops_metrics(path) == {"ticks": 2}


def test_load_ops_metrics_missing_file_returns_empty_dict(tmp_path):
    assert load_ops_metrics(tmp_path / "does_not_exist.json") == {}


def test_save_ops_metrics_no_leftover_tmp_file(tmp_path):
    path = tmp_path / "ops_metrics.json"
    save_ops_metrics({"ticks": 1}, path)
    assert not path.with_suffix(path.suffix + ".tmp").exists()

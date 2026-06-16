"""Streamlit monitoring dashboard for the dispatch system.

Run:
    streamlit run src/dispatch/dashboard.py

Three panels, all sitting on the pure functions in monitoring.py:

  * Operational health — tick throughput, per-tick latency, late pickups.
  * Model performance — predicted vs realized pickup duration, live MAE vs the
    baseline saved at training time.
  * Data drift — per-feature and target PSI against the reference profile, with a
    retraining recommendation that closes the loop.

Two data-source modes (sidebar toggle):

  * **Demo replay** — drives a self-contained, instant in-memory replay so the
    dashboard works without any broker running. Good for exploring the data.
  * **Live (running stack)** — reads the perf log / ops-metrics files the real
    ``dispatcher`` service (consuming Redpanda) writes incrementally as it runs
    (see ``stream/dispatcher.py``), auto-refreshing on a timer. This reflects
    whatever ``docker compose up`` is actually doing right now.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dispatch.config import load_config
from dispatch.data import clean, load_raw
from dispatch.features import add_courier_history, build_features
from dispatch.model import load_model
from dispatch.monitoring import (
    ReferenceProfile,
    classify_psi,
    feature_drift,
    load_ops_metrics,
    load_perf_log,
    regression_metrics,
    retraining_recommendation,
    target_drift,
)
from dispatch.stream.dispatcher import simulate

st.set_page_config(page_title="Courier Dispatch Monitoring", layout="wide")
SEVERITY_COLOR = {"stable": "#2ecc71", "moderate": "#f1c40f", "significant": "#e74c3c"}


@st.cache_resource
def _load_artifacts():
    cfg = load_config()
    model = load_model(cfg)
    profile = ReferenceProfile.from_json(cfg.reference_profile_path)
    return cfg, model, profile


@st.cache_data(show_spinner="Loading + cleaning data...")
def _load_data(nrows: int) -> pd.DataFrame:
    cfg = load_config()
    return clean(load_raw(cfg, nrows=nrows), cfg)


@st.cache_data(show_spinner="Running replay simulation...")
def _run_sim(nrows: int, region: str, day: str, tick_min: float):
    cfg = load_config()
    model = load_model(cfg)
    df = _load_data(nrows)
    sl = df[df["region_id"] == region]
    if day != "(all)":
        sl = sl[sl["accept_time"].dt.normalize() == pd.Timestamp(day)]
    disp = simulate(sl, model, cfg, tick_every_sec=tick_min * 60.0)
    return disp.ops_metrics(), disp.performance_frame(), list(disp.tick_latencies_ms)


@st.cache_data(show_spinner="Computing drift...")
def _live_features(nrows: int, month: int):
    cfg = load_config()
    model = load_model(cfg)
    df = _load_data(nrows)
    live = df[df["accept_time"].dt.month == month]
    live, _ = add_courier_history(live, cfg, global_avg=model.global_avg)
    return build_features(live, cfg), live[cfg.target_name]


def _render_ops_health(ops: dict, latencies: list[float] | None) -> None:
    st.subheader("⚙️ Operational health")
    c = st.columns(5)
    c[0].metric("Ticks", int(ops.get("ticks", 0)))
    c[1].metric("Assignments", int(ops.get("assignments", 0)))
    c[2].metric("Assign / tick", f"{ops.get('avg_assignments_per_tick', 0.0):.2f}")
    c[3].metric("Avg tick latency", f"{ops.get('avg_tick_latency_ms', 0.0):.1f} ms")
    c[4].metric("Late pickups", int(ops.get("late_pickups", 0)),
                help="Orders historically picked up before our tick could assign them.")
    if latencies:
        lat_df = pd.DataFrame({"tick": range(1, len(latencies) + 1), "latency_ms": latencies})
        st.plotly_chart(
            px.line(lat_df, x="tick", y="latency_ms", markers=True,
                    title="Per-tick scoring latency"),
            use_container_width=True,
        )
    elif ops:
        st.caption(
            f"Max tick latency so far: {ops.get('max_tick_latency_ms', 0.0):.1f} ms "
            "(per-tick history isn't persisted for live mode — only the running avg/max)."
        )


def _render_model_performance(perf: pd.DataFrame, profile: ReferenceProfile) -> float | None:
    """Render predicted-vs-realized panel; returns live MAE for the drift section below."""
    st.subheader("🎯 Model performance — predicted vs realized")
    if perf.empty:
        st.info("No completed pickups yet — widen the window, pick another day, or wait for "
                 "more pickups to land.")
        return None
    m = regression_metrics(perf["realized_min"].to_numpy(), perf["predicted_min"].to_numpy())
    c = st.columns(4)
    c[0].metric("Live MAE", f"{m['mae']:.1f} min",
                delta=f"{m['mae'] - profile.baseline_mae:+.1f} vs baseline",
                delta_color="inverse")
    c[1].metric("Live RMSE", f"{m['rmse']:.1f} min")
    c[2].metric("Joined records", len(perf))
    c[3].metric("Baseline MAE", f"{profile.baseline_mae:.1f} min")
    hi = float(max(perf["predicted_min"].max(), perf["realized_min"].max()))
    fig = px.scatter(perf, x="realized_min", y="predicted_min",
                     hover_data=["courier_id", "order_id"],
                     labels={"realized_min": "realized (min)", "predicted_min": "predicted (min)"},
                     title="Predicted vs realized")
    fig.add_shape(type="line", x0=0, y0=0, x1=hi, y1=hi, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Note: in replay, *realized* = pickup_time − our assign_time, so it "
        "reflects tick timing rather than the training target. Treat as an operational "
        "signal, not a substitute for online accuracy."
    )
    return float(np.mean(np.abs(perf["predicted_min"] - perf["realized_min"])))


def _render_drift(profile: ReferenceProfile, X_live: pd.DataFrame, y_live: pd.Series,
                   cfg, live_mae: float | None) -> None:
    st.subheader("📊 Data drift — PSI vs reference profile")
    drift = feature_drift(profile, X_live, cfg.numeric_features, cfg.categorical_features)
    tpsi = target_drift(profile, y_live)
    rec = retraining_recommendation(drift, tpsi, live_mae, profile.baseline_mae)

    if rec["should_retrain"]:
        st.error("**Retraining recommended** — " + "; ".join(rec["reasons"]))
    else:
        st.success("No retraining trigger: features and target within stable PSI bands.")

    c = st.columns([1, 2])
    c[0].metric("Target PSI", f"{tpsi:.3f}", help=classify_psi(tpsi))
    if not drift.empty:
        fig = px.bar(drift, x="psi", y="feature", orientation="h", color="severity",
                     color_discrete_map=SEVERITY_COLOR, title="Feature PSI (live vs reference)")
        fig.add_vline(x=0.10, line_dash="dot"); fig.add_vline(x=0.25, line_dash="dash")
        c[1].plotly_chart(fig, use_container_width=True)
        st.dataframe(drift, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("🚚 Courier Dispatch — Monitoring")
    try:
        cfg, model, profile = _load_artifacts()
    except FileNotFoundError:
        st.error("No model/reference artifacts found. Run `python -m dispatch.train` first.")
        return

    st.caption(
        f"Model trained {model.metadata.get('trained_at', '?')} · "
        f"baseline test MAE **{profile.baseline_mae:.1f} min** · "
        f"reference n={profile.n_train:,}"
    )

    # --- sidebar: mode + controls ---
    with st.sidebar:
        mode = st.radio("Data source", ["Demo replay", "Live (running stack)"], index=0,
                         help="Demo replay runs an instant in-memory simulation here. "
                              "Live mode reads the perf log / ops metrics the real "
                              "dispatcher service writes while consuming Redpanda.")
        st.divider()
        if mode == "Demo replay":
            st.header("Replay controls")
            nrows = st.select_slider("Rows scanned", [50_000, 120_000, 200_000, 400_000], 120_000)
            df0 = _load_data(nrows)
            regions = sorted(df0["region_id"].dropna().unique().tolist())
            region = st.selectbox("Region", regions, index=0)
            # Default to the busiest day in the region so the demo shows real activity.
            day_counts = (
                df0[df0["region_id"] == region]["accept_time"].dt.normalize()
                .dt.strftime("%Y-%m-%d").value_counts()
            )
            days = ["(all)"] + sorted(day_counts.index.tolist())
            busiest = day_counts.index[0] if len(day_counts) else "(all)"
            day = st.selectbox("Day", days, index=days.index(busiest))
            tick_min = st.slider("Tick interval (sim minutes)", 1, 30, 5)
        else:
            st.header("Live controls")
            refresh_sec = st.slider("Auto-refresh every (sec)", 3, 60, 5)
            st.caption(f"Reading `{cfg.artifacts_dir / 'monitoring'}` — start "
                       "`docker compose up` (producer + dispatcher) to see this update.")
            nrows = st.select_slider("Drift sample rows", [50_000, 120_000, 200_000, 400_000],
                                      120_000)
            df0 = _load_data(nrows)
        st.divider()
        months = sorted(df0["accept_time"].dt.month.unique().tolist())
        drift_month = st.selectbox("Drift: live window (month)", months, index=len(months) - 1)

    if mode == "Demo replay":
        ops, perf, latencies = _run_sim(nrows, region, day, float(tick_min))
    else:
        # No caching here — each auto-refresh must re-read the files fresh off disk
        # to reflect what the running dispatcher just wrote.
        monitoring_dir = cfg.artifacts_dir / "monitoring"
        ops = load_ops_metrics(monitoring_dir / "ops_metrics.json")
        perf = load_perf_log(monitoring_dir / "perf_log.jsonl")
        latencies = None   # per-tick history isn't persisted live, only running avg/max
        if not ops:
            st.warning("No live ops metrics yet — the dispatcher hasn't written its first "
                       "tick. This page will auto-refresh once it does.")
        # Reload the page on a timer so new disk writes show up without manual refresh.
        st.markdown(f'<meta http-equiv="refresh" content="{refresh_sec}">',
                    unsafe_allow_html=True)
        st.caption(f"⏱️ Auto-refreshing every {refresh_sec}s")

    _render_ops_health(ops, latencies)
    live_mae = _render_model_performance(perf, profile)
    X_live, y_live = _live_features(nrows, int(drift_month))
    _render_drift(profile, X_live, y_live, cfg, live_mae)


main()

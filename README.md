# Courier Dispatch — Prediction & Optimization

Predict how long each available courier would take to reach a merchant and pick up
an order, then assign one courier per order so that **total predicted pickup time
is minimized** (optimal one-to-one matching via the Hungarian algorithm).

Trained on the [LaDe](https://arxiv.org/abs/2306.10675) Shanghai pickup dataset.
The system is split into an **offline** training pipeline and an **online**
serving + streaming + assignment + monitoring stack.

```
data.py ─ features.py ─ train.py ──► artifacts/{model.joblib, reference_profile.json}
                │                          │
        (shared build_features)            ├─► model.py ──► api/  (FastAPI: /score /debug /health)
                │                          │                         ▲ POST /score
   snapshot.py ─ assignment.py ────────────┘                         │
        (cost matrix + Hungarian)                                    │
                                                                     │
   stream/ producer ─► bus (in-mem | Redpanda) ─► dispatcher ────────┘
                                                       │
                                          monitoring.py ─► dashboard.py (Streamlit)
```

The prediction model and the assignment optimizer are deliberately separate: the
model answers *"how long for courier i → order j?"* one cell at a time (batched),
and the optimizer turns the resulting cost matrix into an assignment.

---

## Results (full dataset: 1,408,240 cleaned rows)

Temporal split — train on months **May–Sep** (1,128,155 rows), test on the
held-out month of **October** (280,085 rows). Target = `pickup_duration` (minutes
from courier accept to pickup).

| Metric | 5-fold `TimeSeriesSplit` CV | Test (October, held out) |
| ------ | --------------------------- | ------------------------ |
| MAE    | 48.2 ± 3.2 min              | **47.5 min**             |
| RMSE   | 86.3 min                    | 87.7 min                 |
| R²     | —                           | **0.895**                |

**Top features (gain):** `slack_min` (0.45), `accept_loc_missing` (0.40),
`minutes_since_midnight`, `window_duration_min`, `distance_to_pickup_km`,
`aoi_id`. The pickup-time window dominates — this is a scheduling task as much as
a routing one (see `DESIGN.md`).

---

## Quickstart

### 1. Environment (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # makes `dispatch` importable
```

Place the dataset at `data/raw/pickup_sh.csv` (git-ignored; never committed).

### 2. Train → writes artifacts/

```bash
python -m dispatch.train             # dev: 150k-row sample, fast iteration
python -m dispatch.train --full      # full dataset (a few minutes)
```

Produces `artifacts/model.joblib` (the `DispatchModel`) and
`artifacts/reference_profile.json` (the drift/perf baseline for monitoring).

### 3. Serve the scoring API

```bash
uvicorn dispatch.api.main:app --reload
# Interactive docs at http://localhost:8000/docs
```

### 4. Run the monitoring dashboard

```bash
streamlit run src/dispatch/dashboard.py
```

### 5. Tests

```bash
pytest                               # 19 tests, ~0.4s, no data/artifact needed
```

### 6. Full streaming stack (Docker + Redpanda)

Train first so `artifacts/` exists, then:

```bash
docker compose up --build
```

This starts **Redpanda** (Kafka API), the **API**, a **producer** that replays the
historic events time-compressed, and a **dispatcher** that consumes them,
snapshots state, and calls the API to score each tick.

---

## Input / output contract

### `POST /score` — batch scoring (the dispatcher's path)

The dispatcher sends the raw snapshot; the API forms the full (courier × order)
cross product internally, scores it in **one** vectorized prediction, and returns
flat pairs. Courier history aggregates (`rolling_avg_min`, `active_load`) are
maintained by the dispatcher and travel with each courier.

```jsonc
// Request
{
  "as_of_time": "2024-10-12T10:00:00",
  "couriers": [
    {"courier_id": "C_8841", "lng": 121.50, "lat": 31.22,
     "rolling_avg_min": 58.0, "active_load": 3}      // lng/lat null => location unknown
  ],
  "orders": [
    {"order_id": "sh_000123", "lng": 121.4737, "lat": 31.2304,
     "region_id": "0", "aoi_id": "457", "aoi_type": "12", "city": "Shanghai",
     "window_start": "2024-10-12T10:30:00", "window_end": "2024-10-12T12:30:00"}
  ]
}

// Response  (one object per courier×order pair; dispatcher reshapes into a matrix)
{
  "as_of_time": "2024-10-12T10:00:00",
  "n_couriers": 1, "n_orders": 1,
  "pairs": [{"courier_id": "C_8841", "order_id": "sh_000123", "predicted_minutes": 92.4}]
}
```

### `POST /debug` — single pair (parity check + manual testing)

Same courier/order shapes, returns the prediction **plus the exact 19-feature row**
used to produce it — for hand-verification in `/docs` and parity against the local
model.

### `GET /health`

Liveness + which artifact is loaded (`trained_at`, baseline test MAE).

### Output meaning

`predicted_minutes` is the model's estimate of `pickup_duration` — minutes from
`as_of_time` (which plays the role of the courier's accept moment) until the order
is picked up. The assignment layer feeds these into `cost[i][j]` and runs
`scipy.linear_sum_assignment` to choose one courier per order at minimum total cost.

---

## Configuration

Everything (paths, Shanghai bounding box, feature lists, target trimming, split
month, XGBoost hyperparameters, snapshot/stream settings) lives in
`config/lade.yaml` and is read through `dispatch.config.load_config`. Training and
serving share that one file so they can never drift apart.

## Project layout

```
src/dispatch/
  config.py        # typed YAML loader
  data.py          # load + clean raw LaDe -> tidy frame
  features.py      # build_features() — THE shared transform (train == serve)
  train.py         # training pipeline -> artifacts/
  model.py         # DispatchModel: load artifact, predict(batch)
  assignment.py    # batched cost matrix + Hungarian (+ greedy baseline)
  snapshot.py      # reconstruct open orders / free couriers at time t
  monitoring.py    # reference profile, PSI drift, perf metrics, retrain trigger
  dashboard.py     # Streamlit monitoring UI
  stream/
    bus.py         # pluggable bus: InMemoryBus | KafkaBus
    producer.py    # replay historic events (time-compressed)
    dispatcher.py  # consume -> snapshot -> assign -> emit; perf log
  api/
    main.py        # FastAPI app
    schemas.py     # pydantic request/response
tests/             # features (incl. leakage), assignment, api
```

See `DESIGN.md` for problem framing, the key decisions, and known limitations.

# Design notes

Problem framing, the decisions that shaped the build, results, and the limits of
what this system can honestly claim.

---

## 1. Problem

Given a stream of open pickup orders and the couriers currently free in an area,
decide **which courier should pick up which order** so that the total time-to-pickup
is as small as possible. Two sub-problems, kept separate:

1. **Prediction** (regression): for a single (courier, order) pair, how many
   minutes until pickup? — an XGBoost model.
2. **Assignment** (combinatorial optimization): given the full cost matrix of
   predicted times, choose one courier per order minimizing total cost — the
   Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).

Separating them keeps each piece testable and lets the model be served as a pure,
stateless, batched function while the optimizer stays a small local component.

---

## 2. The data, as it actually is

`data/raw/pickup_sh.csv` — 1,424,406 rows of LaDe Shanghai pickups. Several things
differ from a naive reading of the schema and drove design choices:

- **Timestamps carry no year** (`MM-DD HH:MM:SS`); data spans **May–Oct**, so no
  year boundary is crossed. We inject a fixed year (2022) at parse time. The target
  is a duration, so the year is irrelevant to it.
- **GPS is sparse**: courier `accept_gps_*` is missing **45.9%** of the time. Rather
  than drop those rows, we keep them and add `accept_loc_missing` — which turns out
  to be one of the two strongest features.
- **Categoricals are integer codes** (`region_id`, `aoi_id`, `aoi_type`), not the
  human strings in the idealized event schema.
- **Order coordinates are already clean** (no `(0,0)`, all within Shanghai). The
  "drop bad coordinates" cleaning therefore targets the GPS columns; the bounding
  box `lng ∈ [120.8, 122.2], lat ∈ [30.6, 31.9]` is applied there.

Cleaning retains **98.9%** of rows.

---

## 3. Target and why it is what it is

`pickup_duration = pickup_time − accept_time`, in minutes. Non-positive durations
are dropped; the long upper tail is trimmed at the **99th percentile** (~1,960 min)
to remove data-entry artifacts and orders abandoned for many hours.

The median is ~120 minutes — large, because in LaDe a courier *accepts* a batch of
orders and then picks them up across a **2-hour delivery window**. So this is a
scheduling problem as much as a travel-time problem, which the feature importances
confirm: `slack_min` (time left in the window at accept) and `accept_loc_missing`
dominate, with `distance_to_pickup_km` a secondary effect.

---

## 4. Leakage prevention — the central correctness concern

The single most important rule: **a feature may depend only on information available
at the moment the courier accepts the order.** Two consequences:

- `pickup_time`, `pickup_gps_time`, `pickup_gps_lng/lat` are **never** features —
  they are the future relative to the accept event.
- The per-courier rolling-average pickup duration is computed with `shift(1)` over
  the courier's strictly-prior pickups, and `courier_active_load` is counted via a
  point-in-time sweep of accepted-but-not-yet-picked orders.

`features.py` enforces this structurally by splitting into two layers:

- `add_courier_history` — the only place that touches the target, used offline,
  point-in-time-safe.
- `build_features` — a **pure, row-wise** transform with no cross-row state. Because
  it carries no state, it *cannot* leak, and it is byte-for-byte identical between
  training and serving.

This is pinned by `test_no_future_leakage`: mutating row *k*'s target leaves all
rows ≤ *k* unchanged and perturbs only later rows. (Writing that test surfaced a
real subtlety — the cold-start global-mean fill is a *global* statistic, so it is
frozen in the artifact at training time and held fixed; otherwise it would couple
rows. The dispatcher uses that same frozen `global_avg` for unseen couriers.)

---

## 5. Model and validation

**XGBoost regressor with native categorical support** (`enable_categorical=True`,
`tree_method="hist"`). Rationale:

- Trees are scale-invariant, so **no standardization/normalization** is applied —
  it would add artifacts to maintain at serve time for zero benefit.
- Native categoricals avoid fragile target/one-hot encoders and, with the
  categories frozen in the artifact, **unseen codes at serve time map to NaN**
  rather than crashing.
- Missing values (e.g. distance when GPS is absent) are handled natively, which is
  why we keep those rows and flag them rather than impute.

**Validation is temporal**, never random: the test set is the held-out final month
(October), and cross-validation uses `TimeSeriesSplit` so no future fold informs a
past one. Full-data results: **test MAE 47.5 min, RMSE 87.7, R² 0.895**; CV MAE
48.2 ± 3.2 (close to test → the temporal split generalizes).

### Courier identity — history features, not raw IDs

`courier_id` is high-cardinality and many couriers at inference are unseen. Feeding
the raw ID would overfit and break on cold couriers. Instead a courier is
represented purely by engineered history (rolling-average duration, active load),
which **generalizes to new couriers** (neutral defaults) and is exactly what the
dispatcher can maintain in its own state.

---

## 6. Empirical validation — stress-testing the model

`slack_min` carries ~45% of the model's gain, which rightly raises a leakage alarm.
We investigated it directly (`notebooks/02_eda_validation.ipynb`); the conclusion is
that it is **legitimate signal**, and the investigation surfaced two deeper truths
about what this model can and cannot do.

### The leakage investigation, settled three ways

**Ablation.** Dropping `slack_min` and retraining on the full data collapses the
model: test MAE **47.5 → 150.0 min**, R² **0.895 → 0.092**. A leaked feature, when
removed, makes a model *slightly* worse because the others secretly re-encode the
same future. Here the opposite happened — the rest of the features explain almost
nothing without it — which is the signature of a feature that is *causally* central,
not leaking.

**Mechanism.** `slack_min = window_end − accept_time`. Both terms are known at the
accept moment: `accept_time` is the prediction reference, and `window_end` is the
order's delivery deadline. Neither touches `pickup_time`.

**Data evidence (three-way).** The EDA confirms `window_end` is a *preset* deadline,
not pickup-derived:
1. `window_duration` is **120 min for 98.3%** of orders (a small fixed menu).
2. `window_end` lands on a **round clock hour 96%** of the time.
3. **97.5% of pickups happen *before* `window_end`** (median 88 min before) — pickups
   sit *inside* a fixed window rather than defining its edge.

So `slack_min` is fully knowable at accept time. The genuine leakage risk —
`pickup_time` / `pickup_gps_*` — is structurally excluded and pinned by
`test_no_future_leakage`.

### The short-pickup failure mode

Breaking test MAE out by target decile reveals the error is not uniform. In absolute
terms it is U-shaped (best ~29 min mid-range, worst ~92 min on the long tail). In
*relative* terms the model is **worst on the shortest pickups**: decile 0 (mean
17 min) has **352% MAPE**. The model systematically **over-predicts fast pickups**,
defaulting toward the window-driven estimate and unable to recognize a quick one —
the opposite of what dispatch wants, since fast pickups are the valuable ones to
identify. It is most reliable (relatively) on long pickups (~11% MAPE).

### Signal-to-noise for the assignment

The cost matrix is only useful if predictions actually *vary across couriers* for the
same order. Scoring many reconstructed snapshots, the per-order std of predictions
across couriers has a **median ≈ 13 min** (mean ~16, up to ~80) — about 28% of the
headline MAE, with example orders spanning ~70→104 min by courier. So the matrix
columns are **not** near-constant: distance, load, and courier history give the
Hungarian step real signal. But that ~13-min discriminating signal is **modest next
to the model's ~47-min absolute error** — much of which is *common-mode* (shared by
all couriers on an order and cancelling in the ranking). The assignment is meaningful
but operates at limited signal-to-noise.

### The data-ceiling conclusion

These findings point at the same root cause: **LaDe's accept→pickup target is
window-scheduling, not courier ETA, and the dataset lacks the features that would
separate couriers finely** — live GPS trajectory, queue depth, traffic, intra-batch
sequencing. The model is near the ceiling of what this data supports. The lever for
better *dispatch* is therefore **richer courier-specific, real-time features**, not a
smaller window-driven MAE — and the honest evaluation metric is assignment regret,
not raw error.

---

## 7. Serving, assignment, and the cross-product split

The deployed scoring path mirrors the local one exactly:

1. The dispatcher holds the raw snapshot (free couriers + open orders + `as_of_time`)
   and the per-courier running aggregates.
2. It POSTs that to `/score`. The API builds **every** (courier, order) feature row
   and runs **one** vectorized `model.predict` — never 20 calls for 20 pairs.
3. The API returns flat scored pairs; the dispatcher reshapes them into `cost[i][j]`
   and runs the Hungarian algorithm locally.

`/debug` scores a single pair and echoes the feature row, so the deployed prediction
can be checked against the local model bit-for-bit (the parity test asserts this).

**Why Hungarian over greedy:** greedy takes the globally cheapest cell first and can
strand later orders with expensive ones. On the matrix `[[1,2],[2,4]]`, greedy picks
`(0,0)=1` then is forced into `(1,1)=4` (total 5); Hungarian picks the off-diagonal
for **4**. The test suite checks Hungarian is strictly better there and ≤ greedy
across 50 random matrices, and a live snapshot showed ~7% lower total predicted time.

---

## 8. Streaming and the event schema

A **pluggable bus** abstracts the broker: `InMemoryBus` for tests and local runs,
`KafkaBus` (Redpanda in docker-compose) for the full stack — the producer and
dispatcher are written once.

The producer replays each historic order as up to three time-ordered events:
`courier_location` (at accept, from `accept_gps`), `order_available` (at
`window_start`), and `order_picked_up` (at `pickup_time`).

**Schema decision:** the original `order_picked_up` event carried no location, but
`distance_to_pickup` needs the courier's current position at tick time, and the
Kafka events were the only live source. We therefore extended the schema to carry
courier GPS (a `courier_location` event plus `lat/lng` on `order_picked_up`). This
keeps serving consistent with training, where the same quantity comes from
`accept_gps`.

---

## 9. Monitoring — closing the loop

At training time `train.py` saves a **reference profile**: per-feature distributions
(quantile bin edges + proportions) and the baseline test MAE.

- **Operational health** — the dispatcher records per-tick scoring latency,
  throughput, assignments per tick, and late pickups.
- **Model performance** — the dispatcher's logger/joiner pairs each booked
  prediction with the realized pickup (predicted vs realized).
- **Data drift** — PSI of live feature/target distributions against the reference
  profile, with conventional bands (0.10 moderate, 0.25 significant).
- **Retraining trigger** — fires on significant feature/target drift or live-MAE
  degradation, which is what closes the loop.

The Streamlit dashboard is pure visualization on top of these functions. Drift
detection demonstrably fires on the region/AOI distribution shift between the
training months and October.

---

## 10. Known limitations (stated honestly)

- **Replay realized ≠ training target.** In offline replay an order's `pickup_time`
  is fixed by history, but the dispatcher assigns at tick boundaries that lag the
  historical accept. So the dispatcher's "realized = pickup_time − assign_time" is an
  **operational signal**, not a clean accuracy number comparable to the 47.5-min test
  MAE. The `late_pickups` counter makes that lag visible. Real online evaluation
  would measure realized pickup against the actual assignment moment.
- **Courier availability is a simulation.** LaDe gives no explicit free/busy state;
  "free couriers" are reconstructed as those recently active near time *t*. The
  mechanism (snapshot → assignment) is faithful; the exact availability set is a
  modeling choice documented in `snapshot.py`.
- **The target is window-dominated, and that is a data ceiling** (see §6). Pickups
  are governed by a fixed 2-hour window, the model over-predicts short pickups
  (352% MAPE in decile 0), and cross-courier discrimination is only ~13 min next to a
  ~47-min error. The model is near the limit of what LaDe supports; better dispatch
  needs richer real-time courier features, not a smaller window-driven MAE.
- **Geographic generalization.** Strong region/AOI PSI between months indicates the
  courier/region mix shifts over time — exactly what the monitoring layer is built to
  catch and act on via retraining.

## 11. Future work

Online/shadow evaluation against true assignment moments; courier-capacity-aware
assignment (penalize overloaded couriers in the cost matrix); incremental/scheduled
retraining wired to the drift trigger; and quantile predictions to expose
uncertainty to the optimizer.

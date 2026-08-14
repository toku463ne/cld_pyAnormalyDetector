# DETECTION.md — Data Collection, Detection and Clustering

Reference for what the code actually does, as implemented. Formulas and defaults
are taken from the source; file references point at the authoritative definition.

Two independent detection axes share the same stores and the same clustering:

| Axis | Entry point | Cadence | Purpose |
|---|---|---|---|
| **Slow** (main) | `anomdec-detect` → `pipeline/detection.py` | hourly | Sweep every item, score, gate, cluster, persist |
| **Fast** | `anomdec-detect-fast` → `pipeline/fast_detection.py` | every 5–10 min | Small watchlist, short window, co-occurrence, JSON for Zabbix |
| **Stats batch** | `anomdec-update-stats` → `pipeline/stats_update.py` | 1–2×/day | Rebuild the baselines both axes read |

The invariant behind the split: **the hourly path never fits a model and never
re-aggregates a full window.** It reads pre-computed statistics and does
arithmetic. All heavy aggregation lives in the daily batch.

---

## 1. Data collection

### 1.1 Sources

`ingestion/base.py` defines the `DataSource` protocol; implementations are
selected by `type` in config (`ingestion/factory.py`).

| type | class | notes |
|---|---|---|
| `zabbix_psql` | `ZabbixPsqlSource` | reads Zabbix PostgreSQL directly |
| `csv` | `CsvSource` | test/offline datasets |
| `logan` | `LoganSource` | goLogAnalyzer log data |

`ZabbixPsqlSource` (`ingestion/zabbix_psql.py`):

- `get_history()` — `UNION ALL` over `history` + `history_uint`, returns
  `(itemid, clock, value)` sorted.
- `get_trends()` — `UNION ALL` over `trends` + `trends_uint`, returns
  `(itemid, clock, value_min, value_avg, value_max)`.
- `get_item_details()` — joins `hosts`/`items`/`hosts_groups`/`hstgrp`. The
  schema version is probed once (`dbversion.mandatory`) to pick `groups` vs
  `hstgrp`.
- `get_events()` — active PROBLEM events (`source=0, object=0, value=1`),
  `DISTINCT` on `eventid` to collapse the trigger→functions fan-out.

Note: `get_item_details` maps `items.key_` into **both** `item_name` and `key_`
on `ItemDetail` (`ingestion/zabbix_psql.py:98-108`). Every pattern match in the
system therefore matches against the technical key, not the display name.

Every fetch is chunked by `batch_size` (default 100) — no full-corpus load.

### 1.2 Management DB (`admdb`)

Tables are named `{ds_name}_{suffix}` (`store/base.py`).

| Table | Columns | Written by |
|---|---|---|
| `{ds}_history` | `itemid, clock, value` | hourly (cache for changepoint/clustering) |
| `{ds}_history_stats` | `itemid, sum, sqr_sum, cnt, mean, std` | hourly |
| `{ds}_trends_stats` | `itemid, sum, sqr_sum, cnt, mean, std, intra_std` | daily |
| `{ds}_hour_stats` | `itemid, hour_of_day, mean, std, cnt` | daily |
| `{ds}_history_updates` | `id=1, startep, endep` | hourly (watermark only) |
| `{ds}_trends_updates` | `id=1, startep, endep` | daily (watermark only) |
| `{ds}_anomalies` | `itemid, created, group_name, hostid, clusterid, host_name, item_name, trend_mean, trend_std, score, detector_scores JSONB, rescued` | hourly |

### 1.3 Sliding-window statistics

`features/rolling_stats.py` recomputes mean/std over the retention window
`[startep, endep]` on every run, from the window both callers already fetch
(`get_trends(startep, endep)` / `get_history(startep, endep)`):

```
mean = sum / cnt
var  = max( (sqr_sum - sum²/cnt) / max(cnt-1, 1), 0 )
std  = sqrt(var)
```

`sum`, `sqr_sum` and `cnt` are stored per item but describe the **current
window**, not an all-time total. The variance is clipped **before** the square
root: floating-point cancellation in `sqr_sum - sum²/cnt` can go slightly
negative and produce `NaN`.

Items that report **nothing** inside the window are handled explicitly. The
caller passes the batch it asked for as `expected_item_ids`; any of those items
missing from the fetched frame has its stats row **deleted**, and the newest
sample in the window is stored as `last_clock`. Leaving the row behind would
freeze `mean` at the last value the item ever reported while the baseline kept
sliding — see §8.5. When `expected_item_ids` is not supplied the old behaviour
holds and an empty window changes nothing.

**`intra_std` — the spread `std` throws away.** Trends rows are hourly
aggregates, so `std` measures how much the hourly *average* moves. It says
nothing about how far an individual sample strays inside an hour, and that is
the quantity every consumer of raw history actually needs. For a metric that
idles most of the hour and bursts for a few minutes the two differ by 8-30x.

```
intra_std = mean(value_max - value_min) / trends_range_to_sigma     # default 4.0
```

The range rule `sigma = E[max - min] / d2(n)` recovers it from the min/max
Zabbix already stores. `d2` depends on samples-per-bucket (2.53 at 6/h, 3.26 at
12/h, 4.64 at 60/h) and that count is not recorded, so the divisor is one
mid-range constant. The **mean** range is deliberate: it is inflated by the
occasional violent hour, which is exactly the metric class this holds back.

Consumers combine the two components in quadrature — independent by the law of
total variance — via `features/baseline.py::baseline_sigma`:

```
sigma = hypot(trends_stats.std, intra_std)
```

and **fail open to `std`** when `intra_std` is NULL: rows written before the
column existed, and `history_stats` rows, which are raw samples and have no
within-bucket spread by construction. Only trends carry it.

The critical property is that samples ageing out of the back of the window stop
contributing: that is what lets a step change be absorbed into the baseline
after `trends_retention` days, so the item stops being flagged. An earlier
version kept the accumulators across runs and tried to subtract the rows that
had aged out — see §8.1.

### 1.4 Seasonal baseline

`features/hour_stats.py` — recomputed from scratch each daily run (a cheap
`GROUP BY`, not incremental):

```
hour_of_day = (clock % 86400) // 3600
(mean, std, cnt) = agg(value_avg) grouped by (itemid, hour_of_day)
```

`hour_of_day` is derived from the raw epoch, i.e. it is **UTC-based**, not local
time. Day-of-week is not modelled.

---

## 2. Detection (hourly)

`pipeline/detection.py:_run_for_source` — order of operations:

```
item_ids
  → update history_stats (window recompute)
  → read trends_stats, history_stats, hour_stats[current_hour]
  → item_filters                       (drop items entirely)
  → ZScoreDetector    (O(1)/item)
  → SeasonalDetector  (O(1)/item)
  → ChangepointDetector (only items already scoring > 0; needs raw history)
  → EnsembleDetector  (weighted combine → raw_score)
  → gating            (category weight × magnitude × duration → effective score)
  → anomaly_filters   (absolute-difference floor)
  → clustering + rescue
  → persist
```

Changepoint runs only on the union of items scored by the two cheap detectors,
so raw history is fetched for a small subset rather than the whole corpus.

### 2.1 The shared score ramp

Both z-score detectors convert a z into a score with the same ramp:

```
score = 0                                        if z <  λ
score = min( (z - λ)/λ × 0.5 + 0.5 , 1.0 )       if z >= λ
```

so `z = λ → 0.5`, `z = 2λ → 1.0`, and **everything at or beyond `2λ` is exactly
1.0**. With the default `λ = 3.0` the score saturates at `z ≥ 6`. This is the
intended shape (a bounded severity), but it means the score carries no ranking
information in the tail — see §8.2.

### 2.2 ZScoreDetector

`detectors/zscore.py` — recent level vs. long-term baseline.

```
z = |history_stats.mean - trends_stats.mean| / trends_stats.std
```

Guards, in order:

- drop items with `trends_stats.cnt <= 0` or `std <= 0`;
- drop items whose relative change is negligible:
  keep only if `trend_mean == 0` **or** `|Δ| / |trend_mean| > min_ignore_rate`
  (default 0.05). When the baseline mean is exactly zero the rule is bypassed
  and the item is always kept.

Features emitted: `z`, `h_mean`, `t_mean`.

### 2.3 SeasonalDetector

`detectors/seasonal.py` — recent level vs. the baseline **for the current hour of
day**, so a metric that always rises at 09:00 is not flagged at 09:00.

```
z = |history_stats.mean - hour_stats.mean| / hour_stats.std      (current hour only)
```

Same ramp as §2.1. Items with no `hour_stats` row, or `std == 0`, are simply
absent from the output (no score) — the detector fails closed here, unlike the
gates which fail open.

Cost is one pre-fetched DB read; the pipeline reads only
`hour_of_day = (endep % 86400) // 3600`.

### 2.4 ChangepointDetector

`detectors/changepoint.py` — two-sided CUSUM over the raw recent history,
centred on the trends baseline.

```
sigma    = hypot(trends_stats.std, intra_std)      # §1.3 — raw-sample scale
w        = sample_interval(clocks) / history_interval
slack    = cusum_k × sigma × w                     (default k = 0.5)
decision = cusum_h × sigma                         (default h = 5.0)

for v in values:
    dev    = (v - trend_mean) × w
    s_pos  = max(0, s_pos + dev - slack)
    s_neg  = max(0, s_neg - dev - slack)
    s_max  = max(s_max, s_pos, s_neg)

score = 0                                                  if s_max < decision
score = min( (s_max - decision)/decision × 0.5 + 0.5, 1.0 ) otherwise
```

This rewards **sustained** deviation: a single outlier adds one `dev` then
decays against the slack, whereas a level shift accumulates every sample. Cost
is O(`history_retention`) per candidate item.

Two properties the test depends on, both of which were once wrong and made the
detector fire on essentially everything (§8.7):

**`sigma` is the raw-sample scale, not `trends_stats.std`.** A CUSUM only works
because the slack `k·sigma` exceeds the typical per-sample deviation, giving the
accumulator negative drift under H0 — that is what makes `s_max` stationary and
lets one fixed `decision` mean the same thing at any window length. Feed it a
sigma computed from hourly *averages* and the slack falls far below the sample
noise of a bursty metric, the drift turns positive, and `s_max` grows linearly
in N. The statistic stops being a changepoint test and becomes a sample counter.

**The accumulator integrates over time, not over samples.** Zabbix items are
collected at whatever interval their template says. Weighting each step by
`dt / history_interval` makes the statistic depend on what the metric did, not on
how often it was polled — otherwise a 60 s item scores 10x a 600 s item watching
the identical event. `history_interval` only anchors what `cusum_h` means; only
the *ratio* to the item's real spacing affects the result.

An item is skipped when `sigma <= 0` (was `trend_std <= 0`).

### 2.5 EnsembleDetector

`detectors/ensemble.py`. Weights of *enabled* detectors are normalised to sum to
1 at construction. Per item, only detectors that actually produced a score > 0
("contributing") take part:

```
final = Σ_contributing (score_d × w_d) / Σ_contributing w_d
is_anomaly = final >= min_score   AND   |contributing| >= require_any
```

**Behavioural consequence worth knowing:** the denominator is the contributing
weight, not the total weight. A silent detector does not dilute the score — one
detector at 1.0 yields `final = 1.0`. `require_any` is the only thing preventing
single-signal flags; the default `require_any: 2` makes the ensemble behave like
a 2-of-3 cascade.

Defaults (`default.yml`): weights zscore 0.3 / changepoint 0.3 / seasonal 0.4,
`min_score: 0.7`, `require_any: 2`.

### 2.6 Gating — category, magnitude, duration, idle baseline

`features/gating.py`. Pure functions, used identically by the pipeline and the
backtester so offline evaluation matches runtime.

```
effective_score = raw_ensemble_score × category_weight × magnitude_scale
                                     × duration_scale × idle_scale
is_anomaly      = effective_score >= ensemble.min_score
```

The raw score and all four multipliers are preserved in `features`
(`raw_score`, `gate_weight`, `mag_scale`, `dur_scale`, `idle_scale`, `delta`).

**Ramp helper** — `ramp(x, lo, hi)`: 0 at/below `lo`, 1 at/above `hi`, linear
between; degenerates to a hard threshold at `hi` when `hi <= lo`.

**Category** (`classify`) — first category whose `key_patterns` fnmatch-match
`items.key_` wins; order in `default.yml` is therefore significant. Unmatched
items get `default_weight`.

fnmatch anchors at the start of the key, so `vfs.dev.*` does **not** match
`vmware.vm.vfs.dev.write[...]`. Every vendor-prefixed variant has to be spelled
out; missing one silently drops the item into the catch-all, which is what
produced 62 of the 93 detections in the 2026-08-14 10:46 cycle (§8.6).

**Magnitude** — always driven by the *change from baseline*
`Δ = |recent_mean - trend_mean|`, never the current absolute level, so a host
sitting steadily at a high value (`Δ ≈ 0`) is not flagged:

| mode | normalised quantity |
|---|---|
| `absolute` | `Δ` in native units |
| `relative` | `Δ / max(|trend_mean|, ε)` |
| `sigma` | `Δ / max(trend_std, ε)` |

`mag = max(ramp(x, lo, hi), floor)`.

A collapse to zero always scores `mag = 1.0` under `relative` mode
(`Δ/baseline ≈ 1`), which is why container/packet/session categories are
declared `relative` and placed **before** the byte-based `network`/`cpu` rules.

**Duration** — how long the item stayed outside
`trend_mean ± sigma·baseline_sigma` inside the window; `count` (total anomalous
samples) or `consecutive` (longest run), times **that item's own median clock
gap**, then ramped `lo_secs → hi_secs`. Default `lo_secs 600 / hi_secs 3600`: a
single 10-minute spike is suppressed, a sustained hour scores full.

Both inputs are per-item measurements rather than config values, for the same
reason as §2.4. The band uses `baseline_sigma` (§1.3) because it is tested
against raw samples, and the spacing comes from
`features/baseline.py::sample_interval` because `history_interval` is a *window
sizing* parameter — `retention × interval` is how far back to fetch — not a
promise about collection frequency. Reading it as a per-sample duration scaled
every count by `configured / real`: 10x on a Zabbix collecting at 60 s under the
default 600, which made the gate a no-op (§8.7). The measured value is recorded
in `features.sample_secs`, the sigma in `features.baseline_sigma`.

**Idle baseline** — a metric that reads zero whenever its resource is idle
(VMware guest disk latency, outstanding-IO depth, a rarely-used counter) has a
baseline mean near zero, so *any* activity is a relative change of tens or
hundreds. No relative threshold can discriminate on such a series, and no
absolute one can either without knowing the unit.

The unit-free question that does discriminate is whether the current level is
**unprecedented**. When `zero_cnt / cnt >= max_zero_ratio` over the trends
window, the item is suppressed unless `recent_mean` exceeds `max_value`, the
highest value anywhere in that window. A normally-zero error counter that spikes
off the scale still fires; the routine idle→busy transitions do not.

`zero_cnt` and `max_value` are computed by the same daily `GROUP BY` that
produces `trends_stats` (§1.3), so this costs nothing at detection time.

All four gates **fail open** (`scale = 1.0`) when the evidence is missing — no
baseline stats, no raw history, `trend_std <= 0`, or a `trends_stats` row
written before `zero_cnt`/`max_value` existed. A real anomaly is never
suppressed for lack of evidence.

### 2.7 Filters

`pipeline/filters.py`. Both match on `key_` (fnmatch glob) AND `units` (exact);
an empty pattern matches everything.

- **`item_filters`** — applied *before* detectors. Unconditional exclusion when
  `min_value` is unset; otherwise exclude only when `recent_mean < min_value`
  (e.g. ignore interfaces below 8 Mbps).
- **`anomaly_filters`** — applied *after* gating. Drop a score when
  `|recent_mean - trend_mean| < min_abs_diff` (e.g. CPU % moves under 8 points).

Filters are the operational-significance layer; gates are the statistical one.

---

## 3. Clustering

`clustering/dbscan.py` — items whose **shapes co-move** belong to the same
incident. Correlation-primary; the older 2-stage Jaccard-then-correlation design
was removed (sparse spikes vanished under resampling and blocked genuinely
co-moving items from ever reaching the correlation stage).

### 3.1 Chart construction

`_build_charts` puts every item on one clock grid, so items collected at
different periods (60 s vs 600 s) are compared at the same wall-clock times:

1. `unitsecs = max over items of (median clock gap)` — the coarsest real
   resolution, so nothing is upsampled beyond its true sampling rate;
2. bucket `clock // unitsecs`, average within bucket;
3. reindex onto the full grid, `interpolate(limit_direction="both")`.

The window is `endep - clustering.detection_period` … `endep` (default 12 h).
Trends are **not** prepended — an earlier version prepended 14 d of hourly
trends for "pre-anomaly shape", but most anomalous items are flat at baseline
for those days, so correlation collapsed onto the single shared spike and
unrelated shapes merged at ~0.9.

### 3.2 Distance — two channels

All series truncated to the shortest length; distance is
`dist = (1 - ρ) / 2`, mapping ρ ∈ [-1, 1] onto [0, 1].

**Channel 1 (primary): Spearman on first differences.**
Raw infra series share a slow non-stationary drift (memory creeping, counters
trending) that makes unrelated items look correlated — differencing removes it.
Ranking (rather than Pearson) matters because the window is short and, during an
incident, many unrelated metrics get one or two large coincident spikes; Pearson
is dominated by those extremes and reports ~0.9 between shapes that only touch
at the spikes. Ranks flatten them, so an item must co-move across the **bulk** of
the window.

**Channel 2 (gated): Spearman on raw levels.**
Differencing cannot group monotonic ramps — a cumulative counter differences into
a roughly constant increment series whose rank order is noise, so genuine ramp
groups fragment into singletons. Raw levels correlate ~1.0 for such ramps, but so
do *unrelated* ramps (everything trending up is rank 1..N over a short window).
So the channel is gated twice: it may only lower a distance when the two items
**share a metric family** and their raw correlation is `>= raw_corr_min` (0.99).

```
family(key) = key up to the first '['      # docker...throttling_periods[c1] → docker...throttling_periods

dist = (1 - spearman(diff_i, diff_j)) / 2
if family_i and family_i == family_j:
    raw = spearman(raw_i, raw_j)
    if raw >= raw_corr_min:
        dist = min(dist, (1 - raw) / 2)
```

A zero-variance (flat) row yields correlation 0 by construction.

### 3.3 Onset constraint

Correlation says nothing about **when** each item started misbehaving. Inside a
12 h window a disk ramp that began a week ago and a step change from yesterday
are both just gentle drift, so unrelated items merge — the raw channel is
especially prone to this, since any two monotone series rank-correlate at ~1.0.

`features/onset.py` computes, per item, **when its current level regime began**,
and `_apply_onset_constraint` forbids an edge between two items whose onsets
differ by more than `max_onset_gap` (default 7200 s):

```
recent_level = median of the last `onset_recent_samples` trends samples
band         = onset_level_tol × |recent_level|        (fallback: sigma × trend_std when level ≈ 0)
onset        = walk back from the newest sample; the regime ends after more than
               `onset_tolerance` consecutive out-of-band samples
```

Onsets come from **trends** (hourly, `trends_retention` days), not history: by
the time an item is flagged, its onset is usually days old and outside the
history window entirely.

Why not the obvious "outside `mean ± sigma·std`" rule: a *sustained* shift
inflates the very std it would be tested against. An item at 1.5 GB for 6 days
then 5.2 GB for 8 days has a window std of ~1.8 GB, so neither level is 2σ from
the window mean and no sample is ever anomalous — the rule resolves nothing for
exactly the case this exists to handle. Robust statistics do not help either:
once the shift occupies most of the window, the median declares the *new* level
normal and the old one the anomaly. Anchoring on the current level and walking
backwards avoids both traps, and for a step change the result is essentially
independent of `onset_level_tol` (`tests/unit/test_onset.py`).

Applied **after** `_normalise` so rescaling cannot compress a forbidden pair back
under `eps`. It can only ever split, never merge. **Fails open**: a pair is
severed only when *both* onsets are known — an unresolved onset never isolates an
item on its own.

Set `max_onset_gap: 0` to disable. The fast axis does not pass onsets (its whole
signal is short-span co-occurrence), so it is unaffected.

### 3.4 DBSCAN

`DBSCAN(eps=corr_eps, min_samples=min_samples, metric="precomputed")` over the
distance matrix; label `-1` = noise. Items with no chart are also `-1`.

`corr_eps` defaults to **0.10**. The older 0.2 was tuned when trends were
prepended; on the short window the distance distribution shifts and 0.2 merges
everything that co-spikes into mega-clusters.

`_normalise` rescales the matrix only when its span exceeds 1.0, and maps `NaN`
to distance 1.0.

Ordering caution: the distance matrix rows follow `list(charts.keys())`, and
`db.labels_` is zipped back against that same order. Any reordering between the
two misattributes labels (`clustering/dbscan.py:70-73`).

### 3.5 Incident rescue

`clustering.rescue_same_incident` (default on). Some items are genuine members of
a confirmed incident but got pushed below threshold by the **magnitude gate
alone**. `magnitude_suppressed()` identifies them:

```
not is_anomaly
AND raw_score >= min_score
AND mag_scale < 1.0
AND raw_score × gate_weight × dur_scale >= min_score      # would have passed without magnitude
```

The pipeline clusters `confirmed ∪ candidates` together, then `select_rescued()`
promotes any candidate sharing a non-noise cluster with a confirmed item. Rescued
rows are persisted with `rescued = TRUE`.

This is the one place where clustering feeds back into the anomaly decision.

---

## 4. Persistence

Anomalies are written to `{ds}_anomalies` with the ensemble score, the
per-detector breakdown as JSONB, and `rescued`. Cluster ids are assigned in a
second pass (`update_cluster_ids`), scoped to the cycle being written — without
that scope the reset step wiped the cluster ids of every retained row from
earlier cycles. Rows older than `anomaly_keep_secs` (default 1 day) are deleted
each run.

---

## 5. Fast axis

`pipeline/fast_detection.py` (shell) + `detectors/fast.py` (pure). Keeps **no DB
state of its own** — it only reads `hour_stats` written by the daily batch.

```
short history window (history_span_secs, default 3600)
  → build_short_stats : last `detect_window` samples = recent; earlier = baseline
  → compute_severity  : ZScoreDetector reused, recent vs. that in-window baseline
  → seasonal_veto     : drop levels expected for this hour-of-day
  → cluster_anomalies : same clustering module, co-occurrence grouping
  → score_events      : noisy-OR over members
  → JSON file for Zabbix to poll ($.max_score)
```

**Seasonal veto** runs *before* event scoring, so recurring backup traffic is
removed item-by-item and never inflates a co-occurrence event. An item is
suppressed when `|recent - hour_mean| / hour_std < seasonal_lambda`. Fail-open:
no usable hour baseline → keep the item.

**Event score** — items sharing a cluster form one event, noise items become
singletons:

```
event_score = 1 - Π(1 - s_i)
```

so corroborating items raise the score (the intended co-occurrence boost).

**Zabbix events** (optional) fold host corroboration into the same noisy-OR:

```
host_weight = 1 - exp( -Σ(clip(severity,0,5)/5) / events_saturation )
```

A storm contributes strongly but saturates below 1; low-severity noise stays
small. Hosts with `weight >= min_event_score` and no metric anomaly produce
standalone `zabbix_events` entries.

---

## 6. Configuration

`config/loader.py`, schema in `config/schema.py` (Pydantic v2). Resolution order:

1. `default.yml`
2. user `config.yml` (deep-merged)
3. secrets from `ANOMDEC_SECRET_PATH` (or `secret_path` in defaults)
4. Jinja2 rendering with `{**os.environ, **secrets}` as context

Jinja2 is rendered on the **raw file text** before YAML parsing — rendering after
`yaml.dump()` breaks expressions like `{{ x | default('y') }}`.

`_cascade_defaults` copies top-level keys into each `data_sources` entry when not
overridden there: `batch_size`, `history_interval`, `history_retention`,
`trends_retention`, `anomaly_keep_secs`, `staleness_secs`, `detectors`, `ensemble`, `clustering`,
`metric_categories`, `item_filters`, `anomaly_filters`, `fast_detect`,
`dashboards`.

### Main tuning knobs

| Key | Default | Effect |
|---|---|---|
| `history_interval` / `history_retention` | 600 / 18 | recent window ≈ 3 h |
| `trends_retention` | 14 | days of baseline |
| `trends_range_to_sigma` | 4.0 | divisor turning the mean hourly range into `intra_std` (§1.3); raise to suppress more |
| `staleness_secs` | 3600 | skip items whose newest history sample is older than this (0 = off, §8.5) |
| `detectors.*.lambda_threshold` | 3.0 | z at which score = 0.5; saturates at 2λ |
| `detectors.changepoint.cusum_h` / `cusum_k` | 5.0 / 0.5 | decision / slack, in std units |
| `ensemble.min_score` | 0.7 | precision-first threshold |
| `ensemble.require_any` | 2 | detectors that must agree |
| `metric_categories.categories[].weight` | per-category | prior importance by metric type |
| `metric_categories.duration.*` | 600 → 3600 s | suppress brief spikes |
| `metric_categories.idle_baseline.max_zero_ratio` | 0.8 | baseline zero more often than this → require an unprecedented level (§2.6) |
| `clustering.corr_eps` | 0.10 | DBSCAN radius on correlation distance |
| `clustering.raw_corr_min` | 0.99 | gate for the raw-level channel |
| `clustering.detection_period` | 43200 | seconds of history used for clustering |
| `clustering.max_onset_gap` | 7200 | max onset difference for two items to share a cluster (0 = off) |
| `clustering.onset_level_tol` | 0.1 | relative band defining "still the same level" |
| `lock_dir` | `/tmp/anomdec/locks` | where the single-instance run locks live (§7) |

---

## 7. Concurrency

Each entry point holds an exclusive `flock` for its whole run
(`pipeline/lock.py`), keyed by command name under `lock_dir`
(default `/tmp/anomdec/locks`):

| Command | Lock file |
|---|---|
| `anomdec-detect` | `<lock_dir>/detect.lock` |
| `anomdec-detect-fast` | `<lock_dir>/detect-fast.lock` |
| `anomdec-update-stats` | `<lock_dir>/update-stats.lock` |

Separate locks, so the fast axis is never blocked by the hourly sweep. A blocked
run does nothing and exits `75` (`EX_TEMPFAIL`); `--wait SECS` queues behind the
holder instead. `--init` runs inside the lock too — recreating tables under a
live run would be worse than racing on accumulators.

`flock` over a PID file because the kernel releases it however the process dies;
a stale PID file from `kill -9` would block every later run. Scope is one host —
see §8.1 for why concurrent runs matter, and note that cross-host coordination
would need a PostgreSQL advisory lock instead.

---

## 8. Known issues and characteristics

### 8.1 The baseline never slid (fixed)

Symptom: items that stepped to a new level **weeks** earlier stayed on the
hourly dashboard indefinitely. Anomaly rows are only kept for
`anomaly_keep_secs` (1 day) and `anomdec-publish-dashboard` renders only the
latest cycle, so these were not stale rows — they were being genuinely
re-detected every hour against a baseline that never moved.

Two independent causes, both in the stats update path:

1. **`{ds}_updates` was shared by both pipelines.** `UpdatesStore` resolved to
   one table with one row (`id = 1`), written by `DetectionPipeline` (≈ 3 h
   window, hourly) *and* `StatsUpdatePipeline` (14 d window, daily). Each
   overwrote the other's watermark. Since `diff_startep = old_endep + 1`, the
   daily trends batch running after an hourly detection saw `diff_startep ≈
   now` and ingested ~1 trends sample per day — `trends_stats` was effectively
   frozen at whatever it held when the item was first inserted.

2. **The subtract-old-data slice could never be populated.** It selected rows in
   `[old_startep, startep)` out of `data_df`, but `data_df` came from
   `get_trends(startep, endep)` / `get_history(startep, endep)` — every row had
   `clock >= startep`. So nothing was ever subtracted and `sum`/`sqr_sum`/`cnt`
   only grew: a cumulative all-time statistic, not a window.

Either one alone keeps the pre-step level in `trend_mean` forever, so the
ZScore detector keeps seeing a large deviation from a baseline that will never
update.

The offline evaluation path never showed this: `evaluation/backtester.py`
computes its stats with a plain `groupby().mean()` over the window and bypasses
`update_rolling_stats` entirely, so backtests measured a correct sliding window
that production did not have.

**Fix:** `{ds}_updates` is split into `{ds}_history_updates` and
`{ds}_trends_updates` (as `CLAUDE.md` always specified), and
`update_rolling_stats` recomputes from the window in `data_df` instead of
carrying accumulators across runs. The incremental path saved nothing in
practice — both callers already fetched the whole window in order to do the
subtraction. Covered by `tests/unit/test_rolling_stats.py`, which previously
did not exist.

**Migration:** self-healing. The new watermark tables start empty and are only
informational now; the first `anomdec-update-stats` run after deploying
recomputes every item's `trends_stats` from the true 14-day window. The orphaned
`{ds}_updates` table can be dropped by hand.

### 8.2 Score saturation

Because the ramp reaches 1.0 at `z = 2λ` (§2.1), most flagged items land on
exactly 1.0. Measured over 12 human-reviewed days of production label queues
(`datasets/queues/`), **341 of 360 queued items had score exactly 1.0** — the
score cannot rank within the flagged set, and per-queue threshold tuning has
nothing to bite on. Raising `lambda_threshold` moves the saturation point; an
unbounded or log-scaled tail would restore ranking.

### 8.3 Precision is metric-type dependent

Over the same 12 reviewed days, precision@30 was **0.508** overall but varied
sharply by key prefix: `docker` 0.91, `unbound` 0.83, `system` 0.71 versus
`vfs` 0.33, `vmware` 0.19, `net` 0.05. `net.*` alone produced a third of all
false positives. This is what `metric_categories[].weight` and the magnitude
gates exist to correct; the current category thresholds in `default.yml` are
explicitly marked as placeholders awaiting backtester tuning.

### 8.4 Seasonal baseline is UTC and day-agnostic

`hour_of_day` comes straight from the epoch (§1.4), so the seasonal baseline is
aligned to UTC rather than site-local time, and weekday/weekend patterns are not
separated. Both are noted as future extensions in `CLAUDE.md`.

### 8.5 Items that stopped reporting were flagged forever (fixed)

Symptom: in the production export of 2026-08-14, **37 of 83 flagged items** came
from three hosts (`new-ubu-24-1`, `new-ubu-24-2`, `IMTDB123`) that had stopped
sending data roughly five days earlier. They had zero history samples in the
3-hour window and their newest trends point was 87–119 hours old, yet every one
of them scored exactly 1.0, every hour.

The signature is visible in the detector breakdown: all 37 fired on
`{zscore, seasonal}` and **none** on `changepoint` — changepoint needs the raw
series, which does not exist for a silent item, so it cannot vote.

Cause: an upsert can only touch items present in the fetched frame, and
`_upsert_window` had no delete path (§1.3). `DetectionPipeline._update_history_stats`
made it worse with `if hist_df.empty: continue`, which skipped a whole batch of
dead items. So `{ds}_history_stats` kept its last-good row indefinitely: a frozen
`h_mean` compared against a `trends_stats` / `hour_stats` baseline that kept
sliding. Both DB-stat detectors saturate on that, and the duration gate fails
open when there is no series to measure (§2.6), so nothing downstream could
catch it.

**Fix**, in two layers:

1. `update_rolling_stats` takes `expected_item_ids` and deletes the stats rows of
   items with no sample in the window; the empty-batch skip is gone. Applied to
   `history_stats` (hourly) and to `trends_stats` + `hour_stats` (daily).
2. A freshness guard: `last_clock` is stored alongside the window stats, and
   items whose newest sample is older than `staleness_secs` (default 3600) are
   dropped before the detectors run. Layer 1 catches complete silence; layer 2
   catches an item still dribbling one sample per window.

Skipped items are counted and logged with their ids rather than silently
discarded. Detecting "collection stopped" as an alert in its own right is a
separate concern — Zabbix's `nodata()` trigger is the right home for it.

**Migration:** self-healing. `last_clock` is added with
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` and a missing value counts as stale;
the first run after deploying purges the accumulated rows.

Covered by `tests/unit/test_rolling_stats.py`, `tests/unit/test_staleness.py`,
and `tests/unit/test_staleness_production_sample.py`, which pins the numbers
above against the real export.

### 8.6 Near-zero baselines and mis-scaled categories (fixed)

The cycle immediately after §8.5 produced 93 flagged items, essentially all of
which a reviewer marked as noise. Three causes:

1. **62 VMware guest storage/disk items** — `vmware.vm.storage.*`,
   `vmware.vm.vfs.dev.*`, `vmware.hv.datastore.*` matched **no** category,
   because fnmatch anchors at the start and the `disk` patterns were
   `vfs.dev.*` / `vfs.fs.*`. They fell into the catch-all (`relative`, lo 0.5)
   and their baselines are zero 81–100 % of the trends window
   (`trend_mean` ≈ 0.003), so the relative change was 10–472×.
2. **9 `vmware.vm.cpu.usage` items** — reported in **Hz** (1e8–1e9) but matched
   by the `cpu` category, whose `absolute` lo 10 / hi 40 is written for
   percentage points. Δ = 7e6 Hz saturates the ramp, so a 0.6 % move scored
   full magnitude. Only `vmware.vm.cpu.usage.perf` is a percentage.
3. **A tail of low-count metrics** — `proc.num[sshd]` going 1 → 3 sessions,
   `mysql.com_insert.rate` moving 0.007/s. Large relative change, trivial
   absolute change.

Fixes: the idle-baseline gate (§2.6) for (1), a `cpu_perf` / `cpu_hz` split for
(2), and key-scoped `min_abs_diff` rules in `anomaly_filters` for (3). Replaying
the export through the real gating code: **93 → 14 after the gates → 6 after the
filters**. Pinned by `tests/unit/test_gating_production_sample.py`.

What remains is dominated by one host (`IMTDB123`, four mssql items) whose cache
behaviour genuinely stepped. There is no host or group predicate in
`item_filters` / `anomaly_filters` — only `key_` and `units` — so "mute this
host" is currently inexpressible.

### 8.7 Everything was measured against the wrong sigma and the wrong clock (fixed)

The 2026-08-14 15:29 cycle flagged 31 items and a reviewer rejected all of them.
They were not a mixed bag: every one was a metric that idles most of the hour and
bursts for a few minutes — mssql latch/page rates, Windows disk-queue counters,
SQL batch requests, VMware guest CPU.

Two numbers gave it away. **All 31 scored `changepoint: 1.0`**, and **all 31 had
`dur_scale = 1.0`**. Neither was measuring anything.

**Cause 1 — `trend_std` is not the sample scale.** `trends_stats.std` is the
spread of hourly *averages*. Averaging an hour of a bursty metric flattens the
bursts, so it is one to two orders of magnitude below the spread of the raw
samples the CUSUM and the duration band are actually tested against. Measured on
the export as `mean(value_max - value_min) / std`:

| items | ratio |
|---|---|
| `cal-qa-tssdb-active-ip` mssql latch/page (7) | 15.6–31.7x |
| `NAS01027` disk queue counters (4) | 8.1–22.6x |
| `IMTDBUAT48121` SQL batch requests/sec | 12.1x |
| `IMTDB123` `mssql.scan_to_search` | 11.2x |

A CUSUM is only stationary under H0 because its slack `k·sigma` outruns the
typical per-sample deviation. With the slack set from the hourly-average std the
drift turned positive and `s_max` grew linearly in N: median `s_max / decision`
was **65**, where **2** already saturates the score. The detector had become a
sample counter. On the 555-item stratified sample (`check_20260813_0921`) it
returned 1.0 for 73% of all items — including **13 of the 25** items the cheap
detectors had scored *low*. A vote that constant is worse than no vote, because
`require_any: 2` plus contributing-weight normalisation (§2.5) lets it promote
any single marginal zscore/seasonal signal to a confirmed anomaly on its own.

**Cause 2 — per-sample accumulation.** The same physical excursion scored 10x
higher on a 60 s item than on a 600 s one, purely from poll frequency.

**Cause 3 — `history_interval` read as a sample duration.** The duration gate
computed `anomalous_secs = n_anomalous × history_interval`, i.e. `× 600`, on a
Zabbix collecting at 60 s. Every duration was inflated 10x and the gate — whose
entire job is suppressing brief spikes — never fired. Real anomalous times on
the export: 9 min (`Processor Queue Length`), 11 min (`checkpoint_pages`,
`lazy_writes`), 14 min (`page_writes`, `Disk Reads/sec`), 17 min
(`Avg. Disk Read Queue`), 18 min (`Batch Requests/sec`) — all booked as ≥ 90 min.

**Fixes:** `intra_std` in `trends_stats` and `baseline_sigma` for the raw-sample
scale (§1.3); time-integrated CUSUM (§2.4); per-item `sample_interval` for the
duration gate (§2.6). Replaying the export through the real code: **31 → 12**.
Against the 19 human-labelled queues in `datasets/queues/` (570 labels) the
change is precision-positive and recall-neutral — 0.889/0.122 → 0.904/0.120,
FP 6 → 5, TP 48 → 47. `cusum_h` was left at 5.0: sweeping it to 10 and 20 moved
those numbers by at most one item, because the gates and filters dominate.

Pinned by `tests/unit/test_changepoint_production_sample.py`,
`tests/unit/test_baseline.py`, and the new cases in
`tests/unit/test_changepoint_detector.py`, `test_gating.py`,
`test_rolling_stats.py`.

**Migration:** self-healing. `intra_std` is added with
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` and every consumer fails open to the
old behaviour while it is NULL, so the fix takes effect for an item the first
time `anomdec-update-stats` runs after deploying. The duration and CUSUM changes
need no migration.

**What this does not fix.** The 12 survivors are held up by two causes outside
this change: `hour_stats.std` is a 10–13 sample standard deviation with no floor
(`system.cpu.intr` has `hour_std` 13.6 against an overall std of 884, giving
z ≈ 200) and is compared against a *3-hour* recent mean rather than the matching
hour; and a detector that abstains does not dilute the ensemble, so
`sug-cdrmediator02 call.stats` is confirmed by changepoint plus a marginal
zscore even though the seasonal detector correctly judged the level normal for
the hour (z = 1.62) and stayed silent.

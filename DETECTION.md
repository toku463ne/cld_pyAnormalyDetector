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
| `{ds}_trends_stats` | `itemid, sum, sqr_sum, cnt, mean, std` | daily |
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

If the window contains no samples at all (source outage), the stored stats are
left unchanged rather than wiped.

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
slack    = cusum_k × trend_std          (default k = 0.5)
decision = cusum_h × trend_std          (default h = 5.0)

for v in values:
    dev    = v - trend_mean
    s_pos  = max(0, s_pos + dev - slack)
    s_neg  = max(0, s_neg - dev - slack)
    s_max  = max(s_max, s_pos, s_neg)

score = 0                                                  if s_max < decision
score = min( (s_max - decision)/decision × 0.5 + 0.5, 1.0 ) otherwise
```

This rewards **sustained** deviation: a single outlier adds one `dev` then
decays against the slack, whereas a level shift accumulates every sample. Cost
is O(`history_retention`) per candidate item.

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

### 2.6 Gating — category, magnitude, duration

`features/gating.py`. Pure functions, used identically by the pipeline and the
backtester so offline evaluation matches runtime.

```
effective_score = raw_ensemble_score × category_weight × magnitude_scale × duration_scale
is_anomaly      = effective_score >= ensemble.min_score
```

The raw score and all three multipliers are preserved in `features`
(`raw_score`, `gate_weight`, `mag_scale`, `dur_scale`, `delta`).

**Ramp helper** — `ramp(x, lo, hi)`: 0 at/below `lo`, 1 at/above `hi`, linear
between; degenerates to a hard threshold at `hi` when `hi <= lo`.

**Category** (`classify`) — first category whose `key_patterns` fnmatch-match
`items.key_` wins; order in `default.yml` is therefore significant. Unmatched
items get `default_weight`.

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

**Duration** — how long the item stayed outside `trend_mean ± sigma·trend_std`
inside the window; `count` (total anomalous samples) or `consecutive` (longest
run), times `history_interval`, then ramped `lo_secs → hi_secs`. Default
`lo_secs 600 / hi_secs 3600`: a single 10-minute spike is suppressed, a
sustained hour scores full.

Both magnitude and duration **fail open** (`scale = 1.0`) when the evidence is
missing — no baseline stats, no raw history, `trend_std <= 0`. A real anomaly is
never suppressed for lack of evidence of brevity.

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
second pass (`update_cluster_ids`), and rows older than `anomaly_keep_secs`
(default 1 day) are deleted each run.

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
`trends_retention`, `anomaly_keep_secs`, `detectors`, `ensemble`, `clustering`,
`metric_categories`, `item_filters`, `anomaly_filters`, `fast_detect`,
`dashboards`.

### Main tuning knobs

| Key | Default | Effect |
|---|---|---|
| `history_interval` / `history_retention` | 600 / 18 | recent window ≈ 3 h |
| `trends_retention` | 14 | days of baseline |
| `detectors.*.lambda_threshold` | 3.0 | z at which score = 0.5; saturates at 2λ |
| `detectors.changepoint.cusum_h` / `cusum_k` | 5.0 / 0.5 | decision / slack, in std units |
| `ensemble.min_score` | 0.7 | precision-first threshold |
| `ensemble.require_any` | 2 | detectors that must agree |
| `metric_categories.categories[].weight` | per-category | prior importance by metric type |
| `metric_categories.duration.*` | 600 → 3600 s | suppress brief spikes |
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

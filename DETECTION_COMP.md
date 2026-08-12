# DETECTION_COMP.md — Hourly Detection: old vs. new

Compares §2 of [DETECTION.md](DETECTION.md) (current implementation) against the
old implementation preserved in `org/pyAnomalyDetector/`.

Old code references are `org/pyAnomalyDetector/…`; new ones are repo-root.
Old defaults come from `org/pyAnomalyDetector/default.yml`, new from `default.yml`.

---

## 1. Paradigm

| | Old | New |
|---|---|---|
| Control flow | **Cascade filter** — `detect1 → detect2 → detect3 → detect4`, each stage narrowing the previous stage's output (`detect_anomalies.py:82-114`) | **Parallel scoring** — every detector scores independently, `EnsembleDetector` combines |
| Combination rule | Implicit logical **AND** across stages; an item must survive all four | Weighted mean over *contributing* detectors + `require_any` |
| Per-item output | `int` (an itemid, or absent) | `AnomalyScore(score, is_anomaly, detector_scores, features)` |
| Failure mode | One over-strict stage silently kills recall for everything downstream | One weak detector is diluted, not fatal |
| Debuggability | Item vanished — which stage dropped it is not recorded | `features` carries `z`, `raw_score`, `gate_weight`, `mag_scale`, `dur_scale`, `delta` |
| Threshold tuning | Four independent λ's, each a hard gate | One `min_score` on a continuous scale + per-detector λ |
| Early exit | `continue` to next data source as soon as any stage empties | No early exit; empty result is a normal outcome |

The cascade's structural problem: because stages run in sequence on the survivors,
`detect1`'s recall is an absolute ceiling on the whole pipeline. The new design
removes that ceiling but replaces it with a different risk — `require_any: 2` is
now the only thing preventing single-signal flags.

---

## 2. Stage → detector mapping

| Old stage | New equivalent | Relationship |
|---|---|---|
| `detect1` | `ZScoreDetector` | Direct descendant — same quantity, continuous output, bug fixed |
| `detect2` | `ChangepointDetector` | **Replaced**, not ported — different algorithm (CUSUM vs. peak-diff) |
| `detect3` | `SeasonalDetector` + `duration` gate | **Replaced** — the pieces map only loosely (see §3.3) |
| `detect4` | *(none)* | **Dropped** — see §5.1 |
| `item_conds` | `item_filters` | Ported, SQL-`LIKE` → fnmatch |
| `item_diff_conds` | `anomaly_filters` | Ported, SQL-`LIKE` → fnmatch |
| — | `metric_categories` gating | **New** — no old equivalent |

---

## 3. Stage-by-stage

### 3.1 detect1 → ZScoreDetector

`detector.py:201-230` vs. `detectors/zscore.py`.

| | Old `detect1` | New `ZScoreDetector` |
|---|---|---|
| Test | `mean_h > mean_t + λ·std` **OR** `mean_h < mean_t - λ·std` | `z = \|mean_h - mean_t\| / std`, `z >= λ` |
| Equivalent? | Yes — algebraically the same condition | — |
| Output | itemid list (binary) | score via ramp: `λ→0.5`, `2λ→1.0`, saturating |
| Baseline guard | `trends_stats.cnt > trends_min_count` (**14**) | `cnt > 0` |
| Std guard | `std > 0` | `std > 0` |
| Small-change filter | `ignore_diff_rate` = 0.2 — **inoperative**, see §4.1 | `min_ignore_rate` = 0.05, applied correctly |
| Zero-baseline handling | `mean_t > 0` required (negative baselines dropped) | `mean_t == 0` bypasses the rate check; negatives kept |
| Cost | 2 DB reads + vector ops per batch | identical |

Two behaviour changes deserve attention.

**Baseline-count guard `cnt > 14` → `cnt > 0`.** `trends_min_count` does not
exist anywhere in the new tree — not in `config/schema.py`, `default.yml`, or any
doc. It was not retuned; it was simply not carried over in the clean-break
rewrite (commit `e9dbeb3`, which created all 95 files at once). See §6.1 for what
the label data says about whether it should come back.

Note that `cnt > 0` is not quite as loose as it looks: `cnt == 1` yields
`variance = 0` (Bessel denominator clipped to 1) and the item is then dropped by
the `std > 0` guard, so the effective floor is `cnt >= 2`.

**`mean_t <= 0` items are now kept.** The old code required `mean_t > 0`
(a side effect of the §4.1 bug); the new code keeps them and only bypasses the
relative-rate check when `mean_t == 0`. Intentional — a metric legitimately
centred on zero should still be detectable — but it widens the candidate set.

### 3.2 detect2 → ChangepointDetector

`detector.py:286-356` vs. `detectors/changepoint.py`. These solve the same
problem — *sudden change* — by different means.

| | Old `detect2` | New `ChangepointDetector` |
|---|---|---|
| Quantity | Adjacent-peak first differences of trends `value_max`/`value_min`, compared against the recent window's `max-first` / `min-first` excursion | CUSUM accumulation of `value - trend_mean` over raw recent history |
| Baseline | mean/std of the **trend diff series**, computed at detection time | `trends_stats.mean` / `.std`, pre-computed |
| Direction | Two explicit passes (`is_up=True/False`) over `value_max` and `value_min` separately | Two CUSUM accumulators (`s_pos`, `s_neg`) in one pass |
| Sustained vs. spike | Not distinguished — a single large jump passes | Inherently favours sustained shift: a lone outlier decays against the slack, a level shift accumulates every sample |
| Small-change filter | `ignore_diff_rate` (active here) | handled downstream by `magnitude` gate |
| Cost | Fetches trends **and** history per batch, builds a per-item diff frame in a Python loop | O(window) per candidate item; trends stats already in memory |
| λ default | 2.0 | `cusum_h` 5.0 / `cusum_k` 0.5 (not comparable) |

The old approach compared *peak excursions*; the new one compares *accumulated
deviation*. The rationale (CLAUDE.md) is that persistence is the better signal
for "needs human attention" than instantaneous jump size.

### 3.3 detect3 / detect4 → SeasonalDetector + duration gate

`detector.py:461-584`. This is the loosest correspondence in the migration — the
old stage bundled three distinct ideas that are now separate or absent.

Old `detect3` per item, per direction:

1. **Threshold count** — count history samples outside `trends_mean ± λ·trends_std`,
   where the baseline is built from `value_max` (up) or `value_min` (down);
   require `anom_cnt / hist_count > anomaly_valid_count_rate` (**0.8**).
2. **Local-peak comparison** (`_calc_local_peak`) — slide a `density_window`
   (= `history_interval × history_retention`) over the trends peak series and
   require the recent mean to exceed every historical local peak (up) or fall
   below every local trough (down).
3. **Two-pass re-test** — items failing pass 1 at `λ1` (1.0) are re-tested on a
   shorter recent window at `λ2` (2.0).

`detect4` is `detect3` with `is_long_trend=True`, i.e. the same logic against
`long_trends_retention` = **60 days** instead of 14.

| Old idea | New home | Fidelity |
|---|---|---|
| Anomalous-sample density (`anomaly_valid_count_rate`) | `metric_categories.duration` gate | Partial — see below |
| Local-peak / historical-extreme comparison | *(none)* | **Lost** (§5.2) |
| Two-pass λ1/λ2 re-test | *(none)* | Lost; the continuous score subsumes the intent |
| Long-trend (60 d) re-check | *(none)* | **Lost** (§5.1) |
| Hour-of-day expectation | `SeasonalDetector` | **New** — the old pipeline had no seasonality model at all |

The density idea survived, but much weaker. Old: ≥80% of window samples must be
anomalous. New (`duration`, defaults `lo_secs 600 → hi_secs 3600`, `count` mode,
18×600 s window): 6 of 18 samples ≈ **33%** already earns full weight, and the
result is a multiplier rather than a gate.

`SeasonalDetector` has no old counterpart — it is the one genuinely new detection
capability, and is exactly the mechanism that stops "rises every morning at 09:00"
from being flagged every morning at 09:00.

### 3.4 Conditions → filters and gating

| | Old | New |
|---|---|---|
| Item exclusion | `item_conds` — SQL fragment `filter` pushed to the DB (`check_itemId_cond`) + `operator`/`value` on recent mean | `item_filters` — fnmatch on `key_` + exact `units`, optional `min_value` |
| Diff suppression | `item_diff_conds` — same shape, evaluated on `\|mean_h - mean_t\|` | `anomaly_filters` — `min_abs_diff` |
| Where evaluated | In the data source (SQL `LIKE`), per condition, per item, in a Python loop | Pure functions on pre-fetched metadata (`pipeline/filters.py`) |
| Applied when | Inside `detect1` only (`_filter_by_conds`) | `item_filters` before detectors; `anomaly_filters` after gating |
| Metric-type weighting | *(none)* | `metric_categories` — per-category weight + magnitude mode |
| Magnitude semantics | *(none)* — absolute conditions only | `absolute` / `relative` / `sigma` modes on `Δ = \|recent-trend\|` |
| Testability | Requires a DB (SQL evaluated remotely) | DB-free, unit-testable |

Porting note: SQL `LIKE 'net.if.%.[%]'` became fnmatch `net.if.*`, because
fnmatch treats `[...]` as a character class. The bracket portion of old patterns
is therefore matched more loosely than before.

The `metric_categories` layer has no ancestor — it exists because the label data
shows precision varies by an order of magnitude across metric types
(DETECTION.md §8.3).

---

## 4. Bugs fixed

### 4.1 `detect1` operator-precedence bug — `ignore_diff_rate` never applied

`detector.py:218`:

```python
h_stats_df = h_stats_df[h_stats_df['mean_t'] > 0 & (abs(...)/h_stats_df['mean_t'] > ignore_diff_rate)]
```

`&` binds tighter than `>`, so Python evaluates `0 & (<bool Series>)` first,
which yields an all-`False` series; the expression collapses to
`h_stats_df['mean_t'] > False`, i.e. **`mean_t > 0`**.

Verified on synthetic data: a row with `mean_t=10, mean_h=10.1` (1% change,
against a 20% ignore rate) passes the buggy filter and is correctly rejected by
the parenthesised version. So in production, `ignore_diff_rate: 0.2` did nothing
in `detect1` — every item clearing the λ test was forwarded, and the only real
effect of that line was dropping non-positive baselines.

New code applies the equivalent rule as an explicit, correctly-parenthesised
mask (`detectors/zscore.py:65-67`) at `min_ignore_rate: 0.05`.

### 4.2 Zero-denominator guards added

`trend_std == 0` and `trend_mean == 0` are handled explicitly in every new
detector. The old `detect2` divided by the trends-diff `mean` with no zero check
(`detector.py:325`, `:332`):

```python
stats_df = stats_df[abs(stats_df['max'] - stats_df['mean'])/stats_df['mean'] > ignore_diff_rate]
```

With `mean == 0` this yields `inf` when the numerator is non-zero — and
`inf > ignore_diff_rate` is `True`, so the item **passes** the small-change
filter unconditionally. When the numerator is also zero it yields `NaN`, which
compares `False` and drops the item. So a zero-centred metric got the opposite of
the intended treatment in one branch and correct treatment in the other,
depending on data it had no control over.

### 4.3 Numerical stability (cleanup, not a behaviour fix)

`features/rolling_stats.py` clips variance at 0 **before** `sqrt`, and clips
`cnt-1` at 1.

The old incremental stats (`data_processing/stats.py:132-133`) computed the same
`sqr_sum - sum²/cnt` without the clip, so float cancellation could produce
`sqrt(negative) → NaN` (and `cnt == 1` produced a divide-by-zero `inf`), but the
next line mopped both up with `.replace([inf, -inf], nan).fillna(0)`.

The end value is therefore the same in both versions — `std = 0`, and the item is
then dropped by the `std > 0` guard. The new code avoids the intermediate
`NaN`/`inf` and the `RuntimeWarning` rather than changing any decision. Worth
noting that the old entry point suppressed warnings globally
(`detect_anomalies.py:170`), so these cases were invisible in operation.

---

## 5. Capabilities lost

Not everything moved across. These are genuine regressions, listed so they are
decisions rather than accidents.

### 5.1 Long-trend (60-day) re-check — `detect4`

`detect4` re-ran the whole `detect3` logic against `long_trends_retention: 60`
days. The new implementation has a single `trends_retention: 14` and no long
baseline at all. Anomalies that are normal on a 2-week view but abnormal on a
2-month view (slow capacity drift, monthly cycles) are no longer detectable.

### 5.2 Trends `value_min` / `value_max` are no longer used

The old detectors deliberately used **asymmetric baselines**: upward tests were
built from `value_max`, downward tests from `value_min` (`detector.py:291-296`,
`:509-521`). The new path fetches all three columns
(`ingestion/zabbix_psql.py:130`) but every statistic is derived from `value_avg`
alone — `trends_stats` (`pipeline/stats_update.py:68`) and `hour_stats`
(`features/hour_stats.py:36`). `value_min`/`value_max` now reach only the
labeling UI's plots.

Consequence: intra-hour volatility is invisible. An item whose hourly average is
unchanged but whose max tripled produces no signal.

### 5.3 Local-peak comparison

`_calc_local_peak` (`detector.py:435-457`) required the recent level to exceed
*every* local peak in the trends window — a "highest ever seen" test that is
strictly stronger than a z-score and is not reproduced by any new component.

### 5.4 Baseline minimum count — dropped, but measured as harmless here

Listed for completeness; unlike 5.1–5.3 the label data argues **against**
restoring it. Quantified in §6.1.

### 5.5 Density requirement weakened

As quantified in §3.3: an 80% hard requirement became a multiplier reaching full
weight at ~33%.

---

## 6. Default parameter diff

### 6.1 Should `trends_min_count` come back?

Measured on the 12 human-reviewed label days in `datasets/queues/`, using each
item's trends row count in its export as a proxy for `trends_stats.cnt`:

| trends rows in window | n | precision |
|---|---|---|
| 1–14 — *what the old guard dropped* | 19 | **1.00** |
| 15–49 | 51 | **1.00** |
| 50–199 | 2 | 0.00 |
| 200+ (full ~336-row coverage) | 288 | 0.39 |

Restoring `cnt > 14` would have removed 19 items, **all 19 of them true
anomalies**, and suppressed zero false positives.

The reason is that low-count items here are not *new* items with a short history
— their timestamps span 266–312 hours, i.e. the full 14-day window. They are
**sparse** items: Zabbix writes a trends row only for hours that received a
value. The population is 62 `docker.*` items (containers that come and go) plus
8 `check_aws_cost_monthly` (a monthly metric — sparse by definition). An item
that normally reports rarely and suddenly reports is, unsurprisingly, usually a
real event.

The within-metric comparison isolates it from metric-type confounding:

| | low coverage | high coverage |
|---|---|---|
| `docker.*` | n=62, precision **1.00** | n=14, precision 0.50 |

Caveats: the queue is flagged-only, so this measures precision *conditional on
being flagged* — it cannot show FPs the guard would have prevented from ever
being queued. But if unstable small-sample std were inflating z into the top-30,
those items would be visible in the queue, and they are not. One site, 12 days.

**Conclusion:** the drop was accidental, but reinstating `14` would be a net
loss on this workload. If the numerical concern is worth addressing, guard on
std reliability rather than row count — the existing `std > 0` check already
excludes `cnt == 1`.

### 6.2 Parameter table

| Concept | Old | New | Note |
|---|---|---|---|
| First-stage λ | `detect1_lambda_threshold: 3.0` | `detectors.zscore.lambda_threshold: 3.0` | unchanged |
| Second-stage λ | `detect2_lambda_threshold: 2.0` | `cusum_h: 5.0`, `cusum_k: 0.5` | different algorithm |
| Third-stage λ | `detect3_lambda_threshold1: 1.0`, `…2: 2.0` | `detectors.seasonal.lambda_threshold: 3.0` | different quantity |
| Small-change floor | `ignore_diff_rate: 0.2` (inoperative in detect1) | `min_ignore_rate: 0.05` + `default` category `relative 0.5→1.0` | |
| Density requirement | `anomaly_valid_count_rate: 0.8` | `duration.lo_secs 600 / hi_secs 3600` | much weaker |
| Baseline min count | `trends_min_count: 14` | *(none — `cnt > 0`)* | dropped in rewrite; see §6.1 — restoring it costs 19 TPs, saves 0 FPs |
| Trends window | `trends_retention: 14`, `long_trends_retention: 60` | `trends_retention: 14` | long window dropped |
| History window | `history_interval: 600`, `history_retention: 18`, `history_recent_retention: 6` | `600` / `18` | recent-retention concept dropped |
| Decision threshold | *(implicit — survive all 4 stages)* | `ensemble.min_score: 0.7`, `require_any: 2` | |

---

## 7. Cost profile (hourly path)

| | Old | New |
|---|---|---|
| Trends fetched at detection time | Yes — `detect2` and `detect3` both call `get_trends_full_data` per batch (`detector.py:276`) | No — `trends_stats` read from admdb |
| Stats computed at detection time | Yes — `_get_trends_stats` re-aggregates mean/std per batch (`detector.py:488-495`) | No — pre-computed daily |
| Raw history fetched | For all `detect1` survivors, plus a full `update_history` normalisation pass | Only for items already scoring > 0 (changepoint), and only when `duration` needs it |
| Per-item Python loops | Several (`_detect_diff_anomalies`, `_calc_local_peak`, `_filter_anomalies` all loop per item) | CUSUM only |
| Seasonality | n/a | O(1) DB lookup |

The old pipeline violated the "no heavy computation at detection time" constraint
that CLAUDE.md now states explicitly: `detect3` alone re-derived trends statistics
from raw trends rows on every hourly run.

---

## 8. Net assessment

**Improved:** interpretability (scores + feature breakdown instead of a
disappearing itemid), seasonality handling (entirely new), runtime cost, testability
(detectors are pure functions), and one silently-dead filter fixed.

**Regressed:** the long-trend axis, min/max-based volatility detection, the
historical-extreme test, and the density requirement — four independent
suppression mechanisms that the old cascade used to reach its (low-recall,
moderate-precision) operating point.

That trade is consistent with the measured outcome: current precision@30 is
**0.508** on 12 reviewed days (DETECTION.md §8.2–8.3), i.e. the new pipeline
flags more and filters less. The `metric_categories` layer is where the lost
suppression is meant to be rebuilt, and its thresholds are still marked as
placeholders in `default.yml`.

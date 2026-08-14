from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class AdmDbConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "anomdec"
    password: str = ""
    dbname: str = "anomdec"
    schema_name: str = Field("public", alias="schema")
    retries: int = 3
    delay: int = 3

    model_config = {"populate_by_name": True}


class ZScoreConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.3
    lambda_threshold: float = 3.0
    min_ignore_rate: float = 0.05


class ChangepointConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.3
    cusum_h: float = 5.0
    cusum_k: float = 0.5


class SeasonalConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.4
    lambda_threshold: float = 3.0


class DetectorsConfig(BaseModel):
    zscore: ZScoreConfig = ZScoreConfig()
    changepoint: ChangepointConfig = ChangepointConfig()
    seasonal: SeasonalConfig = SeasonalConfig()


class EnsembleConfig(BaseModel):
    min_score: float = 0.5
    require_any: int = 1


class ItemFilterRule(BaseModel):
    """Exclude items from detection entirely.

    Matches items where key_ matches key_pattern (fnmatch glob) AND units
    matches units (exact).  If min_value is set, only excludes the item when
    its recent mean is below that value; otherwise always excludes.
    """
    name: str = ""
    key_pattern: str = ""       # fnmatch glob on items.key_; empty = match all
    units: str = ""             # exact match on items.units; empty = match all
    min_value: float | None = None  # exclude if recent_mean < min_value


class AnomalyFilterRule(BaseModel):
    """Suppress anomaly scores whose absolute diff is below a threshold.

    Matches items the same way as ItemFilterRule.  If the absolute difference
    |recent_mean - trend_mean| is below min_abs_diff the anomaly is dropped.
    """
    name: str = ""
    key_pattern: str = ""
    units: str = ""
    min_abs_diff: float | None = None


class ClusteringConfig(BaseModel):
    # Minimum first-difference points a correlation needs before it counts as
    # evidence.  The grid is `detection_period / unitsecs`, and when that comes
    # out tiny the Spearman coefficient stops meaning anything: with three
    # differences two *independent* series land on distance exactly 0.0 one time
    # in six, and only four distinct distances exist at all.  Below this, every
    # item is left unclustered rather than grouped at random.
    min_corr_points: int = 8
    # How members are grouped once the distance matrix exists.
    #
    #   complete : agglomerative, every pair inside a cluster within corr_eps
    #   average  : agglomerative on the mean inter-cluster distance
    #   dbscan   : density-reachable (the original)
    #
    # DBSCAN clusters by *reachability*, so A-B close and B-C close puts A and C
    # together however far apart they are.  That chaining is what let a coincident
    # spike bridge unrelated shapes: a real cycle merged three docker CPU items
    # into a DNS incident 0.52-0.78 away, through intermediate items, and the
    # rescue step then promoted the docker items to anomalies.  Complete linkage
    # bounds the cluster diameter instead, so the bridge cannot form.
    # Measured on the incident-labelled queues: false merges 9 -> 4 and pair
    # precision 0.79 -> 0.89, for 2 of 34 true pairs.
    linkage: Literal["complete", "average", "dbscan"] = "complete"
    # Correlation-distance threshold, correlating first-differences on the
    # history/anomaly window only (see clustering/dbscan.py).
    #
    # It means different things per linkage, which is why the default moved with
    # it: for DBSCAN it is a neighbour radius, and 0.10 was the knee (the older
    # 0.2 was set when 14d of hourly trends were prepended, and on the short
    # window it merged everything that co-spiked).  For complete linkage it is a
    # cap on the *whole cluster*, so it has to be looser to hold a genuine group
    # together.  It also had to move again once the grid stopped collapsing to a
    # handful of points (see _infer_unitsecs): on the finer grid the distance
    # distribution spreads out, and 0.30 dominates 0.20 on the incident labels --
    # true pairs 32 -> 36 for one extra false pair.
    corr_eps: float = 0.30
    # Raw-correlation channel: two items sharing a metric family that correlate
    # in raw levels above this threshold are merged even if their first-difference
    # correlation is weak. This recovers monotonic-ramp groups (cumulative
    # counters like docker throttling_periods) that differencing alone fragments.
    # Kept strict so only near-identical same-family ramps merge.
    raw_corr_min: float = 0.99
    min_samples: int = 2
    # Threshold for "this sample is outside baseline", used by onset detection:
    # a trends sample is anomalous outside trend_mean ± sigma·trend_std.
    sigma: float = 2.0
    detection_period: int = 43200
    rescue_same_incident: bool = True  # pull magnitude-suppressed items into a confirmed cluster
    # Onset constraint: two items may only share a cluster when their anomalies
    # began within this many seconds of each other.  Correlation alone is blind
    # to *when* each item started misbehaving — inside a 12h window a ramp that
    # began a week ago and a step from yesterday are both just gentle drift, so
    # unrelated items merge.  Onsets come from trends (hourly), so the practical
    # resolution is one hour.  0 disables the constraint.
    max_onset_gap: int = 7200
    # Onset = when the item's current level regime began (see features/onset.py).
    # level_tol is the relative band that counts as "still the same level": 0.1
    # means +/-10% of the current value.  For step changes the onset is
    # essentially independent of this; for gradual ramps it sets how far back
    # "the same level" reaches.
    onset_level_tol: float = 0.1
    # Consecutive out-of-band trends samples tolerated before the regime is
    # treated as ended (absorbs brief dips).
    onset_tolerance: int = 2
    # Trailing trends samples whose median defines the current level.
    onset_recent_samples: int = 3


class MagnitudeConfig(BaseModel):
    """Scale a category's weight by the size of the change from baseline.

    The driving quantity is always the *delta* Δ = |recent_mean - trend_mean|,
    never the raw current value, so a host steady at a high level (Δ≈0) is not
    flagged.  `mode` controls how Δ is normalised before the lo→hi ramp:
      - absolute : Δ in native units (use for %, and for byte-rates whose
                   operational floor is unit-coherent within the category)
      - relative : Δ / |trend_mean|   (unit-free; for byte sizes that vary by host)
      - sigma    : Δ / trend_std      (z-units)

    scale = clamp((x - lo) / (hi - lo), 0, 1), floored at `floor`.
    If hi <= lo it degenerates to a hard threshold at `hi`.
    """
    mode: Literal["absolute", "relative", "sigma"] = "absolute"
    lo: float = 0.0    # Δ at/below which scale = 0 (ignore)
    hi: float = 0.0    # Δ at/above which scale = 1 (full weight)
    floor: float = 0.0


class DurationConfig(BaseModel):
    """Down-weight short-lived anomalies; reward sustained ones.

    Within the recent history window, a sample is "anomalous" when it lies
    outside trend_mean ± sigma·trend_std.  Anomalous time is the sample count
    times `history_interval` (count mode) or the longest consecutive run
    (consecutive mode).  scale ramps lo_secs → hi_secs.
    """
    enabled: bool = False
    measure: Literal["count", "consecutive"] = "count"
    sigma: float = 2.0
    lo_secs: int = 600      # ≤ this anomalous time → scale 0 (single spike)
    hi_secs: int = 3600     # ≥ this anomalous time → scale 1 (sustained ≥1h)
    floor: float = 0.0


class IdleBaselineConfig(BaseModel):
    """Suppress items whose baseline is zero most of the time.

    A metric that reads zero whenever its resource is idle — VMware guest disk
    latency, an outstanding-IO depth, a rarely-used counter — has a baseline mean
    near zero, so *any* activity produces a relative change of tens or hundreds.
    Relative magnitude cannot discriminate on such a series, and an absolute
    floor cannot either without knowing the unit.

    The unit-free question that does discriminate: is the current level
    unprecedented?  When the baseline is idle-dominated, the item is suppressed
    unless the recent mean exceeds the highest value seen in the whole baseline
    window.  That keeps the "normally zero, suddenly off the scale" signal — an
    error counter that spikes to a level it has never reached still fires —
    while dropping the routine idle→busy transitions.

    `zero_ratio = zero_cnt / cnt` over the trends retention window.
    """
    enabled: bool = False
    max_zero_ratio: float = 0.8   # baseline zero more often than this = idle-dominated
    floor: float = 0.0            # scale applied when suppressed (0 = hard veto)


class RecurringPeakConfig(BaseModel):
    """Suppress a level this item reaches routinely anyway.

    Some metrics peak as a matter of course — a call counter that climbs every
    business morning, a VM whose CPU spikes on every batch job.  When one of them
    is flagged, the level it has reached is usually one it has reached many times
    before, and a reviewer rejects it on sight.

    The rule the old implementation used for this (`detect3`, see
    `org/pyAnomalyDetector/data_processing/detector.py::_calc_local_peak`) was
    "drop the item unless the current level exceeds every past sustained peak".
    Applied unconditionally that is far too blunt: measured against 19 queues of
    human labels it removed 25 of the 36 confirmed anomalies, because a real
    anomaly is usually not an unprecedented *level* — it is a normal level at an
    abnormal time.

    What makes it safe is a precondition on the baseline's **shape**, which does
    not involve the current level at all: apply the veto only to items that peak
    habitually, `peak_episodes >= min_episodes`.  On the same labels the
    confirmed anomalies sit at a median of 0 episodes and a maximum of 8 (34 of
    36 are at 0 or 1), while the habitually-peaking items reviewers rejected run
    9-72, so the precondition separates them without touching recall.

    Upward excursions only.  A collapse is always below `local_peak`, so a
    symmetric rule would suppress exactly the "the service stopped" signals that
    matter most; a trough counterpart would need its own evidence.

    `exclude_recent_secs` keeps the tail of the baseline window out of the
    precedent, because an excursion still in progress would otherwise be part of
    what it is judged against and clear its own veto.  It defaults to one day,
    the daily batch's cadence, so anything that started since the previous run is
    excluded.  A shift older than that is genuinely part of the baseline by then
    — that is the sliding window working as intended.
    """
    enabled: bool = False
    min_episodes: int = 9        # separate excursions in the baseline = "peaks habitually"
    k_sigma: float = 2.0         # sigma multiple a bucket max must clear to open an episode
    exclude_recent_secs: int = 86400  # tail kept out of the precedent (see below)
    floor: float = 0.0           # scale applied when suppressed (0 = hard veto)


class RecencyConfig(BaseModel):
    """Require the excursion to have *started* recently.

    The job runs hourly over the last `history_retention x history_interval` of
    history, and the product it is meant to deliver is "anomalies that began in
    that window", recorded once and then kept on the dashboard for a few days.

    Without this the detectors deliver something else: they compare the recent
    window's mean against the 14-day baseline, which an excursion keeps
    satisfying every hour until the baseline slides past it.  Measured on four
    real cycles, only 11 of 93, 9 of 31, 3 of 11 and 0 of 12 flagged items had
    actually started inside the window; the rest were the same incidents being
    re-reported, 8 of the 11 items in one cycle having already appeared in the
    previous one.

    `max_age_secs = 0` follows the detection window **plus one trends interval**.
    The margin is not slack: onsets come from trends, so they are quantised to
    the hour, and an excursion that began 2h10m ago is recorded as 3h old.  At
    exactly the window width that quantisation split a single real incident --
    five `IPX012 unbound` counters landed in the 2.1h and 3.1h buckets, and only
    two of the five were ever recorded.  One interval of margin recovers all
    five.

    An incident gets roughly three chances to be caught, since three consecutive
    hourly runs still have its onset inside the window.  An item whose onset
    cannot be resolved is kept (fail-open, like every other gate).
    """
    enabled: bool = False
    max_age_secs: int = 0     # 0 = history_retention x history_interval
    floor: float = 0.0


class MetricCategoryRule(BaseModel):
    """A metric category, matched by fnmatch glob(s) on items.key_ (== item_name
    for CSV sources).  First matching category wins."""
    name: str
    key_patterns: list[str] = []
    weight: float = 1.0
    magnitude: MagnitudeConfig | None = None


class MetricCategoriesConfig(BaseModel):
    default_weight: float = 1.0
    duration: DurationConfig = DurationConfig()
    idle_baseline: IdleBaselineConfig = IdleBaselineConfig()
    recurring_peak: RecurringPeakConfig = RecurringPeakConfig()
    recency: RecencyConfig = RecencyConfig()
    categories: list[MetricCategoryRule] = []


class WatchRule(BaseModel):
    """One watchlist entry for the fast axis.  An item matches if its key_/name
    matches key_pattern (fnmatch glob) AND its host matches host_pattern.  An
    empty pattern matches all on that dimension."""
    key_pattern: str = ""    # fnmatch glob on item key_ (== item name for these sources)
    host_pattern: str = ""   # fnmatch glob on host_name


class FastDetectConfig(BaseModel):
    """High-frequency, short-span detection over a small watchlist.

    Runs every few minutes against a short history window, scores each watched
    item by a short-window z-score, vetoes levels the seasonal baseline considers
    expected (backup-traffic filter), groups co-occurring triggers, and writes a
    JSON event file for Zabbix to poll.
    """
    enabled: bool = False
    watch: list[WatchRule] = []
    history_span_secs: int = 3600   # length of the short baseline window
    detect_window: int = 4          # last N samples form the "recent" mean
    lambda_threshold: float = 3.0   # short-window z -> severity (ZScore ramp)
    min_item_score: float = 0.5     # per-item trigger threshold
    seasonal_veto: bool = True      # suppress levels expected for this hour-of-day
    seasonal_lambda: float = 3.0    # |recent - hour_mean|/hour_std < this => expected
    cooccur: bool = True            # group co-triggers via DBSCAN
    use_zabbix_events: bool = False # fold severity-weighted Zabbix events into the score
    events_window_secs: int = 0     # event lookback window (0 => reuse history_span_secs)
    events_saturation: float = 3.0  # sum of severity-weights mapping to ~full host weight
    min_event_score: float = 0.5    # host event weight >= this -> standalone event alert
    output_path: str = "/tmp/anomdec/fast_events.json"


class DashboardsConfig(BaseModel):
    """Publish detection results to Zabbix dashboards.

    api_url falls back to the data source's api_url (the web base, e.g.
    http://zabbix/); ZabbixAPI normalizes it to .../api_jsonrpc.php and the view
    URL is <web_base>/zabbix.php?action=dashboard.view&dashboardid=<id>.
    """
    enabled: bool = False
    api_url: str = ""
    user: str = ""
    password: str = ""
    hourly_name: str = "anomdec_detected"     # (a) anomdec-detect results, by group
    bycluster_name: str = "anomdec_bycluster" # (a) same results, one page per cluster
    fast_name: str = "anomdec_fast"           # (b) anomdec-detect-fast results
    widget_type: Literal["graph", "svggraph"] = "graph"  # svggraph for Zabbix 7.0+


class LoggingConfig(BaseModel):
    enabled: bool = False
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - %(message)s"
    log_dir: str = "/tmp/anomdec/logs"
    file: str = "anomdec.log"


class DataSourceConfig(BaseModel):
    type: Literal["zabbix_psql", "csv", "logan"]

    # DB-type sources
    host: str = ""
    port: int = 5432
    user: str = ""
    password: str = ""
    dbname: str = ""
    api_url: str = ""

    # CSV source
    data_dir: str = ""

    # DB connection retry/schema (used by PostgreSqlDB for zabbix_psql)
    schema_name: str = Field("public", alias="schema")
    retries: int = 3
    delay: int = 3

    # Collection params (inherit from AppConfig defaults)
    batch_size: int = 100
    history_interval: int = 600
    history_retention: int = 18
    trends_retention: int = 14
    # Trends rows are hourly aggregates, so `std` only measures how much the
    # hourly *average* moves.  `intra_std = mean(value_max - value_min) / this`
    # recovers the within-hour sample spread that raw-history consumers need.
    # The divisor is the range rule's d2 constant: 2.53 at 6 samples/hour, 3.26
    # at 12, 4.64 at 60.  4.0 is a mid-range compromise -- raise it to detect
    # more, lower it to suppress more.
    trends_range_to_sigma: float = 4.0
    anomaly_keep_secs: int = 259200
    staleness_secs: int = 3600

    detectors: DetectorsConfig = DetectorsConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    metric_categories: MetricCategoriesConfig = MetricCategoriesConfig()
    item_filters: list[ItemFilterRule] = []
    anomaly_filters: list[AnomalyFilterRule] = []
    fast_detect: FastDetectConfig = FastDetectConfig()
    dashboards: DashboardsConfig = DashboardsConfig()

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def check_required_fields(self) -> DataSourceConfig:
        if self.type == "zabbix_psql":
            missing = [f for f in ("host", "user", "dbname") if not getattr(self, f)]
            if missing:
                raise ValueError(f"type={self.type} requires: {missing}")
        if self.type == "csv" and not self.data_dir:
            raise ValueError("type=csv requires data_dir")
        return self


class ViewSourceConfig(BaseModel):
    """A read-only view onto an external system (e.g. a Zabbix dashboard).

    Used by tools that need API access to inspect what a human reviewer sees,
    e.g. exporting every item shown on the daily anomaly-review dashboard.
    """
    type: Literal["zabbix_dashboard"]
    dashboard_name: str = ""
    api_url: str = ""
    user: str = ""
    password: str = ""
    data_source_name: str = ""  # key into data_sources for the underlying DB

    @model_validator(mode="after")
    def check_required_fields(self) -> ViewSourceConfig:
        if self.type == "zabbix_dashboard":
            missing = [
                f for f in ("dashboard_name", "api_url", "user", "data_source_name")
                if not getattr(self, f)
            ]
            if missing:
                raise ValueError(f"type={self.type} requires: {missing}")
        return self


class AppConfig(BaseModel):
    admdb: AdmDbConfig
    data_sources: dict[str, DataSourceConfig] = {}
    view_sources: dict[str, ViewSourceConfig] = {}
    logging: LoggingConfig = LoggingConfig()

    # Directory for the single-instance run locks (see pipeline/lock.py).  Each
    # entry point takes its own lock so a cron run and a manual run of the same
    # command cannot overlap and corrupt the shared admdb accumulators.
    lock_dir: str = "/tmp/anomdec/locks"

    # Top-level defaults that cascade into data_sources
    batch_size: int = 100
    history_interval: int = 600
    history_retention: int = 18
    trends_retention: int = 14
    # Trends rows are hourly aggregates, so `std` only measures how much the
    # hourly *average* moves.  `intra_std = mean(value_max - value_min) / this`
    # recovers the within-hour sample spread that raw-history consumers need.
    # The divisor is the range rule's d2 constant: 2.53 at 6 samples/hour, 3.26
    # at 12, 4.64 at 60.  4.0 is a mid-range compromise -- raise it to detect
    # more, lower it to suppress more.
    trends_range_to_sigma: float = 4.0
    anomaly_keep_secs: int = 259200
    staleness_secs: int = 3600
    detectors: DetectorsConfig = DetectorsConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    metric_categories: MetricCategoriesConfig = MetricCategoriesConfig()
    item_filters: list[ItemFilterRule] = []
    anomaly_filters: list[AnomalyFilterRule] = []
    fast_detect: FastDetectConfig = FastDetectConfig()
    dashboards: DashboardsConfig = DashboardsConfig()

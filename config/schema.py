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
    jaccard_eps: float = 0.1
    corr_eps: float = 0.2
    min_samples: int = 2
    sigma: float = 2.0
    detection_period: int = 43200
    rescue_same_incident: bool = True  # pull magnitude-suppressed items into a confirmed cluster


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
    hourly_name: str = "anomdec_detected"   # (a) anomdec-detect results
    fast_name: str = "anomdec_fast"          # (b) anomdec-detect-fast results
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
    anomaly_keep_secs: int = 86400

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

    # Top-level defaults that cascade into data_sources
    batch_size: int = 100
    history_interval: int = 600
    history_retention: int = 18
    trends_retention: int = 14
    anomaly_keep_secs: int = 86400
    detectors: DetectorsConfig = DetectorsConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    metric_categories: MetricCategoriesConfig = MetricCategoriesConfig()
    item_filters: list[ItemFilterRule] = []
    anomaly_filters: list[AnomalyFilterRule] = []
    fast_detect: FastDetectConfig = FastDetectConfig()
    dashboards: DashboardsConfig = DashboardsConfig()

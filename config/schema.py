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

    # Collection params (inherit from AppConfig defaults)
    batch_size: int = 100
    history_interval: int = 600
    history_retention: int = 18
    trends_retention: int = 14
    anomaly_keep_secs: int = 86400

    detectors: DetectorsConfig = DetectorsConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    item_filters: list[ItemFilterRule] = []
    anomaly_filters: list[AnomalyFilterRule] = []

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
    item_filters: list[ItemFilterRule] = []
    anomaly_filters: list[AnomalyFilterRule] = []

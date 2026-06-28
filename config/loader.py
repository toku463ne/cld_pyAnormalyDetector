from __future__ import annotations
import os
import logging
import yaml
from jinja2 import Template
from pathlib import Path

from config.schema import AppConfig, DataSourceConfig

_DEFAULT_YML = Path(__file__).parent.parent / "default.yml"
_SECRET_ENV = "ANOMDEC_SECRET_PATH"

_app_config: AppConfig | None = None


def load_config(config_path: str | None = None) -> AppConfig:
    global _app_config

    # Resolve secrets before rendering (secrets may be referenced in templates)
    raw_defaults = yaml.safe_load(_DEFAULT_YML.read_text()) or {}
    secret_path = os.environ.get(_SECRET_ENV) or raw_defaults.get("secret_path")
    secrets: dict = {}
    if secret_path:
        secrets = yaml.safe_load(Path(secret_path).read_text()) or {}

    # Render Jinja2 on raw file text, then parse YAML.
    # Rendering after yaml.dump() is wrong: yaml.dump() re-quotes strings and
    # breaks Jinja2 expressions like {{ x | default('y') }} → default(''y'').
    context = {**os.environ, **secrets}
    raw = yaml.safe_load(Template(_DEFAULT_YML.read_text()).render(context)) or {}

    if config_path:
        override = yaml.safe_load(Template(Path(config_path).read_text()).render(context)) or {}
        _deep_merge(raw, override)

    # Cascade top-level defaults into each data_source
    _cascade_defaults(raw)

    _app_config = AppConfig.model_validate(raw)
    _setup_logging(_app_config)
    return _app_config


def get_config() -> AppConfig:
    if _app_config is None:
        raise RuntimeError("load_config() has not been called")
    return _app_config


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


_CASCADE_KEYS = (
    "batch_size",
    "history_interval",
    "history_retention",
    "trends_retention",
    "anomaly_keep_secs",
    "detectors",
    "ensemble",
    "clustering",
    "metric_categories",
    "item_filters",
    "anomaly_filters",
    "fast_detect",
    "dashboards",
)


def _cascade_defaults(raw: dict) -> None:
    for ds in raw.get("data_sources", {}).values():
        for key in _CASCADE_KEYS:
            if key not in ds and key in raw:
                ds[key] = raw[key]


def _setup_logging(cfg: AppConfig) -> None:
    log_cfg = cfg.logging
    if log_cfg.enabled:
        os.makedirs(log_cfg.log_dir, exist_ok=True)
        log_file = os.path.join(log_cfg.log_dir, log_cfg.file)
        logging.basicConfig(
            filename=log_file,
            level=getattr(logging, log_cfg.level.upper(), logging.INFO),
            format=log_cfg.format,
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_cfg.format,
        )

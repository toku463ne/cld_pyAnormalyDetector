from config.schema import DataSourceConfig
from ingestion.base import DataSource


def get_data_source(config: DataSourceConfig) -> DataSource:
    if config.type == "csv":
        from ingestion.csv_source import CsvSource
        return CsvSource(config)
    if config.type == "zabbix_psql":
        from ingestion.zabbix_psql import ZabbixPsqlSource
        return ZabbixPsqlSource(config)
    raise ValueError(f"Unknown data source type: {config.type!r}")

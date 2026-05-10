

## Setup for DEV
```
CREATE DATABASE anomdec_test;
CREATE USER anomdec WITH PASSWORD 'anomdec_pass';
GRANT ALL PRIVILEGES ON DATABASE anomdec_test TO anomdec;
\c anomdec_test
GRANT ALL ON SCHEMA public TO anomdec;
```

## Available commands

| Command | Description |
|---|---|
| `anomdec-detect` | Hourly anomaly detection (production) |
| `anomdec-update-stats` | Daily trends / hour-stats batch update |
| `anomdec-sample` | Sample production data for labeling (stratified random sample) |
| `anomdec-dashboard` | Create a Zabbix dashboard from a sample for review |
| `anomdec-export-dashboard` | Export every item shown in an existing Zabbix dashboard |
| `anomdec-label` | Launch labeling UI (Dash) |

## Export prod data
- Step 1: sample from prod (takes ~minutes depending on item count)
    ```bash
    uv run anomdec-sample \
        -c config.yml \
        --source production \
        --output datasets/sample_$(date +%Y%m%d)/psql \
        --n-top 50 --n-mid 20 --n-random 50
    ```

- Step 2: create the Zabbix labeling dashboard
    ```bash
    uv run anomdec-dashboard \
        --scores datasets/sample_.../psql/scores.csv \
        --api-url http://zabbix/api_jsonrpc.php \
        --user Admin --password secret \
        --name labeling_20250510
    ```

- Step 3: review dashboard, edit labels.csv, run backtester
    ```bash
    uv run python -m evaluation.backtester \
        --dataset datasets/sample_.../psql \
        --labels datasets/sample_.../psql/labels.csv
    ```

## Export an existing Zabbix dashboard for labeling

If your team already runs a daily anomaly-review dashboard (e.g. the output of
the old algorithm), labeling exactly those items is far more useful than random
sampling — they are the cases your team actually triages.

Define the dashboard once in `config.yml` under `view_sources`:

```yaml
view_sources:
  zb10:
    type: zabbix_dashboard
    dashboard_name: abnormal_check
    api_url: "{{ ZABBIX_PSQL_API_URL }}"
    user:     "{{ ZABBIX_PSQL_API_USER }}"
    password: "{{ ZABBIX_PSQL_API_PASSWORD }}"
    data_source_name: zb10        # links to data_sources.zb10 for DB access
```

Then export:

```bash
uv run anomdec-export-dashboard \
    -c config.yml \
    --view-source zb10 \
    --output datasets/dashboard_$(date +%Y%m%d)/psql
```

The tool walks every page of the dashboard (resolving graph widgets to their
items) and writes `history.csv.gz`, `trends.csv.gz`, `items.csv.gz`,
`endep.txt`, and a skeleton `labels.csv` (all `label=-1`). Then label with the
UI and run the backtester:

```bash
uv run anomdec-label --dataset datasets/dashboard_$(date +%Y%m%d)/psql

uv run python -m evaluation.backtester \
    --dataset datasets/dashboard_$(date +%Y%m%d)/psql \
    --labels  datasets/dashboard_$(date +%Y%m%d)/psql/labels.csv
```

## Labeling data from UI

Review anomaly data visually and assign labels (anomaly / normal / skip) through
an interactive Dash app.  Labels are auto-saved to `labels.csv` on every click.

### Install UI dependencies (one-time)
```bash
uv sync --extra ui
```

### Launch the UI

With data exported from the old algorithm:
```bash
uv run anomdec-label --dataset tests/testdata/csv/20250508/psql
```

With data sampled from production (see **Export prod data** above):
```bash
uv run anomdec-label --dataset datasets/sample_$(date +%Y%m%d)/psql
```

Then open **http://localhost:8060** in a browser.

### Dataset directory must contain

| File | Description |
|---|---|
| `history.csv.gz` | `itemid, clock, value` |
| `trends.csv.gz` | `itemid, clock, value_min, value_avg, value_max` |
| `anomalies.csv.gz` | Detection records from old algorithm *(optional)* |
| `items.csv.gz` | Item metadata |
| `endep.txt` | End epoch |
| `scores.csv` | Scores from `sample_prod.py` *(optional)* |

### How to label

1. Use the **group dropdown** to navigate between Zabbix host groups.
2. For each item, the chart shows:
   - Grey line — trends (long-term context)
   - Blue line — recent history
   - Green dotted — `trend_mean`; red/blue dashed — `±Nσ`
   - Red `⚡` vertical lines — anomaly detection timestamps
3. Click a label button for each item:
   - **🔴 Anomaly** — confirmed anomaly
   - **🟢 Normal** — confirmed normal (false positive)
   - **⏭ Skip** — exclude from evaluation
4. Progress counter at the top shows how many items remain.

`labels.csv` is written to the dataset directory automatically after each click.

### Run evaluation after labeling

```bash
uv run python -m evaluation.backtester \
    --dataset datasets/sample_.../psql \
    --labels  datasets/sample_.../psql/labels.csv
```

## Production execution

```bash
# Hourly (via cron)
uv run anomdec-detect -c config.yml

# Daily (via cron, off-peak)
uv run anomdec-update-stats -c config.yml
```



## Setup

`scripts/setup.sh` provisions the management DB + account, installs the tools,
and runs a health test.  It is idempotent — safe to re-run.

```bash
# Create role + anomdec DB, install tools, health-test.
# Needs a PostgreSQL superuser; defaults to `sudo -u postgres`.
ANOMDEC_DB_PASSWORD='choose-a-strong-password' \
  ./scripts/setup.sh --with-test-db --write-secret secret.yml

# Health test only (no install, no DB creation) — e.g. in CI or after a deploy:
ANOMDEC_DB_PASSWORD=... ./scripts/setup.sh --check-only

# See all options (env vars + flags):
./scripts/setup.sh --help
```

The health test (`scripts/healthcheck.py`, also runnable standalone) connects as
the app user, creates the full store table set via the real code path — proving
CREATE/ALTER privileges and that the `rescued` column migration applies — then
drops it, and checks the CLI entrypoints.

Copy `secret.example.yml`, fill in real values, and point the app at it:
`export ANOMDEC_SECRET_PATH=/path/to/secret.yml` (or use `--write-secret`).

If a superuser shell isn't available, the role/DB can be created by hand:
```sql
CREATE USER anomdec WITH PASSWORD 'anomdec_pass';
CREATE DATABASE anomdec OWNER anomdec;       -- and/or anomdec_test for tests
\c anomdec
GRANT ALL ON SCHEMA public TO anomdec;
```

## Available commands

| Command | Description |
|---|---|
| `anomdec-detect` | Hourly anomaly detection (production) |
| `anomdec-update-stats` | Daily trends / hour-stats batch update |
| `anomdec-label-queue` | Build the daily stratified labeling queue (recommended) |
| `anomdec-sample` | Sample production data for labeling (stratified random sample) |
| `anomdec-dashboard` | Create a Zabbix dashboard from a sample for review |
| `anomdec-export-dashboard` | Export every item shown in an existing Zabbix dashboard |
| `anomdec-label` | Launch labeling UI (Dash) |

## Daily labeling queue (recommended)

With 30k+ items you can't label everything, and labeling only what the detector
flags hides its own misses. `anomdec-label-queue` builds a small **stratified**
daily queue from *this repo's* output (it does not depend on any old
`abnormal_check` dashboard):

- **flagged** — real alerts from the `{ds}_anomalies` table, collapsed to **one
  item per incident cluster** → measures precision
- **boundary** — items scored just under threshold (0.1–0.5) → recall / threshold
- **control** — a random sample of low-scoring items → miss-rate control

It dedups against a persistent master label file keyed by **(host, key_)** (which
survives Zabbix item-id churn), so each day surfaces only unlabeled items.

Prerequisite: the hourly `anomdec-detect` and daily `anomdec-update-stats` have
run (the queue reads their stored stats + anomalies; only the ~50 selected items
are pulled from Zabbix for the UI charts).

```bash
# Daily cycle:
uv run anomdec-label-queue merge    --dataset datasets/queue_<yesterday>/psql   # fold yesterday in
uv run anomdec-label-queue generate -c config.yml --source production \
    --output datasets/queue_$(date +%Y%m%d)/psql --n-mid 25 --n-random 15
uv run anomdec-label --dataset datasets/queue_$(date +%Y%m%d)/psql              # label
```

The master file defaults to `datasets/master_labels.csv` (override with
`--master`). Run the backtester over accumulated queues to track precision/recall
as the corpus grows.

### Leaner: trust the first-stage z-score

For a smaller, higher-purity daily set, source the flagged stratum from the
**first-stage z-score** (simple recent-vs-trend comparison) instead of the
gated-ensemble anomalies table. These candidates are few and very likely real, and
need only the stored stats (no anomalies table / ensemble):

```bash
uv run anomdec-label-queue generate -c config.yml --source production \
    --output datasets/queue_$(date +%Y%m%d)/psql \
    --flagged-from zscore --n-flagged 30 --n-mid 0 --n-random 10
```

Keep the small `--n-random` control slice: labeling *only* flagged items measures
precision but hides what the detector misses (recall). The control sample keeps
"is detection doing good?" answerable.

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
The labeling UI needs `dash` + `plotly`, which are kept out of the default
install so production hosts running `anomdec-detect` don't carry them. Either
sync them once:

```bash
uv sync --extra ui
```

…or pass `--extra ui` on every `uv run` invocation that launches the UI:

```bash
uv run --extra ui anomdec-label --dataset <path>
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

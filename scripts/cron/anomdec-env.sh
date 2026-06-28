# anomdec-env.sh — shared environment for the cron wrapper scripts.
#
# EDIT THIS FILE for your environment; the wrappers (run-*.sh) source it, so the
# crontab lines stay trivial. Sourced, not executed.

# Repo / working directory (where config.yml lives and datasets/ is written).
export ANOMDEC_HOME="/data/apps/anomdec"

# Installed entrypoints (the venv created by scripts/setup.sh).
export ANOMDEC_BIN="$ANOMDEC_HOME/.venv/bin"

# Log directory (create it + add logrotate).
export ANOMDEC_LOG="/data/apps/anomdec/log"

# Config file, relative to ANOMDEC_HOME (or an absolute path).
export ANOMDEC_CONFIG="config.yml"

# Secrets file consumed by the app (DB passwords etc.).
export ANOMDEC_SECRET_PATH="/data/.creds/anomdec/secret.yml"

# data_sources name used by the daily labeling queue.
export ANOMDEC_SOURCE="production"

# Extra args for `anomdec-label-queue generate` (first-stage z-score + small control).
export ANOMDEC_QUEUE_ARGS="--flagged-from zscore --n-flagged 30 --n-mid 0 --n-random 10"

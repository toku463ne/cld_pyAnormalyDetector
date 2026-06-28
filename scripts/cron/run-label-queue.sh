#!/usr/bin/env bash
# Daily labeling-candidate queue. Cron: once a day, e.g. 07:30.
# Only generates candidates; labeling (anomdec-label) and merge stay manual.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/anomdec-env.sh"
mkdir -p "$ANOMDEC_LOG"
cd "$ANOMDEC_HOME"
exec >> "$ANOMDEC_LOG/queue.log" 2>&1
echo "=== $(date '+%F %T') anomdec-label-queue generate ==="
OUT="datasets/queue_$(date +%Y%m%d)/psql"
# shellcheck disable=SC2086  # ANOMDEC_QUEUE_ARGS is intentionally word-split
exec flock -n /tmp/anomdec-queue.lock "$ANOMDEC_BIN/anomdec-label-queue" generate \
  -c "$ANOMDEC_CONFIG" --source "$ANOMDEC_SOURCE" --output "$OUT" $ANOMDEC_QUEUE_ARGS

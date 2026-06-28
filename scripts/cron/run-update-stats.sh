#!/usr/bin/env bash
# Daily stats batch (trends_stats + hour_stats; heavy). Cron: off-peak, e.g. 02:15.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/anomdec-env.sh"
mkdir -p "$ANOMDEC_LOG"
cd "$ANOMDEC_HOME"
exec >> "$ANOMDEC_LOG/update_stats.log" 2>&1
echo "=== $(date '+%F %T') anomdec-update-stats ==="
exec flock -n /tmp/anomdec-stats.lock "$ANOMDEC_BIN/anomdec-update-stats" -c "$ANOMDEC_CONFIG"

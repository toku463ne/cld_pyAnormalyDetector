#!/usr/bin/env bash
# Slow axis (full hourly detection + clustering). Cron: hourly at :05.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/anomdec-env.sh"
mkdir -p "$ANOMDEC_LOG"
cd "$ANOMDEC_HOME"
exec >> "$ANOMDEC_LOG/detect.log" 2>&1
echo "=== $(date '+%F %T') anomdec-detect ==="
flock -n /tmp/anomdec-detect.lock "$ANOMDEC_BIN/anomdec-detect" -c "$ANOMDEC_CONFIG"
# Publish the hourly Zabbix dashboard (best-effort; never fail the detection job).
"$ANOMDEC_BIN/anomdec-publish-dashboard" -c "$ANOMDEC_CONFIG" || echo "dashboard publish failed"

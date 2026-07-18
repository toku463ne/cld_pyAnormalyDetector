#!/usr/bin/env bash
# Slow axis (full hourly detection + clustering). Cron: hourly at :05.
#
# Overlap protection lives in the tool itself (pipeline/lock.py), not here: a
# shell-level `flock` around this wrapper would only guard cron against cron,
# leaving a manual `anomdec-detect` free to run straight into a cron run.
# Exit 75 (EX_TEMPFAIL) means "another run holds the lock, nothing was done".
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/anomdec-env.sh"
mkdir -p "$ANOMDEC_LOG"
cd "$ANOMDEC_HOME"
exec >> "$ANOMDEC_LOG/detect.log" 2>&1
echo "=== $(date '+%F %T') anomdec-detect ==="

rc=0
"$ANOMDEC_BIN/anomdec-detect" -c "$ANOMDEC_CONFIG" || rc=$?
if [ "$rc" -eq 75 ]; then
  # Already logged above by the tool, with the holding pid and its age.  Not a
  # failure: exit 0 so cron does not mail on every overlap.
  echo "skipped: another anomdec-detect run is in progress"
  exit 0
fi
if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi

# Publish the hourly Zabbix dashboard (best-effort; never fail the detection job).
"$ANOMDEC_BIN/anomdec-publish-dashboard" -c "$ANOMDEC_CONFIG" || echo "dashboard publish failed"

#!/usr/bin/env bash
# Daily stats batch (trends_stats + hour_stats; heavy). Cron: off-peak, e.g. 02:15.
#
# Overlap protection lives in the tool itself (pipeline/lock.py), not here: a
# shell-level `flock` would only guard cron against cron, leaving a manual run
# free to collide.  Exit 75 (EX_TEMPFAIL) = another run holds the lock.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/anomdec-env.sh"
mkdir -p "$ANOMDEC_LOG"
cd "$ANOMDEC_HOME"
exec >> "$ANOMDEC_LOG/update_stats.log" 2>&1
echo "=== $(date '+%F %T') anomdec-update-stats ==="
rc=0
"$ANOMDEC_BIN/anomdec-update-stats" -c "$ANOMDEC_CONFIG" || rc=$?
if [ "$rc" -eq 75 ]; then
  # Already logged by the tool, with the holding pid and its age.  Not a
  # failure: exit 0 so cron does not mail on every overlap.
  echo "skipped: another anomdec-update-stats run is in progress"
  exit 0
fi
exit "$rc"

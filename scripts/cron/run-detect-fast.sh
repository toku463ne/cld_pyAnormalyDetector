#!/usr/bin/env bash
# Fast axis (short-span watchlist). Cron: every ~10 min.
#
# Overlap protection lives in the tool itself (pipeline/lock.py), not here: a
# shell-level `flock` would only guard cron against cron, leaving a manual run
# free to collide.  Exit 75 (EX_TEMPFAIL) = another run holds the lock.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/anomdec-env.sh"
mkdir -p "$ANOMDEC_LOG"
cd "$ANOMDEC_HOME"
exec >> "$ANOMDEC_LOG/fast.log" 2>&1
echo "=== $(date '+%F %T') anomdec-detect-fast ==="
rc=0
"$ANOMDEC_BIN/anomdec-detect-fast" -c "$ANOMDEC_CONFIG" || rc=$?
if [ "$rc" -eq 75 ]; then
  # Already logged by the tool, with the holding pid and its age.  Not a
  # failure: exit 0 so cron does not mail on every overlap.
  echo "skipped: another anomdec-detect-fast run is in progress"
  exit 0
fi
exit "$rc"

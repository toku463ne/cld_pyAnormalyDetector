#!/usr/bin/env bash
# Fast axis (short-span watchlist). Cron: every ~10 min.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/anomdec-env.sh"
mkdir -p "$ANOMDEC_LOG"
cd "$ANOMDEC_HOME"
exec >> "$ANOMDEC_LOG/fast.log" 2>&1
echo "=== $(date '+%F %T') anomdec-detect-fast ==="
exec flock -n /tmp/anomdec-fast.lock "$ANOMDEC_BIN/anomdec-detect-fast" -c "$ANOMDEC_CONFIG"

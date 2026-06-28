#!/usr/bin/env bash
# On-demand: export raw data for the currently-flagged items (the anomdec_detected
# set) for offline inspection, and tar it up to share.
#
# Reuses scripts/cron/anomdec-env.sh for paths/config/source. Extra args are
# passed through to anomdec-export-anomalies (e.g. --all-cycles).
#
#   ./scripts/export-anomalies.sh                # latest detection cycle
#   ./scripts/export-anomalies.sh --all-cycles   # everything in the keep window
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/cron/anomdec-env.sh"
cd "$ANOMDEC_HOME"

TS="$(date +%Y%m%d_%H%M)"
OUT="datasets/check_${TS}/psql"
TARBALL="datasets/check_${TS}.tar.gz"

"$ANOMDEC_BIN/anomdec-export-anomalies" \
  -c "$ANOMDEC_CONFIG" --source "$ANOMDEC_SOURCE" --output "$OUT" "$@"

tar czf "$TARBALL" -C "$OUT" .
echo "dataset  -> $ANOMDEC_HOME/$OUT"
echo "tarball  -> $ANOMDEC_HOME/$TARBALL   (share this)"

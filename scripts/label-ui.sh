#!/usr/bin/env bash
# Start the Dash labeling UI (anomdec-label) on a dataset.
#
# Reuses scripts/cron/anomdec-env.sh for paths. Default dataset is the newest
# datasets/queue_*/psql (produced by anomdec-label-queue). Extra args pass through
# to anomdec-label (--port, --host, --debug).
#
#   ./scripts/label-ui.sh                              # newest queue dataset
#   ./scripts/label-ui.sh datasets/queue_20260628/psql # a specific dataset
#   ./scripts/label-ui.sh --port 8070                  # custom port
#
# Default binds 0.0.0.0:8060 (LAN-accessible) so you can browse it from another
# machine at http://<this-host-IP>:8060. Override the bind with ANOMDEC_LABEL_HOST
# or a trailing --host (e.g. ANOMDEC_LABEL_HOST=127.0.0.1 for tunnel-only access).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/cron/anomdec-env.sh"
cd "$ANOMDEC_HOME"

DATASET=""
if [ $# -gt 0 ] && [ -d "$1" ]; then
  DATASET="$1"; shift
fi
[ -n "$DATASET" ] || DATASET="$(ls -d datasets/queue_*/psql 2>/dev/null | sort | tail -1 || true)"
[ -n "$DATASET" ] || { echo "no dataset given and no datasets/queue_*/psql found" >&2; exit 1; }

BIN="$ANOMDEC_BIN/anomdec-label"
[ -x "$BIN" ] || { echo "anomdec-label not found — install the UI extra: uv pip install -e '.[ui]'" >&2; exit 1; }

HOST="${ANOMDEC_LABEL_HOST:-0.0.0.0}"
echo "labeling dataset: $DATASET"
echo "open from your browser: http://<this-host-IP>:8060   (binding $HOST)"
# A trailing --host in "$@" overrides this default (argparse keeps the last one).
exec "$BIN" --dataset "$DATASET" --host "$HOST" "$@"

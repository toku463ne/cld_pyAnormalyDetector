#!/usr/bin/env bash
# Start the Dash labeling UI (anomdec-label) on a dataset.
#
# Reuses scripts/cron/anomdec-env.sh for paths. Default dataset is the newest
# datasets/queue_*/psql (produced by anomdec-label-queue). Extra args pass through
# to anomdec-label (--port, --host, --debug).
#
#   ./scripts/label-ui.sh                              # newest queue dataset
#   ./scripts/label-ui.sh datasets/queue_20260628/psql # a specific dataset
#   ./scripts/label-ui.sh --host 0.0.0.0 --port 8070   # remote access / custom port
#
# Default binds 127.0.0.1:8060 — for a remote box use an SSH tunnel
# (ssh -L 8060:localhost:8060 host) or pass --host 0.0.0.0.
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

echo "labeling dataset: $DATASET"
exec "$BIN" --dataset "$DATASET" "$@"

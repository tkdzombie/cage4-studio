#!/bin/bash
# Can be run from any working directory. No downloads or shell config mutation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${CAGE4_PYTHON:-$HOME/cage4-studio/.conda-env/bin/python}"
export PYTHONPATH="$ROOT/protocol:$ROOT/cage4-baselines/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m cage4_baselines.runner "$@"

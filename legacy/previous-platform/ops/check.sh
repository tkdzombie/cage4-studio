#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${CAGE4_PYTHON:-$HOME/cage4-studio/.conda-env/bin/python}"
export PYTHONPATH="$ROOT/protocol:$ROOT/server/src:$ROOT/cage4-baselines/src"
cd "$ROOT"
"$PY" -m compileall -q protocol cage4-baselines/src server/src ops tests cage4-baselines/tests cage4-baselines/examples
"$PY" -m unittest discover -s tests -v
"$PY" -m unittest discover -s cage4-baselines/tests -v

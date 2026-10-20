#!/bin/bash
set -Eeuo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
PY="$BASE/.conda-env/bin/python"
PID_FILE="$BASE/.dashboard.pid"
if [[ ! -x "$PY" ]]; then
  echo 'Not installed. First run: ./install.sh'
  exit 1
fi
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Dashboard process already running (pid $(cat "$PID_FILE"))"
  exit 0
fi
# No public binding. Use ssh -L to reach this service.
mkdir -p "$BASE/runs"
cd "$BASE"
nohup "$PY" -m streamlit run cage4studio/ui.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --server.headless true --server.fileWatcherType none \
  >> "$BASE/dashboard.log" 2>&1 < /dev/null &
echo "$!" > "$PID_FILE"
sleep 2
if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo 'Dashboard failed. See dashboard.log'
  exit 1
fi
# TensorBoard reads per-run logs. Logdir_spec displays all run subdirectories.
if [[ -f "$BASE/.tensorboard.pid" ]] && kill -0 "$(cat "$BASE/.tensorboard.pid")" 2>/dev/null; then
  :
else
  nohup "$BASE/.conda-env/bin/tensorboard" --logdir "$BASE/runs" --host 127.0.0.1 --port 6006 \
    >> "$BASE/tensorboard.log" 2>&1 < /dev/null &
  echo "$!" > "$BASE/.tensorboard.pid"
fi
echo 'CAGE4 Studio: http://127.0.0.1:8501 (Mac localhost only)'
echo 'TensorBoard:  http://127.0.0.1:6006 (Mac localhost only)'

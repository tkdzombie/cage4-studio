#!/bin/bash
# Manual, read-only service. Never launches training or a scheduler.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="/tmp/cage4-dashboard-${UID}.pid"
LOG_FILE="/tmp/cage4-dashboard-${UID}.log"
case "${1:-status}" in
 start)
  if curl -fsS --max-time 3 http://127.0.0.1:8501/_stcore/health >/dev/null; then
   echo 'Port 8501 already serves a dashboard; verify status before changing it.'; exit 0
  fi
  nohup "$ROOT/.venv/bin/python" -m streamlit run "$ROOT/cage4/dashboard.py" \
   --server.address 127.0.0.1 --server.port 8501 --server.headless true \
   --server.fileWatcherType none --browser.gatherUsageStats false \
   >>"$LOG_FILE" 2>&1 </dev/null &
  echo "$!" >"$PID_FILE"
  echo 'Started CAGE4 dashboard on loopback 8501.'
  ;;
 status)
  if [[ -f "$PID_FILE" ]]; then ps -p "$(cat "$PID_FILE")" -o pid=,command=; fi
  curl -fsS --max-time 3 http://127.0.0.1:8501/_stcore/health
  ;;
 *) echo 'Usage: dashboard.sh {start|status}' >&2; exit 2 ;;
esac

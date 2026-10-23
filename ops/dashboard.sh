#!/bin/bash
# Manual, read-only service. Never launches training or a scheduler.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-status}" in
 start)
  if curl -fsS --max-time 3 http://127.0.0.1:8501/_stcore/health >/dev/null; then
   echo 'Port 8501 already serves a dashboard; verify status before changing it.'; exit 0
  fi
  nohup "$ROOT/.venv/bin/python" -m streamlit run "$ROOT/dashboard.py" \
   --server.address 127.0.0.1 --server.port 8501 --server.headless true \
   --server.fileWatcherType none --browser.gatherUsageStats false \
   >>"$ROOT/dashboard.log" 2>&1 </dev/null &
  echo "$!" >"$ROOT/dashboard.pid"
  echo 'Started CAGE4 dashboard on loopback 8501.'
  ;;
 status)
  if [[ -f "$ROOT/dashboard.pid" ]]; then ps -p "$(cat "$ROOT/dashboard.pid")" -o pid=,command=; fi
  curl -fsS --max-time 3 http://127.0.0.1:8501/_stcore/health
  ;;
 *) echo 'Usage: dashboard.sh {start|status}' >&2; exit 2 ;;
esac

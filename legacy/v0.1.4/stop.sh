#!/bin/bash
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
for file in .dashboard.pid .tensorboard.pid; do
  if [[ -f "$BASE/$file" ]]; then
    pid="$(cat "$BASE/$file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "Stopped $file (pid $pid)"
    fi
    rm -f "$BASE/$file"
  fi
done
echo 'Training subprocesses are NOT killed. Stop them from the dashboard first.'

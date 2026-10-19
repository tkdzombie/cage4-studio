#!/bin/bash
# Repair missing CybORG import without reinstalling dependencies.
set -Eeuo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
PY="$BASE/.conda-env/bin/python"
REPO="$BASE/cage-challenge-4"
[[ -x "$PY" ]] || { echo "ERROR: Python not found: $PY"; exit 1; }
[[ -f "$REPO/CybORG/__init__.py" ]] || { echo "ERROR: No CybORG module at $REPO/CybORG"; exit 1; }
PURELIB="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
printf '%s\n' "$REPO" > "$PURELIB/cage4_repo.pth"
"$PY" -c 'import CybORG; print("CybORG import OK:", CybORG.__file__)'
"$PY" "$BASE/smoke_test.py"
echo 'Import repair completed; start dashboard with: ./start.sh'

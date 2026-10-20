#!/bin/bash
# Offline smoke test and diagnostics; never changes installed packages.
set -Eeuo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
PY="$BASE/.conda-env/bin/python"
printf '\n=== CAGE4 Studio diagnostic ===\n'
printf 'Host: %s / %s\n' "$(uname -s)" "$(uname -m)"
printf 'Directory: %s\n' "$BASE"
if [[ ! -x "$PY" ]]; then
  echo 'ERROR: Python environment absent; run ./install.sh on Mac Studio.'
  exit 1
fi
"$PY" --version
"$PY" - <<'PY'
from importlib import metadata
for pkg in ('CybORG', 'gymnasium', 'ray', 'torch', 'streamlit', 'setuptools', 'pyarrow'):
    try:
        print(f'{pkg}: {metadata.version(pkg)}')
    except metadata.PackageNotFoundError:
        print(f'{pkg}: MISSING')
try:
    import pkg_resources  # noqa
    print('pkg_resources import: OK')
except ImportError:
    print('pkg_resources import: FAILED; pin setuptools==80.9.0')
PY
PURELIB="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
if [[ -f "$PURELIB/cage4_repo.pth" ]]; then
  echo "CybORG path registration: $(cat "$PURELIB/cage4_repo.pth")"
else
  echo 'CybORG path registration: NOT FOUND (run ./repair_cyborg_import.sh)'
fi
printf '\n=== Official simulation smoke test ===\n'
"$PY" "$BASE/smoke_test.py"
printf '\nDiagnostics PASS. Services can be started with: bash start.sh\n'

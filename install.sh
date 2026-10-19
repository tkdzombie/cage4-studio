#!/bin/bash
# CAGE4 Studio installer for Apple Silicon Mac Studio; run as ordinary user.
set -Eeuo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$BASE/.miniforge3"
ENV_DIR="$BASE/.conda-env"
REPO="$BASE/cage-challenge-4"
LOG="$BASE/install.log"
exec > >(tee -a "$LOG") 2>&1

printf '\n[CAGE4 Studio 0.1.4] Starting install in %s\n' "$BASE"
if [[ "$(uname -s)" != Darwin ]] || [[ "$(uname -m)" != arm64 ]]; then
  echo 'ERROR: This installer targets macOS Apple Silicon (arm64).'
  exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
  echo 'ERROR: Apple Command Line Tools missing. Run: xcode-select --install'
  exit 1
fi
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  echo 'ERROR: git/curl not available.'
  exit 1
fi
# Fast path for an already working server: do not redownload or reinstall
# huge ML dependencies on every package update.
if [[ -x "$ENV_DIR/bin/python" && -f "$REPO/CybORG/__init__.py" && "${CAGE4_FORCE_INSTALL:-0}" != 1 ]]; then
  echo '[reuse] Found existing Python environment and CAGE 4 checkout.'
  PY="$ENV_DIR/bin/python"
  PURELIB="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
  printf '%s\n' "$REPO" > "$PURELIB/cage4_repo.pth"
  if "$PY" "$BASE/smoke_test.py"; then
    echo '[reuse] SMOKE TEST PASS. Skipping heavyweight downloads and dependency installation.'
    "$PY" -m pip freeze > "$BASE/installed-pip-freeze.txt"
    REV="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'CAGE4 official commit: %s\nPython: %s\nVerified at: %s\n' \
      "$REV" "$("$PY" --version 2>&1)" "$(date)" > "$BASE/installed-revision.txt"
    "$BASE/start.sh"
    echo 'Existing working installation reused; no reinstall needed.'
    exit 0
  fi
  echo '[reuse] Existing environment did not pass the smoke test; attempting dependency repair.'
fi

if [[ ! -x "$PREFIX/bin/conda" ]]; then
  echo '[1/5] Downloading Miniforge arm64 (from official conda-forge releases)'
  INSTALLER="$BASE/.Miniforge3-MacOSX-arm64.sh"
  # Fixed official release and published SHA256: https://conda-forge.org/miniforge/
  # "latest/download/<alias>.sha256" is not a valid release asset and returns 404.
  MINIFORGE_VERSION='26.5.3-0'
  WANT='0d765919d3ccfd1f89147aa1cf8133bfc55b3a3c13f5bacdcc091c33132fddd2'
  URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-MacOSX-arm64.sh"
  if [[ -f "$INSTALLER" ]]; then
    HAVE="$(shasum -a 256 "$INSTALLER" | awk '{print $1}')"
    if [[ "$HAVE" != "$WANT" ]]; then
      echo 'Downloaded installer does not match pinned release; downloading correct version.'
      rm -f "$INSTALLER"
    fi
  fi
  if [[ ! -f "$INSTALLER" ]]; then
    curl --fail --location --retry 3 "$URL" --output "$INSTALLER"
  fi
  HAVE="$(shasum -a 256 "$INSTALLER" | awk '{print $1}')"
  if [[ "$WANT" != "$HAVE" ]]; then
    echo "ERROR: Miniforge SHA256 mismatch; expected $WANT, got $HAVE."
    rm -f "$INSTALLER"
    exit 1
  fi
  echo "SHA256 verified for Miniforge ${MINIFORGE_VERSION}."
  bash "$INSTALLER" -b -p "$PREFIX"
  rm -f "$INSTALLER"
else
  echo '[1/5] Existing Miniforge found.'
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo '[2/5] Creating isolated Python 3.10 environment'
  "$PREFIX/bin/conda" create -y -p "$ENV_DIR" python=3.10 pip
else
  echo '[2/5] Existing Python environment found.'
fi
PY="$ENV_DIR/bin/python"
if [[ ! -e "$REPO/.git" ]]; then
  echo '[3/5] Cloning official TTCP CAGE Challenge 4'
  git clone --depth 1 https://github.com/cage-challenge/cage-challenge-4.git "$REPO"
else
  echo '[3/5] Existing official Git repository retained; not updating implicitly.'
fi
if [[ -n "${CAGE4_GIT_REF:-}" ]]; then
  echo 'Switching to explicitly requested git ref (requires fetch access)'
  git -C "$REPO" fetch origin "$CAGE4_GIT_REF"
  git -C "$REPO" checkout --detach FETCH_HEAD
fi
REV="$(git -C "$REPO" rev-parse HEAD)"
echo "CAGE4 commit = $REV"
echo '[4/5] Installing focused core dependencies (macOS arm64)'
"$PY" -m pip install --upgrade pip wheel "setuptools==80.9.0"
"$PY" -m pip install -r "$BASE/requirements-core.txt"
"$PY" -m pip install --no-deps -e "$REPO"
# Upstream CAGE4 setup.py declares py_modules=[] and does not install CybORG
# as an importable Python package. Register the checkout as an import root.
echo '[4/5] Registering upstream CybORG source directory with this Python environment'
if [[ ! -f "$REPO/CybORG/__init__.py" ]]; then
  echo "ERROR: Missing $REPO/CybORG/__init__.py; verify official Git checkout."
  exit 1
fi
PURELIB="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
printf '%s\n' "$REPO" > "$PURELIB/cage4_repo.pth"
"$PY" -c 'import CybORG; print("CybORG import OK:", CybORG.__file__)'
if [[ "${CAGE4_FULL_DEPS:-0}" == 1 ]]; then
  echo 'Optional full official dependency install requested (large, may cause version conflicts)'
  "$PY" -m pip install -r "$REPO/Requirements.txt"
fi
echo '[5/5] Running import + API smoke test'
"$PY" "$BASE/smoke_test.py"
"$PY" -m pip freeze > "$BASE/installed-pip-freeze.txt"
cat > "$BASE/installed-revision.txt" <<EOF_REV
CAGE4 official commit: $REV
Python: $($PY --version 2>&1)
Installed at: $(date)
EOF_REV
echo 'INSTALL SUCCESS. Starting dashboard...'
"$BASE/start.sh"
echo 'On your laptop: python3 local/connect.py --host <Mac-IP> --user <ssh-user>'

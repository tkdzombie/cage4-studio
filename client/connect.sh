#!/bin/bash
# Local Mac only. SSH forwarding + browser; no remote commands or PyCharm control.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_HOST='cage4-studio'; DEFAULT_USER="${USER:-researcher}"; DEFAULT_PORT=18501
if [[ -f "$HERE/settings.local" ]]; then
 { IFS= read -r DEFAULT_HOST; IFS= read -r DEFAULT_USER; IFS= read -r DEFAULT_PORT; } < "$HERE/settings.local"
fi
ACTION="${1:-start}"; HOST="${2:-$DEFAULT_HOST}"; USER_NAME="${3:-$DEFAULT_USER}"
PORT="${4:-$DEFAULT_PORT}"; TB="${5:-0}"
[[ "$HOST" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] || exit 2
[[ "$USER_NAME" =~ ^[A-Za-z_][A-Za-z0-9_-]*$ ]] || exit 2
[[ "$PORT" =~ ^[0-9]+$ && "$TB" =~ ^[0-9]+$ ]] || exit 2
(( PORT >= 1 && PORT <= 65535 && TB >= 0 && TB <= 65535 && TB != PORT )) || exit 2
BASE="$HOME/Library/Application Support/CAGE4-Studio"
umask 077
mkdir -p "$BASE"
KEY="$(printf '%s' "$USER_NAME@$HOST:$PORT:$TB" | shasum -a 256 | cut -c1-16)"
SOCK="$BASE/$KEY.sock"
DEST="$USER_NAME@$HOST"
COMMON=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -S "$SOCK")
case "$ACTION" in
 start)
  if ! ssh "${COMMON[@]}" -O check "$DEST" >/dev/null 2>&1; then
   FORWARDS=(-L "127.0.0.1:$PORT:127.0.0.1:8501")
   if (( TB > 0 )); then FORWARDS+=(-L "127.0.0.1:$TB:127.0.0.1:6006"); fi
   ssh "${COMMON[@]}" -M -fN -o ControlPersist=yes -o ExitOnForwardFailure=yes \
       -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "${FORWARDS[@]}" "$DEST"
  fi
  # Verify tunneled server health; an existing unrelated local listener is not reused.
  curl --fail --silent --max-time 10 "http://127.0.0.1:$PORT/_stcore/health" >/dev/null
  open "http://127.0.0.1:$PORT"
  ;;
 stop) ssh "${COMMON[@]}" -O exit "$DEST" ;;
 status) ssh "${COMMON[@]}" -O check "$DEST" ;;
 *) echo 'Usage: connect.sh {start|stop|status} [host] [user] [local-port] [local-tb-port-or-0]' >&2; exit 2 ;;
esac

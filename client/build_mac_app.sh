#!/bin/bash
set -euo pipefail
[[ "$(uname -s)" == Darwin ]] || exit 1
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE}"
HOST="${2:-cage4-studio}"; REMOTE_USER="${3:-${USER:-researcher}}"; LOCAL_PORT="${4:-18501}"
[[ "$HOST" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ && "$REMOTE_USER" =~ ^[A-Za-z_][A-Za-z0-9_-]*$ ]] || exit 2
[[ "$LOCAL_PORT" =~ ^[0-9]+$ ]] && (( LOCAL_PORT > 0 && LOCAL_PORT <= 65535 )) || exit 2
APP="$OUT/CAGE4 Studio Client.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$HERE/connect.sh" "$APP/Contents/Resources/"
printf '%s\n' "$HOST" "$REMOTE_USER" "$LOCAL_PORT" > "$APP/Contents/Resources/settings.local"
# Native executable: Finder/LaunchServices must launch a real macOS application.
xcrun clang -fobjc-arc -framework Cocoa -mmacosx-version-min=11.0 \
  "$HERE/launcher.m" -o "$APP/Contents/MacOS/launcher"
chmod +x "$APP/Contents/MacOS/launcher" "$APP/Contents/Resources/connect.sh"
/usr/bin/python3 - "$APP/Contents/Info.plist" <<'PLIST'
import plistlib,sys
with open(sys.argv[1],'wb') as f:
 plistlib.dump(dict(CFBundleName='CAGE4 Studio Client',CFBundleDisplayName='CAGE4 Studio Client',
 CFBundleIdentifier='org.cage4studio.localclient',CFBundleExecutable='launcher',
 CFBundlePackageType='APPL',CFBundleShortVersionString='0.4.1',CFBundleVersion='41',
 LSMinimumSystemVersion='11.0',LSUIElement=True),f)
PLIST
plutil -lint "$APP/Contents/Info.plist"
codesign --force --sign - "$APP"
codesign --verify --strict "$APP"
echo "$APP"

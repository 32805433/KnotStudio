#!/bin/sh
# Register this extracted application for the current user; no sudo or Python.
set -eu
APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$DESKTOP_DIR"
# Desktop-entry Exec quoting has different escaping rules from shell quoting.
APP_EXEC=$(printf '%s' "$APP_DIR/KnotStudio" | sed 's/[\\`"$]/\\&/g; s/\\/\\\\/g; s/%/%%/g')
APP_ICON=$(printf '%s' "$APP_DIR/KnotStudio.png" | sed 's/\\/\\\\/g')
cat > "$DESKTOP_DIR/org.knotstudio.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Knot Studio
Comment=Draw, recognize, and edit knot and link diagrams
Exec="$APP_EXEC" %f
Icon=$APP_ICON
Terminal=false
Categories=Education;Science;Graphics;
StartupNotify=true
EOF
printf '%s\n' 'Knot Studio is now in your application menu.'

#!/bin/sh
set -eu
case "$0" in
  */*) APP_PATH=${0%/*} ;;
  *) APP_PATH=. ;;
esac
APP_DIR=$(CDPATH= cd -- "$APP_PATH" && pwd)
exec "$APP_DIR/KnotStudio" "$@"

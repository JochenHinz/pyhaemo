#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pyhaemo-gui.log"
cd "/home/fabio/pyhaemo"
{
  echo "==== $(date) ====
"
  echo "DISPLAY=${DISPLAY-}"
  echo "WAYLAND_DISPLAY=${WAYLAND_DISPLAY-}"
  echo "XDG_SESSION_TYPE=${XDG_SESSION_TYPE-}"
  echo "Launching PyHaemo GUI..."
  /home/fabio/miniconda3/envs/pyhaemo/bin/python "/home/fabio/pyhaemo/partery/gui_compact.py"
} >> "$LOG_FILE" 2>&1

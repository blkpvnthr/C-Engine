#!/usr/bin/env bash
# Install the C-Engine continuous runtime as a macOS LaunchAgent (addendum §2).
#
# Substitutes the repo + venv-python paths into the plist template, installs it to
# ~/Library/LaunchAgents, and loads it. The agent keeps `python -m cengine.runtime run` alive
# 24/7 and restarts it on crash. Running it never places live orders by itself (no trading-engine
# runner is wired; acquisition is dry-run) — it drives off-session research + continuous
# governance only.
#
# Usage:  scripts/install_runtime_service.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.cengine.runtime"
TEMPLATE="$REPO/deploy/launchd/$LABEL.plist.template"
AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST="$AGENTS_DIR/$LABEL.plist"

# Prefer the project venv python, fall back to python3 on PATH.
PYTHON="$REPO/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

if [ ! -f "$TEMPLATE" ]; then
  echo "error: template not found: $TEMPLATE" >&2
  exit 1
fi

mkdir -p "$AGENTS_DIR" "$REPO/data/runtime/logs"

echo "Repo:    $REPO"
echo "Python:  $PYTHON"
echo "Plist:   $PLIST"

sed -e "s#__REPO__#$REPO#g" -e "s#__PYTHON__#$PYTHON#g" "$TEMPLATE" > "$PLIST"

# Reload if already present.
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "Loaded LaunchAgent $LABEL."
echo "  status:  launchctl list | grep $LABEL"
echo "  logs:    tail -f $REPO/data/runtime/logs/runtime.out.log"
echo "  stop:    scripts/uninstall_runtime_service.sh"

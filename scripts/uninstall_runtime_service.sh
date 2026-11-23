#!/usr/bin/env bash
# Uninstall the C-Engine continuous runtime LaunchAgent (addendum §2).
# Unloads the agent and removes the installed plist. Does not touch data/ or logs.
set -euo pipefail

LABEL="com.cengine.runtime"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ -f "$PLIST" ]; then
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Unloaded and removed $LABEL."
else
  echo "No installed plist at $PLIST (nothing to do)."
fi

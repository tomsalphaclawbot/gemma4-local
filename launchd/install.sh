#!/usr/bin/env bash
# launchd/install.sh — Install/uninstall the Gemma 4 MLX launchd service
#
# Usage:
#   ./launchd/install.sh            # install + start
#   ./launchd/install.sh --uninstall  # stop + uninstall

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$PROJECT_DIR/launchd/work.tomsalphaclawbot.gemma4-mlx.plist"
PLIST_LABEL="work.tomsalphaclawbot.gemma4-mlx"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
PLIST_DEST="$LAUNCH_AGENTS/$PLIST_LABEL.plist"

case "${1:-}" in
  --uninstall)
    echo "Uninstalling $PLIST_LABEL..."
    launchctl bootout "gui/$UID/$PLIST_LABEL" 2>/dev/null && echo "  Stopped." || echo "  (was not running)"
    if [ -L "$PLIST_DEST" ]; then
      rm "$PLIST_DEST"
      echo "  Symlink removed: $PLIST_DEST"
    elif [ -f "$PLIST_DEST" ]; then
      rm "$PLIST_DEST"
      echo "  File removed: $PLIST_DEST"
    fi
    echo "Done."
    ;;

  *)
    echo "Installing $PLIST_LABEL..."

    # Ensure state dir exists (launchd writes logs here)
    mkdir -p "$PROJECT_DIR/state"

    # Symlink plist into LaunchAgents (symlink so edits in repo take effect)
    mkdir -p "$LAUNCH_AGENTS"
    if [ -e "$PLIST_DEST" ]; then
      echo "  Removing existing plist at $PLIST_DEST"
      launchctl bootout "gui/$UID/$PLIST_LABEL" 2>/dev/null || true
      rm -f "$PLIST_DEST"
      sleep 1
    fi
    # Note: launchd requires a real file, not a symlink
    cp "$PLIST_SRC" "$PLIST_DEST"
    echo "  Installed: $PLIST_DEST"

    # Bootstrap and start
    launchctl bootstrap "gui/$UID" "$PLIST_DEST"
    echo "  Service loaded. Model takes ~30s to warm up."
    echo ""
    echo "Check status:  launchctl list $PLIST_LABEL"
    echo "View logs:     tail -f $PROJECT_DIR/state/gemma4-server.log"
    echo "Test:          curl http://127.0.0.1:8890/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"mlx-community/gemma-4-26b-a4b-it-4bit\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":20}'"
    ;;
esac

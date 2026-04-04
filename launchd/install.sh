#!/usr/bin/env bash
# launchd/install.sh — Install/uninstall Gemma 4 MLX server + serializing proxy
#
# Usage:
#   ./launchd/install.sh              # install + start both services
#   ./launchd/install.sh --uninstall  # stop + uninstall both services

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

# Service definitions: label + source plist
SERVICES=(
  "work.tomsalphaclawbot.gemma4-mlx"
  "work.tomsalphaclawbot.gemma4-proxy"
)

install_service() {
  local label="$1"
  local src="$PROJECT_DIR/launchd/${label}.plist"
  local dest="$LAUNCH_AGENTS/${label}.plist"

  if [ ! -f "$src" ]; then
    echo "  ⚠ Plist not found: $src — skipping"
    return 1
  fi

  echo "  Installing $label..."
  if [ -e "$dest" ]; then
    launchctl bootout "gui/$UID/$label" 2>/dev/null || true
    rm -f "$dest"
    sleep 1
  fi

  cp "$src" "$dest"
  launchctl bootstrap "gui/$UID" "$dest"
  echo "    ✓ Loaded"
}

uninstall_service() {
  local label="$1"
  local dest="$LAUNCH_AGENTS/${label}.plist"

  echo "  Uninstalling $label..."
  launchctl bootout "gui/$UID/$label" 2>/dev/null && echo "    Stopped." || echo "    (was not running)"
  if [ -f "$dest" ] || [ -L "$dest" ]; then
    rm -f "$dest"
    echo "    Plist removed."
  fi
}

case "${1:-}" in
  --uninstall)
    echo "Uninstalling Gemma 4 services..."
    for svc in "${SERVICES[@]}"; do
      uninstall_service "$svc"
    done
    echo "Done."
    ;;

  *)
    echo "Installing Gemma 4 services..."
    mkdir -p "$PROJECT_DIR/state"
    mkdir -p "$LAUNCH_AGENTS"

    for svc in "${SERVICES[@]}"; do
      install_service "$svc"
    done

    echo ""
    echo "Both services installed. MLX server takes ~30s to warm up."
    echo ""
    echo "Check status:"
    echo "  launchctl list work.tomsalphaclawbot.gemma4-mlx"
    echo "  launchctl list work.tomsalphaclawbot.gemma4-proxy"
    echo ""
    echo "View logs:"
    echo "  tail -f $PROJECT_DIR/state/gemma4-server.log"
    echo "  tail -f $PROJECT_DIR/state/gemma4-proxy.log"
    echo ""
    echo "Test (always use proxy port 8891):"
    echo "  curl http://127.0.0.1:8891/v1/chat/completions -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\":\"mlx-community/gemma-4-26b-a4b-it-4bit\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":20}'"
    echo ""
    echo "Proxy stats:"
    echo "  curl -s http://127.0.0.1:8891/proxy/stats | python3 -m json.tool"
    ;;
esac

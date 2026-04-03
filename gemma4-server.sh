#!/usr/bin/env bash
# gemma4-server.sh — Start Gemma 4 26B MoE via mlx-vlm HTTP server
# OpenAI-compatible endpoint at http://127.0.0.1:8890
# TurboQuant KV-3 compression, ~15.5GB peak RAM
#
# Usage:
#   ./gemma4-server.sh            # start in background
#   ./gemma4-server.sh --fg       # start in foreground
#   ./gemma4-server.sh --status   # check if running
#   ./gemma4-server.sh --stop     # stop server

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="/Users/openclaw/.openclaw/workspace"
VENV="$WORKSPACE/.venvs/gemma4-mlx"
MODEL="mlx-community/gemma-4-26b-a4b-it-4bit"
PORT=8890
HOST="127.0.0.1"
PID_FILE="$PROJECT_DIR/state/gemma4-server.pid"
LOG_FILE="$PROJECT_DIR/state/gemma4-server.log"

# Ensure state dir exists
mkdir -p "$PROJECT_DIR/state"

case "${1:-}" in
  --status)
    if [ -f "$PID_FILE" ]; then
      PID=$(cat "$PID_FILE")
      if kill -0 "$PID" 2>/dev/null; then
        echo "RUNNING (pid $PID)"
        echo "  Endpoint: http://$HOST:$PORT/v1/chat/completions"
        echo "  Log: $LOG_FILE"
        exit 0
      else
        echo "STOPPED (stale pid file)"
        rm -f "$PID_FILE"
        exit 1
      fi
    else
      echo "STOPPED"
      exit 1
    fi
    ;;

  --stop)
    if [ -f "$PID_FILE" ]; then
      PID=$(cat "$PID_FILE")
      if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping Gemma 4 server (pid $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "Stopped."
      else
        echo "Not running (stale pid file removed)"
        rm -f "$PID_FILE"
      fi
    else
      echo "Not running."
    fi
    exit 0
    ;;

  --fg)
    echo "Starting Gemma 4 26B MoE server in foreground on http://$HOST:$PORT ..."
    exec "$VENV/bin/python" -m mlx_vlm server \
      --model "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      --kv-bits 3 \
      --kv-quant-scheme turboquant
    ;;

  *)
    # Check if already running
    if [ -f "$PID_FILE" ]; then
      PID=$(cat "$PID_FILE")
      if kill -0 "$PID" 2>/dev/null; then
        echo "Already running (pid $PID) — http://$HOST:$PORT"
        exit 0
      else
        rm -f "$PID_FILE"
      fi
    fi

    echo "Starting Gemma 4 26B MoE server in background..."
    echo "  Model: $MODEL"
    echo "  Endpoint: http://$HOST:$PORT/v1/chat/completions"
    echo "  Log: $LOG_FILE"
    echo "  TurboQuant KV-3 compression enabled"

    nohup "$VENV/bin/python" -m mlx_vlm server \
      --model "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      --kv-bits 3 \
      --kv-quant-scheme turboquant \
      >> "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" > "$PID_FILE"
    echo "Started (pid $PID). Model loading takes ~30s..."
    echo ""
    echo "Test with:"
    echo "  curl http://$HOST:$PORT/v1/chat/completions \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":100}'"
    ;;
esac

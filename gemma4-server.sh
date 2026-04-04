#!/usr/bin/env bash
# gemma4-server.sh — Start Gemma 4 26B MoE via mlx-vlm HTTP server
# OpenAI-compatible endpoint at http://127.0.0.1:8890
# TurboQuant KV-3 compression, ~15.5GB peak RAM
# Context capped at 16k tokens (--max-kv-size) to keep prefill times manageable
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

# MLX wired memory cap in MB — limits GPU wired memory so Docker and other
# services can coexist on 32GB. 16384 = 16GB for MLX, ~16GB for everything else.
# Previous 20GB cap caused OOM kills on OpenClaw gateway under concurrent load.
# Requires: sudo sysctl iogpu.wired_limit_mb=16384 (or set in /etc/sysctl.conf)
MLX_WIRED_LIMIT_MB=${MLX_WIRED_LIMIT_MB:-16384}

# Minimum free RAM (GB) required before starting
MIN_FREE_GB=${MIN_FREE_GB:-6}

# Ensure state dir exists
mkdir -p "$PROJECT_DIR/state"

# --- Preflight: check available RAM ---
check_ram() {
  AVAIL_GB=$(python3 -c "
import subprocess, re
out = subprocess.check_output(['vm_stat']).decode()
page = 16384
free = int(re.search(r'Pages free:\\s+(\\d+)', out).group(1))
inactive = int(re.search(r'Pages inactive:\\s+(\\d+)', out).group(1))
print(f'{(free+inactive)*page/1e9:.1f}')
")
  echo "Available RAM: ${AVAIL_GB} GB (minimum required: ${MIN_FREE_GB} GB)"
  python3 -c "exit(0 if float('${AVAIL_GB}') >= ${MIN_FREE_GB} else 1)" || {
    echo "ERROR: Not enough free RAM to start safely. Free up memory first."
    exit 1
  }
}

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
    check_ram
    echo "Starting Gemma 4 26B MoE server in foreground on http://$HOST:$PORT ..."
    exec "$VENV/bin/python" -m mlx_vlm server \
      --model "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      --kv-bits 3 \
      --kv-quant-scheme turboquant \
      --max-kv-size 16384
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

    check_ram
    echo "Starting Gemma 4 26B MoE server in background..."
    echo "  Model: $MODEL"
    echo "  Endpoint: http://$HOST:$PORT/v1/chat/completions"
    echo "  Log: $LOG_FILE"
    echo "  TurboQuant KV-3 compression enabled"
    echo "  MLX wired memory cap: ${MLX_WIRED_LIMIT_MB} MB"

    # Set MLX wired memory limit before starting (requires iogpu.wired_limit_mb sysctl)
    "$VENV/bin/python" -c "
import mlx.core as mx
if mx.metal.is_available():
    limit = ${MLX_WIRED_LIMIT_MB} * 1024 * 1024
    mx.set_wired_limit(limit)
    print(f'MLX wired limit set to ${MLX_WIRED_LIMIT_MB} MB')
" 2>/dev/null || true

    nohup "$VENV/bin/python" -m mlx_vlm server \
      --model "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      --kv-bits 3 \
      --kv-quant-scheme turboquant \
      --max-kv-size 16384 \
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

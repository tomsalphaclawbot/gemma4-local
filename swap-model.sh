#!/usr/bin/env bash
# swap-model.sh — Switch between Gemma 4 E4B and 26B service models
#
# Usage:
#   ./swap-model.sh e4b      # Switch to E4B (4.5B, fast, 6GB)
#   ./swap-model.sh 26b      # Switch to 26B MoE (default, quality, 18GB)
#   ./swap-model.sh status   # Show which model is currently running
#
# Why swap? Only one model at a time on a 32GB machine — running both
# would exhaust RAM and hit swap. See MODEL-SELECTION.md for guidance.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$WORKSPACE_DIR/.venvs/gemma4-mlx/bin/python3.11"

E4B_MODEL="mlx-community/gemma-4-e4b-it-4bit"
MODEL_26B="mlx-community/gemma-4-26b-a4b-it-4bit"
PORT=8890
HOST=127.0.0.1

# ── Helpers ──────────────────────────────────────────────────────────────────

current_model() {
    ps aux | grep "mlx_vlm server" | grep -v grep | grep -oE 'mlx-community/[^ ]+' | head -1
}

server_pid() {
    ps aux | grep "mlx_vlm server" | grep -v grep | awk '{print $2}' | head -1
}

wait_ready() {
    local model="$1"
    local label="$2"
    echo -n "  Waiting for server to be ready"
    local i=0
    while [ $i -lt 60 ]; do
        if curl -sf "http://$HOST:$PORT/v1/models" > /dev/null 2>&1; then
            echo " ready ✓"
            return 0
        fi
        echo -n "."
        sleep 2
        i=$((i + 1))
    done
    echo " timed out ✗"
    return 1
}

stop_server() {
    local pid
    pid=$(server_pid)
    if [ -n "$pid" ]; then
        echo "  Stopping current server (pid $pid)..."
        kill "$pid" 2>/dev/null || true
        # Wait for it to fully exit and release memory
        local i=0
        while kill -0 "$pid" 2>/dev/null && [ $i -lt 20 ]; do
            sleep 1
            i=$((i + 1))
        done
        echo "  Stopped. Waiting 3s for memory to free..."
        sleep 3
    fi
}

start_server() {
    local model="$1"
    local label="$2"
    local state_dir="$SCRIPT_DIR/state"
    mkdir -p "$state_dir"

    # Source HF token if available
    if [ -f "$SCRIPT_DIR/.env" ]; then
        set -a; source "$SCRIPT_DIR/.env"; set +a
    elif command -v rbw &>/dev/null; then
        export HF_TOKEN
        HF_TOKEN=$(rbw get huggingface.co --field token 2>/dev/null || echo "")
    fi

    echo "  Starting $label..."
    nohup "$VENV_PYTHON" -m mlx_vlm server \
        --model "$model" \
        --host "$HOST" \
        --port "$PORT" \
        --kv-bits 3 \
        --kv-quant-scheme turboquant \
        >> "$state_dir/gemma4-server.log" 2>&1 &

    echo "  Server pid: $!"
}

# ── Commands ─────────────────────────────────────────────────────────────────

cmd_status() {
    local model
    model=$(current_model)
    local pid
    pid=$(server_pid)

    if [ -z "$model" ]; then
        echo "Status: no Gemma 4 server running"
        echo "  Start with: $0 26b   (or e4b)"
        return
    fi

    echo "Status: running"
    echo "  Model: $model"
    echo "  PID:   $pid"
    echo "  Port:  $PORT"

    # Check if responding
    if curl -sf "http://$HOST:$PORT/v1/models" > /dev/null 2>&1; then
        echo "  API:   healthy ✓"
    else
        echo "  API:   loading... (may take up to 20s)"
    fi

    # Memory
    local ram_gb
    ram_gb=$(ps aux | grep "mlx_vlm server" | grep -v grep | awk '{printf "%.1f", $6/1024/1024}' | head -1)
    [ -n "$ram_gb" ] && echo "  RAM:   ~${ram_gb} GB (CPU view; GPU unified memory may differ)"
}

cmd_switch() {
    local target="$1"
    local model label

    case "$target" in
        e4b|E4B)
            model="$E4B_MODEL"
            label="Gemma 4 E4B (4.5B, 4-bit)"
            ;;
        26b|26B)
            model="$MODEL_26B"
            label="Gemma 4 26B-A4B MoE (4-bit)"
            ;;
        *)
            echo "Unknown model: $target (use 'e4b' or '26b')"
            exit 1
            ;;
    esac

    local current
    current=$(current_model)

    if [ "$current" = "$model" ]; then
        echo "Already running $label"
        cmd_status
        return
    fi

    echo "Switching to $label"

    if [ -n "$current" ]; then
        echo "  Current: $current"
        stop_server
    fi

    start_server "$model" "$label"
    wait_ready "$model" "$label"

    echo ""
    echo "Active model: $label"
    echo "Endpoint:     http://$HOST:$PORT/v1/chat/completions"
    echo "Model ID:     $model"
}

# ── Main ─────────────────────────────────────────────────────────────────────

case "${1:-status}" in
    status)     cmd_status ;;
    e4b|E4B)    cmd_switch "e4b" ;;
    26b|26B)    cmd_switch "26b" ;;
    stop)
        echo "Stopping Gemma 4 server..."
        stop_server
        echo "Done."
        ;;
    *)
        echo "Usage: $0 {e4b|26b|status|stop}"
        echo ""
        echo "  e4b    — Gemma 4 E4B (4.5B, 4-bit) — fast, 6GB RAM"
        echo "  26b    — Gemma 4 26B-A4B MoE (4-bit) — quality, 18GB RAM  [default]"
        echo "  status — show current model and health"
        echo "  stop   — stop the server"
        exit 1
        ;;
esac

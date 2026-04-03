#!/usr/bin/env bash
# infer.sh — One-shot inference via mlx-vlm CLI
#
# Usage:
#   ./infer.sh "Your prompt here"
#   ./infer.sh "Your prompt" --model 26b          # 26B MoE (default)
#   ./infer.sh "Your prompt" --model e4b          # E4B fast model
#   ./infer.sh "Your prompt" --model 31b          # 31B dense
#   ./infer.sh "Describe this" --image /path/to/image.jpg
#   ./infer.sh "Your prompt" --max-tokens 500
#
# Note: For repeated use, prefer the HTTP server (gemma4-server.sh) —
#       it keeps the model warm and runs 3x faster.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
VENV="$WORKSPACE_DIR/.venvs/gemma4-mlx"

if [ $# -eq 0 ]; then
  echo "Usage: ./infer.sh \"Your prompt\" [--model 26b|e4b|31b] [--image /path] [--max-tokens N]"
  exit 1
fi

PROMPT="$1"
shift

MODEL_KEY="26b"
IMAGE=""
MAX_TOKENS=512

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_KEY="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

case "$MODEL_KEY" in
  e4b)   MODEL="mlx-community/gemma-4-e4b-it-8bit" ;;
  26b)   MODEL="mlx-community/gemma-4-26b-a4b-it-4bit" ;;
  31b)   MODEL="mlx-community/gemma-4-31b-it-4bit" ;;
  *)     MODEL="$MODEL_KEY" ;;  # allow full model ID passthrough
esac

ARGS=(
  --model "$MODEL"
  --max-tokens "$MAX_TOKENS"
  --prompt "$PROMPT"
  --kv-bits 3
  --kv-quant-scheme turboquant
)

if [ -n "$IMAGE" ]; then
  ARGS+=(--image "$IMAGE")
fi

exec "$VENV/bin/python" -m mlx_vlm.generate "${ARGS[@]}"

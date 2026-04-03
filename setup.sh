#!/usr/bin/env bash
# setup.sh — Install mlx-vlm venv and download Gemma 4 models
#
# Usage:
#   ./setup.sh              # install venv + download default model (26B MoE 4bit)
#   ./setup.sh --all        # download all three models (~29GB)
#   ./setup.sh --e4b        # download E4B only (~9GB, fast)
#
# Requirements: Python 3.11+, ~16GB free RAM, Apple Silicon Mac

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
VENV="$WORKSPACE_DIR/.venvs/gemma4-mlx"

echo "=== Gemma 4 Local Setup ==="
echo "  Venv: $VENV"
echo ""

# Create venv if needed
if [ ! -d "$VENV" ]; then
  echo "Creating venv..."
  python3 -m venv "$VENV"
fi

echo "Installing mlx-vlm (includes TurboQuant)..."
"$VENV/bin/pip" install -q -U mlx-vlm
echo "  ✅ mlx-vlm $(\"$VENV/bin/python\" -c 'import mlx_vlm; print(mlx_vlm.__version__)' 2>/dev/null || echo 'installed')"

echo ""
echo "Downloading models (this will take a while)..."

download_model() {
  local model="$1"
  local label="$2"
  echo "  Fetching $label ($model)..."
  "$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$model', repo_type='model')
print('  ✅ $label downloaded')
"
}

case "${1:-}" in
  --all)
    download_model "mlx-community/gemma-4-e4b-it-8bit"           "E4B 8bit (~9GB)"
    download_model "mlx-community/gemma-4-26b-a4b-it-4bit"       "26B MoE 4bit (~16GB)"
    download_model "mlx-community/gemma-4-31b-it-4bit"           "31B Dense 4bit (~19GB)"
    ;;
  --e4b)
    download_model "mlx-community/gemma-4-e4b-it-8bit"           "E4B 8bit (~9GB)"
    ;;
  *)
    download_model "mlx-community/gemma-4-26b-a4b-it-4bit"       "26B MoE 4bit (~16GB)"
    ;;
esac

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start the HTTP server:"
echo "  ./gemma4-server.sh"
echo ""
echo "Or run one-off inference:"
echo "  ./infer.sh 'Your prompt here'"

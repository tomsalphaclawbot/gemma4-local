# gemma4-local

Local Gemma 4 inference on Apple Silicon via MLX + TurboQuant. Runs as a persistent OpenAI-compatible HTTP service wired into OpenClaw as a model fallback.

## Hardware

- MacBook Air, 32GB unified RAM, Apple Silicon (arm64)

## Architecture

```
OpenClaw gateway
  └── fallback: gemma4-mlx/mlx-community/gemma-4-26b-a4b-it-4bit
        └── http://127.0.0.1:8890/v1/chat/completions
              └── mlx-vlm server (launchd: work.tomsalphaclawbot.gemma4-mlx)
                    └── mlx-community/gemma-4-26b-a4b-it-4bit
                          └── TurboQuant KV-3 compression
```

## Models Downloaded

All cached in `~/.cache/huggingface/hub/` (~29GB total).

| Model | Quant | Peak RAM | CLI speed | Server speed |
|---|---|---|---|---|
| gemma-4-e4b-it-8bit | 8bit | ~9 GB | ~7.6 tok/s | — |
| gemma-4-26b-a4b-it-4bit ⭐ | 4bit | ~15.5 GB | ~13.7 tok/s | **~42 tok/s** |
| gemma-4-31b-it-4bit | 4bit | ~19 GB | ~1.6 tok/s | — |

⭐ = default / active service model

## Venv

```
.venvs/gemma4-mlx/     # Python 3.11, mlx-vlm v0.4.3 + TurboQuant built-in
```

## HTTP Service (launchd)

The 26B MoE model runs as a persistent background service via launchd.

**Service label:** `work.tomsalphaclawbot.gemma4-mlx`  
**Plist:** `~/Library/LaunchAgents/work.tomsalphaclawbot.gemma4-mlx.plist`  
**Endpoint:** `http://127.0.0.1:8890/v1/chat/completions`  
**Log:** `state/gemma4-server.log`

Behavior:
- Starts automatically on login (RunAtLoad)
- KeepAlive — launchd restarts on crash (30s throttle)
- Model takes ~30s to load after start before it accepts requests

### launchd commands

```bash
# Status
launchctl list work.tomsalphaclawbot.gemma4-mlx

# Restart
launchctl kickstart -k gui/$UID/work.tomsalphaclawbot.gemma4-mlx

# Stop/unload (won't auto-restart until next login)
launchctl bootout gui/$UID/work.tomsalphaclawbot.gemma4-mlx

# Re-load after bootout
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/work.tomsalphaclawbot.gemma4-mlx.plist
```

### Helper script

`scripts/gemma4-server.sh` also works for start/stop/status (manages the launchd-owned process or runs standalone):

```bash
bash scripts/gemma4-server.sh            # start (or check if already running)
bash scripts/gemma4-server.sh --status   # status
bash scripts/gemma4-server.sh --stop     # stop
bash scripts/gemma4-server.sh --fg       # run in foreground (for debugging)
```

## OpenClaw Integration

**Provider entry** in `~/.openclaw/openclaw.json`:
```json
"gemma4-mlx": {
  "baseUrl": "http://127.0.0.1:8890/v1",
  "apiKey": "none",
  "api": "openai-responses",
  "models": [...]
}
```

**Fallback chain** (agents.defaults.model.fallbacks):
```
anthropic/claude-opus-4-6
  → openai-codex/gpt-5.3-codex
  → anthropic/claude-sonnet-4-6
  → gemma4-mlx/mlx-community/gemma-4-26b-a4b-it-4bit  ← last resort local
```

**Alias:** `gemma4` — switch to it directly with `/model gemma4`

**Important:** Always use the full model ID in API requests:
```
"model": "mlx-community/gemma-4-26b-a4b-it-4bit"
```

## Testing

```bash
curl http://127.0.0.1:8890/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 50
  }'
```

## TurboQuant KV Compression

mlx-vlm v0.4.3 includes TurboQuant natively (PR #858, #894). The server runs with `--kv-bits 3 --kv-quant-scheme turboquant`.

- KV-3: 4.6x compression, +1.06% PPL degradation
- At 128K context on 31B: KV memory 13.3 GB → 4.9 GB (-63%)
- Quality preserved at typical conversation lengths

## Gemma 4 Architecture Notes

- Released: April 2, 2026 by Google DeepMind (Apache 2.0)
- 26B-A4B: Mixture of Experts — only 4B params active per token
- 256K context window, multimodal (text + image)
- Per-Layer Embeddings (PLE), Shared KV Cache, alternating attention
- Thinks via `<|channel>thought</channel>` (activated automatically)
- LMArena: 26B MoE = 1441, 31B dense = 1452

## Running Other Models (CLI)

For one-off inference with any downloaded model:

```bash
cd /Users/openclaw/.openclaw/workspace
source .venvs/gemma4-mlx/bin/activate

# 26B MoE
python -m mlx_vlm.generate \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --max-tokens 200 --prompt "Your prompt here"

# E4B (fast, 8GB)
python -m mlx_vlm.generate \
  --model mlx-community/gemma-4-e4b-it-8bit \
  --max-tokens 200 --prompt "Your prompt here"

# With image
python -m mlx_vlm.generate \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --prompt "Describe this image" --image /path/to/image.jpg
```

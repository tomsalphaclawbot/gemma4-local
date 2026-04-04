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

## Models

All cached in `~/.cache/huggingface/hub/` (~34GB total).

| Model | Quant | Peak RAM | Load time | Notes |
|---|---|---|---|---|
| gemma-4-e4b-it-4bit ⚡ | 4bit | ~5–7 GB | **1.75s** | Best for low-latency tasks |
| gemma-4-26b-a4b-it-4bit ⭐ | 4bit | ~16–18 GB | 10.8s | Default service model, best quality |
| gemma-4-31b-it-4bit | 4bit | ~19 GB | ~15s | Dense; slower, highest quality |

⭐ = default service model &nbsp; ⚡ = recommended for fast/cheap tasks

## Benchmark Results (mlx-vlm 0.4.4, 2026-04-04)

> **Note:** mlx-vlm 0.4.4 fixed chunked prefill for Gemma 4's shared KV cache architecture,
> delivering ~2x faster prompt processing at long context vs 0.4.3.

| Task | E4B prompt | E4B gen | E4B time | 26B prompt | 26B gen | 26B time | E4B RAM | 26B RAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Short Q&A | 137/s | 67/s | ~0.4s | 80/s | 23/s | ~0.6s | 5.3 GB | 15.7 GB |
| Reasoning | 172/s | 34/s | ~1.5s | 112/s | 35/s | ~1.0s | 5.3 GB | 15.7 GB |
| Code Generation | 161/s | 34/s | ~3.0s | 102/s | 35/s | ~2.8s | 5.3 GB | 15.7 GB |
| Medium Context (3K tok) | 544/s | 34/s | ~5s | 337/s | 33/s | ~10s | 6.1 GB | 16.9 GB |
| Long Context (12K tok) | 516/s | 31/s | ~25s | 300/s | 29/s | ~43s | 6.8 GB | 18.0 GB |
| Summarization | 259/s | 34/s | ~4s | 227/s | 35/s | ~4.3s | 6.8 GB | 18.0 GB |
| Instruction Following | 202/s | 36/s | ~0.6s | 99/s | 39/s | ~0.6s | 6.8 GB | 18.0 GB |

**Load times:** E4B = 1.75s &nbsp;|&nbsp; 26B = 10.8s

**Key takeaways:**
- E4B is 1.5–2× faster on prompt throughput and loads in under 2s — great for high-frequency calls
- 26B has better reasoning depth and consistent generation quality
- Long context (12K tokens): E4B prompts at 516 tok/s, 26B at 300 tok/s — both usable
- Generation speed is nearly identical (~30–40 tok/s) on both — bottleneck is model size, not tokenization
- Both stay under 18GB RAM, leaving headroom on a 32GB machine

### Sample outputs (26B)

**Reasoning** (bat-and-ball): _"The ball costs $0.05. If the ball costs x, then the bat costs x + $1, and x + (x + $1) = $1.10, so 2x = $0.10, x = $0.05."_

**Instruction following**: `C++ / Java / Python / Rust / Swift` — alphabetical, no numbering, perfect.

**Summarization** (meeting notes):
- Performance Review: Q4 saw positive growth with DAU up 23%, revenue up 18%, churn down 4 points
- Technical Debt: Auth system debt to be addressed over 3 weeks, delaying feature X by 2 weeks
- Design Decision: Onboarding design B approved (34% better completion), assets due Wednesday

## Switching Models

Only one model runs at a time — 32GB RAM is enough for one, not both. See **[MODEL-SELECTION.md](MODEL-SELECTION.md)** for the full decision guide.

```bash
./swap-model.sh status   # what's running
./swap-model.sh 26b      # switch to 26B MoE (default — quality)
./swap-model.sh e4b      # switch to E4B (fast — 6GB RAM, 1.75s load)
./swap-model.sh stop     # stop server, free memory
```

**TL;DR:** Run 26B for quality work. Swap to E4B for batch/high-volume jobs or when you need the RAM for something else.

## Venv

```
.venvs/gemma4-mlx/     # Python 3.11, mlx-vlm v0.4.4 + TurboQuant built-in
```

## Chat Template (required for instruction models)

Always use `apply_chat_template` — raw prompts produce garbled output:

```python
from mlx_vlm import load, generate, apply_chat_template
from mlx_vlm.utils import load_config

model, processor = load("mlx-community/gemma-4-26b-a4b-it-4bit")
config = load_config("mlx-community/gemma-4-26b-a4b-it-4bit")

prompt = apply_chat_template(
    processor, config,
    prompt="What color is the sky?",
    add_generation_prompt=True
)
result = generate(model, processor, prompt=prompt, max_tokens=50, verbose=True)
print(result.text)
```

Gemma 4 turn format: `<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n`

## HTTP Service (launchd)

The 26B MoE model runs as a persistent background service.

**Service:** `work.tomsalphaclawbot.gemma4-mlx`  
**Endpoint:** `http://127.0.0.1:8890/v1/chat/completions`  
**Log:** `state/gemma4-server.log`

```bash
# Status
launchctl list work.tomsalphaclawbot.gemma4-mlx

# Restart
launchctl kickstart -k gui/$UID/work.tomsalphaclawbot.gemma4-mlx

# Stop
launchctl bootout gui/$UID/work.tomsalphaclawbot.gemma4-mlx
```

Helper script:
```bash
bash projects/gemma4-local/gemma4-server.sh            # start
bash projects/gemma4-local/gemma4-server.sh --status   # status
bash projects/gemma4-local/gemma4-server.sh --stop     # stop
```

## OpenClaw Integration

**Provider** in `~/.openclaw/openclaw.json`: `gemma4-mlx`  
**Alias:** `gemma4` — use `/model gemma4` to switch  
**Fallback chain:** `claude-opus-4-6 → gpt-5.3-codex → claude-sonnet-4-6 → gemma4`

Always use the full model ID in API requests:
```
"model": "mlx-community/gemma-4-26b-a4b-it-4bit"
```

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

mlx-vlm 0.4.4 includes TurboQuant natively. Server runs with `--kv-bits 3 --kv-quant-scheme turboquant`.

| Mode | Compression | PPL impact | Notes |
|---|---|---|---|
| turbo2 | 6.4× | +6.48% PPL | Aggressive |
| turbo3 | 4.6× | +1.06% PPL | **Default — sweet spot** |
| turbo4 | 3.8× | +0.23% PPL | Conservative |

At 128K context on 31B: KV memory 13.3 GB → 4.9 GB (-63%), quality preserved.

## HuggingFace Auth

Required for model downloads. Token in Bitwarden under `huggingface.co` → field `token`.

```bash
# Set for session
export HF_TOKEN=$(rbw get huggingface.co --field token)
```

Saved persistently to `~/.cache/huggingface/token` and `projects/gemma4-local/.env`.

## Benchmarking

```bash
HF_TOKEN=$(rbw get huggingface.co --field token) \
  .venvs/gemma4-mlx/bin/python3.11 projects/gemma4-local/benchmark.py
```

Results saved to `projects/gemma4-local/benchmark_results.json`.

## Gemma 4 Architecture Notes

- Released: April 2, 2026 by Google DeepMind (Apache 2.0)
- 26B-A4B: Mixture of Experts — only 4B params active per token
- 256K context window, multimodal (text + image + audio)
- Per-Layer Embeddings (PLE), Shared KV Cache, alternating attention
- Thinking mode via `<|channel>thought</channel>` (automatic)
- LMArena: 26B MoE = 1441 (#6 open), 31B dense = 1452 (#3 open)

## Running Alongside Docker (32GB)

MLX and Docker Desktop can coexist on 32GB, but you **must** set a system-level GPU memory cap first — otherwise MLX will claim the full recommended 22.9 GB and Docker's VM will destabilize under memory pressure.

**One-time setup (requires sudo):**
```bash
# Cap MLX GPU wired memory at 20GB, leaving ~12GB for Docker + OS
sudo sysctl iogpu.wired_limit_mb=20480

# Persist across reboots
echo 'iogpu.wired_limit_mb=20480' | sudo tee -a /etc/sysctl.conf
```

The `gemma4-server.sh` script also calls `mx.set_wired_limit(20480 MB)` at startup as an application-level guard. Both layers together prevent the crash.

**What happens without the cap:** MLX wires up to 22.9 GB (Apple's `max_recommended_working_set_size`). macOS swaps aggressively. Docker's VM triggers the disk write watchdog (~2.1 GB dirtied in 40-60s). Hard reset. This was confirmed empirically — diagnostic logs: `ResetCounter-*.diag` in `/Library/Logs/DiagnosticReports/`.

**Tuning:** default cap is 20480 MB. To adjust: `MLX_WIRED_LIMIT_MB=18432 bash gemma4-server.sh`

---

## Credits

| Project | Link |
|---|---|
| **Gemma 4** (Google DeepMind) | [deepmind.google/models/gemma](https://deepmind.google/models/gemma/gemma-4/) |
| **MLX** (Apple) | [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx) |
| **mlx-vlm** (Blaizzy / [@prince_canuma](https://x.com/prince_canuma)) | [github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) |
| **TurboQuant+** ([@no_stp_on_snek](https://x.com/no_stp_on_snek)) | [github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) |
| **mlx-community** (HuggingFace) | [huggingface.co/mlx-community](https://huggingface.co/mlx-community) |
| **OpenClaw** | [github.com/openclaw/openclaw](https://github.com/openclaw/openclaw) |

# gemma4-local

Local Gemma 4 inference on Apple Silicon via MLX + TurboQuant. Runs as a persistent OpenAI-compatible HTTP service wired into OpenClaw as a model fallback.

## Hardware

- MacBook Air Mac16,13, Apple M4 (10 cores: 4P+6E), 10-core GPU (Metal 3), 32GB RAM, 1TB SSD, macOS 15.7.4
- mlx-lm 0.31.1, mlx-vlm 0.4.3

## Architecture

```
Clients (OpenClaw, Hermes, scripts, curl)
  └── http://127.0.0.1:8891/v1  ← gemma4-proxy.py (serializing queue)
        │
        │  • Only 1 inference request forwarded at a time
        │  • Concurrent requests queued (FIFO)
        │  • Memory pressure check before forwarding
        │  • 503 with clean error instead of OOM
        │  • Stats: http://127.0.0.1:8891/proxy/stats
        │  • Health: http://127.0.0.1:8891/proxy/health
        │
        └── http://127.0.0.1:8890/v1  ← mlx-vlm server (raw, never hit directly)
              └── mlx-community/gemma-4-26b-a4b-it-4bit
                    └── TurboQuant KV-3 compression
                    └── --max-kv-size 16384 (16K context cap)
```

**Why the proxy exists:** MLX server is single-threaded for inference. Concurrent requests cause parallel KV cache allocations that spike memory past 32GB, triggering macOS Jetsam OOM kills on _other_ processes (including the OpenClaw gateway). The proxy serializes requests so this can never happen.

### Services (launchd)

| Service | Port | LaunchAgent | Role |
|---|---|---|---|
| MLX server | 8890 | `work.tomsalphaclawbot.gemma4-mlx` | Raw inference engine |
| Serializing proxy | 8891 | `work.tomsalphaclawbot.gemma4-proxy` | Queue + memory gate |

Both auto-start on boot with KeepAlive.

## Context Cap: 16K tokens

Set via `--max-kv-size 16384`. Gemma 4 supports 256K natively, but prefix cache is broken on hybrid sliding-window attention models (mlx-lm [issue #980](https://github.com/ml-explore/mlx-lm/issues/980)). Without prefix cache reuse, every turn does full recomputation — TTFT scales linearly:

| Context | TTFT | Prefill rate |
|---|---|---|
| 1K tokens | ~9s | 120-140 tok/s |
| 4K tokens | ~29s | 120-140 tok/s |
| 8K tokens | ~58s | 120-140 tok/s |
| 16K tokens | ~131s | 120-140 tok/s |
| 24K tokens | ~198s | 120-140 tok/s |

16K is the practical ceiling where TTFT stays under ~2 minutes. This cap will be raised when mlx-lm fixes prefix cache for hybrid attention models.

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

### Server mode performance (26B, via proxy)

Sequential requests through the serializing proxy:
- Prompt: 60–123 tok/s
- Generation: 33–35 tok/s steady, 60 tok/s peak
- Peak RAM: 15.7 GB

### Sample outputs (26B)

**Reasoning** (bat-and-ball): _"The ball costs $0.05. If the ball costs x, then the bat costs x + $1, and x + (x + $1) = $1.10, so 2x = $0.10, x = $0.05."_

**Instruction following**: `C++ / Java / Python / Rust / Swift` — alphabetical, no numbering, perfect.

**Summarization** (meeting notes):
- Performance Review: Q4 saw positive growth with DAU up 23%, revenue up 18%, churn down 4 points
- Technical Debt: Auth system debt to be addressed over 3 weeks, delaying feature X by 2 weeks
- Design Decision: Onboarding design B approved (34% better completion), assets due Wednesday

## Quick Start

```bash
# Start server + proxy (both managed by launchd)
bash gemma4-server.sh              # start MLX server
# Proxy auto-starts via LaunchAgent

# Check status
bash gemma4-server.sh --status     # MLX server
curl -s http://127.0.0.1:8891/proxy/health | python3 -m json.tool  # proxy

# Stop
bash gemma4-server.sh --stop       # MLX server
launchctl bootout gui/$UID/work.tomsalphaclawbot.gemma4-proxy  # proxy

# Test inference (always through proxy)
curl http://127.0.0.1:8891/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 50
  }'

# Check proxy stats
curl -s http://127.0.0.1:8891/proxy/stats | python3 -m json.tool
```

## Serializing Proxy

`gemma4-proxy.py` — a lightweight HTTP proxy that prevents the OOM footgun.

### Features

- **Request serialization:** Only 1 inference request hits MLX at a time. Concurrent requests queue with FIFO ordering.
- **Memory pressure gate:** Checks `vm_stat` before forwarding. Rejects with 503 if free RAM < 4 GB (configurable).
- **Queue timeout:** Requests waiting longer than 120s get a clean 503 instead of hanging forever.
- **Passthrough for non-inference:** `/v1/models`, health checks, etc. go straight through (no lock).
- **Stats & health endpoints:** `/proxy/stats` (counters, queue depth, free RAM), `/proxy/health` (quick MLX liveness check).
- **Zero dependencies:** stdlib-only Python, runs anywhere.

### Configuration (env vars)

| Var | Default | Description |
|---|---|---|
| `GEMMA_PROXY_PORT` | 8891 | Proxy listen port |
| `GEMMA_MLX_PORT` | 8890 | Backend MLX server port |
| `GEMMA_QUEUE_TIMEOUT` | 120 | Max seconds a request will wait in queue |
| `GEMMA_MIN_FREE_RAM_GB` | 4.0 | Reject if free RAM drops below this |

### Stats response example

```json
{
  "requests_total": 42,
  "requests_completed": 40,
  "requests_rejected_memory": 1,
  "requests_rejected_timeout": 1,
  "requests_failed": 0,
  "current_queue_depth": 0,
  "free_ram_gb": 15.8,
  "mlx_healthy": true,
  "uptime_s": 3600.0
}
```

## Switching Models

Only one model runs at a time — 32GB RAM is enough for one, not both. See **[MODEL-SELECTION.md](MODEL-SELECTION.md)** for the full decision guide.

```bash
./swap-model.sh status   # what's running
./swap-model.sh 26b      # switch to 26B MoE (default — quality)
./swap-model.sh e4b      # switch to E4B (fast — 6GB RAM, 1.75s load)
./swap-model.sh stop     # stop server, free memory
```

**TL;DR:** Run 26B for quality work. Swap to E4B for batch/high-volume jobs or when you need the RAM for something else.

## OpenClaw Integration

**Provider** in `~/.openclaw/openclaw.json`: `gemma4-mlx`
**Endpoint:** `http://127.0.0.1:8891/v1` (proxy, not raw MLX)
**Alias:** `gemma4` — use `/model gemma4` to switch
**Fallback chain:** `claude-opus-4-6 → gpt-5.3-codex → claude-sonnet-4-6 → gemma4`

Always use the full model ID in API requests:
```
"model": "mlx-community/gemma-4-26b-a4b-it-4bit"
```

## Memory Management

### Wired memory cap

MLX will claim all available GPU memory by default. The server script sets a system-level cap:

```bash
# In gemma4-server.sh:
sudo sysctl iogpu.wired_limit_mb=16384  # 16 GB for MLX, 16 GB for everything else
```

Previous 20 GB cap caused OOM kills under concurrent load. 16 GB is the safe operating point.

### What happens without safeguards

1. Multiple concurrent requests → parallel KV cache allocations → memory spike past 32 GB
2. macOS Jetsam OOM killer activates → kills largest non-kernel process (usually OpenClaw gateway)
3. Gateway dies, all channels (Telegram, Discord) go dark

The proxy prevents step 1. The wired cap prevents runaway allocation even if the proxy is bypassed.

## Venv

```
.venvs/gemma4-mlx/     # Python 3.11, mlx-vlm v0.4.3 + TurboQuant built-in
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

## TurboQuant KV Compression

mlx-vlm includes TurboQuant natively. Server runs with `--kv-bits 3 --kv-quant-scheme turboquant`.

| Mode | Compression | PPL impact | Notes |
|---|---|---|---|
| turbo2 | 6.4× | +6.48% PPL | Aggressive |
| turbo3 | 4.6× | +1.06% PPL | **Default — sweet spot** |
| turbo4 | 3.8× | +0.23% PPL | Conservative |

At 128K context on 31B: KV memory 13.3 GB → 4.9 GB (-63%), quality preserved.

## HuggingFace Auth

Required for model downloads. Token in Bitwarden under `huggingface.co` → field `token`.

```bash
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
- Per-Layer Embeddings (PLE), Shared KV Cache, alternating attention (5:1 sliding+global, sliding_window=1024)
- Thinking mode via `<|channel>thought</channel>` (automatic)
- LMArena: 26B MoE = 1441 (#6 open), 31B dense = 1452 (#3 open)
- **Prefix cache broken** on this architecture (hybrid sliding window) — see [mlx-lm #980](https://github.com/ml-explore/mlx-lm/issues/980)

## File Map

```
gemma4-server.sh        # Server start/stop/status script
gemma4-proxy.py         # Serializing proxy (the safety layer)
swap-model.sh           # Switch between E4B/26B/31B models
infer.sh                # Quick CLI inference test
setup.sh                # Initial environment setup
MODEL-SELECTION.md      # Decision guide for model selection
benchmark.py            # Full benchmark suite
benchmark_context.py    # Context length scaling benchmarks
benchmark_kv_cache.py   # KV cache behavior tests
benchmark_ttft.py       # Time-to-first-token measurements
benchmark_results.json  # Saved benchmark data
openclaw-config.json    # Reference OpenClaw provider config
launchd/                # LaunchAgent plists + install script
state/                  # Runtime state (logs, pid files)
```

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

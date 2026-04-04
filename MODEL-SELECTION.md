# Model Selection Guide

Two models are available locally. Only run one at a time — the machine has 32GB unified RAM and both loaded simultaneously would exhaust it and hit swap.

```bash
./swap-model.sh status   # what's running now
./swap-model.sh 26b      # switch to 26B (default)
./swap-model.sh e4b      # switch to E4B (fast)
./swap-model.sh stop     # stop server entirely
```

---

## The Two Models

### Gemma 4 26B-A4B MoE — Default
`mlx-community/gemma-4-26b-a4b-it-4bit`

| Spec | Value |
|---|---|
| Parameters | 26B total, **4B active per token** (MoE) |
| Quantization | 4-bit |
| Peak RAM | ~16–18 GB |
| Load time | ~11s |
| Prompt speed | 80–337 tok/s (scales with context) |
| Generation speed | ~33 tok/s |
| Context window | 256K |

**Use when:**
- Quality matters (reasoning, code, analysis, summarization)
- You're doing interactive back-and-forth conversation
- Tasks need careful instruction following or nuanced judgment
- You have a few seconds to spare on the first response

---

### Gemma 4 E4B — Fast
`mlx-community/gemma-4-e4b-it-4bit`

| Spec | Value |
|---|---|
| Parameters | 4.5B effective |
| Quantization | 4-bit |
| Peak RAM | ~5–7 GB |
| Load time | **1.75s** |
| Prompt speed | 137–544 tok/s |
| Generation speed | ~35 tok/s |
| Context window | 128K |

**Use when:**
- High-volume batch processing (100s of short calls)
- Fast classification, tagging, extraction
- Prototyping — iteration speed matters more than output quality
- RAM is needed for something else simultaneously
- You need the server up in under 2 seconds

---

## Why Not Both?

32GB sounds like enough. It isn't, once you account for:

| What | RAM |
|---|---|
| macOS + system processes | ~4–5 GB |
| OpenClaw gateway + services | ~2–3 GB |
| Active browser / apps | ~3–5 GB |
| Gemma 4 E4B | ~6–7 GB |
| Gemma 4 26B | ~16–18 GB |
| **Total (both models)** | **~33–38 GB** ← exceeds physical RAM |

Running both would spill into swap, which destroys inference performance (swap is orders of magnitude slower than unified RAM for this workload).

**The right pattern:** run 26B by default, swap to E4B when you need speed or RAM headroom, swap back when done.

---

## Benchmark Summary (mlx-vlm 0.4.4, 2026-04-04)

| Task | E4B | 26B | Winner |
|---|---|---|---|
| Short Q&A | 0.4s | 0.6s | E4B (barely) |
| Reasoning | ~1.5s | ~1.0s | 26B + better answer |
| Code generation | ~3s | ~2.8s | Tie on speed, 26B on quality |
| 3K context | 5s | 10s | E4B on speed |
| 12K context | 25s | 43s | E4B on speed |
| Summarization | 4s | 4.3s | Tie on speed |
| Instruction following | 0.6s | 0.6s | Tie — both perfect |
| **Load time** | **1.75s** | **10.8s** | E4B |
| **Peak RAM** | **6.8 GB** | **18.0 GB** | E4B |

**Generation speed is nearly identical** on both (~33–35 tok/s). The E4B advantage is on prompt processing (prefill) and load time — not on how fast you see tokens stream out.

---

## OpenClaw Integration

The server always runs on `http://127.0.0.1:8890/v1`. The model ID in requests must match what's actually running:

```bash
# Check what's running before making API calls
./swap-model.sh status
```

OpenClaw config (`~/.openclaw/openclaw.json`) references the 26B by default. If you switch to E4B, pass the E4B model ID explicitly in your API call — OpenClaw doesn't auto-detect the swap.

---

## Quick Reference

```bash
# Default: start 26B (quality)
./swap-model.sh 26b

# Switch to E4B (speed/batch)
./swap-model.sh e4b

# Check what's running
./swap-model.sh status

# Stop everything (free ~18GB RAM)
./swap-model.sh stop

# Test the active model
curl http://127.0.0.1:8890/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10
  }'
```

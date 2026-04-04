#!/usr/bin/env python3
"""
KV cache / conversation resumption benchmark for Gemma 4 on Apple Silicon.

Tests the speedup from reusing a prefilled KV cache across conversation turns.

Methodology:
  - "cold" run: fresh prefill of full prompt (base + delta)
  - "warm" run: prefill base once, cache KV state, then only prefill delta

The warm TTFT should be ~(delta_tokens / prefill_tps) — nearly instant for small deltas.

Usage:
    python benchmark_kv_cache.py --model 26b
    python benchmark_kv_cache.py --model e4b
"""

import time, os, sys, json, argparse, gc
from datetime import datetime

os.environ.setdefault("HF_HUB_OFFLINE", "1")
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    for line in open(_env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from mlx_vlm import load, stream_generate, apply_chat_template
from mlx_vlm.generate import PromptCacheState
from mlx_vlm.utils import load_config

MODELS = {
    "e4b": ("mlx-community/gemma-4-e4b-it-4bit",      "Gemma 4 E4B"),
    "26b": ("mlx-community/gemma-4-26b-a4b-it-4bit",  "Gemma 4 26B MoE"),
}

# Base context sizes (tokens, approximate) and delta sizes to append
BASE_SIZES  = [2_000, 4_000, 8_000, 12_000]
DELTA_SIZES = [200, 500, 1_000]

FILLER = (
    "The study of memory in artificial intelligence systems reveals fundamental "
    "architectural constraints that differ from biological neural networks. "
    "Unlike the hippocampal-neocortical system in mammals, current transformer models "
    "process context as a flat sequence without episodic tagging or temporal indexing. "
    "This limits their ability to retrieve specific past events and reason causally "
    "across long time horizons. Retrieval-augmented generation attempts to address "
    "this gap but introduces its own failure modes around relevance scoring. "
)  # ~80 tokens

DELTA_FILLER = (
    "Additional context appended after the initial prefill: "
    "Recent experiments show that KV cache reuse dramatically reduces latency "
    "for multi-turn conversations where a long shared prefix exists. "
)  # ~40 tokens


def build_prompt(base_tokens: int, delta_tokens: int = 0) -> str:
    base_reps = max(1, base_tokens // 80)
    base_ctx = FILLER * base_reps
    if delta_tokens > 0:
        delta_reps = max(1, delta_tokens // 40)
        delta_ctx = DELTA_FILLER * delta_reps
        full_ctx = base_ctx + "\n\n" + delta_ctx
    else:
        full_ctx = base_ctx
    return (
        f"Background context:\n\n{full_ctx}\n\n"
        "Based on the above context, what is the single most important architectural "
        "limitation of current transformer memory systems? One sentence only."
    )


def collect_stream(gen) -> tuple[str, int, int, float, float]:
    """Drain a stream_generate generator, return (text, prompt_tok, gen_tok, prompt_tps, gen_tps)."""
    text = ""
    prompt_tokens = generation_tokens = 0
    prompt_tps = gen_tps = 0.0
    for chunk in gen:
        if hasattr(chunk, "text"):
            text += chunk.text
        if hasattr(chunk, "prompt_tokens") and chunk.prompt_tokens:
            prompt_tokens = chunk.prompt_tokens
        if hasattr(chunk, "generation_tokens") and chunk.generation_tokens:
            generation_tokens = chunk.generation_tokens
        if hasattr(chunk, "prompt_tps") and chunk.prompt_tps:
            prompt_tps = chunk.prompt_tps
        if hasattr(chunk, "generation_tps") and chunk.generation_tps:
            gen_tps = chunk.generation_tps
    return text.strip(), prompt_tokens, generation_tokens, prompt_tps, gen_tps


def run_benchmark(model_key: str):
    model_id, label = MODELS[model_key]

    print(f"\n{'='*70}", flush=True)
    print(f"  {label} — KV Cache Resumption Benchmark", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    print("Loading model...", flush=True)
    model, processor = load(model_id)
    config = load_config(model_id)
    print(f"Loaded in {time.time()-t0:.1f}s\n", flush=True)

    results = {
        "model_id": model_id,
        "label": label,
        "timestamp": datetime.utcnow().isoformat(),
        "runs": [],
    }

    print(
        f"  {'Base':>6}  {'Delta':>6}  {'Mode':>6}  "
        f"{'Prompt tok':>11}  {'Prefill':>10}  {'TTFT':>7}  {'Gen':>8}  {'Speedup':>8}",
        flush=True
    )
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*11}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}", flush=True)

    for base_tok in BASE_SIZES:
        for delta_tok in DELTA_SIZES:
            # --- Reset state between iterations to avoid KV cache accumulation ---
            gc.collect()
            try:
                import mlx.core as mx
                mx.metal.reset_peak_memory()
            except Exception:
                pass

            # --- COLD RUN: full prefill of base+delta ---
            full_prompt_raw = build_prompt(base_tok, delta_tok)
            full_prompt = apply_chat_template(processor, config,
                                              prompt=full_prompt_raw,
                                              add_generation_prompt=True)

            t_cold = time.time()
            try:
                gen_cold = stream_generate(model, processor, full_prompt, max_tokens=40, verbose=False)
                _, p_tok_cold, _, p_tps_cold, g_tps_cold = collect_stream(gen_cold)
                cold_elapsed = time.time() - t_cold
                cold_ttft = p_tok_cold / p_tps_cold if p_tps_cold > 0 else cold_elapsed
            except Exception as e:
                print(f"  {base_tok:>5,}  {delta_tok:>5,}  cold  OOM/ERR: {e!s:.60}", flush=True)
                continue

            print(
                f"  {base_tok:>5,}  {delta_tok:>5,}  {'cold':>6}  "
                f"{p_tok_cold:>10,}  {p_tps_cold:>8.0f}/s  {cold_ttft:>6.1f}s  "
                f"{g_tps_cold:>6.0f}/s  {'baseline':>8}",
                flush=True
            )

            # --- WARM RUN: prefill base, cache, then only prefill delta ---
            # Fresh cache state — never reuse across iterations
            cache_state = PromptCacheState()
            base_prompt_raw = build_prompt(base_tok, 0)
            base_prompt = apply_chat_template(processor, config,
                                              prompt=base_prompt_raw,
                                              add_generation_prompt=True)
            try:
                gen_prime = stream_generate(
                    model, processor, base_prompt,
                    max_tokens=5,          # minimal generation to fill cache
                    verbose=False,
                    prompt_cache_state=cache_state,
                )
                collect_stream(gen_prime)  # drain to populate cache_state
            except Exception as e:
                print(f"  {base_tok:>5,}  {delta_tok:>5,}  prime  ERR: {e!s:.60}", flush=True)
                continue

            # Step 2: time turn 2 with delta appended — should skip cached prefix
            t_warm = time.time()
            try:
                gen_warm = stream_generate(
                    model, processor, full_prompt,  # same full prompt as cold
                    max_tokens=40,
                    verbose=False,
                    prompt_cache_state=cache_state,
                )
                _, p_tok_warm, _, p_tps_warm, g_tps_warm = collect_stream(gen_warm)
                warm_elapsed = time.time() - t_warm
                warm_ttft = p_tok_warm / p_tps_warm if p_tps_warm > 0 else warm_elapsed
            except Exception as e:
                print(f"  {base_tok:>5,}  {delta_tok:>5,}  warm   OOM/ERR: {e!s:.60}", flush=True)
                continue

            speedup = cold_ttft / warm_ttft if warm_ttft > 0 else float("inf")

            print(
                f"  {base_tok:>5,}  {delta_tok:>5,}  {'warm':>6}  "
                f"{p_tok_warm:>10,}  {p_tps_warm:>8.0f}/s  {warm_ttft:>6.1f}s  "
                f"{g_tps_warm:>6.0f}/s  {speedup:>7.1f}x",
                flush=True
            )

            results["runs"].append({
                "base_tokens": base_tok,
                "delta_tokens": delta_tok,
                "cold": {
                    "prompt_tokens": p_tok_cold,
                    "prompt_tps": round(p_tps_cold, 1),
                    "ttft_s": round(cold_ttft, 2),
                    "gen_tps": round(g_tps_cold, 1),
                },
                "warm": {
                    "prompt_tokens": p_tok_warm,
                    "prompt_tps": round(p_tps_warm, 1),
                    "ttft_s": round(warm_ttft, 2),
                    "gen_tps": round(g_tps_warm, 1),
                },
                "speedup_x": round(speedup, 2),
            })

        print(flush=True)  # blank line between base sizes

    del model, processor
    gc.collect()

    out = os.path.join(os.path.dirname(__file__), "benchmark_kv_cache_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["e4b", "26b"], default="26b")
    args = parser.parse_args()
    run_benchmark(args.model)

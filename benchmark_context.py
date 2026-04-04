#!/usr/bin/env python3
"""
Context-length scaling benchmark for Gemma 4 on Apple Silicon.
Tests prompt throughput, generation speed, RAM, and TTFT across context sizes.

Usage:
    python benchmark_context.py --model 26b
    python benchmark_context.py --model e4b
    python benchmark_context.py --model both   # runs sequentially
"""

import time, os, sys, json, argparse, gc
from datetime import datetime

os.environ.setdefault("HF_HUB_OFFLINE", "1")
if os.path.exists(os.path.join(os.path.dirname(__file__), ".env")):
    for line in open(os.path.join(os.path.dirname(__file__), ".env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from mlx_vlm import load, generate, apply_chat_template
from mlx_vlm.utils import load_config

MODELS = {
    "e4b":  ("mlx-community/gemma-4-e4b-it-4bit",       "Gemma 4 E4B (4.5B, 4-bit)",  128_000),
    "26b":  ("mlx-community/gemma-4-26b-a4b-it-4bit",   "Gemma 4 26B-A4B MoE (4-bit)", 256_000),
}

# Context lengths to test (in tokens, approximate — actual will vary with tokenizer)
# We build prompts that hit roughly these sizes
CONTEXT_SIZES = [1_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000]

# Filler text (~6 tokens per sentence)
FILLER = (
    "The study of memory in artificial intelligence systems reveals fundamental "
    "architectural constraints that differ significantly from biological neural networks. "
    "Unlike the hippocampal-neocortical system in mammals, current transformer models "
    "process context as a flat sequence without episodic tagging or temporal indexing. "
    "This limits their ability to retrieve specific past events and reason about causality "
    "across long time horizons. Research into retrieval-augmented generation attempts to "
    "address this gap but introduces its own failure modes around relevance scoring. "
)  # ~80 tokens

def make_prompt(target_tokens: int) -> str:
    """Build a prompt that hits approximately target_tokens in the context."""
    reps = max(1, target_tokens // 80)
    context = FILLER * reps
    return (
        f"Background context:\n\n{context}\n\n"
        "Based on the above, identify the single most important architectural "
        "limitation described. One sentence only."
    )

def run_benchmark(model_id: str, label: str, max_ctx: int, sizes=None):
    if sizes is None:
        sizes = [s for s in CONTEXT_SIZES if s <= max_ctx]

    print(f"\n{'='*65}", flush=True)
    print(f"  {label}", flush=True)
    print(f"  Max context: {max_ctx:,} tokens", flush=True)
    print(f"{'='*65}", flush=True)

    t_load = time.time()
    print("Loading model...", flush=True)
    model, processor = load(model_id)
    config = load_config(model_id)
    load_s = time.time() - t_load
    print(f"Loaded in {load_s:.1f}s\n", flush=True)

    results = {
        "model_id": model_id,
        "label": label,
        "load_time_s": round(load_s, 2),
        "max_context": max_ctx,
        "runs": [],
        "timestamp": datetime.utcnow().isoformat(),
        "mlx_vlm_version": "0.4.4",
    }

    print(f"  {'Context':>10}  {'Prompt tok':>11}  {'Prefill':>10}  {'Gen':>8}  {'TTFT':>7}  {'RAM':>8}", flush=True)
    print(f"  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*8}", flush=True)

    for target in sizes:
        raw_prompt = make_prompt(target)
        prompt = apply_chat_template(
            processor, config,
            prompt=raw_prompt,
            add_generation_prompt=True
        )

        try:
            t0 = time.time()
            result = generate(
                model, processor,
                prompt=prompt,
                max_tokens=40,
                verbose=False,
            )
            elapsed = time.time() - t0

            # TTFT estimate: prompt_tokens / prompt_tps
            ttft = result.prompt_tokens / result.prompt_tps if result.prompt_tps > 0 else 0

            row = {
                "target_tokens": target,
                "prompt_tokens": result.prompt_tokens,
                "generation_tokens": result.generation_tokens,
                "prompt_tps": round(result.prompt_tps, 1),
                "generation_tps": round(result.generation_tps, 1),
                "ttft_s": round(ttft, 2),
                "total_time_s": round(elapsed, 2),
                "peak_memory_gb": round(result.peak_memory, 3),
                "output": result.text.strip()[:200],
                "oom": False,
            }

            print(
                f"  {target:>9,}  {result.prompt_tokens:>10,}  "
                f"{result.prompt_tps:>8.0f}/s  {result.generation_tps:>6.0f}/s  "
                f"{ttft:>6.1f}s  {result.peak_memory:>7.2f}GB",
                flush=True
            )

        except Exception as e:
            err = str(e)
            if "memory" in err.lower() or "oom" in err.lower() or "metal" in err.lower():
                print(f"  {target:>9,}  {'OOM':>11}  {'—':>10}  {'—':>8}  {'—':>7}  {'—':>8}", flush=True)
                row = {"target_tokens": target, "oom": True, "error": err[:200]}
                results["runs"].append(row)
                break  # no point testing larger sizes
            else:
                print(f"  {target:>9,}  ERROR: {err[:60]}", flush=True)
                row = {"target_tokens": target, "oom": False, "error": err[:200]}

        results["runs"].append(row)

    del model, processor
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.reset_peak_memory()
    except Exception:
        pass

    return results


def print_summary(all_results):
    print("\n\n" + "="*80, flush=True)
    print("CONTEXT SCALING SUMMARY", flush=True)
    print("="*80, flush=True)
    print(f"  {'Context':>10}  {'Model':>30}  {'Prefill':>10}  {'Gen':>8}  {'TTFT':>7}  {'RAM':>8}", flush=True)
    print(f"  {'-'*10}  {'-'*30}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*8}", flush=True)

    for res in all_results:
        for run in res["runs"]:
            if run.get("oom"):
                print(f"  {run['target_tokens']:>9,}  {res['label']:>30}  {'OOM':>10}", flush=True)
            elif "error" in run and not run.get("prompt_tps"):
                print(f"  {run['target_tokens']:>9,}  {res['label']:>30}  {'ERR':>10}", flush=True)
            else:
                print(
                    f"  {run['target_tokens']:>9,}  {res['label']:>30}  "
                    f"{run['prompt_tps']:>8.0f}/s  {run['generation_tps']:>6.0f}/s  "
                    f"{run['ttft_s']:>6.1f}s  {run['peak_memory_gb']:>7.2f}GB",
                    flush=True
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["e4b", "26b", "both"], default="26b")
    parser.add_argument("--max-size", type=int, default=None,
                        help="Override max context size to test (tokens)")
    args = parser.parse_args()

    to_run = ["e4b", "26b"] if args.model == "both" else [args.model]
    all_results = []

    for key in to_run:
        model_id, label, max_ctx = MODELS[key]
        sizes = [s for s in CONTEXT_SIZES if s <= (args.max_size or max_ctx)]
        res = run_benchmark(model_id, label, max_ctx, sizes)
        all_results.append(res)

    if len(all_results) > 1:
        print_summary(all_results)

    out = os.path.join(os.path.dirname(__file__), "benchmark_context_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}", flush=True)

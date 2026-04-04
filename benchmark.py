#!/usr/bin/env python3
"""
Gemma 4 Local Benchmark — E4B vs 26B
Tests: context lengths, task types, chat template correctness
mlx-vlm 0.4.4 + TurboQuant KV-3
"""

import time
import os
import json
import sys
from datetime import datetime

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["HF_HUB_OFFLINE"] = "1"  # use cache only

from mlx_vlm import load, generate, apply_chat_template
from mlx_vlm.utils import load_config

RESULTS = []

TASKS = [
    {
        "name": "short_qa",
        "label": "Short Q&A",
        "prompt": "What is the capital of France? Answer in one word.",
        "max_tokens": 10,
    },
    {
        "name": "reasoning",
        "label": "Reasoning",
        "prompt": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost? Show your reasoning briefly.",
        "max_tokens": 80,
    },
    {
        "name": "code_gen",
        "label": "Code Generation",
        "prompt": "Write a Python function that returns the nth Fibonacci number using memoization. Just code, no explanation.",
        "max_tokens": 150,
    },
    {
        "name": "medium_context",
        "label": "Medium Context (2K tokens)",
        "prompt": (
            "Background: " + ("The episodic memory system stores specific events with their temporal and spatial context. "
            "The semantic memory system stores general facts and concepts. "
            "Working memory holds information temporarily for immediate use. "
            "Long-term potentiation (LTP) is the cellular mechanism underlying memory consolidation. ") * 60
            + "\nBased on the above, list the three key memory systems in one sentence each."
        ),
        "max_tokens": 80,
    },
    {
        "name": "long_context",
        "label": "Long Context (8K tokens)",
        "prompt": (
            "Context: " + ("Agents operating across sessions face the reconstruction problem: "
            "each experience passes through a decision to write, a summarization step, and a rebuild phase, "
            "losing texture and reasoning quality at every step. "
            "The hippocampus handles episodic memory while the neocortex handles semantic consolidation. "
            "Current LLM memory architectures are all cortex and no hippocampus. ") * 200
            + "\nIn two sentences, what is the core problem with current LLM memory?"
        ),
        "max_tokens": 60,
    },
    {
        "name": "summarization",
        "label": "Summarization",
        "prompt": """Summarize the following meeting notes in 3 bullet points:

Meeting: Q1 Planning Session
Attendees: Alice (PM), Bob (Eng), Carol (Design)

Alice opened by reviewing Q4 metrics: DAU up 23%, revenue up 18%, churn down 4 points. 
Bob raised concerns about technical debt in the auth system — estimated 3 weeks to address, 
risk of security incident if deferred. Carol presented three new onboarding flow designs; 
user testing showed design B had 34% better completion rate. 
Decision: delay feature X by 2 weeks to address auth debt. Design B approved for onboarding. 
Alice to communicate timeline change to stakeholders by Friday. Bob to start auth sprint Monday. 
Carol to prepare final design B assets by Wednesday.""",
        "max_tokens": 120,
    },
    {
        "name": "instruction_follow",
        "label": "Instruction Following",
        "prompt": "List exactly 5 programming languages, one per line, alphabetically, no numbering, no extra text.",
        "max_tokens": 30,
    },
]

MODELS = [
    ("mlx-community/gemma-4-e4b-it-4bit", "Gemma 4 E4B (4.5B, 4-bit)"),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "Gemma 4 26B-A4B (MoE, 4-bit)"),
]

def run_model_benchmarks(model_id, model_label):
    print(f"\n{'='*60}", flush=True)
    print(f"MODEL: {model_label}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    print("Loading...", flush=True)
    model, processor = load(model_id)
    config = load_config(model_id)
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s", flush=True)

    model_results = {
        "model_id": model_id,
        "model_label": model_label,
        "load_time_s": round(load_time, 2),
        "tasks": [],
        "mlx_vlm_version": "0.4.4",
        "timestamp": datetime.utcnow().isoformat(),
    }

    for task in TASKS:
        print(f"\n  [{task['label']}]", flush=True)
        prompt = apply_chat_template(
            processor, config,
            prompt=task["prompt"],
            add_generation_prompt=True
        )

        t1 = time.time()
        result = generate(
            model, processor,
            prompt=prompt,
            max_tokens=task["max_tokens"],
            verbose=False,
        )
        elapsed = time.time() - t1

        task_result = {
            "name": task["name"],
            "label": task["label"],
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "prompt_tps": round(result.prompt_tps, 1),
            "generation_tps": round(result.generation_tps, 1),
            "total_time_s": round(elapsed, 2),
            "peak_memory_gb": round(result.peak_memory, 3),
            "output": result.text.strip()[:300],
        }
        model_results["tasks"].append(task_result)

        print(f"    Prompt: {result.prompt_tokens} tok @ {result.prompt_tps:.0f} tok/s", flush=True)
        print(f"    Gen:    {result.generation_tokens} tok @ {result.generation_tps:.0f} tok/s", flush=True)
        print(f"    Time:   {elapsed:.2f}s | RAM: {result.peak_memory:.2f}GB", flush=True)
        print(f"    Output: {result.text.strip()[:120]}", flush=True)

    RESULTS.append(model_results)
    del model, processor  # free memory before next model
    import gc; gc.collect()
    try:
        import mlx.core as mx
        mx.metal.reset_peak_memory()
    except Exception:
        pass

    return model_results


if __name__ == "__main__":
    print(f"Gemma 4 Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M')}", flush=True)
    print(f"mlx-vlm 0.4.4 | TurboQuant KV-3 | Apple Silicon", flush=True)

    for model_id, model_label in MODELS:
        try:
            run_model_benchmarks(model_id, model_label)
        except Exception as e:
            print(f"ERROR on {model_label}: {e}", flush=True)
            import traceback; traceback.print_exc()

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\n\nResults saved to {out_path}", flush=True)

    # Print summary table
    print("\n\n=== SUMMARY ===")
    print(f"{'Task':<30} {'E4B prompt':>12} {'E4B gen':>9} {'26B prompt':>12} {'26B gen':>9} {'E4B RAM':>9} {'26B RAM':>9}")
    print("-" * 100)

    if len(RESULTS) >= 2:
        e4b = {t["name"]: t for t in RESULTS[0]["tasks"]}
        m26b = {t["name"]: t for t in RESULTS[1]["tasks"]}
        for task in TASKS:
            n = task["name"]
            if n in e4b and n in m26b:
                print(
                    f"{task['label']:<30}"
                    f"{e4b[n]['prompt_tps']:>10.0f}/s"
                    f"{e4b[n]['generation_tps']:>8.0f}/s"
                    f"{m26b[n]['prompt_tps']:>10.0f}/s"
                    f"{m26b[n]['generation_tps']:>8.0f}/s"
                    f"{e4b[n]['peak_memory_gb']:>8.2f}GB"
                    f"{m26b[n]['peak_memory_gb']:>8.2f}GB"
                )

    print(f"\nLoad times: E4B={RESULTS[0]['load_time_s']}s | 26B={RESULTS[1]['load_time_s']}s" if len(RESULTS) >= 2 else "")

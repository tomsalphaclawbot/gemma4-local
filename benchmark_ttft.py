#!/usr/bin/env python3
"""
TTFT benchmark for mlx-vlm 0.4.4 vs our baseline (0.4.3)
Tests: 1K, 8K, 32K, 64K token contexts
Model: gemma-4-26b-a4b-it-4bit
"""
import time
import sys
import subprocess

MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"

# Build prompts of roughly target token sizes
# ~4 chars per token is a rough estimate
def make_prompt(target_tokens):
    # Use a realistic long document — repeat a paragraph
    para = (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence contains all the letters of the alphabet. "
        "Memory systems in AI agents require careful design to balance "
        "recall accuracy, retrieval speed, and storage efficiency. "
        "The hippocampus in biological brains stores episodic memories "
        "while the cortex handles semantic consolidation over time. "
    )
    chars_needed = target_tokens * 4
    repeated = (para * (chars_needed // len(para) + 1))[:chars_needed]
    return f"Summarize the following text in one sentence:\n\n{repeated}\n\nSummary:"

def benchmark_context(tokens):
    prompt = make_prompt(tokens)
    actual_chars = len(prompt)
    print(f"\n--- Context: ~{tokens}K tokens ({actual_chars:,} chars) ---")

    cmd = [
        sys.executable, "-m", "mlx_vlm.generate",
        "--model", MODEL,
        "--max-tokens", "50",
        "--temp", "0.0",
        "--prompt", prompt,
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/Users/openclaw/.openclaw/workspace"
        )
        elapsed = time.time() - start

        # Extract timing from mlx_vlm output
        output = result.stdout + result.stderr
        print(f"  Total time: {elapsed:.1f}s")

        # Look for tok/s in output
        for line in output.split('\n'):
            if 'tok' in line.lower() or 'prompt' in line.lower() or 'speed' in line.lower():
                print(f"  {line.strip()}")

        # Show first line of response
        for line in output.split('\n'):
            line = line.strip()
            if line and not line.startswith('=') and 'tok' not in line.lower() and len(line) > 10:
                print(f"  Response: {line[:100]}")
                break

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 300s")
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    print(f"mlx-vlm TTFT Benchmark")
    print(f"Model: {MODEL}")
    print(f"Testing context sizes: 1K, 8K, 32K tokens")

    for k in [1, 8, 32]:
        benchmark_context(k * 1000)

    print("\nDone.")

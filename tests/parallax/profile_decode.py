"""Per-op profiling of decode steps: parallax (cute/fla) vs fa-baseline.

Boots an 8B-shape dummy-weight engine, runs a steady-state decode workload
under torch.profiler, and prints the top CUDA ops plus a component rollup
(attention kernel, gather/staging, projections, norms/rope, store, other).
Used to attribute the e2e gap between the parallax backend and fa.

Usage (on a GPU node):
    python -m tests.parallax.profile_decode <cute|fla|fa-baseline> [bs] [prompt] [gen]
"""

from __future__ import annotations

import json
import os
import sys

import torch

from .bench_serving import _PARALLAX_8B


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cute"
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    prompt_len = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    gen_len = int(sys.argv[4]) if len(sys.argv) > 4 else 32

    from tests.parallax.make_tiny_ckpt import _write_tokenizer

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    ckpt = f"/tmp/parallax_prof_{mode}"
    os.makedirs(ckpt, exist_ok=True)
    cfg = dict(_PARALLAX_8B)
    if mode == "fa-baseline":
        cfg["architectures"] = ["Qwen3ForCausalLM"]
    with open(os.path.join(ckpt, "config.json"), "w") as f:
        json.dump(cfg, f)
    _write_tokenizer(ckpt, cfg["vocab_size"])
    if mode in ("cute", "fla"):
        os.environ["MINISGL_PARALLAX_DECODE_IMPL"] = mode
    os.environ.setdefault("MINISGL_DIST_PORT", str(20000 + os.getpid() % 20000))

    llm = LLM(
        model_path=ckpt,
        dtype=torch.bfloat16,
        use_dummy_weight=True,
        max_running_req=bs,
        max_seq_len_override=16384,
        memory_ratio=0.8,
        max_extend_tokens=131072,
        cache_type="naive",
        cuda_graph_max_bs=0,
    )
    prompts = [[(7 * i + 13 * j) % 500 + 2 for j in range(prompt_len)] for i in range(bs)]
    sp = SamplingParams(max_tokens=gen_len, ignore_eos=True)
    llm.generate(prompts, sp)  # warmup / JIT

    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=False,
    ) as prof:
        llm.generate(prompts, sp)

    events = prof.key_averages()
    rows = [
        (e.key, e.device_time_total / 1e3, e.count)
        for e in events
        if e.device_time_total > 0
    ]
    rows.sort(key=lambda r: -r[1])
    total_cuda = sum(r[1] for r in rows)
    print(f"\n=== mode={mode} bs={bs} prompt={prompt_len} gen={gen_len} "
          f"| total CUDA {total_cuda:.1f} ms (prefill+decode of one generate) ===")
    print(f"{'op':70s} {'cuda ms':>9} {'%':>5} {'count':>7}")
    for key, ms, count in rows[:25]:
        print(f"{key[:70]:70s} {ms:>9.1f} {100*ms/total_cuda:>5.1f} {count:>7}")

    # component rollup by name heuristics
    buckets = {
        "attention kernel": ("ParallaxDecode", "flash", "fwd_kernel", "parallax", "onestep", "fmha"),
        "gather/staging": ("index_select", "indexSelect", "copy_", "CopyD2D", "elementwise_kernel"),
        "gemm (proj/lm_head)": ("gemm", "cutlass", "nvjet", "matmul", "Cijk"),
        "norm/rope/act": ("rms", "Rope", "rope", "silu", "act_and_mul", "norm"),
        "store_kv": ("store",),
    }
    rollup: dict[str, float] = {k: 0.0 for k in buckets}
    rollup["other"] = 0.0
    for key, ms, _ in rows:
        for name, pats in buckets.items():
            if any(p.lower() in key.lower() for p in pats):
                rollup[name] += ms
                break
        else:
            rollup["other"] += ms
    print("\n--- rollup ---")
    for name, ms in sorted(rollup.items(), key=lambda kv: -kv[1]):
        print(f"{name:24s} {ms:>9.1f} ms  {100*ms/total_cuda:>5.1f}%")


if __name__ == "__main__":
    main()

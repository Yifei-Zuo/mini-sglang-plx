"""Benchmarks for the parallax serving integration (run on one H100).

Modes:
  kernel       — CuTeDSL serving decode vs FLA Triton onestep (uniform/skewed)
  kernel-grid  — CuTeDSL vs FA3 (sgl_kernel.flash_attn_with_kvcache, the exact
                 kernel minisgl's `fa` backend uses) vs FLA Triton, over a
                 (batch, seqlen) grid; identical contiguous KV for all three
  serving      — e2e sweep on an 8B-shape model (dummy weights) through
                 minisgl.llm.LLM: one engine per mode, several (bs, prompt)
                 configs; prefill latency and steady-state decode tok/s are
                 separated by differencing gen=1 vs gen=N runs

Usage:
    python -m tests.parallax.bench_serving kernel
    python -m tests.parallax.bench_serving kernel-grid
    python -m tests.parallax.bench_serving serving [cute|fla|fa-baseline]
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

HQ, HKV, D = 32, 8, 128  # Qwen3-8B-ish local heads (TP=1)


def _time_cuda(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def _make_inputs(B, L, lens, gen):
    q = torch.randn((B, 1, HQ, D), generator=gen, device="cuda").to(torch.bfloat16)
    r = torch.randn_like(q)
    k = torch.randn((B, L, HKV, D), generator=gen, device="cuda").to(torch.bfloat16)
    v = torch.randn_like(k)
    for b, n in enumerate(lens):
        if n < L:
            k[b, n:] = k[b, n - 1]
            v[b, n:] = v[b, n - 1]
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return q, r, k, v, seq


def bench_kernel():
    from fla.ops.parallax import parallax_decode_onestep
    from parallax.cute.parallax_decode import parallax_decode_serving

    g = torch.Generator(device="cuda").manual_seed(0)
    cases = {
        "uniform B=8 L=4096": (8, 4096, [4096] * 8),
        "skewed  B=8 (1x4096 + 7x128)": (8, 4096, [4096] + [128] * 7),
        "uniform B=1 L=16384": (1, 16384, [16384]),
        "uniform B=64 L=1024": (64, 1024, [1024] * 64),
    }
    for name, (B, bucket, lens) in cases.items():
        q, r, k, v, seq = _make_inputs(B, bucket, lens, g)
        shift = bucket - seq
        scale = D**-0.5
        out = torch.empty_like(q)
        t_cute = _time_cuda(
            lambda: parallax_decode_serving(q, r, k, v, scale, seqused_k=seq, out=out)
        )
        t_fla = _time_cuda(
            lambda: parallax_decode_onestep(q, r, k, v, scale=scale, cache_start=shift)
        )
        print(f"{name:36s} cute {t_cute*1e3:8.1f} us   fla {t_fla*1e3:8.1f} us")


def bench_kernel_grid():
    """cute vs FA3 vs fla-triton on identical contiguous KV, uniform full lengths."""
    from fla.ops.parallax import parallax_decode_onestep
    from parallax.cute.parallax_decode import parallax_decode_serving
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    g = torch.Generator(device="cuda").manual_seed(0)
    scale = D**-0.5
    grid = [
        (1, 1024), (1, 4096), (1, 16384), (1, 65536),
        (8, 1024), (8, 4096), (8, 16384),
        (32, 1024), (32, 4096), (32, 16384),
        (64, 1024), (64, 4096),
        (128, 1024), (128, 4096),
    ]
    print(f"{'B':>4} {'L':>6} | {'cute us':>9} {'FA3 us':>9} {'fla us':>9} | cute/FA3")
    for B, L in grid:
        if B * L > (1 << 21):  # cap KV at ~8 GiB per tensor pair
            continue
        lens = [L] * B
        q, r, k, v, seq = _make_inputs(B, L, lens, g)
        out = torch.empty_like(q)
        t_cute = _time_cuda(
            lambda: parallax_decode_serving(q, r, k, v, scale, seqused_k=seq, out=out)
        )
        t_fa = _time_cuda(
            lambda: flash_attn_with_kvcache(
                q=q, k_cache=k, v_cache=v, cache_seqlens=seq,
                softmax_scale=scale, causal=True, ver=3,
            )
        )
        t_fla = _time_cuda(
            lambda: parallax_decode_onestep(q, r, k, v, scale=scale, cache_start=None)
        )
        print(
            f"{B:>4} {L:>6} | {t_cute*1e3:>9.1f} {t_fa*1e3:>9.1f} {t_fla*1e3:>9.1f} "
            f"| {t_cute/t_fa:>7.2f}x"
        )


_PARALLAX_8B = {
    "architectures": ["ParallaxQwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 40960,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
    "eos_token_id": 0,
}

# (bs, prompt_len, gen_len) e2e sweep configs
_SWEEP = [
    (1, 512, 128),
    (8, 512, 128),
    (32, 512, 128),
    (1, 2048, 128),
    (8, 2048, 128),
    (32, 2048, 128),
    (8, 8192, 128),
]


def bench_serving(mode: str):
    from tests.parallax.make_tiny_ckpt import _write_tokenizer

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    ckpt = f"/tmp/parallax_bench_{mode}"
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

    max_bs = max(c[0] for c in _SWEEP)
    max_tokens_needed = max(bs * (p + g) for bs, p, g in _SWEEP)
    llm = LLM(
        model_path=ckpt,
        dtype=torch.bfloat16,
        use_dummy_weight=True,
        max_running_req=max_bs,
        max_seq_len_override=16384,
        memory_ratio=0.8,  # pool auto-sized (page_size differs per mode)
        max_extend_tokens=131072,  # no chunked prefill: keep prefill one-shot
        cache_type="naive",
        cuda_graph_max_bs=0,  # graphs off for all modes: apples-to-apples v1
    )
    print(f"mode={mode}  (graphs off, naive cache)")
    print(f"{'bs':>4} {'prompt':>7} {'gen':>5} | {'prefill ms':>11} {'decode ms/step':>15} "
          f"{'decode tok/s':>13} {'total s':>8}")
    for bs, prompt_len, gen_len in _SWEEP:
        prompts = [
            [(7 * i + 13 * j) % 500 + 2 for j in range(prompt_len)] for i in range(bs)
        ]
        sp1 = SamplingParams(max_tokens=1, ignore_eos=True)
        spN = SamplingParams(max_tokens=gen_len, ignore_eos=True)
        llm.generate(prompts, spN)  # warmup: JIT/triton compiles for these shapes
        t0 = time.perf_counter()
        llm.generate(prompts, sp1)
        t_prefill = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = llm.generate(prompts, spN)
        t_total = time.perf_counter() - t0
        n_decode = sum(len(o["token_ids"]) for o in out) - bs  # steps after the first
        dec_time = t_total - t_prefill
        per_step = dec_time / (gen_len - 1) * 1e3
        print(
            f"{bs:>4} {prompt_len:>7} {gen_len:>5} | {t_prefill*1e3:>11.1f} "
            f"{per_step:>15.2f} {n_decode/dec_time:>13.1f} {t_total:>8.2f}"
        )


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "kernel"
    if what == "kernel":
        bench_kernel()
    elif what == "kernel-grid":
        bench_kernel_grid()
    else:
        bench_serving(sys.argv[2] if len(sys.argv) > 2 else "cute")

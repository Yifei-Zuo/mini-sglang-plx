"""End-to-end greedy generation through the scheduler/engine (GPU).

Each engine run executes in its own spawned subprocess because Engine asserts
CUDA is uninitialized at startup.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PROMPTS = {
    "short": list(range(2, 14)),
    "mixed": [list(range(2, 2 + n)) for n in (3, 9, 17, 26, 40, 64)],
    "long": [2 + (i * 7) % 500 for i in range(150)],
}
NUM_TOKENS = 8

_LLM_DEFAULTS = dict(
    max_running_req=8,
    max_seq_len_override=2048,
    memory_ratio=0.5,  # pool auto-sized: page count depends on the mode's page_size
    cache_type="naive",
)


def _gen_proc(ckpt, prompt_lists, num_tokens, llm_kwargs, repeats, queue):
    try:
        # unique store port per engine process: concurrent engines on one
        # node (parallel test sessions, leaked children) must not collide
        os.environ.setdefault("MINISGL_DIST_PORT", str(20000 + os.getpid() % 20000))
        from minisgl.core import SamplingParams
        from minisgl.llm import LLM

        kwargs = {**_LLM_DEFAULTS, **llm_kwargs}
        llm = LLM(model_path=ckpt, dtype=torch.bfloat16, **kwargs)
        sp = SamplingParams(max_tokens=num_tokens, ignore_eos=True)
        results = []
        for _ in range(repeats):
            outs = llm.generate(prompt_lists, sp)
            results.append([o["token_ids"] for o in outs])
        queue.put(("ok", results))
    except Exception:
        queue.put(("err", traceback.format_exc()))


def _generate(ckpt, prompt_lists, num_tokens=NUM_TOKENS, repeats=1, **llm_kwargs):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_gen_proc, args=(ckpt, prompt_lists, num_tokens, llm_kwargs, repeats, queue)
    )
    p.start()
    try:
        status, payload = queue.get(timeout=600)
    finally:
        # always reap: a leaked child wedges pytest at exit (atexit joins
        # non-daemon children) and keeps holding the distributed port
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
            p.join(timeout=10)
    assert status == "ok", f"engine subprocess failed:\n{payload}"
    return payload if repeats > 1 else payload[0]


@pytest.fixture(scope="session")
def ckpt(tmp_path_factory):
    from .make_tiny_ckpt import build_tiny_ckpt

    path = str(tmp_path_factory.mktemp("ckpt") / "parallax_tiny")
    build_tiny_ckpt(path)
    return path


@requires_cuda
def test_single_request_matches_reference(ckpt):
    from .reference_model import TinyParallaxReference

    engine_out = _generate(ckpt, [PROMPTS["short"]])[0]
    ref = TinyParallaxReference(ckpt).greedy_generate(PROMPTS["short"], NUM_TOKENS)
    assert engine_out == ref, f"engine {engine_out} != reference {ref}"


@requires_cuda
def test_batched_mixed_lengths_match_single(ckpt):
    batched = _generate(ckpt, PROMPTS["mixed"])
    singles = [_generate(ckpt, [p])[0] for p in PROMPTS["mixed"]]
    for i, (b, s) in enumerate(zip(batched, singles)):
        assert b == s, f"prompt {i}: batched {b} != single {s}"


@requires_cuda
def test_chunked_prefill_matches_unchunked(ckpt):
    unchunked = _generate(ckpt, [PROMPTS["long"]])[0]
    chunked = _generate(ckpt, [PROMPTS["long"]], max_extend_tokens=64)[0]
    assert chunked == unchunked


@requires_cuda
def test_radix_prefix_hit_matches(ckpt):
    # same prompt twice in one engine: second run hits the radix prefix cache
    # and goes through the extend path
    first, second = _generate(
        ckpt, [PROMPTS["long"]], repeats=2, cache_type="radix"
    )
    assert first == second


if __name__ == "__main__":
    # standalone smoke run: python -m tests.parallax.test_e2e_generate <ckpt_dir>
    import sys

    from .make_tiny_ckpt import build_tiny_ckpt

    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/parallax_tiny_ckpt"
    if not os.path.exists(os.path.join(path, "config.json")):
        build_tiny_ckpt(path)
    out = _generate(path, [PROMPTS["short"]])
    print("engine:", out)

    from .reference_model import TinyParallaxReference

    print("ref   :", [TinyParallaxReference(path).greedy_generate(PROMPTS["short"], NUM_TOKENS)])

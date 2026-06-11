"""Paged decode path (page_size=64, cute): backend-level parity vs naive (GPU).

Emulates the engine's page-aligned allocator: each request owns whole pool
pages; the token-granular global page table maps sequence position -> token
slot (page * 64 + offset). The backend converts that to per-tile page entries
and the kernel TMA-reads the pool directly.
"""

import os

import pytest
import torch
from minisgl.core import Batch

from .utils import make_req, setup_ctx, tiny_parallax_config

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
not_forced_fla = pytest.mark.skipif(
    os.environ.get("MINISGL_PARALLAX_DECODE_IMPL") == "fla",
    reason="paged path requires the cute decode impl",
)

HQ, HKV, D = 8, 2, 64
DTYPE = torch.bfloat16
PAGE = 64


def _cute_paged_available() -> bool:
    try:
        from parallax.cute.parallax_decode import parallax_decode_serving_paged  # noqa: F401

        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
    except Exception:
        return False


@requires_cuda
@not_forced_fla
@pytest.mark.skipif(not _cute_paged_available(), reason="needs SM90 + paged cute kernel")
@pytest.mark.parametrize("window_size", [None, 16])
def test_paged_decode_mixed_lengths(window_size):
    from minisgl.attention.parallax import ParallaxBackend

    config = tiny_parallax_config(
        num_layers=1, num_qo_heads=HQ, num_kv_heads=HKV, head_dim=D, window_size=window_size
    )
    device = torch.device("cuda:0")
    ctx = setup_ctx(config, device, num_tokens=64 * 1024, table_width=4096, page_size=PAGE)
    backend = ParallaxBackend(config)
    assert backend.paged, "expected the paged cute decode path at page_size=64"
    ctx.attn_backend = backend

    gen = torch.Generator().manual_seed(21)
    lens = [1, 17, 63, 64, 65, 300]
    bs = len(lens)

    # page-aligned allocation: distinct random pages per request
    num_pages = ctx.kv_cache.k_cache(0).shape[0] - 1
    pages = torch.randperm(num_pages, generator=gen)
    k_pool = ctx.kv_cache.k_cache(0).view(-1, HKV, D)  # token-granular view
    v_pool = ctx.kv_cache.v_cache(0).view(-1, HKV, D)

    full_k, full_v, out_loc = [], [], []
    page_cursor = 0
    for i, n in enumerate(lens):
        n_pages = (n + PAGE - 1) // PAGE
        req_pages = pages[page_cursor : page_cursor + n_pages]
        page_cursor += n_pages
        pos = torch.arange(n)
        slots = (req_pages[pos // PAGE] * PAGE + pos % PAGE).to(torch.int32)
        ctx.page_table[i, :n] = slots.to(device)
        kf = torch.randn((n, HKV, D), generator=gen).to(DTYPE).to(device)
        vf = torch.randn((n, HKV, D), generator=gen).to(DTYPE).to(device)
        k_pool[slots[: n - 1].to(device).long()] = kf[: n - 1]
        v_pool[slots[: n - 1].to(device).long()] = vf[: n - 1]
        full_k.append(kf)
        full_v.append(vf)
        out_loc.append(slots[n - 1].to(device))

    reqs = [make_req(n, n - 1, i, uid=i) for i, n in enumerate(lens)]
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.out_loc = torch.stack(out_loc)
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert md.page_entries is not None and md.gather_idx is None

    q = torch.randn((bs, HQ, D), generator=gen).to(DTYPE).to(device)
    r = torch.randn((bs, HQ, D), generator=gen).to(DTYPE).to(device)
    k_new = torch.stack([fk[-1] for fk in full_k]).view(bs, HKV * D)
    v_new = torch.stack([fv[-1] for fv in full_v]).view(bs, HKV * D)

    out = backend.forward_parallax(q, r, k_new, v_new, 0, batch)

    from fla.ops.parallax.naive import naive_parallax

    for i, n in enumerate(lens):
        q_full = torch.zeros((n, HQ, D), dtype=torch.float32, device=device)
        r_full = torch.zeros_like(q_full)
        q_full[-1] = q[i].float()
        r_full[-1] = r[i].float()
        ref = naive_parallax(
            q_full.unsqueeze(0),
            r_full.unsqueeze(0),
            full_k[i].float().unsqueeze(0),
            full_v[i].float().unsqueeze(0),
            scale=D**-0.5,
            window_size=window_size,
            causal=True,
        ).squeeze(0)[-1]
        err = (out[i].float() - ref).abs().max().item()
        scale_ref = max(ref.abs().max().item(), 1.0)
        assert err <= 3e-2 * scale_ref + 0.1, (
            f"req {i} (len {n}, window {window_size}): max err {err:.4g}"
        )

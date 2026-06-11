"""ParallaxBackend.forward_parallax vs the fla naive reference, all three modes (GPU)."""

import pytest
import torch
from minisgl.core import Batch

from .utils import make_req, setup_ctx, tiny_parallax_config

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HQ, HKV, D = 4, 2, 64
DTYPE = torch.bfloat16


def _cute_available() -> bool:
    try:
        from parallax.cute.parallax_decode import parallax_decode_serving  # noqa: F401

        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
    except Exception:
        return False


DECODE_IMPLS = ["fla"] + (["cute"] if _cute_available() else [])


def _setup(window_size=None):
    from minisgl.attention.parallax import ParallaxBackend

    config = tiny_parallax_config(
        num_layers=1, num_qo_heads=HQ, num_kv_heads=HKV, head_dim=D, window_size=window_size
    )
    device = torch.device("cuda:0")
    ctx = setup_ctx(config, device, num_tokens=16384, table_width=4096)
    backend = ParallaxBackend(config)
    ctx.attn_backend = backend
    return backend, ctx, device


def _flat_pool(ctx):
    k = ctx.kv_cache.k_cache(0).view(-1, HKV, D)
    v = ctx.kv_cache.v_cache(0).view(-1, HKV, D)
    return k, v


def _scatter_history(ctx, lens, gen):
    """Assign each request shuffled pool slots and pre-write all but its last token.

    Returns (full_k, full_v) per request: the request's complete K/V sequence
    including the (yet-unwritten) last token, plus the new-token tensors and
    out_loc pointing at each request's last slot.
    """
    device = ctx.page_table.device
    total = sum(lens)
    slots = torch.randperm(16384, generator=gen)[:total].to(torch.int32).to(device)
    k_pool, v_pool = _flat_pool(ctx)
    full_k, full_v, out_loc = [], [], []
    off = 0
    for i, n in enumerate(lens):
        req_slots = slots[off : off + n]
        off += n
        ctx.page_table[i, :n] = req_slots
        kf = torch.randn((n, HKV, D), generator=gen).to(DTYPE).to(device)
        vf = torch.randn((n, HKV, D), generator=gen).to(DTYPE).to(device)
        # pre-write the history (all but the last token); the last token goes
        # through store_kv inside forward_parallax, like a real decode step
        k_pool[req_slots[: n - 1].to(torch.int64)] = kf[: n - 1]
        v_pool[req_slots[: n - 1].to(torch.int64)] = vf[: n - 1]
        full_k.append(kf)
        full_v.append(vf)
        out_loc.append(req_slots[n - 1])
    return full_k, full_v, torch.stack(out_loc)


def _naive_ref(q, r, k, v, window_size):
    from fla.ops.parallax.naive import naive_parallax

    out = naive_parallax(
        q.float().unsqueeze(0),
        r.float().unsqueeze(0),
        k.float().unsqueeze(0),
        v.float().unsqueeze(0),
        scale=D**-0.5,
        window_size=window_size,
        causal=True,
    )
    return out.squeeze(0)


def _assert_close(out, ref, what):
    err = (out.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err <= 2e-2 * max(scale, 1.0), f"{what}: max abs err {err:.4g} (ref scale {scale:.3g})"


@requires_cuda
@pytest.mark.parametrize("impl", DECODE_IMPLS)
@pytest.mark.parametrize("window_size", [None, 16])
def test_decode_mixed_lengths(impl, window_size):
    backend, ctx, device = _setup(window_size)
    backend.decode_impl = impl
    gen = torch.Generator().manual_seed(7)
    lens = [1, 17, 63, 64, 65, 300]
    bs = len(lens)
    full_k, full_v, out_loc = _scatter_history(ctx, lens, gen)

    reqs = [make_req(n, n - 1, i, uid=i) for i, n in enumerate(lens)]
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.out_loc = out_loc
    backend.prepare_metadata(batch)

    q = torch.randn((bs, HQ, D), generator=gen).to(DTYPE).to(device)
    r = torch.randn((bs, HQ, D), generator=gen).to(DTYPE).to(device)
    k_new = torch.stack([fk[-1] for fk in full_k]).view(bs, HKV * D)
    v_new = torch.stack([fv[-1] for fv in full_v]).view(bs, HKV * D)

    out = backend.forward_parallax(q, r, k_new, v_new, 0, batch)

    # reference: per request, q/r as the only (last-position) query over its full KV
    for i, n in enumerate(lens):
        from fla.ops.parallax.naive import naive_parallax

        # emulate decode: single query at absolute position n-1 over n keys ==
        # last row of full causal attention where q-row count == key count;
        # naive_parallax needs q,k with same T, so build T=n queries but only
        # compare the last row (other rows use zero queries, irrelevant).
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
        _assert_close(out[i], ref, f"req {i} (len {n}, impl {impl}, window {window_size})")


@requires_cuda
def test_prefill_varlen():
    backend, ctx, device = _setup()
    gen = torch.Generator().manual_seed(11)
    lens = [5, 64, 33]
    total = sum(lens)
    slots = torch.randperm(16384, generator=gen)[:total].to(torch.int32).to(device)
    reqs = []
    off = 0
    for i, n in enumerate(lens):
        ctx.page_table[i, :n] = slots[off : off + n]
        reqs.append(make_req(n, 0, i, uid=i))
        off += n
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    batch.out_loc = slots
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.mode == "prefill_varlen"

    q = torch.randn((total, HQ, D), generator=gen).to(DTYPE).to(device)
    r = torch.randn((total, HQ, D), generator=gen).to(DTYPE).to(device)
    k = torch.randn((total, HKV, D), generator=gen).to(DTYPE).to(device)
    v = torch.randn((total, HKV, D), generator=gen).to(DTYPE).to(device)

    out = backend.forward_parallax(q, r, k.view(total, -1), v.view(total, -1), 0, batch)

    off = 0
    for i, n in enumerate(lens):
        s = slice(off, off + n)
        ref = _naive_ref(q[s], r[s], k[s], v[s], None)
        _assert_close(out[s], ref, f"prefill seq {i} (len {n})")
        off += n

    # store_kv must have written this step's K/V into the pool
    k_pool, _ = _flat_pool(ctx)
    assert torch.equal(k_pool[slots[:5].to(torch.int64)], k[:5])


@requires_cuda
def test_extend_with_cached_prefix():
    backend, ctx, device = _setup()
    gen = torch.Generator().manual_seed(13)
    # req0: 7 cached + 9 new; req1: pure prefill (0 cached + 5 new)
    cached = [7, 0]
    new = [9, 5]
    lens = [c + e for c, e in zip(cached, new)]
    k_pool, v_pool = _flat_pool(ctx)
    total_new = sum(new)
    slots = torch.randperm(16384, generator=gen)[: sum(lens)].to(torch.int32).to(device)

    full_k, full_v, out_loc_parts = [], [], []
    off = 0
    for i, (c, n) in enumerate(zip(cached, lens)):
        req_slots = slots[off : off + n]
        off += n
        ctx.page_table[i, :n] = req_slots
        kf = torch.randn((n, HKV, D), generator=gen).to(DTYPE).to(device)
        vf = torch.randn((n, HKV, D), generator=gen).to(DTYPE).to(device)
        k_pool[req_slots[:c].to(torch.int64)] = kf[:c]
        v_pool[req_slots[:c].to(torch.int64)] = vf[:c]
        full_k.append(kf)
        full_v.append(vf)
        out_loc_parts.append(req_slots[c:])
    out_loc = torch.cat(out_loc_parts)

    reqs = [make_req(n, c, i, uid=i) for i, (c, n) in enumerate(zip(cached, lens))]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    batch.out_loc = out_loc
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.mode == "extend"

    q = torch.randn((total_new, HQ, D), generator=gen).to(DTYPE).to(device)
    r = torch.randn((total_new, HQ, D), generator=gen).to(DTYPE).to(device)
    k_new = torch.cat([fk[c:] for fk, c in zip(full_k, cached)]).view(total_new, -1)
    v_new = torch.cat([fv[c:] for fv, c in zip(full_v, cached)]).view(total_new, -1)

    out = backend.forward_parallax(q, r, k_new, v_new, 0, batch)

    lo = 0
    for i, (c, n) in enumerate(zip(cached, lens)):
        e = n - c
        from fla.ops.parallax.naive import naive_parallax

        q_full = torch.zeros((n, HQ, D), dtype=torch.float32, device=device)
        r_full = torch.zeros_like(q_full)
        q_full[c:] = q[lo : lo + e].float()
        r_full[c:] = r[lo : lo + e].float()
        ref = naive_parallax(
            q_full.unsqueeze(0),
            r_full.unsqueeze(0),
            full_k[i].float().unsqueeze(0),
            full_v[i].float().unsqueeze(0),
            scale=D**-0.5,
            causal=True,
        ).squeeze(0)[c:]
        _assert_close(out[lo : lo + e], ref, f"extend req {i}")
        lo += e

"""ParallaxBackend.prepare_metadata across the three modes (needs CUDA for pinned H2D)."""

import pytest
import torch
from minisgl.core import Batch

from .utils import make_req, setup_ctx, tiny_parallax_config

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture()
def backend_and_ctx():
    from minisgl.attention.parallax import ParallaxBackend

    config = tiny_parallax_config(num_layers=1)
    device = torch.device("cuda:0")
    ctx = setup_ctx(config, device)
    backend = ParallaxBackend(config)
    ctx.attn_backend = backend
    return backend, ctx, device


def _fill_page_table(ctx, table_idx, length, offset):
    # distinct, recognizable pool slots per request
    ctx.page_table[table_idx, :length] = torch.arange(
        offset, offset + length, dtype=torch.int32, device=ctx.page_table.device
    )


@requires_cuda
def test_decode_metadata_fla_right_aligned(backend_and_ctx):
    backend, ctx, device = backend_and_ctx
    backend.decode_impl = "fla"
    lens = [5, 1, 17, 130]
    reqs = [make_req(n, n - 1, i, uid=i) for i, n in enumerate(lens)]
    for i, n in enumerate(lens):
        _fill_page_table(ctx, i, n, 1000 * (i + 1))
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    backend.prepare_metadata(batch)
    md = batch.attn_metadata

    assert md.mode == "decode"
    bucket = md.kv_bucket_len
    assert bucket == 256  # next pow2 >= 130
    assert md.cache_start.tolist() == [bucket - n for n in lens]
    assert md.seqused_k.tolist() == lens
    assert md.get_last_indices(len(lens)).tolist() == list(range(len(lens)))

    gather = md.gather_idx.cpu()
    for i, n in enumerate(lens):
        shift = bucket - n
        row = gather[i]
        # valid region right-aligned: ends with the request's tokens in order
        assert row[shift:].tolist() == list(range(1000 * (i + 1), 1000 * (i + 1) + n))
        # left padding clamps to the request's own first token (finite K/V)
        assert (row[:shift] == 1000 * (i + 1)).all()


@requires_cuda
def test_decode_metadata_cute_left_aligned(backend_and_ctx):
    backend, ctx, device = backend_and_ctx
    backend.decode_impl = "cute"
    lens = [3, 64]
    reqs = [make_req(n, n - 1, i, uid=i) for i, n in enumerate(lens)]
    for i, n in enumerate(lens):
        _fill_page_table(ctx, i, n, 500 * (i + 1))
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    backend.prepare_metadata(batch)
    md = batch.attn_metadata

    assert md.mode == "decode"
    assert md.kv_bucket_len == 64
    assert md.cache_start is None
    gather = md.gather_idx.cpu()
    for i, n in enumerate(lens):
        row = gather[i]
        assert row[:n].tolist() == list(range(500 * (i + 1), 500 * (i + 1) + n))
        # right padding clamps to the request's own last token (finite K/V)
        assert (row[n:] == 500 * (i + 1) + n - 1).all()


@requires_cuda
def test_prefill_varlen_metadata(backend_and_ctx):
    backend, ctx, _ = backend_and_ctx
    lens = [3, 7]
    reqs = [make_req(n, 0, i, uid=i) for i, n in enumerate(lens)]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    backend.prepare_metadata(batch)
    md = batch.attn_metadata

    assert md.mode == "prefill_varlen"
    assert md.cu_seqlens_q.tolist() == [0, 3, 10]
    assert md.cu_seqlens_k.tolist() == [0, 3, 10]
    assert md.get_last_indices(2).tolist() == [2, 9]


@requires_cuda
def test_extend_metadata(backend_and_ctx):
    backend, ctx, _ = backend_and_ctx
    # one chunked/radix-hit request (cached prefix) + one fresh request
    reqs = [make_req(6, 2, 0, uid=0), make_req(4, 0, 1, uid=1)]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    backend.prepare_metadata(batch)
    md = batch.attn_metadata

    assert md.mode == "extend"
    assert md.cu_seqlens_q.tolist() == [0, 4, 8]
    assert md.seqlens_k_cpu == [6, 4]
    assert md.get_last_indices(2).tolist() == [3, 7]


@requires_cuda
def test_single_token_prefix_hit_routes_to_decode(backend_and_ctx):
    # radix hit on all but the last token: prefill phase but extend_len == 1
    backend, ctx, _ = backend_and_ctx
    backend.decode_impl = "fla"
    reqs = [make_req(9, 8, 0, uid=0)]
    _fill_page_table(ctx, 0, 9, 100)
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.mode == "decode"

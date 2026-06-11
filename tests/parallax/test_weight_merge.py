"""Checkpoint-key merge and TP-shard behavior for parallax weights (CPU-only)."""

import torch
from minisgl.models.weight import (
    _MERGE_GROUPS,
    _PARALLAX_MERGE_GROUPS,
    _get_merge_info,
    _shard_tensor,
)


def _run_merge(pairs, groups):
    """Emulate load_weight's streaming merge accumulation over (name, tensor) pairs."""
    merge_buf = {}
    out = {}
    for name, tensor in pairs:
        if (info := _get_merge_info(name, groups)) is None:
            out[name] = tensor
            continue
        merged_key, slot, all_slots = info
        merge_buf.setdefault(merged_key, {})[slot] = tensor
        if not all(s in merge_buf[merged_key] for s in all_slots):
            continue
        parts = [merge_buf[merged_key][s] for s in all_slots]
        del merge_buf[merged_key]
        out[merged_key] = torch.cat(parts, dim=0)
    assert not merge_buf, f"incomplete merge groups: {list(merge_buf)}"
    return out


def test_parallax_qrkv_merge_key_and_order():
    hidden, qo_dim, kv_dim = 16, 8, 4
    prefix = "model.layers.0.self_attn"
    parts = {
        "q": torch.full((qo_dim, hidden), 0.0),
        "r": torch.full((qo_dim, hidden), 1.0),
        "k": torch.full((kv_dim, hidden), 2.0),
        "v": torch.full((kv_dim, hidden), 3.0),
    }
    # arrival order deliberately scrambled
    pairs = [(f"{prefix}.{s}_proj.weight", parts[s]) for s in ("v", "q", "k", "r")]
    out = _run_merge(pairs, _PARALLAX_MERGE_GROUPS)
    assert set(out) == {f"{prefix}.qrkv_proj.weight"}
    merged = out[f"{prefix}.qrkv_proj.weight"]
    assert merged.shape == (2 * qo_dim + 2 * kv_dim, hidden)
    expected = torch.cat([parts["q"], parts["r"], parts["k"], parts["v"]], dim=0)
    assert torch.equal(merged, expected)


def test_default_groups_unchanged():
    hidden, qo_dim, kv_dim = 16, 8, 4
    prefix = "model.layers.0.self_attn"
    pairs = [
        (f"{prefix}.q_proj.weight", torch.zeros(qo_dim, hidden)),
        (f"{prefix}.k_proj.weight", torch.ones(kv_dim, hidden)),
        (f"{prefix}.v_proj.weight", torch.full((kv_dim, hidden), 2.0)),
    ]
    out = _run_merge(pairs, _MERGE_GROUPS)
    assert set(out) == {f"{prefix}.qkv_proj.weight"}
    assert out[f"{prefix}.qkv_proj.weight"].shape == (qo_dim + 2 * kv_dim, hidden)
    # r_proj is not part of the default (non-parallax) groups
    assert _get_merge_info("x.r_proj.weight", _MERGE_GROUPS) is None


def test_mlp_merge_same_in_parallax_groups():
    pairs = [
        ("model.layers.0.mlp.gate_proj.weight", torch.zeros(4, 2)),
        ("model.layers.0.mlp.up_proj.weight", torch.ones(4, 2)),
    ]
    out = _run_merge(pairs, _PARALLAX_MERGE_GROUPS)
    assert set(out) == {"model.layers.0.mlp.gate_up_proj.weight"}


def test_norm_keys_pass_through():
    for s in ("q_norm", "r_norm", "k_norm"):
        key = f"model.layers.0.self_attn.{s}.weight"
        assert _get_merge_info(key, _PARALLAX_MERGE_GROUPS) is None


def test_r_proj_shards_dim0_like_q():
    hidden, num_heads, head_dim, tp = 16, 4, 4, 2
    w = torch.arange(num_heads * head_dim * hidden, dtype=torch.float32)
    w = w.view(num_heads * head_dim, hidden)
    for key in ("a.q_proj.weight", "a.r_proj.weight"):
        for rank in range(tp):
            shard = _shard_tensor(key, w, rank, tp, num_kv_heads=2)
            assert torch.equal(shard, w.chunk(tp, dim=0)[rank])


def test_kv_proj_gqa_replication_unaffected():
    # num_kv_heads < tp: each rank gets a (replicated) single-head slice
    hidden, num_kv_heads, head_dim, tp = 16, 2, 4, 4
    w = torch.randn(num_kv_heads * head_dim, hidden)
    for rank in range(tp):
        shard = _shard_tensor("a.k_proj.weight", w, rank, tp, num_kv_heads=num_kv_heads)
        head_idx = rank * num_kv_heads // tp
        assert torch.equal(shard, w[head_idx * head_dim : (head_idx + 1) * head_dim])

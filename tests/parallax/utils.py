"""Shared helpers for parallax tests."""

from __future__ import annotations

import torch
from minisgl.core import Context, Req, SamplingParams
from minisgl.models.config import ModelConfig, RotaryConfig


def tiny_parallax_config(
    num_layers: int = 2,
    num_qo_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    hidden_size: int = 128,
    vocab_size: int = 256,
    window_size: int | None = None,
) -> ModelConfig:
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        intermediate_size=2 * hidden_size,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=4096,
            base=10000.0,
            scaling=None,
        ),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="qwen3",
        architectures=["ParallaxQwen3ForCausalLM"],
        window_size=window_size,
    )


def install_ctx(ctx: Context) -> None:
    """Install (or replace) the global context; tests may re-install freely."""
    import minisgl.core as core

    core._GLOBAL_CTX = ctx


def setup_ctx(
    config: ModelConfig,
    device: torch.device,
    *,
    num_tokens: int = 8192,
    max_running_req: int = 16,
    table_width: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
    page_size: int = 1,
) -> Context:
    from minisgl.kvcache.mha_pool import MHAKVCache

    ctx = Context(page_size=page_size)
    ctx.kv_cache = MHAKVCache(
        num_kv_heads=config.num_kv_heads,
        num_layers=config.num_layers,
        head_dim=config.head_dim,
        num_pages=num_tokens // page_size + 1,  # +1 dummy page, mirroring the engine
        page_size=page_size,
        dtype=dtype,
        device=device,
    )
    ctx.page_table = torch.zeros(
        (max_running_req + 1, table_width), dtype=torch.int32, device=device
    )
    install_ctx(ctx)
    return ctx


def make_req(device_len: int, cached_len: int, table_idx: int, uid: int = 0) -> Req:
    return Req(
        input_ids=torch.zeros(device_len, dtype=torch.int32, device="cpu"),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=8,
        uid=uid,
        sampling_params=SamplingParams(),
        cache_handle=None,  # type: ignore[arg-type]
    )

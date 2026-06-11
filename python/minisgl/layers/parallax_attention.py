from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class ParallaxAttentionLayer(StateLessOP):
    """Attention layer for parallax models (softmax attention + secondary query R).

    Splits the merged [q | r | k | v] projection, applies the per-head norms and
    RoPE (`r` is rotated with the same cos/sin as `q`, matching
    fla.layers.parallax), then dispatches to the parallax attention backend,
    which needs all four tensors.
    """

    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        r_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.r_norm = r_norm
        self.k_norm = k_norm

    def forward(self, qrkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qo, kv = self.qo_attn_dim, self.kv_attn_dim
        q, r, k, v = qrkv.split([qo, qo, kv, kv], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.r_norm is not None:
            self.r_norm.forward_inplace(r.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        # One fused RoPE call on the [q | r] slice: rope is applied per head, so
        # treating q and r as a single 2*H_q-head query rotates both with the
        # identical per-position cos/sin.
        qr = qrkv[:, : 2 * qo]
        qr, k = self.rotary.forward(ctx.batch.positions, qr, k)
        q = qr[:, :qo].view(-1, self.num_qo_heads, self.head_dim)
        r = qr[:, qo:].view(-1, self.num_qo_heads, self.head_dim)
        backend = ctx.attn_backend
        forward_parallax = getattr(backend, "forward_parallax", None)
        if forward_parallax is None:
            raise RuntimeError(
                f"Parallax models require the parallax attention backend, "
                f"got {type(backend).__name__}. Launch with --attention-backend parallax."
            )
        o = forward_parallax(q, r, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)

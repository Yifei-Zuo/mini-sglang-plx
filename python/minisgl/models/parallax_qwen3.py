"""Qwen3-style model with parallax attention.

Checkpoint convention (HF safetensors):
- config.json: `architectures: ["ParallaxQwen3ForCausalLM"]` with an
  AutoConfig-compatible `model_type` (e.g. "qwen3"); standard Qwen3 fields
  (num_attention_heads, num_key_value_heads, head_dim, ...), plus an optional
  `window_size` (int, fla semantics: attend to the most recent `window_size`
  keys including the current one; absent/null = full causal attention).
- weights: Qwen3 keys, with the attention projections extended by the
  secondary query stream `r` (mirrors fla.layers.parallax naming):
    model.layers.N.self_attn.{q_proj,r_proj,k_proj,v_proj,o_proj}.weight
    model.layers.N.self_attn.{q_norm,r_norm,k_norm}.weight
  The loader fuses q/r/k/v into `self_attn.qrkv_proj` (row order [q | r | k | v]).

Checkpoints trained with fla's `ParallaxForCausalLM` (`attn.` prefixes) need a
one-off key-rename conversion to this convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import (
    BaseOP,
    LinearOProj,
    LinearQRKVMerged,
    OPList,
    ParallaxAttentionLayer,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as ParallaxQwen3MLP

if TYPE_CHECKING:
    from .config import ModelConfig


class ParallaxQwen3Attn(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        head_dim = config.head_dim
        self.qrkv_proj = LinearQRKVMerged(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=False,
        )
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.r_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.attn = ParallaxAttentionLayer(
            layer_id=layer_id,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=head_dim,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            r_norm=self.r_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("ParallaxAttn")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qrkv = self.qrkv_proj.forward(x)
        del x
        o = self.attn.forward(qrkv)
        return self.o_proj.forward(o)


class ParallaxQwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = ParallaxQwen3Attn(config, layer_id)
        self.mlp = ParallaxQwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class ParallaxQwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [ParallaxQwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class ParallaxQwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = ParallaxQwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["ParallaxQwen3ForCausalLM"]

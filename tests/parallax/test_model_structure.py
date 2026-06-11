"""Model structure: state_dict keys must equal the post-merge checkpoint keys (CPU/meta)."""

import torch
from minisgl.layers import set_rope_device

from .utils import tiny_parallax_config


def test_state_dict_keys_match_checkpoint_convention():
    from minisgl.models.parallax_qwen3 import ParallaxQwen3ForCausalLM

    set_rope_device(torch.device("cpu"))
    config = tiny_parallax_config(num_layers=2)
    with torch.device("meta"):
        model = ParallaxQwen3ForCausalLM(config)
    keys = set(model.state_dict().keys())

    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(config.num_layers):
        p = f"model.layers.{i}"
        expected |= {
            f"{p}.self_attn.qrkv_proj.weight",
            f"{p}.self_attn.q_norm.weight",
            f"{p}.self_attn.r_norm.weight",
            f"{p}.self_attn.k_norm.weight",
            f"{p}.self_attn.o_proj.weight",
            f"{p}.mlp.gate_up_proj.weight",
            f"{p}.mlp.down_proj.weight",
            f"{p}.input_layernorm.weight",
            f"{p}.post_attention_layernorm.weight",
        }
    assert keys == expected, (
        f"missing: {sorted(expected - keys)}\nunexpected: {sorted(keys - expected)}"
    )

    qrkv = model.state_dict()["model.layers.0.self_attn.qrkv_proj.weight"]
    qo = config.num_qo_heads * config.head_dim
    kv = config.num_kv_heads * config.head_dim
    assert qrkv.shape == (2 * qo + 2 * kv, config.hidden_size)


def test_config_is_parallax():
    config = tiny_parallax_config()
    assert config.is_parallax
    assert not config.is_moe

    plain = tiny_parallax_config()
    object.__setattr__(plain, "architectures", ["Qwen3ForCausalLM"])
    assert not plain.is_parallax


def test_model_registry_resolves():
    from minisgl.models.register import _MODEL_REGISTRY

    assert "ParallaxQwen3ForCausalLM" in _MODEL_REGISTRY

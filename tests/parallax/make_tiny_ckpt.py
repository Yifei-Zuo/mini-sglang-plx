"""Build a tiny random-weight ParallaxQwen3 checkpoint (HF layout) for tests."""

from __future__ import annotations

import json
import os

import torch

BASE_CONFIG = {
    "architectures": ["ParallaxQwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 256,
    "vocab_size": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
    "eos_token_id": 0,
    "bos_token_id": 1,
}


def build_tiny_ckpt(path: str, seed: int = 1234, window_size: int | None = None) -> dict:
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    cfg = dict(BASE_CONFIG)
    if window_size is not None:
        cfg["window_size"] = window_size
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    g = torch.Generator().manual_seed(seed)
    hid, inter, vocab = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    hq, hkv, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]

    def w(out_dim: int, in_dim: int, std: float = 0.05) -> torch.Tensor:
        return (torch.randn((out_dim, in_dim), generator=g) * std).to(torch.bfloat16)

    def norm_w(dim: int) -> torch.Tensor:
        return (1.0 + 0.1 * torch.randn((dim,), generator=g)).to(torch.bfloat16)

    sd = {"model.embed_tokens.weight": w(vocab, hid, std=0.1)}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        sd[f"{p}.self_attn.q_proj.weight"] = w(hq * d, hid)
        sd[f"{p}.self_attn.r_proj.weight"] = w(hq * d, hid)
        sd[f"{p}.self_attn.k_proj.weight"] = w(hkv * d, hid)
        sd[f"{p}.self_attn.v_proj.weight"] = w(hkv * d, hid)
        sd[f"{p}.self_attn.o_proj.weight"] = w(hid, hq * d)
        sd[f"{p}.self_attn.q_norm.weight"] = norm_w(d)
        sd[f"{p}.self_attn.r_norm.weight"] = norm_w(d)
        sd[f"{p}.self_attn.k_norm.weight"] = norm_w(d)
        sd[f"{p}.mlp.gate_proj.weight"] = w(inter, hid)
        sd[f"{p}.mlp.up_proj.weight"] = w(inter, hid)
        sd[f"{p}.mlp.down_proj.weight"] = w(hid, inter)
        sd[f"{p}.input_layernorm.weight"] = norm_w(hid)
        sd[f"{p}.post_attention_layernorm.weight"] = norm_w(hid)
    sd["model.norm.weight"] = norm_w(hid)
    sd["lm_head.weight"] = w(vocab, hid, std=0.1)
    save_file(sd, os.path.join(path, "model.safetensors"))

    _write_tokenizer(path, vocab)
    return cfg


def _write_tokenizer(path: str, vocab_size: int) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"<eos>": 0, "<bos>": 1}
    for i in range(2, vocab_size):
        vocab[f"t{i}"] = i
    tok = Tokenizer(WordLevel(vocab, unk_token="<eos>"))
    tok.pre_tokenizer = Whitespace()
    tok.save(os.path.join(path, "tokenizer.json"))
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "<eos>",
                "bos_token": "<bos>",
                "unk_token": "<eos>",
                "model_max_length": 4096,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    import sys

    build_tiny_ckpt(sys.argv[1] if len(sys.argv) > 1 else "/tmp/parallax_tiny_ckpt")

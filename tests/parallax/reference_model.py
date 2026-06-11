"""Pure-torch fp32 reference for the tiny ParallaxQwen3 checkpoint.

Mirrors the serving model's exact dataflow (pre-norm residuals, per-head q/r/k
norms before RoPE, neox-style rotary matching flashinfer's
apply_rope_with_cos_sin_cache_inplace, parallax attention via
fla.ops.parallax.naive.naive_parallax) so greedy generations are comparable.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class TinyParallaxReference:
    def __init__(self, ckpt_dir: str, device: torch.device | str = "cuda"):
        from safetensors.torch import load_file

        with open(os.path.join(ckpt_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.device = torch.device(device)
        sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        self.sd = {k: v.to(self.device, torch.float32) for k, v in sd.items()}
        self.eps = self.cfg["rms_norm_eps"]
        self.window_size = self.cfg.get("window_size")
        d = self.cfg["head_dim"]
        inv_freq = 1.0 / (
            self.cfg["rope_theta"] ** (torch.arange(0, d, 2, dtype=torch.float32) / d)
        )
        t = torch.arange(self.cfg["max_position_embeddings"], dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.cos = freqs.cos().to(self.device)  # (max_pos, d/2)
        self.sin = freqs.sin().to(self.device)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (T, H, D); neox half-split convention
        d = x.shape[-1]
        cos = self.cos[positions].unsqueeze(1)  # (T, 1, d/2)
        sin = self.sin[positions].unsqueeze(1)
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    @torch.no_grad()
    def last_logits(self, ids: list[int]) -> torch.Tensor:
        from fla.ops.parallax.naive import naive_parallax

        cfg, sd = self.cfg, self.sd
        hq, hkv, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        T = len(ids)
        positions = torch.arange(T, device=self.device)
        idx = torch.tensor(ids, device=self.device, dtype=torch.long)
        x = sd["model.embed_tokens.weight"][idx]
        residual = None
        for i in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{i}"
            if residual is None:
                residual = x
                h = _rmsnorm(x, sd[f"{p}.input_layernorm.weight"], self.eps)
            else:
                residual = residual + x
                h = _rmsnorm(residual, sd[f"{p}.input_layernorm.weight"], self.eps)

            q = (h @ sd[f"{p}.self_attn.q_proj.weight"].T).view(T, hq, d)
            r = (h @ sd[f"{p}.self_attn.r_proj.weight"].T).view(T, hq, d)
            k = (h @ sd[f"{p}.self_attn.k_proj.weight"].T).view(T, hkv, d)
            v = (h @ sd[f"{p}.self_attn.v_proj.weight"].T).view(T, hkv, d)
            q = _rmsnorm(q, sd[f"{p}.self_attn.q_norm.weight"], self.eps)
            r = _rmsnorm(r, sd[f"{p}.self_attn.r_norm.weight"], self.eps)
            k = _rmsnorm(k, sd[f"{p}.self_attn.k_norm.weight"], self.eps)
            q, r, k = (self._rope(t, positions) for t in (q, r, k))

            o = naive_parallax(
                q.unsqueeze(0),
                r.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                scale=d**-0.5,
                window_size=self.window_size,
                causal=True,
            ).squeeze(0)
            x = o.reshape(T, hq * d) @ sd[f"{p}.self_attn.o_proj.weight"].T

            residual = residual + x
            h = _rmsnorm(residual, sd[f"{p}.post_attention_layernorm.weight"], self.eps)
            gate = h @ sd[f"{p}.mlp.gate_proj.weight"].T
            up = h @ sd[f"{p}.mlp.up_proj.weight"].T
            x = (F.silu(gate) * up) @ sd[f"{p}.mlp.down_proj.weight"].T

        residual = residual + x
        h = _rmsnorm(residual, sd["model.norm.weight"], self.eps)
        return h[-1] @ sd["lm_head.weight"].T

    @torch.no_grad()
    def greedy_generate(self, prompt_ids: list[int], num_tokens: int) -> list[int]:
        ids = list(prompt_ids)
        out = []
        for _ in range(num_tokens):
            nxt = int(self.last_logits(ids).argmax().item())
            out.append(nxt)
            ids.append(nxt)
        return out

"""Attention backend for parallax models.

Parallax is softmax attention with a secondary query stream R (see
fla.ops.parallax.naive.naive_parallax for the math). It keeps a standard K/V
cache — R is per-query and never cached — so the regular paged pool, page
table, and prefix caching all apply. What differs from FA/FI backends:

- the model layer hands the backend four tensors (q, r, k, v) via
  ``forward_parallax``; the base-class ``forward(q, k, v)`` is unsupported;
- the parallax kernels need contiguous per-request K/V, so decode gathers each
  request's tokens from the paged pool into a padded ``(B, L_bucket, H_kv, D)``
  buffer (kept persistent so the CuTeDSL kernel's input cache, keyed on
  ``data_ptr``, stays bounded);
- prefill/extend run on the FLA Triton kernels; decode runs on the CuTeDSL
  SM90 kernel when available (``MINISGL_PARALLAX_DECODE_IMPL=auto|cute|fla``).

Padding contract: the gather builds indices clamped into each request's own
valid tokens, so the padded region contains real (finite) K/V values. The
kernels mask padded columns from the softmax branch, but the unmasked
``r @ k`` branch multiplies them by zero — which is only NaN-safe if the
padding is finite. Never point padding at uninitialized pool slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.env import ENV
from minisgl.utils import div_even, init_logger, is_sm90_supported

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

logger = init_logger(__name__)

# Smallest KV bucket; buckets match the CuTeDSL decode kernel's _KV_BUCKETS
# so gathered buffers never need re-padding by the kernel.
_MIN_KV_BUCKET = 64


def _kv_bucket(max_len: int) -> int:
    """Round ``max_len`` up to the KV bucket grid: powers of two up to 1024,
    then multiples of 1024 up to 16384, then multiples of 4096 beyond (the
    cute kernel precompiles these up to 131072). Every bucket is a multiple
    of 64 (the kernel tile size). MUST agree exactly with _KV_BUCKETS /
    _snap_kv_len in parallax.cute.parallax_decode."""
    if max_len <= 1024:
        return max(_MIN_KV_BUCKET, 1 << (max_len - 1).bit_length())
    if max_len <= 16384:
        return (max_len + 1023) // 1024 * 1024
    return (max_len + 4095) // 4096 * 4096


def _bs_bucket(bs: int) -> int:
    """Pad decode batch size to a power of two: the CuTeDSL compile cache keys
    on B, so bucketing bounds the number of kernel variants."""
    return 1 << (bs - 1).bit_length() if bs > 1 else 1


@dataclass
class ParallaxMetadata(BaseAttnMetadata):
    mode: Literal["decode", "prefill_varlen", "extend"]
    cu_seqlens_q: torch.Tensor  # (padded_bs + 1,) int32, device
    seqlens_q_cpu: List[int]
    seqlens_k_cpu: List[int]
    # decode only; decode_rows >= batch size when the cute path pads B to a bucket
    decode_rows: int = 0
    seqused_k: torch.Tensor | None = None  # (decode_rows,) int32, device
    kv_bucket_len: int = 0
    gather_idx: torch.Tensor | None = None  # (decode_rows, kv_bucket_len) int32, device
    cache_start: torch.Tensor | None = None  # (decode_rows,) int32; right-aligned layout (fla)
    # paged cute decode (page_size == 64): pool page index per (row, tile);
    # every entry is clamped to a valid page (incl. tiles past seqused and
    # B-bucket pad rows). gather_idx is None in this mode.
    page_entries: torch.Tensor | None = None  # (decode_rows, kv_bucket_len // 64) int32
    # prefill_varlen only
    cu_seqlens_k: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class ParallaxBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        from minisgl.distributed import get_tp_info

        ctx = get_global_ctx()
        assert ctx.page_size in (1, 64), "parallax backend requires page_size 1 or 64"
        self.page_size = ctx.page_size
        self.config = config
        self.kvcache = ctx.kv_cache
        self.scale = config.head_dim**-0.5
        self.window_size = config.window_size
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim = config.head_dim
        self.decode_impl = self._select_decode_impl()
        logger.info_rank0(f"parallax decode kernel: {self.decode_impl}")
        # Persistent gather arenas, flat (capacity, H_kv, D), grown geometrically and
        # reused across layers/steps. Stable storage is required by the CuTeDSL
        # kernel's input cache (keyed on data_ptr).
        self._k_arena: torch.Tensor | None = None
        self._v_arena: torch.Tensor | None = None
        # Persistent decode staging buffers for the CuTeDSL path (q/r/out per
        # row, per-request lengths); zero-initialized so pad rows are finite.
        # q/r/out are written on the engine stream inside the forward itself, so
        # one buffer suffices; seqused is written in prepare_metadata on the
        # scheduler stream while the PREVIOUS batch may still be running on the
        # engine stream — ping-pong two buffers (only one batch is ever in
        # flight under overlap scheduling).
        self._qro_bufs: torch.Tensor | None = None  # (3, capacity, 1, H_q, D)
        self._seqused_bufs: list[torch.Tensor | None] = [None, None]
        self._seqused_flip = 0
        # Paged cute decode: at page_size == 64 (one pool page == one kernel KV
        # tile) the kernel reads the pool directly via TMA — no gather at all.
        self.paged = self.decode_impl == "cute" and self.page_size == 64
        # Page-table staging, ping-pong like seqused (written in
        # prepare_metadata on the scheduler stream while the previous batch may
        # still be running). Strides are baked into the kernel compile key, so
        # buffers are allocated at fixed capacity to avoid stride churn.
        self._pt_bufs: list[torch.Tensor | None] = [None, None]
        self._pt_flip = 0
        if self.paged:
            # Finite-tails contract: page tails past each request's seqused are
            # read by the kernel (masked from softmax, but multiplied by zero in
            # the unmasked R·K branch — 0 * NaN poisons the output). A one-time
            # zero-init keeps every recycled page finite forever.
            for layer in range(config.num_layers):
                self.kvcache.k_cache(layer).zero_()
                self.kvcache.v_cache(layer).zero_()
        if self.decode_impl == "cute":
            self._maybe_warmup_cute()

    @staticmethod
    def _select_decode_impl() -> str:
        choice = ENV.PARALLAX_DECODE_IMPL.value
        if choice not in ("auto", "cute", "fla"):
            logger.warning(f"Unknown MINISGL_PARALLAX_DECODE_IMPL={choice!r}, using 'auto'")
            choice = "auto"
        if choice == "fla":
            return "fla"
        cute_err = None
        try:
            from parallax.cute.parallax_decode import parallax_decode_serving  # noqa: F401

            cute_ok = is_sm90_supported()
            if not cute_ok:
                cute_err = "CuTeDSL parallax decode requires SM90 (Hopper)"
        except ImportError as e:
            cute_ok, cute_err = False, f"cannot import parallax_decode_serving: {e}"
        if choice == "cute" and not cute_ok:
            raise RuntimeError(f"MINISGL_PARALLAX_DECODE_IMPL=cute but {cute_err}")
        if choice == "auto" and not cute_ok:
            logger.info(f"Falling back to fla decode kernel ({cute_err})")
        return "cute" if cute_ok else "fla"

    def _maybe_warmup_cute(self) -> None:
        """Pre-compile CuTeDSL decode variants (opt-in via MINISGL_PARALLAX_CUTE_WARMUP).

        Each (B bucket, KV bucket) pair is a separate JIT compile taking tens of
        seconds; without warmup the first decode step hitting a new shape stalls.
        """
        spec = ENV.PARALLAX_CUTE_WARMUP.value
        if not spec:
            return
        from parallax.cute.parallax_decode import _KV_BUCKETS, parallax_decode_serving

        max_bs, max_len = (int(x) for x in spec.split(":"))
        bs_list = [b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256) if b <= _bs_bucket(max_bs)]
        kv_list = [b for b in _KV_BUCKETS if b <= _kv_bucket(max_len)]
        device, dtype = self.kvcache.device, self.kvcache.dtype
        window_left = -1 if self.window_size is None else self.window_size
        if not self.paged:
            self._ensure_arena(max(bs_list) * max(kv_list), dtype, device)
            self._k_arena.zero_()  # warmup reads the arena before any gather: must be finite
            self._v_arena.zero_()
        logger.info_rank0(
            f"parallax cute warmup ({'paged' if self.paged else 'gather'}): compiling "
            f"{len(bs_list) * len(kv_list)} variants (bs {bs_list} x kv {kv_list})"
        )
        # descending: the largest allocation happens first, so every smaller
        # bucket is a slice of the same buffers (stable data_ptrs, shared with
        # runtime use)
        for bs in sorted(bs_list, reverse=True):
            staged = self._staged_qro(bs, dtype, device)
            seqused = self._staged_seqused(bs, device)  # ones: valid minimal lengths
            kv_shape_tail = (self.num_kv_heads, self.head_dim)
            for kv in kv_list:
                if self.paged:
                    from parallax.cute.parallax_decode import parallax_decode_serving_paged

                    # staged pt zeros -> page 0, valid; pool was zero-inited
                    pt = self._staged_pt(bs, kv // 64, device)
                    parallax_decode_serving_paged(
                        staged[0], staged[1],
                        self.kvcache.k_cache(0), self.kvcache.v_cache(0),
                        self.scale,
                        page_table=pt, seqused_k=seqused, kv_len_bucket=kv,
                        window_size_left=window_left,
                        out=staged[2],
                    )
                else:
                    n = bs * kv
                    k_g = self._k_arena[:n].view(bs, kv, *kv_shape_tail)
                    v_g = self._v_arena[:n].view(bs, kv, *kv_shape_tail)
                    parallax_decode_serving(
                        staged[0], staged[1], k_g, v_g, self.scale,
                        seqused_k=seqused,
                        window_size_left=window_left,
                        out=staged[2],
                    )
        torch.cuda.synchronize(device)
        logger.info_rank0("parallax cute warmup done")

    # ------------------------------------------------------------------ #
    # BaseAttnBackend interface
    # ------------------------------------------------------------------ #

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        raise RuntimeError(
            "The parallax backend only serves parallax models, whose attention layers "
            "call forward_parallax(q, r, k, v, ...). A non-parallax model selected this "
            "backend — launch it with fa/fi/trtllm instead."
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        padded_bs = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        device = self.kvcache.device
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        if all(c == 0 for c in cached_lens):
            # Pure prefill: every request attends to exactly its own new tokens,
            # which is full causal self-attention over a packed varlen batch.
            cu_seqlens = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens = cu_seqlens.to(torch.int32).to(device, non_blocking=True)
            batch.attn_metadata = ParallaxMetadata(
                mode="prefill_varlen",
                cu_seqlens_q=cu_seqlens,
                seqlens_q_cpu=seqlens_q,
                seqlens_k_cpu=seqlens_k,
                cu_seqlens_k=cu_seqlens,
            )
            return

        if max_seqlen_q == 1:
            # Decode (single new token per request over its full cache). Build
            # per-request gather indices into the token-granular pool. Padded
            # columns are clamped into the request's own valid tokens so the
            # gathered padding is finite (see module docstring).
            max_seqlen_k = max(seqlens_k)
            bucket = _kv_bucket(max_seqlen_k)
            rows_k = list(seqlens_k)
            table_idxs = [req.table_idx for req in reqs]
            if self.decode_impl == "cute":
                # Pad B to a bucket so the CuTeDSL compile cache stays bounded.
                # Pad rows reuse request 0's table with seqused 1: the kernel
                # reads exactly one real (finite) token and the rows are
                # dropped before returning to the model.
                rows = _bs_bucket(padded_bs)
                rows_k += [1] * (rows - padded_bs)
                table_idxs += [table_idxs[0]] * (rows - padded_bs)
            else:
                rows = padded_bs
            seqused_cpu = torch.tensor(rows_k, **CPU_KWARGS)
            seqused_k = self._staged_seqused(rows, device).copy_(seqused_cpu, non_blocking=True)
            table_idx = torch.tensor(
                table_idxs, device="cpu", dtype=torch.int64, pin_memory=True
            ).to(device, non_blocking=True)
            page_table = get_global_ctx().page_table
            seqused_64 = seqused_k.to(torch.int64).unsqueeze(1)
            cache_start = None
            gather_idx = None
            page_entries = None
            if self.paged:
                # Pool page index per (row, tile): token at sequence position p
                # lives in pool page global_table[idx, p] // 64 (the allocator
                # fills pages sequentially, so one tile == one page). Clamping
                # the probe position to seqused-1 makes every entry — tiles
                # past seqused and B-bucket pad rows alike — a valid page.
                tiles = bucket // 64
                cols_t = (torch.arange(tiles, device=device, dtype=torch.int64) * 64).unsqueeze(0)
                src = torch.minimum(cols_t, seqused_64 - 1)
                flat_pos = table_idx.unsqueeze(1) * page_table.shape[1] + src
                entries = page_table.view(-1)[flat_pos] // 64
                page_entries = self._staged_pt(rows, tiles, device)
                page_entries.copy_(entries)
            else:
                cols = torch.arange(bucket, device=device, dtype=torch.int64).unsqueeze(0)
                if self.decode_impl == "fla":
                    # Right-aligned (left-padded) layout: the fla onestep kernel
                    # masks [0, cache_start) per request. Left padding clamps to
                    # token 0.
                    shift = bucket - seqused_64
                    src = (cols - shift).clamp_(min=0)
                    cache_start = shift.to(torch.int32).squeeze(1)
                else:
                    # Left-aligned layout for the CuTeDSL gather path (masks via
                    # seqused_k); right padding clamps to the last valid token.
                    src = torch.minimum(cols, seqused_64 - 1)
                flat_pos = table_idx.unsqueeze(1) * page_table.shape[1] + src
                gather_idx = page_table.view(-1)[flat_pos]
            batch.attn_metadata = ParallaxMetadata(
                mode="decode",
                cu_seqlens_q=torch.arange(padded_bs + 1, device=device, dtype=torch.int32),
                seqlens_q_cpu=seqlens_q,
                seqlens_k_cpu=seqlens_k,
                decode_rows=rows,
                seqused_k=seqused_k,
                kv_bucket_len=bucket,
                gather_idx=gather_idx,
                cache_start=cache_start,
                page_entries=page_entries,
            )
            return

        # Extend (chunked prefill or prefix-cache hit): new tokens attend to the
        # cached prefix plus themselves; handled per request in the forward.
        cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_q = cu_seqlens_q.to(torch.int32).to(device, non_blocking=True)
        batch.attn_metadata = ParallaxMetadata(
            mode="extend",
            cu_seqlens_q=cu_seqlens_q,
            seqlens_q_cpu=seqlens_q,
            seqlens_k_cpu=seqlens_k,
        )

    # ------------------------------------------------------------------ #
    # Parallax forward (called by ParallaxAttentionLayer)
    # ------------------------------------------------------------------ #

    def forward_parallax(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, ParallaxMetadata)
        # Store first: out_loc was allocated before the forward, so gathers over
        # [0, device_len) below already include this step's tokens.
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        if metadata.mode == "decode":
            return self._forward_decode(q, r, layer_id, metadata, batch.padded_size)
        if metadata.mode == "prefill_varlen":
            return self._forward_prefill_varlen(q, r, k, v, metadata)
        return self._forward_extend(q, r, layer_id, metadata, batch)

    def _flat_kv(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (-1, self.num_kv_heads, self.head_dim)
        return self.kvcache.k_cache(layer_id).view(shape), self.kvcache.v_cache(layer_id).view(shape)

    def _ensure_arena(self, num_tokens: int, dtype: torch.dtype, device: torch.device) -> None:
        if self._k_arena is not None and self._k_arena.shape[0] >= num_tokens:
            return
        cap = num_tokens
        if self._k_arena is not None:
            cap = max(cap, 2 * self._k_arena.shape[0])
        shape = (cap, self.num_kv_heads, self.head_dim)
        self._k_arena = torch.empty(shape, dtype=dtype, device=device)
        self._v_arena = torch.empty(shape, dtype=dtype, device=device)

    def _staged_seqused(self, rows: int, device: torch.device) -> torch.Tensor:
        self._seqused_flip ^= 1
        buf = self._seqused_bufs[self._seqused_flip]
        if buf is None or buf.shape[0] < rows:
            cap = max(rows, 2 * buf.shape[0] if buf is not None else 0)
            buf = torch.ones(cap, dtype=torch.int32, device=device)
            self._seqused_bufs[self._seqused_flip] = buf
        return buf[:rows]

    def _staged_pt(self, rows: int, tiles: int, device: torch.device) -> torch.Tensor:
        """Ping-pong page-table staging slice (rows, tiles), zero-initialized.

        Fixed generous capacity: the slice's row stride is baked into the cute
        compile key, so growth (which changes the stride) would trigger fresh
        JIT variants. 256 rows x 2048 tiles (128k tokens) = 2 MB per buffer.
        """
        self._pt_flip ^= 1
        buf = self._pt_bufs[self._pt_flip]
        if buf is None or buf.shape[0] < rows or buf.shape[1] < tiles:
            cap_r = max(256, rows)
            cap_t = max(2048, tiles)
            buf = torch.zeros((cap_r, cap_t), dtype=torch.int32, device=device)
            self._pt_bufs[self._pt_flip] = buf
        return buf[:rows, :tiles]

    def _staged_qro(self, rows: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self._qro_bufs is None or self._qro_bufs.shape[1] < rows:
            cap = max(rows, 2 * self._qro_bufs.shape[1] if self._qro_bufs is not None else 0)
            # zeros, not empty: pad rows must hold finite values
            self._qro_bufs = torch.zeros(
                (3, cap, 1, self.num_qo_heads, self.head_dim), dtype=dtype, device=device
            )
        return self._qro_bufs[:, :rows]

    def _forward_decode(
        self, q: torch.Tensor, r: torch.Tensor, layer_id: int, md: ParallaxMetadata, bs: int
    ) -> torch.Tensor:
        bucket = md.kv_bucket_len
        rows = md.decode_rows  # == bs on the fla path; B-bucket-padded on cute

        if md.page_entries is not None:
            # Paged cute path: the kernel TMA-reads the pool directly — no
            # gather, no arena. q/r/out still stage into persistent buffers.
            from parallax.cute.parallax_decode import parallax_decode_serving_paged

            staged = self._staged_qro(rows, q.dtype, q.device)
            q_st, r_st, o_st = staged[0], staged[1], staged[2]
            q_st[:bs].copy_(q.unsqueeze(1))
            r_st[:bs].copy_(r.unsqueeze(1))
            window_left = -1 if self.window_size is None else self.window_size
            o = parallax_decode_serving_paged(
                q_st, r_st,
                self.kvcache.k_cache(layer_id),  # (num_pages, 64, H_kv, D)
                self.kvcache.v_cache(layer_id),
                self.scale,
                page_table=md.page_entries,
                seqused_k=md.seqused_k,
                kv_len_bucket=bucket,
                window_size_left=window_left,
                out=o_st,
            )
            return o.view(rows, self.num_qo_heads, self.head_dim)[:bs]

        flat_k, flat_v = self._flat_kv(layer_id)
        n = rows * bucket
        self._ensure_arena(n, flat_k.dtype, flat_k.device)
        idx = md.gather_idx.view(-1)
        torch.index_select(flat_k, 0, idx, out=self._k_arena[:n])
        torch.index_select(flat_v, 0, idx, out=self._v_arena[:n])
        kv_shape = (rows, bucket, self.num_kv_heads, self.head_dim)
        k_g = self._k_arena[:n].view(kv_shape)
        v_g = self._v_arena[:n].view(kv_shape)
        if self.decode_impl == "fla":
            from fla.ops.parallax import parallax_decode_onestep

            o = parallax_decode_onestep(
                q.unsqueeze(1),  # fla wrappers handle non-contiguous inputs
                r.unsqueeze(1),
                k_g,
                v_g,
                scale=self.scale,
                window_size=self.window_size,
                cache_start=md.cache_start,
            )
        else:
            from parallax.cute.parallax_decode import parallax_decode_serving

            # All six kernel-facing tensors are persistent backend-owned buffers
            # (the CuTeDSL input cache keys on data_ptr and holds strong refs).
            staged = self._staged_qro(rows, q.dtype, q.device)
            q_st, r_st, o_st = staged[0], staged[1], staged[2]
            q_st[:bs].copy_(q.unsqueeze(1))
            r_st[:bs].copy_(r.unsqueeze(1))
            # Despite the FA2-style name, the cute kernel's window_size_left keeps
            # exactly W keys ending at the diagonal ([seqused-W, seqused-1]) — the
            # same semantics as fla's window_size. No off-by-one adjustment.
            window_left = -1 if self.window_size is None else self.window_size
            o = parallax_decode_serving(
                q_st, r_st, k_g, v_g, self.scale,
                seqused_k=md.seqused_k,
                window_size_left=window_left,
                out=o_st,
            )
        return o.view(rows, self.num_qo_heads, self.head_dim)[:bs]

    def _forward_prefill_varlen(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        md: ParallaxMetadata,
    ) -> torch.Tensor:
        from fla.ops.parallax import parallel_parallax

        kv_shape = (-1, self.num_kv_heads, self.head_dim)
        o = parallel_parallax(
            q.unsqueeze(0),
            r.unsqueeze(0),
            k.view(kv_shape).unsqueeze(0),
            v.view(kv_shape).unsqueeze(0),
            scale=self.scale,
            window_size=self.window_size,
            cu_seqlens=md.cu_seqlens_k,
        )
        return o.view(-1, self.num_qo_heads, self.head_dim)

    def _forward_extend(
        self, q: torch.Tensor, r: torch.Tensor, layer_id: int, md: ParallaxMetadata, batch: Batch
    ) -> torch.Tensor:
        from fla.ops.parallax import parallax_decode as fla_parallax_decode

        flat_k, flat_v = self._flat_kv(layer_id)
        page_table = get_global_ctx().page_table
        out = torch.empty(
            (q.shape[0], self.num_qo_heads, self.head_dim), dtype=q.dtype, device=q.device
        )
        lo = 0
        for i, req in enumerate(batch.padded_reqs):
            sq, skv = md.seqlens_q_cpu[i], md.seqlens_k_cpu[i]
            idx = page_table[req.table_idx, :skv].to(torch.int64)
            k_g = flat_k.index_select(0, idx).unsqueeze(0)  # (1, skv, H_kv, D)
            v_g = flat_v.index_select(0, idx).unsqueeze(0)
            o_i = fla_parallax_decode(
                q[lo : lo + sq].unsqueeze(0),
                r[lo : lo + sq].unsqueeze(0),
                k_g,
                v_g,
                scale=self.scale,
                window_size=self.window_size,
            )
            out[lo : lo + sq] = o_i.view(sq, self.num_qo_heads, self.head_dim)
            lo += sq
        return out

    # ------------------------------------------------------------------ #
    # CUDA graphs: unsupported in v1 (run with --cuda-graph-max-bs 0; the
    # engine forces this for parallax models).
    # ------------------------------------------------------------------ #

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        raise NotImplementedError("parallax v1 does not support CUDA graphs")

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError("parallax v1 does not support CUDA graphs")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError("parallax v1 does not support CUDA graphs")

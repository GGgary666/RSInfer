# Copyright 2023-2026 SGLang Team
# Copyright contributors to the vLLM project (causal_conv1d_ref, causal_conv1d_update_ref)
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference kernels for Qwen3.8, ported from SGLang ``e838360`` (LICENSE.sglang).

Each function keeps the signature and the arithmetic of the SGLang code it comes from; only
the CUDA / CPU-extension dispatch is gone:
- ``gemma_rms_norm``: ``GemmaRMSNorm.forward_native`` (srt/layers/layernorm.py);
- ``rms_norm_gated``: ``rms_norm_ref`` (kernels/ops/attention/fla/layernorm_gated.py);
- ``rope``: ``RotaryEmbedding.forward_native`` and ``apply_rotary_emb``
  (srt/layers/rotary_embedding/base.py, utils.py);
- ``fused_qk_gemma_rmsnorm_rope_gate``: ``Qwen3_5AttentionDecoderLayer.forward_prepare_native``
  (srt/models/qwen3_5.py), behind the signature of the kernel its CUDA path calls;
- ``silu_and_mul``: ``SiluAndMul.forward_native``; ``fused_sigmoid_mul``: the attention output
  gate; ``linear``: ``UnquantizedLinearMethod``; ``embedding``: ``UnquantizedEmbeddingMethod``;
- ``kv_store``: the fallback of ``_set_kv_buffer_impl`` (srt/mem_cache/memory_pool.py);
- ``attention_extend`` / ``attention_decode``: ``TorchNativeAttnBackend._run_sdpa_forward_*``;
- GDN: the torch references that SGLang checks its CPU kernels against
  (test/registered/cpu/test_mamba.py, test/registered/layers/mamba/test_causal_conv1d.py),
  behind the pool-indexed signatures of the kernels ``GDNAttnBackend`` calls
  (``causal_conv1d_fn``, ``fused_gdn_gating``, ``chunk_gated_delta_rule``, and for a decode step
  ``fused_qkvzba_causal_conv1d_update`` and ``fused_recurrent_gated_delta_rule_packed_decode``,
  which unpack the projections around ``causal_conv1d_update`` and
  ``fused_sigmoid_gating_delta_rule_update``);
- ``sample``: the native path of ``Sampler.forward`` with ``multinomial_with_seed``
  (srt/layers/sampler.py); ``murmur_hash32`` is the Triton kernel of
  kernels/ops/sampling/murmur_hash.py written in torch.

State layouts are SGLang's. KV: one ``[slots, kv_heads, head_dim]`` tensor per layer and per K/V,
indexed by token slot (``MHATokenToKVPool``). GDN: per layer, conv ``[slots, conv_dim, K - 1]``
and temporal ``[slots, HV, V, K]`` (``MambaPool``; the temporal state is ``[V, K]``, the
transpose of HF's ``[K, V]``).

Batch invariance (``Definition.batch_invariant``): attention and the chunked delta rule loop over
sequences, and the KV store and the embedding only move data, so a row's result does not depend
on the other rows. The rest run one torch op over the whole batch, which may take another path
or accumulation order for another number of rows, so they do not claim it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention, softplus

TOP_K_ALL = 1 << 30
"""SGLang's ``top_k`` for the whole vocabulary (srt/sampling/sampling_params.py)."""
SAMPLING_EPS = 1e-6


def _on_device(values: Sequence[float] | Sequence[int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Per-request sampling values on ``device`` without waiting for the queued work: from pinned
    memory, ``non_blocking``, as SGLang's ``SamplingBatchInfo`` copies them."""
    if device.type != "cuda":
        return torch.tensor(list(values), dtype=dtype, device=device)
    return torch.tensor(list(values), dtype=dtype, pin_memory=True).to(device, non_blocking=True)


# ------------------------------------------------------------------ norms


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float,
                   residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """``norm.gemma_rms@1``: ``GemmaRMSNorm.forward_native``. With ``residual``, adds it first and
    returns ``(normed, x + residual)``. SGLang's CUDA path computes the two in place
    (``gemma_fused_add_rmsnorm``: ``residual`` becomes the sum, ``x`` its norm), so the definition
    lets a solution overwrite ``x`` and ``residual``: a caller uses what is returned, not what it
    passed. The reference returns new tensors."""
    orig_dtype = x.dtype
    if residual is not None:
        x = x + residual
        residual = x

    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + weight.float())
    x = x.to(orig_dtype)
    return x if residual is None else (x, residual)


def rms_norm_gated(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, z: torch.Tensor | None = None,
                   eps: float = 1e-6, group_size: int | None = None, norm_before_gate: bool = True,
                   upcast: bool = True) -> torch.Tensor:
    """``norm.rms_gated@1``: ``rms_norm_ref``; ``"... (g d) -> ... g d"`` is a ``view``. ``z`` has
    ``x``'s shape, or is ``[tokens, heads, dim]`` for an ``x`` of ``[tokens * heads, dim]``, as
    ``rms_norm_gated`` takes the view of the projection a prefill step hands it."""
    dtype = x.dtype
    if z is not None and z.shape != x.shape:
        z = z.reshape(x.shape)
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = x.view(*x.shape[:-1], -1, group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x_group * rstd).view(x.shape) * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


# ------------------------------------------------------------------ rope


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox_style: bool) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def rope(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, head_size: int, rotary_dim: int,
         cos_sin_cache: torch.Tensor, is_neox_style: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """``RotaryEmbedding.forward_native`` on the first ``rotary_dim`` of each head;
    ``cos_sin_cache`` ``[positions, rotary_dim]``, cos then sin (``layers.rope`` builds it). Text
    positions are the same in every mrope section, so Qwen3.8's mrope is this."""
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    query_rot = _apply_rotary_emb(query[..., :rotary_dim], cos, sin, is_neox_style)
    query = torch.cat((query_rot, query[..., rotary_dim:]), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, head_size)
    key_rot = _apply_rotary_emb(key[..., :rotary_dim], cos, sin, is_neox_style)
    key = torch.cat((key_rot, key[..., rotary_dim:]), dim=-1).reshape(key_shape)
    return query, key


def fused_qk_gemma_rmsnorm_rope_gate(q_gate: torch.Tensor, k: torch.Tensor, q_weight: torch.Tensor,
                                     k_weight: torch.Tensor, cos_sin_cache: torch.Tensor, positions: torch.Tensor,
                                     eps: float, num_q_heads: int, num_kv_heads: int, head_dim: int,
                                     rotary_dim: int, has_gate: bool = True
                                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``attn.qk_norm_rope_gate@1``: what an attention layer does to its projection before it
    attends (``forward_prepare_native``). ``q_gate`` ``[tokens, num_q_heads * (1 + has_gate) *
    head_dim]`` holds per head the query, then its output gate; ``k`` ``[tokens, num_kv_heads *
    head_dim]``. GemmaRMSNorm of q and k per head, RoPE on the first ``rotary_dim`` of each head
    (``cos_sin_cache`` as ``rope`` takes it), and the gate taken apart from the queries. Returns q
    ``[tokens, num_q_heads * head_dim]``, k, and the gate in q's shape (``None`` without one)."""
    num_tokens = q_gate.shape[0]
    gate: torch.Tensor | None = None
    if has_gate:
        q, gate = torch.chunk(q_gate.view(num_tokens, num_q_heads, -1), 2, dim=-1)
        q, gate = q.reshape(num_tokens, -1), gate.reshape(num_tokens, -1)
    else:
        q = q_gate
    q_by_head = gemma_rms_norm(q.reshape(-1, head_dim), q_weight, eps)
    k_by_head = gemma_rms_norm(k.reshape(-1, head_dim), k_weight, eps)
    assert isinstance(q_by_head, torch.Tensor) and isinstance(k_by_head, torch.Tensor)
    q, k = rope(positions, q_by_head.view(q.shape), k_by_head.view(k.shape), head_dim, rotary_dim, cos_sin_cache)
    return q, k, gate


# ------------------------------------------------------------------ dense


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``act.silu_and_mul@1``: ``SiluAndMul.forward_native``."""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def fused_sigmoid_mul(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``act.sigmoid_mul@1``: the attention output gate, ``attn_output * sigmoid(gate)``, both
    ``[tokens, hidden]``. SGLang's kernel writes the result over ``attn_output``, so the
    definition lets a solution do that: a caller uses what is returned. The reference returns a
    new tensor."""
    return attn_output * torch.sigmoid(gate)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``gemm.linear@1``: ``UnquantizedLinearMethod.apply``, one GEMM over all rows."""
    return F.linear(x, weight, bias)


def embedding(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``embed.lookup@1``: ``UnquantizedEmbeddingMethod.embedding``."""
    return F.embedding(input_, weight)


# ------------------------------------------------------------------ full attention


def kv_store(k_cache: torch.Tensor, v_cache: torch.Tensor, loc: torch.Tensor, cache_k: torch.Tensor,
             cache_v: torch.Tensor) -> None:
    """``kv.store@1``: ``MHATokenToKVPool.set_kv_buffer`` without scales, its naive store."""
    if cache_k.dtype != k_cache.dtype:
        cache_k = cache_k.to(k_cache.dtype)
        cache_v = cache_v.to(v_cache.dtype)
    k_cache[loc] = cache_k
    v_cache[loc] = cache_v


def create_flashinfer_kv_indices(req_to_token: torch.Tensor, req_pool_indices: torch.Tensor,
                                 page_kernel_lens: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor,
                                 out_cache_loc: torch.Tensor | None = None) -> None:
    """``attn.kv_indices@1``: what ``create_flashinfer_kv_indices_triton`` writes
    (kernels/ops/attention/kv_indices.py): row ``r``'s first ``page_kernel_lens[r]`` slots of
    ``req_to_token[req_pool_indices[r]]`` at ``kv_indices[kv_indptr[r]:]``; with ``out_cache_loc``,
    also row ``r``'s last slot at ``out_cache_loc[r]`` (where a decode step stores). Reads nothing
    on the host: ``kv_indices`` past ``kv_indptr[rows]`` keeps what it held."""
    rows = req_pool_indices.numel()
    if not rows:
        return
    width = req_to_token.shape[1]
    j = torch.arange(kv_indices.numel(), device=kv_indices.device)
    starts = kv_indptr[:rows + 1].long()
    row = torch.searchsorted(starts[1:], j, right=True).clamp_(max=rows - 1)
    slots = req_to_token[req_pool_indices[row], (j - starts[row]).clamp_(0, width - 1)]
    kv_indices.copy_(torch.where(j < starts[rows], slots.to(kv_indices.dtype), kv_indices))
    if out_cache_loc is not None:
        out_cache_loc.copy_(req_to_token[req_pool_indices, page_kernel_lens.long() - 1])


def attention_extend(query: torch.Tensor, output: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     req_to_token: torch.Tensor, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor,
                     extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, scaling: float | None = None,
                     enable_gqa: bool = False, causal: bool = False) -> torch.Tensor:
    """``attn.extend@1``: ``TorchNativeAttnBackend._run_sdpa_forward_extend`` without encoders
    and sliding windows. ``query``, ``output`` ``[num_tokens, num_heads, head_size]``; caches
    ``[slots, kv_heads, head_size]``; ``req_to_token`` ``[rows, max_context_len]``."""
    assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
    assert seq_lens.shape[0] == extend_seq_lens.shape[0]

    # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
    query = query.movedim(0, query.dim() - 2)

    start_q = 0
    for seq_idx in range(seq_lens.shape[0]):
        extend_seq_len_q = int(extend_seq_lens[seq_idx])
        prefill_seq_len_q = int(extend_prefix_lens[seq_idx])
        seq_len_kv = int(seq_lens[seq_idx])
        end_q = start_q + extend_seq_len_q

        per_req_query = query[:, start_q:end_q, :]
        per_req_query_redudant = torch.empty(
            (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
            dtype=per_req_query.dtype, device=per_req_query.device)
        per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

        # get key and value from cache. per_req_tokens contains the kv cache
        # index for each token in the sequence.
        req_pool_idx = req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
        per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
        if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
            per_req_key = per_req_key.to(per_req_query.dtype)
            per_req_value = per_req_value.to(per_req_query.dtype)

        per_req_out_redudant = (
            scaled_dot_product_attention(
                per_req_query_redudant.unsqueeze(0), per_req_key.unsqueeze(0), per_req_value.unsqueeze(0),
                enable_gqa=enable_gqa, scale=scaling, is_causal=causal)
            .squeeze(0)
            .movedim(query.dim() - 2, 0)
        )
        output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
        start_q = end_q
    return output


def attention_decode(query: torch.Tensor, output: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     req_to_token: torch.Tensor, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor,
                     scaling: float | None = None, enable_gqa: bool = False, causal: bool = False) -> torch.Tensor:
    """``attn.decode@1``: ``TorchNativeAttnBackend._run_sdpa_forward_decode``, one query token per
    sequence, without encoders and sliding windows."""
    query = query.movedim(0, query.dim() - 2)

    start_q = 0
    for seq_idx in range(seq_lens.shape[0]):
        seq_len_q = 1
        seq_len_kv = int(seq_lens[seq_idx])
        end_q = start_q + seq_len_q

        per_req_query = query[:, start_q:end_q, :]
        req_pool_idx = req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
        per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
        if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
            per_req_key = per_req_key.to(per_req_query.dtype)
            per_req_value = per_req_value.to(per_req_query.dtype)

        per_req_out = (
            scaled_dot_product_attention(
                per_req_query.unsqueeze(0), per_req_key.unsqueeze(0), per_req_value.unsqueeze(0),
                enable_gqa=enable_gqa, scale=scaling, is_causal=causal)
            .squeeze(0)
            .movedim(query.dim() - 2, 0)
        )
        output[start_q:end_q, :, :] = per_req_out
        start_q = end_q
    return output


# ------------------------------------------------------------------ GDN: causal conv


def causal_conv1d_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                      initial_states: torch.Tensor | None = None, return_final_states: bool = False,
                      final_states_out: torch.Tensor | None = None,
                      activation: str | None = "silu") -> tuple[torch.Tensor, torch.Tensor | None]:
    """x: (batch, dim, seqlen); weight: (dim, width); bias: (dim,); initial_states,
    final_states_out: (batch, dim, width - 1). out: (batch, dim, seqlen)."""
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor,
                             bias: torch.Tensor | None = None, activation: str | None = None) -> torch.Tensor:
    """x: (batch, dim) or (batch, dim, seqlen); conv_state: (batch, dim, state_len), updated in
    place; weight: (dim, width). out: like x. (``cache_seqlens`` circular buffers left out.)"""
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)  # (batch, dim, state_len + seqlen)
    conv_state.copy_(x_new[:, :, -state_len:])
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def causal_conv1d_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str | None,
                     conv_states: torch.Tensor, has_initial_state: torch.Tensor, cache_indices: torch.Tensor,
                     query_start_loc: torch.Tensor, seq_lens_cpu: Sequence[int] | None = None) -> torch.Tensor:
    """``gdn.conv1d_fn@1``, the prefill conv ``GDNAttnBackend._convolve_prefill`` calls: varlen
    ``x`` ``(dim, total_tokens)``, sequence ``i`` is ``x[:, query_start_loc[i]:query_start_loc[i + 1]]``
    and continues from ``conv_states[cache_indices[i]]`` if ``has_initial_state[i]``; the states
    are updated in place to each sequence's last ``width - 1`` inputs. ``seq_lens_cpu`` is the
    sequences' lengths on the host, which SGLang's kernel sizes its launch by; the reference has
    no use for them."""
    out = torch.empty_like(x)
    for i in range(cache_indices.shape[0]):
        start, end = int(query_start_loc[i]), int(query_start_loc[i + 1])
        slot = int(cache_indices[i])
        initial = conv_states[slot:slot + 1] if bool(has_initial_state[i]) else None
        seq_out, final = causal_conv1d_ref(x[None, :, start:end], weight, bias, initial, return_final_states=True,
                                           activation=activation)
        out[:, start:end] = seq_out[0]
        assert final is not None
        conv_states[slot] = final[0]
    return out


def causal_conv1d_update(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                         activation: str | None, conv_state_indices: torch.Tensor) -> torch.Tensor:
    """The decode conv: ``x`` ``(batch, dim)``, row ``i`` steps
    ``conv_state[conv_state_indices[i]]`` in place. A negative index is a padded lane (SGLang's
    ``pad_slot_id``), whose result is undefined: the reference runs it on the padding slot."""
    idx = conv_state_indices.to(torch.long).clamp_min(0)
    states = conv_state[idx]
    out = causal_conv1d_update_ref(x, states, weight, bias, activation)
    conv_state[idx] = states
    return out


def fused_qkvzba_causal_conv1d_update(mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor, conv_state: torch.Tensor,
                                      conv_weight: torch.Tensor, conv_bias: torch.Tensor | None,
                                      conv_state_indices: torch.Tensor, qkv_dim: int, v_dim: int, num_v_heads: int,
                                      head_v_dim: int, activation: str | None
                                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``gdn.qkvzba_conv1d_update@1``, the first half of a GDN decode step
    (``fused_qkvzba_causal_conv1d_update_contiguous``): the two input projections taken apart and
    the conv stepped. ``mixed_qkvz`` ``[batch, qkv_dim + v_dim]`` is ``[q | k | v | z]``,
    ``mixed_ba`` ``[batch, 2 * num_v_heads]`` is ``[b | a]``. Returns the convolved ``[q | k | v]``
    (``causal_conv1d_update`` on the conv state pool), ``z`` ``[batch, num_v_heads, head_v_dim]``,
    ``b`` and ``a``. The reference returns ``z``, ``b`` and ``a`` as views of the projections."""
    assert mixed_qkvz.shape[1] == qkv_dim + v_dim and v_dim == num_v_heads * head_v_dim
    z = mixed_qkvz[:, qkv_dim:].view(mixed_qkvz.shape[0], num_v_heads, head_v_dim)
    mixed_qkv = causal_conv1d_update(mixed_qkvz[:, :qkv_dim], conv_state, conv_weight, conv_bias, activation,
                                     conv_state_indices)
    return mixed_qkv, z, mixed_ba[:, :num_v_heads], mixed_ba[:, num_v_heads:]


# ------------------------------------------------------------------ GDN: gating and delta rule


def fused_gdn_gating(A_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                     dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``gdn.gating@1``: ``torch_gdn_gating``; ``g`` and ``beta`` ``[1, tokens, HV]``."""
    return -A_log.float().exp() * softplus(a.float() + dt_bias).unsqueeze(0), b.sigmoid().unsqueeze(0)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Aligned with the l2norm of the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor,
                                 beta: torch.Tensor, chunk_size: int = 64, initial_state: torch.Tensor | None = None,
                                 output_final_state: bool = False, use_qk_l2norm_in_kernel: bool = False,
                                 chunk_states: list[torch.Tensor] | None = None
                                 ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """fla's naive chunked gated delta rule; ``[B, T, H, D]`` in, state ``[B, H, K, V]``. With
    ``chunk_states``, appends the state before each chunk, as fla's chunk kernel returns ``h``."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        if chunk_states is not None:
            chunk_states.append(last_recurrent_state)
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    final_state = last_recurrent_state if output_final_state else None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, final_state


def torch_recurrent_gated_delta_rule(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor,
                                     beta: torch.Tensor, initial_state: torch.Tensor | None, output_final_state: bool,
                                     use_qk_l2norm_in_kernel: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """fla's naive recurrent gated delta rule; ``[B, T, H, D]`` in, state ``[B, H, K, V]``."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = value.new_zeros((batch_size, num_heads, sequence_length, v_head_dim))
    last_recurrent_state = (
        value.new_zeros((batch_size, num_heads, k_head_dim, v_head_dim))
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    final_state = last_recurrent_state if output_final_state else None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, final_state


def chunk_gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                           initial_state: torch.Tensor, initial_state_indices: torch.Tensor, cu_seqlens: torch.Tensor,
                           use_qk_l2norm_in_kernel: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``gdn.chunk_delta_rule@1``, the prefill rule ``TritonGDNKernel.extend`` calls:
    ``chunk_gated_delta_rule_update`` of the CPU test, on the pool ``initial_state``
    ``[slots, HV, V, K]`` at ``initial_state_indices``, updated in place. ``q``, ``k``
    ``[1, T, HK, K]``; ``v`` ``[1, T, HV, V]``; ``g``, ``beta`` ``[1, T, HV]``. Returns
    ``(output [1, T, HV, V], final states [N, HV, V, K], h)``: ``h`` ``[1, chunks, HV, V, K]`` is the
    state before each chunk of 64 tokens of each sequence, in order, as fla's ``h`` (what the prefix
    cache keeps from inside a step, ``_track_mamba_state_extend``)."""
    idx = initial_state_indices.to(torch.long)
    num_heads, num_value_heads = q.shape[2], v.shape[2]
    initial_state_kv = initial_state[idx].transpose(-1, -2).contiguous()
    if num_value_heads // num_heads > 1:
        q = q.repeat_interleave(num_value_heads // num_heads, dim=2)
        k = k.repeat_interleave(num_value_heads // num_heads, dim=2)
    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state[idx])
    states: list[torch.Tensor] = []
    start_q = 0
    for i in range(idx.shape[0]):
        end_q = int(cu_seqlens[i + 1])
        core_attn_outi, last_recurrent_state = torch_chunk_gated_delta_rule(
            query=q[:, start_q:end_q, :, :], key=k[:, start_q:end_q, :, :], value=v[:, start_q:end_q, :, :],
            g=g[:, start_q:end_q, :], beta=beta[:, start_q:end_q, :], initial_state=initial_state_kv[i],
            output_final_state=True, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, chunk_states=states)
        output[:, start_q:end_q, :, :] = core_attn_outi
        assert last_recurrent_state is not None
        final_state[i] = last_recurrent_state.transpose(-1, -2).contiguous()
        start_q = end_q
    initial_state[idx] = final_state.to(initial_state.dtype)
    h = torch.stack([state.squeeze(0).transpose(-1, -2) for state in states]).unsqueeze(0)  # every row has a chunk
    return output, final_state, h


def fused_sigmoid_gating_delta_rule_update(A_log: torch.Tensor, dt_bias: torch.Tensor, q: torch.Tensor,
                                           k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                                           initial_state_source: torch.Tensor, initial_state_indices: torch.Tensor,
                                           cu_seqlens: torch.Tensor, use_qk_l2norm_in_kernel: bool = True
                                           ) -> torch.Tensor:
    """The decode rule ``TritonGDNKernel.decode`` calls: one token per
    sequence, ``sigmoid_gating_delta_rule_update`` of the CPU test on the pool at
    ``initial_state_indices``, updated in place. ``q``, ``k`` ``[1, B, HK, K]``; ``v``
    ``[1, B, HV, V]``; ``a``, ``b`` ``[B, HV]``. Returns ``[1, B, HV, V]``. A negative index is a
    padded lane, whose result is undefined: the reference runs it on the padding slot."""
    assert cu_seqlens.shape[0] == initial_state_indices.shape[0] + 1
    idx = initial_state_indices.to(torch.long).clamp_min(0)
    num_heads, num_value_heads = q.shape[2], v.shape[2]
    if num_value_heads // num_heads > 1:
        q = q.repeat_interleave(num_value_heads // num_heads, dim=2)
        k = k.repeat_interleave(num_value_heads // num_heads, dim=2)
    beta = b.sigmoid()
    g = -A_log.float().exp() * softplus(a.float() + dt_bias)
    initial_state_kv = initial_state_source[idx].transpose(-1, -2).contiguous()
    core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), g.unsqueeze(1), beta.unsqueeze(1),
        initial_state_kv, True, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
    assert last_recurrent_state is not None
    initial_state_source[idx] = last_recurrent_state.transpose(-1, -2).contiguous().to(initial_state_source.dtype)
    return core_attn_out.transpose(0, 1)


def fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                                                   A_log: torch.Tensor, dt_bias: torch.Tensor, scale: float,
                                                   ssm_states: torch.Tensor, cache_indices: torch.Tensor,
                                                   num_v_heads: int, head_v_dim: int) -> torch.Tensor:
    """``gdn.packed_decode@1``, the second half of a GDN decode step (``TritonGDNKernel.
    packed_decode``): ``mixed_qkv`` ``[batch, qkv_dim]`` is the convolved ``[q | k | v]``, which the
    kernel reads in place of separate q, k and v; then ``fused_sigmoid_gating_delta_rule_update``
    on the temporal state pool at ``cache_indices``. ``scale`` is ``head_k_dim ** -0.5``. Returns
    ``[1, batch, num_v_heads, head_v_dim]``; a padded lane's (negative index) is undefined."""
    batch, head_k_dim = mixed_qkv.shape[0], ssm_states.shape[-1]
    assert abs(scale - head_k_dim ** -0.5) < 1e-9, "the reference scales q by head_k_dim ** -0.5"
    v_dim = num_v_heads * head_v_dim
    k_dim = (mixed_qkv.shape[1] - v_dim) // 2
    query, key, value = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim=-1)
    return fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, query.reshape(1, batch, -1, head_k_dim), key.reshape(1, batch, -1, head_k_dim),
        value.reshape(1, batch, num_v_heads, head_v_dim), a, b, ssm_states, cache_indices,
        torch.arange(batch + 1, dtype=torch.int32, device=mixed_qkv.device))


# ------------------------------------------------------------------ sampling


def _fmix32(h: torch.Tensor) -> torch.Tensor:
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    return h ^ (h >> 16)


def _rotl32(x: torch.Tensor, r: int) -> torch.Tensor:
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


def _murmur3_mix(h: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    k = (k * 0xCC9E2D51) & 0xFFFFFFFF
    k = _rotl32(k, 15)
    k = (k * 0x1B873593) & 0xFFFFFFFF
    h = h ^ k
    h = _rotl32(h, 13)
    return (h * 5 + 0xE6546B64) & 0xFFFFFFFF


def murmur_hash32(seed: torch.Tensor, positions: torch.Tensor, col_indices: torch.Tensor) -> torch.Tensor:
    """``murmur_hash32_kernel``: the 64-bit seed, 32-bit position and 32-bit column as four
    32-bit blocks; ``[n, m]`` hashes, kept in int64 (every value fits in 32 bits)."""
    assert seed.shape == positions.shape, "Seed and positions must have the same shape (n,)"
    seed = seed.to(torch.int64)
    rows = seed.shape[0]
    h = torch.zeros((rows, 1), dtype=torch.int64, device=seed.device)
    h = _murmur3_mix(h, (seed & 0xFFFFFFFF)[:, None])
    h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF)[:, None])
    h = _murmur3_mix(h, (positions.to(torch.int64) & 0xFFFFFFFF)[:, None])
    h = _murmur3_mix(h.expand(rows, col_indices.shape[0]), (col_indices.to(torch.int64) & 0xFFFFFFFF)[None, :])
    return _fmix32(h ^ 16)


def multinomial_with_seed(logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """A deterministic batched alternative to ``torch.multinomial``: Gumbel-max with noise from
    (seed, position, token)."""
    _, m = logprobs.shape
    col_indices = torch.arange(m, device=logprobs.device)
    hashed = murmur_hash32(seed, positions, col_indices)

    # NOTE (sehoon): it is critical to keep gumbel noise calculation in float64 to avoid numerical instability.
    x = hashed.to(torch.float64) / 0xFFFFFFFF
    # clamp both ends: x == 1 gives gumbel +inf (NaN at -inf logprobs); the cap is
    # the hash spacing so that bucket matches its neighbor instead of dominating
    x.log_().clamp_(min=torch.finfo(x.dtype).min, max=-(2.0 ** -32)).neg_()
    x.log_().neg_()  # -log(-log(x)) == gumbel noise
    x.add_(logprobs.to(torch.float64))
    return torch.argmax(x, dim=1, keepdim=True)


def _top_k_top_p_sampling_from_probs_torch(probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor,
                                           sampling_seed: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``top_k_top_p_min_p_sampling_from_probs_torch`` with a seed and without min-p."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[torch.arange(0, probs_sort.shape[-1], device=probs.device).view(1, -1) >= top_ks.view(-1, 1)] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    logprobs = probs_sort.to(torch.float64)
    logprobs.log_()
    sampled_index = multinomial_with_seed(logprobs, sampling_seed, positions)
    probs_idx = probs_idx.to(torch.int32)
    return torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)


def sample(logits: torch.Tensor, temperatures: Sequence[float], top_ps: Sequence[float], top_ks: Sequence[int],
           seeds: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
    """``sample.seeded@1``: ``Sampler.forward``'s native path, always seeded (deterministic
    inference). Requests are normalized as ``SamplingParams`` does: temperature 0 is top-k 1,
    top-k -1 the whole vocabulary. ``positions``: of each row's last input token (the model
    runner passes ``positions`` for decode, ``seq_lens - 1`` for extend). ``int32`` tokens."""
    temps = [1.0 if 0 <= t < SAMPLING_EPS else float(t) for t in temperatures]
    ks = [1 if 0 <= t < SAMPLING_EPS else (TOP_K_ALL if k == -1 else int(k)) for t, k in zip(temperatures, top_ks)]
    if all(k == 1 for k in ks):  # sampling_info.is_all_greedy
        return torch.argmax(logits, dim=-1).to(torch.int32)
    device = logits.device
    logits = logits.float().div_(_on_device(temps, torch.float32, device).view(-1, 1))
    probs = torch.softmax(logits, dim=-1)
    seed = _on_device(seeds, torch.int64, device)
    pos = _on_device(positions, torch.int64, device)
    if all(k == TOP_K_ALL for k in ks) and all(p == 1.0 for p in top_ps):  # simple_sampling_case
        return multinomial_with_seed(torch.log(probs), seed, pos).view(-1).to(torch.int32)
    return _top_k_top_p_sampling_from_probs_torch(
        probs, _on_device(ks, torch.int64, device), _on_device(top_ps, torch.float32, device), seed, pos)


def track_states(conv: torch.Tensor, temporal: torch.Tensor, cache_indices: torch.Tensor, mask: torch.Tensor,
                 dst: torch.Tensor) -> None:
    """``gdn.track_states@1``: for each lane with ``mask`` set, copy every layer's conv and temporal
    state from slot ``cache_indices[i]`` to slot ``dst[i]`` of the pools ``conv`` ``[layers, slots,
    ...]`` and ``temporal`` (``track_mamba_states_all_layers``). Without a host round trip, so a
    graph can hold it: every lane is written, a lane without ``mask`` its ``dst`` with what it holds;
    those lanes name the padding slot 0, so lanes that share it write the same thing."""
    src, to = cache_indices.long(), dst.long()
    for pool in (conv, temporal):
        keep = mask.view(1, -1, *([1] * (pool.dim() - 2)))
        pool[:, to] = torch.where(keep, pool[:, src], pool[:, to])

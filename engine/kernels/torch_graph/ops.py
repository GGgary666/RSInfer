"""PyTorch kernels that keep nothing of a step on the host, for steps replayed from a graph.

``attention_decode`` computes what the reference of ``attn.decode@1`` computes (SGLang's
``TorchNativeAttnBackend._run_sdpa_forward_decode``), but over a fixed extent: every row attends
over the whole width of ``req_to_token`` under a length mask, so the lengths and the page mappings
are read on the device at replay. Its cost follows that width (``batch.max_context_len``), not the
rows' lengths; a paged decode kernel (FlashInfer) replaces it."""

from __future__ import annotations

import torch


def attention_decode(query: torch.Tensor, output: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     req_to_token: torch.Tensor, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor,
                     scaling: float | None = None, enable_gqa: bool = False, causal: bool = False) -> torch.Tensor:
    """``attn.decode@1``. ``query`` and ``output`` ``[rows, heads, dim]``; a group of
    ``heads / kv_heads`` query heads shares one K/V head, which is never expanded."""
    if causal:
        raise ValueError("a decode step attends under a length mask, not a causal triangle")
    rows, heads, dim = query.shape
    kv_heads = k_cache.shape[1]
    if heads != kv_heads and not enable_gqa:
        raise ValueError(f"{heads} query heads over {kv_heads} K/V heads need enable_gqa")
    slots = req_to_token[req_pool_indices.long()].long()
    mask = torch.arange(slots.shape[1], device=slots.device) < seq_lens[:, None]
    # Past a row's length the table holds whatever an earlier request left: read the padding slot.
    slots = torch.where(mask, slots, 0)
    key = k_cache[slots].to(query.dtype)    # [rows, width, kv_heads, dim]
    value = v_cache[slots].to(query.dtype)
    # Inactive lanes may leave non-finite padding values; zero weight times NaN is still NaN.
    value.masked_fill_(~mask[:, :, None, None], 0)
    q = query.reshape(rows, kv_heads, heads // kv_heads, dim) * (dim ** -0.5 if scaling is None else scaling)
    scores = torch.einsum("rhgd,rwhd->rhgw", q, key).masked_fill(~mask[:, None, None, :], float("-inf"))
    output.copy_(torch.einsum("rhgw,rwhd->rhgd", torch.softmax(scores, dim=-1), value).reshape(rows, heads, dim))
    return output

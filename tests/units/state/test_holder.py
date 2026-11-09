"""``core.state.holder``: the state pools' share of the device, as SGLang sizes it."""

from __future__ import annotations

from typing import Any

import pytest

from core.state.holder.unit import Holder, item_bytes
from models.qwen38.config import Qwen3_5TextConfig
from models.qwen38.unit import states
from tests.units.support.worker import Worker, tiny_config

GiB = 2**30
QWEN38_27B = {  # text_config of Qwen3.8-27B (the Qwen3_5 architecture, 64 layers)
    "vocab_size": 248320, "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
    "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "linear_num_value_heads": 48,
    "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "full_attention_interval": 4, "max_position_embeddings": 262144,
    "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}}


def decls() -> Any:
    return states(Qwen3_5TextConfig.from_hf(QWEN38_27B), "bf16", 16)


def test_the_item_sizes_of_qwen38_27b() -> None:
    """stage2-contract.md §5.4: a request's GDN state is about 147 MiB at TP1; KV is 64 KiB a token."""
    by_kind: dict[str, Any] = {}
    for d in decls():
        by_kind.setdefault(d.kind, []).append(d)
    assert item_bytes(by_kind["gdn.slots"]) == 48 * (10240 * 3 * 2 + 48 * 128 * 128 * 4)  # 146.8 MiB
    assert item_bytes(by_kind["kv.paged"]) == 16 * 2 * 16 * 4 * 256 * 2  # 1 MiB a page of 16 tokens


def test_a_sm100_split_as_sglang_splits_it() -> None:
    """0.85 of the device less the weights for the states; 0.9 : 1 between GDN state and KV."""
    h = Holder(180 * GiB, cpu=False, mem_fraction=0.85, state_ratio=0.9, max_slots=0, cpu_bytes=0)
    weights = 54 * 10**9
    sizes = h.size(decls(), weights)
    slot, page = 153_944_064, 1 << 20
    budget = int(180 * GiB * 0.85) - weights - slot - page
    assert sizes["gdn.slots"] == int(budget * 0.9 / 1.9) // slot
    assert sizes["kv.paged"] == (budget - sizes["gdn.slots"] * slot) // page
    assert (sizes["gdn.slots"], sizes["kv.paged"] * 16) == (h.items("gdn.slots"), h.items("kv.paged") * 16)
    assert sizes["gdn.slots"] == 338 and sizes["kv.paged"] * 16 == 886_448  # about 48 GiB of GDN state, 54 GiB of KV


def test_a_fixed_number_of_slots_leaves_the_rest_to_kv() -> None:
    h = Holder(180 * GiB, cpu=False, mem_fraction=0.85, state_ratio=0.9, max_slots=64, cpu_bytes=0)
    sizes = h.size(decls(), 54 * 10**9)
    assert sizes["gdn.slots"] == 64
    assert sizes["kv.paged"] == (int(180 * GiB * 0.85) - 54 * 10**9 - 153_944_064 - (1 << 20) - 64 * 153_944_064) // (1 << 20)


def test_on_a_cpu_the_budget_is_cpu_bytes() -> None:
    h = Holder(512 * GiB, cpu=True, mem_fraction=0.85, state_ratio=1.0, max_slots=0, cpu_bytes=GiB)
    sizes = h.size(decls(), 54 * 10**9)  # the weights live elsewhere in RAM: not this budget
    slot = 153_944_064
    assert sizes["gdn.slots"] == int((GiB - slot - (1 << 20)) / 2) // slot


def test_states_that_do_not_fit_are_refused_with_the_numbers() -> None:
    h = Holder(24 * GiB, cpu=False, mem_fraction=0.85, state_ratio=0.9, max_slots=0, cpu_bytes=0)
    with pytest.raises(RuntimeError, match="the states do not fit.*0.85 of 24.0 GiB less 50.29 GiB of weights"):
        h.size(decls(), 54 * 10**9)


def test_pools_without_a_count_take_the_holders() -> None:
    """The whole worker on the CPU with no pool sizes set: the model sizes them through the holder
    once its weights are allocated, and the executor reports what the pools got."""
    w = Worker(tiny_config().to_dict(), **{"state.kv.pages": 0, "state.gdn.slots": 0, "holder.cpu_bytes": 1 << 22,
                                           "holder.state_ratio": 1.0})
    try:
        holder = w.service("state.holder@1")
        got = w.service("executor@1").capacity().items
        assert got == {"kv.paged": holder.items("kv.paged"), "gdn.slots": holder.items("gdn.slots")}
        assert got["kv.paged"] > 0 and got["gdn.slots"] > 0
    finally:
        w.close()

"""A state store rebuilt in place (a change applies to an empty engine, doc/specs/empty-apply.md)
gives the old pool's memory back before allocating the new one: a new pool a little larger than
the old one, on a device with room for one of them only, still fits. The case of
doc/framework-bugs/2026-11-10-gdn-ssm-dtype-rebuild-oom.md: the GDN states switched from f32 to bf16,
twice the slots in about the same bytes. Any GPU; the sizes follow its free memory.

The weight store does the same, so a rollback after a failed reload fits again: the case of
doc/framework-bugs/2026-11-13-a-weight-dtype-change-runs-out-of-memory-and-its-rollback-is-unconfirmed.md."""

from __future__ import annotations

import math

import torch

from core.contracts.model import WeightDecl
from core.contracts.state import StateDecl
from core.state.weights.unit import WeightStore
from platforms.torch_eager import TorchEagerPlatform
from state.gdn_slots.unit import MambaPool

LAYERS = 4
TEMPORAL = (48, 128, 128)  # Qwen3.8's GDN state per layer and slot
CONV = (10240, 3)


def decls(ssm_dtype: str) -> list[StateDecl]:
    return [StateDecl("gdn.slots", i, {"conv": (CONV, "bf16"), "temporal": (TEMPORAL, ssm_dtype)}, "slot")
            for i in range(LAYERS)]


def test_a_gdn_pool_rebuilt_a_little_larger_fits_where_only_one_of_them_fits() -> None:
    platform = TorchEagerPlatform(cuda=True)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    slots = int(free * 0.55) // (LAYERS * math.prod(TEMPORAL) * 4)
    old = MambaPool(platform, slots)
    old.declare(decls("f32"))
    old.free()  # as the unit's effect does when a change reloads it
    # bf16: twice the slots, and a few more, so the new temporal tensor is a little larger; the new
    # conv tensor is larger than the old conv block and would be cut out of the old temporal one
    new = MambaPool(platform, 2 * slots + 4)
    try:
        new.declare(decls("bf16"))
        assert new.temporal is not None and new.temporal.dtype == torch.bfloat16
        assert new.temporal.shape == (LAYERS, 2 * slots + 5, *TEMPORAL)
        assert new.temporal.numel() * 2 > (slots + 1) * LAYERS * math.prod(TEMPORAL) * 4
    finally:
        new.free()
        torch.cuda.empty_cache()


def weights(free: int, dtype: str) -> list[WeightDecl]:
    """Two "vocab" matrices of a tenth of ``free`` and one of a fifth in bf16; twice that in f32."""
    elems = [free // 10 // 2, free // 10 // 2, free // 5 // 2]
    return [WeightDecl(f"w{i}", (n,), dtype) for i, n in enumerate(elems)]


def test_old_weights_and_pool_fit_again_after_a_larger_attempt_failed() -> None:
    """A reload to f32 weights fails, and the rollback loads the bf16 weights and the old pool
    again. Without giving the cache back first, the old weights were cut out of the blocks the
    attempt freed, those blocks stayed pinned, and the pool fit neither in them nor on the device."""
    platform = TorchEagerPlatform(cuda=True)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    slots = int(free * 0.45) // (LAYERS * math.prod(TEMPORAL) * 4)
    old_weights, old_pool = WeightStore(platform, "", 0), MambaPool(platform, slots)
    old_weights.declare(weights(free, "bf16"))
    old_pool.declare(decls("f32"))
    old_pool.free()  # the change unloads the old instances
    old_weights.free()

    new_weights, new_pool = WeightStore(platform, "", 0), MambaPool(platform, slots)
    failed = False
    try:
        new_weights.declare(weights(free, "f32"))
        new_pool.declare(decls("f32"))
    except torch.OutOfMemoryError:
        failed = True
    assert failed  # the attempt does not fit: 0.8 of the free memory in weights, then the pool
    new_pool.free()  # the rollback unloads the new instances, then loads the old ones again
    new_weights.free()

    weights_again, pool_again = WeightStore(platform, "", 0), MambaPool(platform, slots)
    try:
        weights_again.declare(weights(free, "bf16"))
        pool_again.declare(decls("f32"))
        assert pool_again.temporal is not None and pool_again.temporal.shape == (LAYERS, slots + 1, *TEMPORAL)
    finally:
        pool_again.free()
        weights_again.free()
        torch.cuda.empty_cache()


def test_a_weight_store_redeclared_in_place_gives_the_old_tensors_back_before_allocating() -> None:
    """When a model reloads on the same ``WeightStore`` (e.g. ``eplb.extra_experts`` changing the expert
    stacks' slot count), ``declare`` drops the old tensors from ``self.tensors`` before ``empty_cache`` so
    the larger stacks fit where only one copy fits."""
    platform = TorchEagerPlatform(cuda=True)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    store = WeightStore(platform, "", 0)
    try:
        store.declare([WeightDecl("experts", (int(free * 0.55) // 2,), "bf16")])
        store.declare([WeightDecl("experts", (int(free * 0.58) // 2,), "bf16")])
        assert store.tensors["experts"].numel() == int(free * 0.58) // 2
    finally:
        store.declare([])
        for buf in list(store.buffers_.values()):
            platform.free(buf)
        store.buffers_.clear()
        store.tensors.clear()
        torch.cuda.empty_cache()

"""A state store rebuilt in place (a change applies to an empty engine, doc/specs/empty-apply.md)
gives the old pool's memory back before allocating the new one: a new pool a little larger than
the old one, on a device with room for one of them only, still fits. The case of
doc/framework-bugs/2026-11-10-gdn-ssm-dtype-rebuild-oom.md: the GDN states switched from f32 to bf16,
twice the slots in about the same bytes. Any GPU; the sizes follow its free memory."""

from __future__ import annotations

import math

import torch

from core.contracts.state import StateDecl
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

"""core.kernels: what selection picks, and what a reload of a kernel unit does to the catalog."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from core.common.framework import FrameworkModules
from core.common.ids import Identity, UnitId
from core.common.manifest import load_manifest
from core.contracts.kernels import Definition, Implementation, Solution
from core.kernels.unit import KernelCatalog
from core.manifest.checks import check_tree
from kernels.reference import ops
from kernels.reference.unit import DEFINITIONS, reference_id
from platforms.torch_eager import TorchEagerPlatform
from tests.framework.support import PROC
from unit.runtime import ProcessPlan, ReloadPlan, UnitRuntime, UnitSpec

ENGINE = Path(__file__).resolve().parents[3] / "engine"
CATALOG, REFERENCE = UnitId("core.kernels"), UnitId("kernels.reference")


def spec(uid: UnitId) -> UnitSpec:
    return UnitSpec(load_manifest(ENGINE / Path(*uid.split(".")) / "unit.toml").unwrap())


def test_selection_takes_a_fitting_solution_and_falls_back_to_the_reference() -> None:
    cat = KernelCatalog(["cuda", "sm_100"])
    undo_def = cat.register(Definition("attn.paged@1", "", "reference.attn.paged"))
    cat.register(Implementation(Solution("reference.attn.paged", "attn.paged@1", "m:f"), lambda: "ref"))
    assert (cat.select("attn.paged@1", {"head_dim": 256}).solution, cat.select("attn.paged@1", {}).fallback) == \
        ("reference.attn.paged", True)
    undo = cat.register(Implementation(Solution("fast.attn", "attn.paged@1", "m:g", shapes={"head_dim": [128, 256]},
                                                requires=["sm_100"]), lambda: "fast"))
    cat.register(Implementation(Solution("tpu.attn", "attn.paged@1", "m:h", requires=["tpu"]), lambda: "tpu"))
    assert cat.select("attn.paged@1", {"head_dim": 256}).solution == "fast.attn"
    assert cat.select("attn.paged@1", {"head_dim": 64}).solution == "reference.attn.paged"  # shape does not fit
    assert cat.call("attn.paged@1", {"head_dim": 128}) == "fast"
    undo()
    assert cat.call("attn.paged@1", {"head_dim": 128}) == "ref"
    undo_def()
    with pytest.raises(LookupError, match="no kernel definition"):
        cat.select("attn.paged@1", {})


def test_two_versions_of_a_definition_each_run_their_own_reference() -> None:
    """The reference id keeps the version, so ``@1`` and ``@2`` sit side by side; a name or id
    registered twice is refused, since a lookup would take whichever came first."""
    cat = KernelCatalog([])
    for version, result in (("gemm.linear@1", 1), ("gemm.linear@2", 2)):
        cat.register(Definition(version, "", reference_id(version)))
        cat.register(Implementation(Solution(reference_id(version), version, "m:f"), lambda result=result: result))
    assert (cat.call("gemm.linear@1", {}), cat.call("gemm.linear@2", {})) == (1, 2)
    assert cat.solutions("gemm.linear@2") == ["reference.gemm.linear@2"]
    with pytest.raises(ValueError, match="definition 'gemm.linear@1' is already registered"):
        cat.register(Definition("gemm.linear@1", "", "other"))
    with pytest.raises(ValueError, match="solution 'reference.gemm.linear@2' is already registered"):
        cat.register(Implementation(Solution("reference.gemm.linear@2", "gemm.linear@2", "m:g"), lambda: 3))
    undo = cat.register(Definition("attn.paged@1", "", "reference.attn.paged@1"))
    undo()
    cat.register(Definition("attn.paged@1", "", "reference.attn.paged@1"))  # free again once undone


def test_the_kernel_units_pass_the_manifest_rules() -> None:
    errors, _ = check_tree(ENGINE, None, FrameworkModules.default())
    assert [e for e in errors if e.subject in (CATALOG, REFERENCE)] == []


def test_a_reloaded_kernel_unit_registers_again_and_an_unloaded_one_leaves_nothing() -> None:
    rt = UnitRuntime(PROC, platform=TorchEagerPlatform())
    rt.load_process_plan(ProcessPlan(PROC, Identity("k"), [spec(CATALOG), spec(REFERENCE)],
                                     code_root=str(ENGINE))).unwrap()
    unit = rt.instance(CATALOG)
    assert unit is not None
    cat: KernelCatalog = getattr(unit, "catalog")
    names = [name for name, *_ in DEFINITIONS]
    assert [d.name for d in cat.definitions()] == sorted(names)
    x, w = torch.randn(3, 8), torch.randn(8)
    assert torch.equal(cat.call("norm.gemma_rms@1", {}, x, w, 1e-6), cast(torch.Tensor, ops.gemma_rms_norm(x, w, 1e-6)))
    assert cat.select("norm.gemma_rms@1", {}).fallback
    rt.reload(ReloadPlan(units_to_reload=[spec(REFERENCE)])).unwrap()
    assert [d.name for d in cat.definitions()] == sorted(names)
    assert all(cat.solutions(n) == [f"reference.{n}"] for n in names)
    assert {d.name for d in cat.definitions() if d.batch_invariant} == {n for n, *_, inv in DEFINITIONS if inv}
    rt.unload_all().unwrap()
    assert cat.definitions() == [] and cat.solutions("norm.gemma_rms@1") == []


def test_a_kept_choice_follows_a_reload_of_the_kernel_unit_that_registered_it() -> None:
    """Choices are fixed once made (``call`` per definition and shape, a backend at start); a
    registration or undo moves ``version``, so the next call after a kernel unit's reload (undo,
    then register again) runs the new code instead of the function kept before."""
    from layers.solutions import Kernels

    cat = KernelCatalog([])
    cat.register(Definition("attn.paged@1", "", "reference.attn.paged"))
    cat.register(Implementation(Solution("reference.attn.paged", "attn.paged@1", "m:f"), lambda: "ref"))
    kernels = Kernels(cat, {"attn.backend": "auto", "attn.prefill_backend": "", "attn.decode_backend": ""}, "attn")
    kernels.resolve({"prefill": {"attn.paged@1": {}}, "decode": {"attn.paged@1": {}}})
    assert (kernels("decode", "attn.paged@1")(), cat.call("attn.paged@1", {})) == ("ref", "ref")
    undo = cat.register(Implementation(Solution("fast.attn", "attn.paged@1", "m:g"), lambda: "fast v1"))
    assert (kernels("decode", "attn.paged@1")(), cat.call("attn.paged@1", {})) == ("fast v1", "fast v1")
    undo()  # the kernel unit reloads: its old registration goes, the new one comes
    cat.register(Implementation(Solution("fast.attn", "attn.paged@1", "m:g"), lambda: "fast v2"))
    assert (kernels("prefill", "attn.paged@1")(), cat.call("attn.paged@1", {})) == ("fast v2", "fast v2")

"""core.kernels: what selection picks, and what a reload of a kernel unit does to the catalog."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.common.framework import FrameworkModules
from core.common.ids import Identity, UnitId
from core.common.manifest import load_manifest
from core.contracts.kernels import Definition, Implementation, Solution
from core.kernels.unit import KernelCatalog
from core.manifest.checks import check_tree
from kernels.reference import ops
from kernels.reference.unit import DEFINITIONS
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
    assert torch.equal(cat.call("norm.gemma_rms@1", {}, x, w, 1e-6), ops.gemma_rms_norm(x, w, 1e-6))
    assert cat.select("norm.gemma_rms@1", {}).fallback
    rt.reload(ReloadPlan(units_to_reload=[spec(REFERENCE)])).unwrap()
    assert [d.name for d in cat.definitions()] == sorted(names)
    assert all(cat.solutions(n) == [f"reference.{n.partition('@')[0]}"] for n in names)
    rt.unload_all().unwrap()
    assert cat.definitions() == [] and cat.solutions("norm.gemma_rms@1") == []

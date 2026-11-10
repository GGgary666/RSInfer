"""The worker's start-up configuration, as SGLang's (stage2-contract.md §5.5.3): the state dtypes and
the attention / GDN backends per phase are fixed once at start and logged by ``core.plan``; an
explicitly named backend that cannot run here, or a state dtype stage 2 does not keep, fails the
start instead of falling back."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from core.change.types import ProcActions
from core.common.result import Failed
from tests.units.support.worker import Driver, FailingAlloc, Worker, tiny_config


def worker(config: dict[str, Any] | None = None, capsys: pytest.CaptureFixture[str] | None = None,
           hf: dict[str, Any] | None = None) -> tuple[Worker, dict[str, Any]]:
    w = Worker(hf or tiny_config().to_dict(), **(config or {}))
    logged: dict[str, Any] = {}
    if capsys is not None:
        lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("core.plan: ")]
        assert len(lines) == 1
        logged = json.loads(lines[0].removeprefix("core.plan: "))
    return w, logged


def test_the_start_up_log_says_what_runs_in_each_phase(capsys: pytest.CaptureFixture[str]) -> None:
    w, logged = worker(capsys=capsys)
    try:
        assert logged == w.service("plan@1").resolved
        assert logged["platform"] == "torch_eager" and logged["verify"] == "disabled"
        assert logged["states"] == {"kv.paged": {"k": "f32", "v": "f32", "page_size": 4},
                                    "gdn.slots": {"conv": "f32", "temporal": "f32"}}
        assert logged["prefill"]["layers.attn@1"] == {
            "backend": "auto", "kv.store@1": "reference.kv.store@1 (fallback)",
            "attn.extend@1": "reference.attn.extend@1 (fallback)"}
        assert logged["decode"]["layers.gdn@1"] == {
            "backend": "auto", "gdn.conv1d_update@1": "reference.gdn.conv1d_update@1 (fallback)",
            "gdn.recurrent_update@1": "reference.gdn.recurrent_update@1 (fallback)"}
    finally:
        w.close()


def test_a_backend_named_per_phase_is_used_for_that_phase(capsys: pytest.CaptureFixture[str]) -> None:
    w, logged = worker({"attn.decode_backend": "reference", "gdn.backend": "reference"}, capsys)
    try:
        assert logged["decode"]["layers.attn@1"] == {"backend": "reference", "kv.store@1": "reference.kv.store@1",
                                                     "attn.decode@1": "reference.attn.decode@1"}
        assert logged["prefill"]["layers.attn@1"]["backend"] == "auto"
        assert {logged[p]["layers.gdn@1"]["backend"] for p in ("prefill", "decode")} == {"reference"}
    finally:
        w.close()


@pytest.mark.parametrize("key", ["attn.backend", "attn.prefill_backend", "gdn.decode_backend"])
def test_a_named_backend_without_a_fitting_solution_fails_the_start(key: str) -> None:
    with pytest.raises(Exception, match=rf"{key} = 'flashinfer': backend 'flashinfer' has no solution"):
        Worker(tiny_config().to_dict(), **{key: "flashinfer"})


def test_state_dtypes_follow_sglang_defaults() -> None:
    """KV follows the model's dtype, the conv state is bf16, the SSM state f32 or what the
    checkpoint's config asks for; SGLang's spellings are accepted."""
    w, _ = worker({"state.gdn.conv_dtype": "bf16"})
    try:
        gdn = w.service("state.gdn.slots@1")
        assert (w.service("state.kv.paged@1").k_buffer[3].dtype, gdn.conv.dtype, gdn.temporal.dtype) == \
            (torch.float32, torch.bfloat16, torch.float32)
    finally:
        w.close()
    w, _ = worker({"state.kv.dtype": "bfloat16"}, hf=tiny_config(mamba_ssm_dtype="bfloat16").to_dict())
    try:
        assert (w.service("state.kv.paged@1").k_buffer[3].dtype, w.service("state.gdn.slots@1").temporal.dtype) == \
            (torch.bfloat16, torch.bfloat16)
    finally:
        w.close()


def test_a_state_dtype_stage_2_does_not_keep_fails_the_start() -> None:
    with pytest.raises(Exception, match=r"state.kv.dtype = 'fp8_e4m3': stage 2 keeps states in bf16, f16 or f32"):
        Worker(tiny_config().to_dict(), **{"state.kv.dtype": "fp8_e4m3"})


READER = {"state.kv.dtype": "state.kv_paged", "state.gdn.ssm_dtype": "state.gdn_slots",
          "state.gdn.conv_dtype": "state.gdn_slots", "model.dtype": "core.state.weights",
          "state.kv.pages": "state.kv_paged", "state.gdn.slots": "state.gdn_slots",
          "holder.mem_fraction": "core.state.holder", "holder.state_ratio": "core.state.holder",
          "holder.max_slots": "core.state.holder", "holder.cpu_bytes": "core.state.holder"}
"""The unit that reads each key the state pools are built from: the stores their dtype keys, the
weight store the model config ``auto`` follows, the holder its sizing keys."""


def plan_change(w: Worker, key: str, value: Any) -> ProcActions:
    """``core.deps``'s plan for setting ``key`` on the running worker ``w``, from its own introspection."""
    from core.change.state import IdentityManifest
    from core.change.types import ChangeSet
    from core.common.framework import FrameworkModules
    from core.common.ids import Identity
    from core.common.result import Ok
    from core.deps.planner import World, plan
    from core.sourcetree import read_tree
    from core.supervisor import HostSpec, Mesh, RankPlacement, RoleSpec, Topology
    from tests.framework.support import PROC
    from tests.units.support.worker import ENGINE
    from unit.runtime.host import introspection

    intro = introspection(w.runtime, PROC, Identity("b"))
    topo = Topology([HostSpec(PROC.host, "addr", "/snapshots", gpus=[0])], Mesh({"tp": 1}),
                    [RankPlacement(0, PROC.host, 0)], [RoleSpec("worker", [], per_rank=True)])
    tree = read_tree(ENGINE)
    config = {**intro.config, key: value}
    r = plan(World(ChangeSet(base=Identity("b"), config=config), IdentityManifest(topology=topo),
                   IdentityManifest(config=config, topology=topo), tree, tree, {PROC: intro}, None,
                   FrameworkModules.default()))
    assert isinstance(r, Ok) and not r.value.restart_engine, r
    (pa,) = r.value.procs
    return pa


@pytest.mark.parametrize("key", sorted(READER))
def test_changing_what_the_state_pools_are_built_from_rebuilds_them_in_place(key: str) -> None:
    """A change applies to an empty engine (doc/specs/empty-apply.md): ``core.deps`` reloads the
    unit reading the key, and the runtime rebuilds the stores with it (their users go along),
    instead of restarting the engine as it did while requests could be in it."""
    value = {"state.kv.pages": 16, "state.gdn.slots": 4, "holder.mem_fraction": 0.5, "holder.state_ratio": 0.5,
             "holder.max_slots": 3, "holder.cpu_bytes": 1 << 24, "state.gdn.conv_dtype": "f16"}.get(key, "bf16")
    w, _ = worker()
    try:
        pa = plan_change(w, key, value)
        assert pa.reload is not None and READER[key] in [u.id for u in pa.reload.units_to_reload]
    finally:
        w.close()


@pytest.mark.parametrize(("key", "value"), [("state.kv.dtype", "bf16"), ("state.gdn.conv_dtype", "bf16"),
                                            ("state.kv.pages", 16)])
def test_a_worker_rebuilt_for_new_state_config_generates_as_one_started_with_it(key: str, value: Any) -> None:
    """The plan carried out on an empty worker: the next request's tokens equal those of a worker
    built with the new value from the start, and the capacity follows the new pools."""
    prompt = [5, 17, 3, 44, 9, 2]
    rebuilt, _ = worker()
    fresh, _ = worker({key: value})
    try:
        pa = plan_change(rebuilt, key, value)
        assert pa.reload is not None
        rebuilt.runtime.reload(pa.reload, values=pa.config).unwrap()
        tokens = []
        for w in (rebuilt, fresh):
            d = Driver(w)
            d.add(0, prompt)
            d.generate([0], 4)
            tokens.append(d.requests[0].outputs)
        assert tokens[0] == tokens[1]
        assert (rebuilt.service("executor@1").capacity().items
                == fresh.service("executor@1").capacity().items)
    finally:
        rebuilt.close()
        fresh.close()


def test_a_kv_rebuild_out_of_memory_rolls_back_to_a_worker_that_generates_as_before() -> None:
    """``model.dtype`` changes the KV the model declares (``state.kv.dtype`` is ``auto``), while the
    KV store itself is not reloaded: its running instance rebuilds the buffers. Out of memory there,
    the reload rolls back, the old model declares the old KV again, and the worker generates as
    before (#46 review: the store took the old declarations as unchanged, its buffers gone)."""
    prompt = [5, 17, 3, 44, 9, 2]
    w, _ = worker()
    try:
        d = Driver(w)
        d.add(0, prompt)
        d.generate([0], 4)
        before = list(d.requests[0].outputs)
        pa = plan_change(w, "model.dtype", "bf16")
        assert pa.reload is not None and "state.kv_paged" not in [u.id for u in pa.reload.units_to_reload]
        kv = w.service("state.kv.paged@1")
        kv.platform = failing = FailingAlloc(kv.platform)
        failing.arm()
        r = w.runtime.reload(pa.reload, values=pa.config)
        assert isinstance(r, Failed) and any("injected" in diag.reason for diag in r.diagnostics), r
        assert not failing.armed  # the KV store's allocation is what failed
        d.remove(0)
        d.rebuilt()
        d.add(1, prompt)
        d.generate([1], 4)
        assert d.requests[1].outputs == before
    finally:
        w.close()

"""The worker's start-up configuration, as SGLang's (stage2-contract.md §5.5.3): the state dtypes and
the attention / GDN backends per phase are fixed once at start and logged by ``core.plan``; an
explicitly named backend that cannot run here, or a state dtype stage 2 does not keep, fails the
start instead of falling back."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from tests.units.support.worker import Worker, tiny_config


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


@pytest.mark.parametrize(("key", "value"), [("state.kv.dtype", "bf16"), ("state.gdn.ssm_dtype", "bf16"),
                                            ("state.gdn.conv_dtype", "f16"), ("model.dtype", "bf16")])
def test_changing_what_a_state_dtype_comes_from_under_a_request_restarts_the_engine(key: str, value: str) -> None:
    """The stores read the keys the model resolves their dtypes from, so ``core.deps``, planning
    from the running worker's own introspection, refuses to reload them in place (§5.5.5): the
    model's reload would otherwise reallocate them empty under the request."""
    from core.change.state import IdentityManifest
    from core.change.types import ChangeSet
    from core.common.framework import FrameworkModules
    from core.common.ids import Identity
    from core.common.result import Ok
    from core.deps.planner import World, plan
    from core.sourcetree import read_tree
    from core.supervisor import HostSpec, Mesh, RankPlacement, RoleSpec, Topology
    from tests.framework.support import PROC
    from tests.units.support.worker import ENGINE, Driver
    from unit.runtime.host import introspection

    w, _ = worker()
    try:
        d = Driver(w)
        d.add(0, [5, 17, 3, 44])
        d.step({0: 4})
        intro = introspection(w.runtime, PROC, Identity("b"))
        topo = Topology([HostSpec(PROC.host, "addr", "/snapshots", gpus=[0])], Mesh({"tp": 1}),
                        [RankPlacement(0, PROC.host, 0)], [RoleSpec("worker", [], per_rank=True)])
        tree = read_tree(ENGINE)
        config = {**intro.config, key: value}
        r = plan(World(ChangeSet(base=Identity("b"), config=config), IdentityManifest(topology=topo),
                       IdentityManifest(config=config, topology=topo), tree, tree, {PROC: intro}, None,
                       FrameworkModules.default()))
        assert isinstance(r, Ok) and r.value.restart_engine, r
        assert key in r.value.notes[0] and "restart the engine" in r.value.notes[0]
    finally:
        w.close()

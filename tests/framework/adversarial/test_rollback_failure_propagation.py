"""Config and runtime rollback failures must survive the outer Abort acknowledgement."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

import pytest

from core.change.protocol import Abort
from core.change.types import ActionPlan, ChangeSet, ProcActions
from core.common.manifest import ConfigKeyDecl
from core.common.result import Failed, Ok, Result, fail
from tests.framework.change.harness import Harness, OLD, harness as harness
from unit.runtime import InstanceState, ReloadPlan


@pytest.mark.parametrize("failure", ["hook", "validation", "none"])
def test_config_rollback_failure_is_sticky(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    h = harness
    host = h.hosts[h.target]
    rt = host.runtime
    key = "audit.value"
    rt.config.declare(key, h.unit.id, ConfigKeyDecl("int", 1, "live"), 1).unwrap()
    instance = rt.snapshot().unit(h.unit.id).instance
    assert instance is not None
    runtime_state = {"value": 1}
    hooks: list[int] = []

    def on_config(value: int) -> None:
        hooks.append(value)
        if failure == "hook" and value == 1:
            raise RuntimeError("restoring old resource layout failed")
        runtime_state["value"] = value

    stage = rt.config.stage

    def checked_stage(values: dict[str, Any]) -> Result[dict[str, Any]]:
        if failure == "validation" and values == {key: 1}:
            return fail("context", "validate", "old value rejected", code="config.type")
        return stage(values)

    monkeypatch.setattr(rt.config, "stage", checked_stage)
    rt.config.add_hook(key, instance, on_config)
    h.action_plan = ActionPlan(procs=[ProcActions(h.target, config={key: 2})])

    async def main() -> None:
        await h.coord.start()
        h.coord.faults.add("after_execute")
        result = await h.coord.submit(ChangeSet(base=OLD, config={key: 2}))
        failed = failure != "none"
        assert result.status == ("failed" if failed else "aborted")
        assert h.coord.identity == host.identity == OLD
        assert host.paused == h.coord.recovery_required == failed
        assert runtime_state["value"] == (2 if failed else 1)
        assert rt.config.entries()[key].value == (2 if failure == "validation" else 1)
        if failed:
            expected = "config.hook_failed" if failure == "hook" else "config.type"
            assert {expected, "change.recovery_required"} <= {d.code for d in result.diagnostics}
        before = list(hooks)
        for _ in range(2):
            repeated = await h.call(h.target, Abort(result.change_id))
            assert isinstance(repeated, Failed if failed else Ok)
        assert hooks == before  # a repeated Abort must neither retry nor hide failed cleanup
        h.action_plan = ActionPlan()
        following = await h.coord.submit(ChangeSet(base=OLD))
        assert following.status == ("failed" if failed else "committed")

    asyncio.run(main())


@pytest.mark.parametrize("restores_old", [False, True], ids=["rollback-fails", "rollback-succeeds"])
def test_reload_failure_distinguishes_whether_old_instances_were_restored(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, restores_old: bool,
) -> None:
    h = harness
    host = h.hosts[h.target]
    unit_class = type(h.unit_of(h.target))
    original_init = cast(Callable[[object, Any, Any], None], unit_class.__init__)
    attempts = 0

    def init(self: object, ctx: Any, config: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1 or not restores_old:
            raise RuntimeError("resource unavailable")
        original_init(self, ctx, config)

    monkeypatch.setattr(unit_class, "__init__", init)
    h.action_plan = ActionPlan(procs=[ProcActions(h.target, reload=ReloadPlan(units_to_reload=[h.unit]))])

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        failed = not restores_old
        assert result.status == ("failed" if failed else "aborted")
        assert h.coord.identity == host.identity == OLD
        assert host.paused == h.coord.recovery_required == failed
        state = host.runtime.snapshot().unit(h.unit.id).state
        assert state is (InstanceState.FAILED if failed else InstanceState.ACTIVE)
        if failed:
            assert {"escalate.restart_process", "change.recovery_required"} <= {
                d.code for d in result.diagnostics}
        for _ in range(2):
            repeated = await h.call(h.target, Abort(result.change_id))
            assert isinstance(repeated, Failed if failed else Ok)
        assert attempts == 2  # one failed replacement, one attempt to restore the old instance
        h.action_plan = ActionPlan()
        following = await h.coord.submit(ChangeSet(base=OLD))
        assert following.status == ("failed" if failed else "committed")

    asyncio.run(main())

"""CPU checks for graph configuration, revocation and bounded decode inputs."""

from types import SimpleNamespace

import pytest
import torch

from core.common.ids import InstanceId, UnitId
from core.contracts.columns import Column
from core.contracts.records import BatchPlan
from core.plan.graph import Dependencies, buckets
from executors.pytorch.unit import check_plan
from tests.framework.support import PROC
from tests.units.support.worker import Worker, tiny_config
from unit.runtime.proxy import Provision, ServiceProxy


@pytest.mark.parametrize("values", [[0], [-1], [2, 2], [9], [True], [1.5]])
def test_bad_buckets_fail(values: list[int]) -> None:
    with pytest.raises(ValueError, match="graph_buckets"):
        buckets("auto", values, 8, 1024)


def test_default_buckets_and_required_coverage() -> None:
    assert buckets("auto", [], 6, 1024) == [1, 2, 4]
    assert buckets("required", [6, 1, 4], 6, 1024) == [1, 4, 6]
    with pytest.raises(ValueError, match="cover"):
        buckets("required", [], 6, 1024)
    with pytest.raises(ValueError, match="cuda_graph"):
        buckets("silent_fallback", [], 4, 1024)


def test_required_on_cpu_fails_at_start() -> None:
    with pytest.raises(Exception, match="platform has no CUDA graph support"):
        Worker(tiny_config().to_dict(), **{"plan.cuda_graph": "required"})


def test_decode_length_is_validated_before_state_changes() -> None:
    plan = BatchPlan(Column.of("i32", [0]), Column.of("i32", [2]), Column.of("i32", [0]),
                     Column.of("i64", [4, 5]), Column.of("u8", [0]), "decode", 0)
    with pytest.raises(ValueError, match="exactly one"):
        check_plan(plan)


def test_guard_checks_current_binding_and_provider_revocation() -> None:
    tensor = torch.zeros(2, 3)
    state = SimpleNamespace(generation=0, buffers=lambda: {"x": tensor})
    catalog = SimpleNamespace(version=0)
    old = Provision("test@1", InstanceId(PROC, UnitId("test"), 1), state)
    proxy = ServiceProxy(old)
    guard = Dependencies([proxy], [lambda: proxy], catalog, debug=True)
    assert guard.valid()
    tensor.add_(1)  # In-place contents are not dependencies.
    assert guard.valid()
    tensor = tensor.transpose(0, 1)  # Same address, different strides, missing generation bump.
    assert not guard.valid()
    tensor = tensor.transpose(0, 1)
    state.generation += 1
    assert not guard.valid()
    state.generation = 0
    old.revoked = True
    replacement = Provision("test@1", InstanceId(PROC, UnitId("test"), 2), state)
    assert not guard.valid()
    assert Dependencies([ServiceProxy(replacement)], [lambda: state], catalog).valid()

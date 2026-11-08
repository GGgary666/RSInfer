from pathlib import Path

from core.common.framework import FrameworkModules
from core.common.ids import HostId, UnitId
from core.manifest.checks import check_tree
from core.supervisor.topology import HostSpec, Mesh, RankPlacement, RoleSpec, Topology
from unit.runtime.imports import ImportGraph, scan_root


def test_frontend_and_request_table_units_obey_the_engine_rules(tmp_path: Path) -> None:
    host = HostId("front-host")
    topology = Topology([HostSpec(host, "test-address", str(tmp_path))], Mesh({"tp": 1}),
                        [RankPlacement(0, host, 0)], [
                            RoleSpec("frontend", [UnitId(f"frontend.{name}") for name in ("http", "tokenizer", "detok")],
                                     hosts=[host]),
                            RoleSpec("sched", [UnitId("sched.reqs")], hosts=[host]),
                        ])
    engine = Path(__file__).resolve().parents[3] / "engine"
    errors, _ = check_tree(engine, topology, FrameworkModules.default())
    assert errors == []


def test_frontend_imports_do_not_reach_worker_contracts() -> None:
    """Editing scheduler/worker records must not reload the HTTP or tokenizer units."""
    engine = Path(__file__).resolve().parents[3] / "engine"
    graph = ImportGraph(scan_root(engine))
    reached = graph.imports_closure(["frontend.http.unit", "frontend.tokenizer.unit", "frontend.detok.unit"])
    assert "core.contracts.requests" in reached
    assert "core.contracts.serving" in reached
    assert not {"core.contracts.records", "core.contracts.plan", "core.contracts.storage"} & reached

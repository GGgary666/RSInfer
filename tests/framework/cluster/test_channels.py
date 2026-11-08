"""ipc.md §6 with real processes: data channels follow their peers when the supervisor replaces
the process at either end."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import msgspec
import pytest

from core.common.ids import ProcId
from core.supervisor import DesiredProc
from tests.framework.cluster.harness import Cluster, build_tree, cluster, proc, role, topology


async def until(check: Callable[[], bool], what: str, within: float = 30.0) -> None:
    deadline = time.monotonic() + within
    while not check():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.05)  # never block: the supervisor runs in this event loop


def runs(c: Cluster, sink: ProcId) -> list[tuple[str, int, list[int]]]:
    """What the current sink incarnation received: (sender token, reader generation, numbers)."""
    lines = c.log_of(sink)
    starts = [i for i, line in enumerate(lines) if line.startswith("sink up")]
    out: list[tuple[str, int, list[int]]] = []
    for line in lines[starts[-1] + 1:] if starts else []:
        if not line.startswith("got "):
            continue  # what toyc.tunable logs
        _, record, _, gen = line.split()
        token, n = record.split(":")
        if not out or out[-1][:2] != (token, int(gen)):
            out.append((token, int(gen), []))
        out[-1][2].append(int(n))
    return out


def tokens(c: Cluster, sender: ProcId) -> list[str]:
    return [line.split()[2] for line in c.log_of(sender) if line.startswith("sender up")]


@pytest.mark.parametrize("sink_host", ["host-a", "host-b"], ids=["shm", "tcp"])
def test_data_channels_follow_their_peers_across_process_restarts(tmp_path: Path, sink_host: str) -> None:
    async def main() -> None:
        hosts = sorted({"host-a", sink_host})
        async with cluster(tmp_path, hosts) as c:
            identity, files = c.ingest(build_tree(tmp_path))
            sender, sink = proc("host-a", "sched", 0), proc(sink_host, "worker", 0)

            def extra(d: DesiredProc) -> dict[str, object]:
                return {"toyc.sink": msgspec.to_builtins(sink), "toyc.source_host": "host-a"}

            # toyc.tunable owns toyc.log_dir, where the toy units write what they did
            roles = [role("sched", ["toyc.tunable", "toyc.sender"], hosts=["host-a"]),
                     role("worker", ["toyc.tunable", "toyc.sink"], hosts=[sink_host])]
            topo = topology(tmp_path, hosts, ["host-a"], roles)
            (await c.sup.apply_topology(topo, identity, files, c.plan_for(identity, extra))).unwrap()
            first = tokens(c, sender)[0]
            await until(lambda: any(t == first and len(ns) >= 5 for t, _, ns in runs(c, sink)), "the first records")

            # The reader's process is replaced: a new ring or port, published by the new process;
            # the sender finds it again and keeps sending.
            (await c.sup.restart([sink])).unwrap()
            await until(lambda: any(t == first and len(ns) >= 5 for t, _, ns in runs(c, sink)),
                        "records at the new reader")
            assert "sender gen 2" in c.log_of(sender)

            # The writer's process is replaced: the same reader takes the new producer.
            (await c.sup.restart([sender])).unwrap()
            second = tokens(c, sender)[-1]
            assert second != first
            await until(lambda: any(t == second and len(ns) >= 5 for t, _, ns in runs(c, sink)),
                        "records of the new sender")

            got = runs(c, sink)
            assert [(t, g) for t, g, _ in got] == [(first, 1), (second, 2)]
            for _, _, numbers in got:  # nothing lost or reordered while a connection lasted
                assert numbers == list(range(numbers[0], numbers[0] + len(numbers)))
            await c.sup.stop_all()

    asyncio.run(main())


def test_a_reader_opened_after_check_in_is_published(tmp_path: Path) -> None:
    """The sink opens its reader once it runs: its ring name comes from the node agent through
    the supervisor, the host publishes the address on its own, and the sender finds it."""
    async def main() -> None:
        async with cluster(tmp_path, ["host-a"]) as c:
            identity, files = c.ingest(build_tree(tmp_path))
            sender, sink = proc("host-a", "sched", 0), proc("host-a", "worker", 0)

            def extra(d: DesiredProc) -> dict[str, object]:
                return {"toyc.sink": msgspec.to_builtins(sink), "toyc.source_host": "host-a",
                        "toyc.late_reader": True}

            roles = [role("sched", ["toyc.tunable", "toyc.sender"], hosts=["host-a"]),
                     role("worker", ["toyc.tunable", "toyc.sink"], hosts=["host-a"])]
            topo = topology(tmp_path, ["host-a"], ["host-a"], roles)
            (await c.sup.apply_topology(topo, identity, files, c.plan_for(identity, extra))).unwrap()
            checkin = c.sup.checkin_of(sink)
            assert checkin is not None and "in" not in checkin.endpoints
            first = tokens(c, sender)[0]
            await until(lambda: any(t == first and len(ns) >= 5 for t, _, ns in runs(c, sink)), "records")
            address = c.sup.endpoint(sink, "in")
            assert address is not None and address.shm is not None
            await c.sup.stop_all()

    asyncio.run(main())

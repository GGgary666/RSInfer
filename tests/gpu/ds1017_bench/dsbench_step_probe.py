"""Expose existing worker timing events in a disposable benchmark source tree.

This writes one log record per completed inference step. It adds no CUDA events,
device synchronization, profiler session, or periodic hardware query.
"""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    path = Path(sys.argv[1]) / "engine/core/sched/loop.py"
    source = path.read_text()
    if 'print("DSBENCH_STEP"' in source:
        return
    changes = {
        "from core.ipc.records import": "from core.contracts.records import StepStats\nfrom core.ipc.records import",
        "parts = read_sections(record, {}, {Capacity.SECTION:":
            "parts = read_sections(record, {}, {StepStats.SECTION: StepStats, Capacity.SECTION:",
        "        flight.answers[rank] = answer": """        # PROBE: consume timing events already produced by the executor.
        stat = parts.get(StepStats.SECTION)
        if isinstance(stat, StepStats) and stat.step == flight.epoch:
            stats = getattr(flight, "_dsbench_stats", {})
            stats[rank] = (stat.host_us, stat.device_us, stat.idle_before_us)
            setattr(flight, "_dsbench_stats", stats)
        flight.answers[rank] = answer""",
        "    def _retire(self, flight: Flight) -> None:\n": """    def _retire(self, flight: Flight) -> None:
        # PROBE: step accounting, active only while benchmark requests execute.
        print("DSBENCH_STEP", flight.epoch, self.clock(),
              [sum(n for row, n in flight.counts.items() if row in flight.prefills and self._domain(row) == d)
               for d in range(len(self.domains))], getattr(flight, "_dsbench_stats", {}), flush=True)
""",
    }
    for old, new in changes.items():
        assert source.count(old) == 1, f"benchmark probe needs updating: {old}"
        source = source.replace(old, new)
    path.write_text(source)


if __name__ == "__main__":
    main()

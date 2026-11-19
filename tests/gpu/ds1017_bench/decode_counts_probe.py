"""Switch DSBENCH_STEP counts to full-DP pure-decode row counts in a disposable staged source tree."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1]) / "engine/core/sched/loop.py"
    source = path.read_text(encoding="utf-8")
    if "_decode_bench_counts" in source:
        return
    prefill_expr = (
        "[sum(n for row, n in flight.counts.items() if row in flight.prefills and self._domain(row) == d)\n"
        "               for d in range(len(self.domains))]"
    )
    assert source.count(prefill_expr) == 1, "apply dsbench_step_probe.py before decode_counts_probe.py"
    source = source.replace(prefill_expr, 'getattr(flight, "_decode_bench_counts")')
    anchor = "                                   unhomed=unhomed))"
    assert source.count(anchor) == 1, "decode_counts_probe needs updating: flight anchor not unique"
    source = source.replace(
        anchor,
        anchor + "\n        self.flights[-1]._decode_bench_counts = [len(step.rows) if not step.prefills else -1 for step in steps]",
    )
    path.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()

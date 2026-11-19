"""Record true admission prefix-hit positions in a disposable staged source tree."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1]) / "engine/core/sched/loop.py"
    source = path.read_text(encoding="utf-8")
    if "AGENTX_ADMIT" in source:
        return
    needle = "        self.table.start_step(epoch, rows, counts)\n"
    assert source.count(needle) == 1, "admit probe needs updating: start_step anchor not unique"
    injection = (
        needle
        + "        for step in steps:\n"
        + "            for request in step.admitted:\n"
        + '                print("AGENTX_ADMIT", request.input.request, request.computed, flush=True)\n'
    )
    path.write_text(source.replace(needle, injection), encoding="utf-8")


if __name__ == "__main__":
    main()

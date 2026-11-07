"""One log file per process, so a test can see what each unit did (a library, not a unit)."""

from __future__ import annotations

import os
from typing import Any


class Journal:
    def __init__(self, directory: str, ctx: Any) -> None:
        p = ctx.instance.proc
        self.path = os.path.join(directory, f"{p.host}-{p.role}-{p.index}.log") if directory else ""

    def write(self, text: str) -> None:
        if self.path:
            with open(self.path, "a") as f:
                f.write(text + "\n")

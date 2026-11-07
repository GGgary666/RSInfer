"""A scheduling policy: first come, first served, one request at a time (modules.html ③)."""

from __future__ import annotations

from typing import Any

from ..journal import Journal

VERSION = 1


class Policy:
    version = VERSION

    def pick(self, progress: dict[str, int]) -> list[str]:
        return sorted(progress)[:1]


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.register("scn.policies@1", Policy())
        Journal(config["scn.log_dir"], ctx).write(f"policy v{VERSION} gen{ctx.instance.generation}")

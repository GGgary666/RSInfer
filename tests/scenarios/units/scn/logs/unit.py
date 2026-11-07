"""Owns the log directory key every scenario unit reads."""

from __future__ import annotations

from typing import Any


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.directory = config["scn.log_dir"]

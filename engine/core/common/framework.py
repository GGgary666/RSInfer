"""Which modules belong to the framework core (``engine/framework.toml``)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

FRAMEWORK_TOML = "framework.toml"


@dataclass(frozen=True)
class FrameworkModules:
    prefixes: frozenset[str]
    exact: frozenset[str]

    def contains(self, module: str) -> bool:
        if module in self.exact:
            return True
        parts = module.split(".")
        for i in range(len(parts), 0, -1):
            head = ".".join(parts[:i])
            if head in self.prefixes and head not in self.exact:
                return True
        return False

    @classmethod
    def load(cls, engine_root: Path) -> FrameworkModules:
        raw = tomllib.loads((engine_root / FRAMEWORK_TOML).read_text())["framework"]
        exact = frozenset[str](raw.get("exact", []))
        return cls(frozenset[str](raw["modules"]), exact)

    @classmethod
    def default(cls) -> FrameworkModules:
        """The list shipped next to this file's package root."""
        return cls.load(Path(__file__).resolve().parents[2])

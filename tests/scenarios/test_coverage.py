"""Roadmap stage 1 checks over the whole test suite and engine tree: coverage and teardown."""

from __future__ import annotations

import ast

from tests.scenarios.coverage import EDGES, KINDS, LADDER
from tests.scenarios.harness import ENGINE, REPO


def declared() -> dict[str, set[object]]:
    """Union of every ``@pytest.mark.covers(...)`` in tests/, read from the source."""
    out: dict[str, set[object]] = {"edges": set(), "ladder": set(), "kinds": set()}
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "covers":
                for kw in node.keywords:
                    if kw.arg in out:
                        out[kw.arg] |= set(ast.literal_eval(kw.value))
    return out


def test_every_edge_rung_and_module_kind_is_covered() -> None:
    """Every kind of edge, every rung of the action ladder and every kind of module is exercised by
    at least one test that says so (and asserts it)."""
    got = declared()
    assert set(EDGES) - got["edges"] == set()
    assert set(LADDER) - got["ladder"] == set()
    assert set(KINDS) - got["kinds"] == set()
    assert got["edges"] <= set(EDGES) and got["ladder"] <= set(LADDER) and got["kinds"] <= set(KINDS)


def test_the_engine_does_not_need_the_rsi_plane() -> None:
    """Teardown check: nothing in engine/ imports ``rsi``; the scenario tests start the engine,
    serve and apply without it."""
    offenders: list[str] = []
    for path in sorted(ENGINE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else []
            if any(n == "rsi" or n.startswith("rsi.") for n in names):
                offenders.append(str(path.relative_to(REPO)))
    assert offenders == []
    assert not (ENGINE / "rsi").exists()

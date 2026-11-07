"""What the coverage check counts (roadmap stage 1, "覆盖检查"; modules.html ③).

A test declares what it exercises with ``@pytest.mark.covers(edges=..., ladder=..., kinds=...)``
and asserts it (scenario tests check the action plan that actually ran).
``test_coverage.py`` reads these declarations from the source and requires every
item below to be covered by at least one test.
"""

from __future__ import annotations

EDGES = ("data", "call", "state", "derive", "ctrl", "decl")
"""The six kinds of edges between modules (modules.html ③ §2)."""
LADDER = (0, 1, 2, 3, 4, 5, 6)
"""The action ladder: 0 nothing, 1 config in place, 2 reload a unit, 3 rebuild a derived artifact,
4 state action, 5 restart a process, 6 restart the engine (modules.html ③ §3)."""
KINDS = ("logic", "state", "framework")
"""The three kinds of modules (modules.html ③ §1)."""

"""core.identity (doc/specs/control-plane.md §2): candidate identity = f(base, change-set)."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from core.change.state import IdentityManifest, apply_changeset
from core.change.types import ChangeSet
from core.common.ids import Identity
from core.identity.unit import IdentityFunction
from platforms.base import PlatformInfo

paths = st.text("abc/", min_size=1, max_size=6).map(lambda p: f"engine/{p}.py")
hashes = st.text("0123456789abcdef", min_size=4, max_size=8)
files = st.dictionaries(paths, hashes, max_size=6)
config = st.dictionaries(st.sampled_from(["a.k", "b.k", "c.k"]), st.integers(0, 9), max_size=3)


@given(files, files, config)
def test_the_same_base_and_change_set_always_give_the_same_identity(base: dict[str, str], patch: dict[str, str],
                                                                    cfg: dict[str, int]) -> None:
    f = IdentityFunction()
    start = IdentityManifest(files=base)
    cs = ChangeSet(base=Identity("x"), files=dict(patch), config=cfg)
    reordered = ChangeSet(base=Identity("x"), files=dict(reversed(list(patch.items()))),
                          config=dict(reversed(list(cfg.items()))))
    assert f.identity(apply_changeset(start, cs)) == f.identity(apply_changeset(start, reordered))
    # the path does not matter, only where it ends: starting fresh from the result gives the same name
    fresh = IdentityManifest(files={**base, **patch}, config=dict(cfg))
    assert f.identity(apply_changeset(start, cs)) == f.identity(fresh)


def test_contents_and_platform_change_the_identity() -> None:
    f = IdentityFunction()
    m = IdentityManifest(files={"engine/a.py": "h1"})
    assert f.identity(m) != f.identity(IdentityManifest(files={"engine/a.py": "h2"}))
    assert f.identity(m) != f.identity(IdentityManifest(files={"engine/a.py": "h1"}, config={"k": 1}))
    assert f.identity(m, PlatformInfo("fake", {"driver": "1"})) != f.identity(m, PlatformInfo("fake", {"driver": "2"}))


def test_diff_names_what_differs() -> None:
    f = IdentityFunction()
    a = IdentityManifest(files={"engine/a.py": "h1", "engine/b.py": "h2"}, config={"k": 1})
    b = IdentityManifest(files={"engine/a.py": "h1", "engine/c.py": "h3"}, config={"k": 2})
    d = f.diff(a, b)
    assert d["files"] == {"engine/b.py": ("h2", None), "engine/c.py": (None, "h3")}
    assert d["config"] == {"k": (1, 2)} and d["topology"] is False

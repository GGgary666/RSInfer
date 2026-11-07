"""The coordinator's fault points (core.change.faults): a misspelt point fails the test that arms it."""

from __future__ import annotations

import pytest

from core.change.faults import POINTS, Faults


def test_a_misspelt_fault_point_is_refused() -> None:
    f = Faults()
    with pytest.raises(ValueError, match="after_exceute"):
        f.add("after_exceute")
    with pytest.raises(ValueError):
        f |= {"after_pause", "skip_abrot"}


def test_one_shot_points_fire_once_and_held_points_hold() -> None:
    f = Faults()
    f |= {"after_pause", "no_decisions"}
    assert f.take("after_pause") and not f.take("after_pause")
    assert f.armed("no_decisions") and f.armed("no_decisions")
    assert set(POINTS) >= {"after_pause", "no_decisions", "delay_commit"}

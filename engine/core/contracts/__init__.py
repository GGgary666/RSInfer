"""Interfaces between the stage-2 units (doc/specs/stage2-contract.md). Draft: a change
here needs the review of both sides of the interface it touches.

This package is a library, not a unit and not part of the framework core. Units on
both sides of an interface import it: they may not import each other
(``manifest.cross_unit_import``), and a change to an interface module reloads every
importer in the same epoch, which is what an interface change needs (modules.html ⑥).

Nothing here imports torch: CI runs without it. Tensors are typed ``Tensor`` (an
alias of ``Any``): a ``torch.Tensor`` on the torch platforms.
"""

from __future__ import annotations

from typing import Any

type Tensor = Any
"""A ``torch.Tensor`` on ``torch_cpu`` and ``cuda_sm100``; never constructed by the contracts."""

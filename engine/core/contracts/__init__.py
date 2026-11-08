"""Interfaces between the stage-2 units (doc/specs/stage2-contract.md). Draft: a change
here needs the review of both sides of the interface it touches.

This package is a library, not a unit and not part of the framework core. Units on
both sides of an interface import it: they may not import each other
(``manifest.cross_unit_import``), and a change to an interface module reloads every
importer in the same epoch, which is what an interface change needs (modules.html ⑥).

One module per seam, importing only what its seam needs: a unit reloads when any module
its imports reach changes (``core.deps``), so a change to one seam must not reach the
units of another (stage2-contract.md §3). This file stays empty: every module of the
package depends on it.

Nothing here imports torch: CI runs without it. Tensors are typed ``Tensor``
(``core.contracts.tensor``, an alias of ``Any``): a ``torch.Tensor`` on the torch platforms.
"""

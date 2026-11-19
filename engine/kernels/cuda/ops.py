"""The C++/CUDA kernels of ``kernels.cuda`` and how they are built.

``build`` compiles ``kv_store_cu.CUDA_SOURCE`` with PyTorch's C++ extension loader (``ninja`` and
the CUDA compiler on ``PATH``, as FlashInfer's JIT needs) into its cache directory
(``TORCH_EXTENSIONS_DIR``, by default under the user's home). Keeping the CUDA C++ source in a
Python module lets ``core.deps``'s import graph detect C++ kernel edits during hot ``apply``. The
compiled module is named after a digest of the source: a changed source builds and loads a fresh
module beside the old one (native code cannot be unloaded); an unchanged one loads from cache.

A build takes about a minute. ``build`` is also the unit's build step (``[unit] build`` in
``unit.toml``): a change that edits the source runs it while the engine serves, in a process of its
own, and the reload at the pause loads from cache. Before build steps (2026-11-11) the reload built
it while the worker was paused, outlasted the change's time limit, and the change was rolled back.
Building by hand ahead of a change still works, in the worker's environment:

    PYTHONPATH=engine uv run python -m kernels.cuda.ops
"""

from __future__ import annotations

from typing import Any

import torch

from kernels.cuda_build import build_cuda_extension, only_builder as only_builder
from kernels.cuda.kv_store_cu import CUDA_SOURCE


def build() -> Any:
    """The compiled module; ``None`` without a CUDA device (a build step runs on every worker)."""
    return build_cuda_extension("agx_kernels_cuda", "kv_store.cu", CUDA_SOURCE)


def kv_store(module: Any, k_cache: torch.Tensor, v_cache: torch.Tensor, loc: torch.Tensor, cache_k: torch.Tensor,
             cache_v: torch.Tensor) -> None:
    """``kv.store@1`` in one kernel (``kv_store_cu``). Converts first, as the reference does, when
    the pool has another dtype; stores as the reference does when the kernel does not take the
    tensors (``store`` says so before it launches anything)."""
    if cache_k.dtype != k_cache.dtype:
        cache_k = cache_k.to(k_cache.dtype)
        cache_v = cache_v.to(v_cache.dtype)
    if not module.store(k_cache, v_cache, loc, cache_k, cache_v):
        k_cache[loc] = cache_k
        v_cache[loc] = cache_v


if __name__ == "__main__":
    build()

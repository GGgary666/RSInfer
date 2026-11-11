"""The C++/CUDA kernels of ``kernels.cuda`` and how they are built.

``build`` compiles the sources next to this file with PyTorch's C++ extension loader (``ninja`` and
the CUDA compiler on ``PATH``, as FlashInfer's JIT needs) into its cache directory
(``TORCH_EXTENSIONS_DIR``, by default under the user's home). The module is named after a digest of
the sources: a changed source is another module, built and loaded beside the old one, which stays
in the process (native code cannot be unloaded); an unchanged one is loaded from the cache.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import torch

SOURCES = ("kv_store.cu",)


def build() -> Any:
    from torch.utils.cpp_extension import load

    paths = [Path(__file__).with_name(name) for name in SOURCES]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()[:16]
    name = f"agx_kernels_cuda_{digest}"
    started = time.perf_counter()
    module = load(name=name, sources=[str(p) for p in paths], extra_cuda_cflags=["-O3"], verbose=False)
    print(f"kernels.cuda: {name} ready in {time.perf_counter() - started:.1f} s", file=sys.stderr, flush=True)
    return module


def kv_store(module: Any, k_cache: torch.Tensor, v_cache: torch.Tensor, loc: torch.Tensor, cache_k: torch.Tensor,
             cache_v: torch.Tensor) -> None:
    """``kv.store@1`` in one kernel (``kv_store.cu``). Converts first, as the reference does, when
    the pool has another dtype; stores as the reference does when the kernel does not take the
    tensors (``store`` says so before it launches anything)."""
    if cache_k.dtype != k_cache.dtype:
        cache_k = cache_k.to(k_cache.dtype)
        cache_v = cache_v.to(v_cache.dtype)
    if not module.store(k_cache, v_cache, loc, cache_k, cache_v):
        k_cache[loc] = cache_k
        v_cache[loc] = cache_v

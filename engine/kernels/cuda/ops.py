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

import contextlib
import fcntl
import hashlib
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import torch

from kernels.cuda.kv_store_cu import CUDA_SOURCE


def build() -> Any:
    """The compiled module; ``None`` without a CUDA device (a build step runs on every worker)."""
    from torch.utils.cpp_extension import _get_build_directory, load

    if not torch.cuda.is_available():
        return None

    raw = CUDA_SOURCE.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    name = f"agx_kernels_cuda_{digest}"
    build_dir = Path(_get_build_directory(name, False))
    build_dir.mkdir(parents=True, exist_ok=True)
    with only_builder(build_dir):
        cu_path = build_dir / "kv_store.cu"
        if not cu_path.exists() or cu_path.read_bytes() != raw:
            # Written beside and moved into place: a module loaded from here meanwhile (only
            # ``load`` builds under the lock) sees the whole file or none of it.
            with tempfile.NamedTemporaryFile(dir=build_dir, suffix=".cu", delete=False) as tmp:
                tmp.write(raw)
            Path(tmp.name).replace(cu_path)
        if not any(build_dir.glob(f"{name}*.so")):
            print(f"kernels.cuda: building {name}, about a minute", file=sys.stderr, flush=True)
        started = time.perf_counter()
        module = load(name=name, sources=[str(cu_path)], build_directory=str(build_dir), extra_cuda_cflags=["-O3"],
                      verbose=False)
    print(f"kernels.cuda: {name} ready in {time.perf_counter() - started:.2f} s", file=sys.stderr, flush=True)
    return module


@contextlib.contextmanager
def only_builder(build_dir: Path) -> Generator[None]:
    """Hold ``build_dir`` alone: another process building the same source (a worker of another
    rank, a build step, a unit's load) waits here, then finds it built.

    PyTorch locks the directory too, with a file it creates (``lock``) and deletes when it is done;
    a builder killed before it deleted it leaves it there, and every later build of the source waits
    on it forever (2026-11-12, Codex's review of #55). This lock (``flock``) goes with its holder
    however it ends, and every build of the directory takes it first: whoever holds it, finding a
    ``lock`` there, finds one a dead builder left."""
    with open(build_dir / ".agx-build.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        stale = build_dir / "lock"
        if stale.exists():
            print(f"kernels.cuda: removing {stale}, left by a build that did not finish", file=sys.stderr,
                  flush=True)
            stale.unlink(missing_ok=True)
        yield


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

"""Shared PyTorch CUDA extension builder for worker kernels and device state stores."""

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

def build_cuda_extension(prefix: str, filename: str, source: str) -> Any:
    """Compile independent CUDA extensions into separate, source-keyed build directories."""
    from torch.utils.cpp_extension import _get_build_directory, load

    if not torch.cuda.is_available():
        return None

    raw = source.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    name = f"{prefix}_{digest}"
    build_dir = Path(_get_build_directory(name, False))
    build_dir.mkdir(parents=True, exist_ok=True)
    with only_builder(build_dir):
        cu_path = build_dir / filename
        if not cu_path.exists() or cu_path.read_bytes() != raw:
            # Written beside and moved into place: a module loaded from here meanwhile (only
            # ``load`` builds under the lock) sees the whole file or none of it.
            with tempfile.NamedTemporaryFile(dir=build_dir, suffix=".cu", delete=False) as tmp:
                tmp.write(raw)
            Path(tmp.name).replace(cu_path)
        if not any(build_dir.glob(f"{name}*.so")):
            print(f"kernels.cuda: building {name}, about a minute", file=sys.stderr, flush=True)
        started = time.perf_counter()
        module = load(name=name, sources=[str(cu_path)], build_directory=str(build_dir), with_cuda=True, extra_cuda_cflags=["-O3"],
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
    with open(build_dir / ".rsinfer-build.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        stale = build_dir / "lock"
        if stale.exists():
            print(f"kernels.cuda: removing {stale}, left by a build that did not finish", file=sys.stderr,
                  flush=True)
            stale.unlink(missing_ok=True)
        yield

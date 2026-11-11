"""For every GPU test: the kernel libraries of the ``gpu`` extra are imported once per process,
before any test. The tests' import isolation (tests/conftest.py) forgets the modules a test
imported, so the next test's kernel units would import the libraries again, and FlashInfer's
CuTe DSL dependency is a native extension that aborts the process when it is imported twice. An
engine process imports them once: a reload drops the units' own modules only."""

from __future__ import annotations

for _library in ("triton", "flashinfer"):
    try:
        __import__(_library)
    except ImportError:  # an environment without them: the kernel units register nothing
        pass

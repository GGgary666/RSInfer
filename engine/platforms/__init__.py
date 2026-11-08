"""Platforms (doc/specs/platform.md). A topology role names the one its processes run on."""

PLATFORMS = ("fake", "torch_eager")
"""``fake``: CI's platform for the engine's mechanisms, no tensors. ``torch_eager``: real tensors
on the CPU or the process's CUDA devices, no graphs."""

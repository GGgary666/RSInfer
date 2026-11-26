"""Platforms (doc/specs/platform.md). A topology role names the one its processes run on."""

PLATFORMS = ("fake", "torch_eager", "cuda_graph", "ascend")
"""``fake``: CI's platform for the engine's mechanisms, no tensors. ``torch_eager``: real tensors
on the CPU or the process's CUDA devices, no graphs. ``cuda_graph``: CUDA tensors and graphs.
``ascend``: eager tensors on the process's NPU devices."""

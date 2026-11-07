"""platform.md §4 tests 1–3 on the fake platform."""

from __future__ import annotations

import pytest

from core.context import RuntimeFlags, ValueDep, report_value
from platforms.base import StaleGraph
from platforms.fake import FakeKernel, FakeKernelError, FakeOOM, FakePlatform


def test_grow_keeps_addresses_and_graph_still_replays() -> None:
    p = FakePlatform()
    vr = p.reserve(1 << 20, 0)
    p.map(vr, 0, 4096)
    kv = p.handle("kv_pool", vr.view(0, (1024,), "f32"))
    with p.graph().capture() as g:
        p.launch("attn", [kv])
    addr = kv.buffer.addr

    p.map(vr, 4096, 4096)  # grow: map more physical memory behind the same range
    tail = vr.view(4096, (1024,), "f32")

    assert kv.buffer.addr == addr and tail.addr == addr + 4096
    g.replay()
    assert p.kernels["attn"].calls == 1


def test_by_value_change_without_recapture_is_stale() -> None:
    p = FakePlatform()
    flags = RuntimeFlags()
    flags.set("deepep.mode", "normal")
    selected = {"attn": "flashinfer"}
    buf = p.alloc(1024, 0)
    with p.graph().capture() as g:
        flags.get("deepep.mode")
        report_value(ValueDep("solution", "attn", selected["attn"], lambda: selected["attn"]))
        p.launch("attn", [buf], solution="flashinfer", scalars={"scale": 0.5})
    g.replay()

    selected["attn"] = "triton"
    with pytest.raises(StaleGraph, match="solution 'attn'"):
        g.replay()
    selected["attn"] = "flashinfer"
    flags.set("deepep.mode", "low_latency")
    with pytest.raises(StaleGraph, match="deepep.mode"):
        g.replay()
    flags.set("deepep.mode", "normal")
    g.replay()


def test_same_address_new_contents_replays() -> None:
    p = FakePlatform()
    w = p.handle("weights", p.alloc(4096, 0, shape=(1024,), dtype="f32"))
    with p.graph().capture() as g:
        p.launch("matmul", [w])
    w.buffer.write()  # refit: same address, same shape, new contents
    g.replay()


def test_reallocated_buffer_without_recapture_is_stale() -> None:
    p = FakePlatform()
    kv = p.handle("kv_pool", p.alloc(4096, 0, shape=(1024,), dtype="f32"))
    with p.graph().capture() as g:
        p.launch("attn", [kv])
    old = kv.buffer
    kv.buffer = p.alloc(4096, 0, shape=(1024,), dtype="f32")
    p.free(old)
    with pytest.raises(StaleGraph, match="moved"):
        g.replay()


def test_shape_change_is_stale() -> None:
    p = FakePlatform()
    vr = p.reserve(8192, 0)
    p.map(vr, 0, 8192)
    kv = p.handle("kv_pool", vr.view(0, (1024,), "f32"))
    with p.graph().capture() as g:
        p.launch("attn", [kv])
    kv.buffer = vr.view(0, (2048,), "f32")
    with pytest.raises(StaleGraph, match="is now"):
        g.replay()


def test_graph_reading_freed_memory_is_caught() -> None:
    p = FakePlatform()
    buf = p.alloc(1024, 0)
    with p.graph().capture() as g:
        p.launch("attn", [buf])
    p.free(buf)
    with pytest.raises(StaleGraph, match="freed"):
        g.replay()


def test_kernel_fault_injection_and_memory_limits() -> None:
    p = FakePlatform(total_bytes=8192)
    p.kernels["attn"] = FakeKernel(fail=True)
    with pytest.raises(FakeKernelError):
        p.launch("attn")
    p.alloc(4096, 0)
    with pytest.raises(FakeOOM):
        p.alloc(8192, 0)

"""The platform contract on every platform (platform.md §3), what torch_eager adds (§2.3), and
how a process picks its platform."""

from __future__ import annotations

import pytest
import torch

from core.common.ids import Identity
from core.supervisor.protocol import ProcSpec
from platforms import PLATFORMS
from platforms.base import Platform
from platforms.fake import FakePlatform
from platforms.torch_eager import DTYPES, TorchEagerPlatform
from tests.framework.support import PROC
from unit.runtime import ProcessPlan
from unit.runtime.host import make_platform

MAKERS = {"fake": FakePlatform, "torch_eager": TorchEagerPlatform}


@pytest.fixture(params=MAKERS)
def platform(request: pytest.FixtureRequest) -> Platform:
    return MAKERS[request.param]()


def test_every_platform_name_has_a_maker() -> None:
    assert set(MAKERS) | {"ascend", "cuda_graph"} == set(PLATFORMS)


def test_a_buffer_has_the_shape_dtype_and_size_asked_for(platform: Platform) -> None:
    buf = platform.alloc(0, 0, shape=(4, 8), dtype="bf16")
    assert (buf.device, buf.shape, buf.dtype, buf.nbytes) == (0, (4, 8), "bf16", 64)
    raw = platform.alloc(4096, 0)
    assert raw.shape == (4096,) and raw.dtype == "u8" and raw.nbytes == 4096
    assert raw.addr != buf.addr


def test_a_buffer_covers_every_byte_asked_for(platform: Platform) -> None:
    """``nbytes`` is a floor: more than ``shape`` needs (padding), or not a whole number of
    elements, is still all there (PR #24 review)."""
    padded = platform.alloc(4096, 0, shape=(1,), dtype="f32")
    assert (padded.nbytes, padded.shape) == (4096, (1,))
    ragged = platform.alloc(5, 0, dtype="f32")
    assert (ragged.nbytes, ragged.shape) == (5, (1,))


def test_a_buffer_without_elements_still_has_an_address(platform: Platform) -> None:
    """Bytes asked for with no whole element in them, or an empty shape with room reserved,
    are still memory at an address; even an empty buffer has one (PR #24 review)."""
    for nbytes, shape in [(1, None), (4096, (0,)), (0, (0,))]:
        buf = platform.alloc(nbytes, 0, shape=shape, dtype="f32")
        assert buf.shape == (0,) and buf.nbytes == nbytes and buf.addr != 0, (nbytes, shape)


def test_negative_sizes_offsets_and_shapes_are_refused(platform: Platform) -> None:
    """A negative (inferred) dimension once sized a view from the reservation behind the
    mapped prefix: mapped 64 bytes, a tensor of 4092 (PR #24 review)."""
    vr = platform.reserve(4096, 0)
    platform.map(vr, 0, 64)
    for bad in [lambda: platform.alloc(-1, 0), lambda: platform.alloc(0, 0, shape=(2, -1), dtype="f32"),
                lambda: platform.reserve(-1, 0), lambda: platform.map(vr, 64, -32),
                lambda: vr.view(0, (-1,), "f32"), lambda: vr.view(-4, (1,), "f32"),
                lambda: vr.view(2, (4,), "f32")]:
        with pytest.raises(ValueError, match="negative|not aligned"):
            bad()
    assert vr.mapped == 64


def test_every_call_naming_a_device_checks_it(platform: Platform) -> None:
    n = len(platform.devices())
    for bad in [lambda: platform.alloc(64, n), lambda: platform.reserve(64, -1), lambda: platform.stream(n),
                lambda: platform.synchronize(-1)]:
        with pytest.raises(ValueError, match="no device"):
            bad()


def test_freeing_twice_is_an_error(platform: Platform) -> None:
    buf = platform.alloc(1024, 0)
    platform.free(buf)
    with pytest.raises(RuntimeError, match="double free"):
        platform.free(buf)


def test_grow_keeps_the_base_and_views_stay_inside_the_mapped_prefix(platform: Platform) -> None:
    vr = platform.reserve(1 << 16, 0)
    platform.map(vr, 0, 4096)
    head = vr.view(0, (1024,), "f32")
    with pytest.raises(ValueError, match="beyond mapped"):
        vr.view(4096, (1024,), "f32")
    with pytest.raises(ValueError, match="extend the mapped prefix"):
        platform.map(vr, 8192, 4096)
    platform.map(vr, 4096, 4096)
    tail = vr.view(4096, (1024,), "f32")
    assert vr.mapped == 8192 and head.addr == vr.base and tail.addr == vr.base + 4096
    with pytest.raises(ValueError, match="beyond reserved"):
        platform.map(vr, 8192, 1 << 16)


def test_streams_synchronize_devices_and_info(platform: Platform) -> None:
    assert platform.stream(0).device == 0
    platform.synchronize(0)
    n = len(platform.devices())
    assert n >= 1 and [d.index for d in platform.devices()] == list(range(n))
    with pytest.raises(ValueError, match="no device"):
        platform.synchronize(n)
    assert platform.info().name in PLATFORMS


def test_the_fake_platform_has_no_tensors() -> None:
    p = FakePlatform()
    with pytest.raises(TypeError, match="no tensors"):
        p.tensor(p.alloc(64, 0))


def test_a_torch_eager_buffer_is_a_zeroed_tensor_at_its_address() -> None:
    p = TorchEagerPlatform()
    for name, dtype in DTYPES.items():
        buf = p.alloc(0, 0, shape=(3, 5), dtype=name)
        t = p.tensor(buf)
        assert (t.shape, t.dtype, t.device.type) == ((3, 5), dtype, "cpu")
        assert t.data_ptr() == buf.addr and t.nbytes == buf.nbytes and not t.any()
    for nbytes, shape, numel in [(4096, (1,), 1), (5, None, 1), (1, None, 0), (4096, (0,), 0), (0, (0,), 0)]:
        b = p.alloc(nbytes, 0, shape=shape, dtype="f32")
        storage = p.tensor(b).untyped_storage()
        assert p.tensor(b).numel() == numel and storage.nbytes() >= nbytes
        assert b.addr == storage.data_ptr() != 0, (nbytes, shape)  # the storage's, even with no elements
    buf = p.alloc(0, 0, shape=(4,), dtype="f32")
    p.tensor(buf).copy_(torch.arange(4.0))
    assert p.tensor(buf).tolist() == [0.0, 1.0, 2.0, 3.0]  # the same tensor every time
    p.free(buf)
    with pytest.raises(RuntimeError, match="freed"):
        p.tensor(buf)
    with pytest.raises(TypeError, match="not a torch_eager buffer"):
        p.tensor(FakePlatform().alloc(64, 0))


def test_views_of_a_grown_range_share_its_memory() -> None:
    """What a KV pool that grows relies on: the old view still reads and writes the same bytes
    after more of the range is mapped, and a new view sits right behind it."""
    p = TorchEagerPlatform()
    vr = p.reserve(1 << 12, 0)
    p.map(vr, 0, 1024)
    head = p.tensor(vr.view(0, (256,), "f32"))
    head.fill_(1.0)
    p.map(vr, 1024, 1024)
    tail = p.tensor(vr.view(1024, (256,), "f32"))
    tail.fill_(2.0)
    whole = p.tensor(vr.view(0, (512,), "f32"))
    assert whole[:256].eq(1.0).all() and whole[256:].eq(2.0).all()
    assert head.data_ptr() == vr.base and tail.data_ptr() == vr.base + 1024
    with pytest.raises(ValueError, match="not aligned"):
        vr.view(2, (4,), "f32")


def test_torch_eager_records_no_graphs() -> None:
    with pytest.raises(NotImplementedError, match="eagerly"):
        TorchEagerPlatform().graph()


def test_a_process_runs_on_the_platform_its_spec_names(monkeypatch: pytest.MonkeyPatch) -> None:
    def spec(platform: str, options: dict[str, int] | None = None, gpus: list[int] | None = None) -> ProcSpec:
        return ProcSpec(PROC, Identity("i"), ProcessPlan(PROC, Identity("i"), []), "control", "127.0.0.1",
                        gpus=gpus or [], platform=platform, platform_options=options or {})

    assert isinstance(make_platform(spec("fake")), FakePlatform)
    cpu = make_platform(spec("torch_eager", gpus=[0]))  # GPUs placed, but CUDA not asked for
    assert isinstance(cpu, TorchEagerPlatform) and cpu.info().versions["device"] == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        make_platform(spec("torch_eager", {"cuda": 1}, gpus=[0]))
    with pytest.raises(ValueError, match="unknown platform"):
        make_platform(spec("tpu"))

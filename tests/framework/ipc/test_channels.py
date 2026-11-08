"""ipc.md §6: the host's data channels, without processes: the directory is a dict here."""

from __future__ import annotations

import socket
import time
import uuid
from collections.abc import Iterator

import msgspec
import pytest
import zmq

from core.common.ids import HostId, ProcId
from core.ipc import StreamAddress
from core.ipc.channels import ChannelReader, Channels, ChannelWriter
from tests.framework.ipc.peers import BIND, PATIENT

TUN = msgspec.structs.replace(PATIENT, lookup_retry_s=0.05)
BACKENDS = {"shm": ("host-a", "host-a"), "tcp": ("host-a", "host-b")}
READER = ProcId(HostId("host-a"), "dst", 0)


@pytest.fixture(params=sorted(BACKENDS))
def hosts(request: pytest.FixtureRequest) -> tuple[str, str]:
    return BACKENDS[request.param]


@pytest.fixture
def zctx() -> Iterator[zmq.Context[zmq.Socket[bytes]]]:
    ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    yield ctx
    ctx.destroy(linger=0)  # whatever a failed test left open


class Directory:
    """What the supervisor does for the channels: names rings, publishes and looks up readers."""

    def __init__(self) -> None:
        self.endpoints: dict[tuple[ProcId, str], StreamAddress] = {}
        self.published: list[tuple[str, StreamAddress]] = []
        self.allocated: list[str] = []
        self.lookups = 0
        self.opened: list[Channels] = []

    def close(self) -> None:
        for channels in self.opened:
            channels.close()

    def channels(self, zctx: zmq.Context[zmq.Socket[bytes]], host: str, proc: ProcId = READER) -> Channels:
        def allocate(name: str) -> str:
            ring = f"agx-test-{name}-{uuid.uuid4().hex[:8]}"
            self.allocated.append(ring)
            return ring

        def publish(name: str, address: StreamAddress) -> None:
            self.published.append((name, address))
            self.endpoints[proc, name] = address

        def lookup(p: ProcId, name: str) -> StreamAddress | None:
            self.lookups += 1
            return self.endpoints.get((p, name))

        channels = Channels(zctx, local=HostId(host), bind=BIND, tun=TUN, allocate=allocate, lookup=lookup,
                            publish=publish)
        self.opened.append(channels)
        return channels


@pytest.fixture
def directory() -> Iterator[Directory]:
    d = Directory()
    yield d
    d.close()


def send(writer: ChannelWriter, data: bytes, until: bool = True, within: float = 10.0) -> None:
    """Write until ``write`` returns ``until``."""
    deadline = time.monotonic() + within
    while writer.write(data, timeout=5) is not until:
        assert time.monotonic() < deadline, f"write never returned {until}"
        time.sleep(0.01)


def take(reader: ChannelReader, within: float = 10.0) -> bytes:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        data = reader.read(timeout=0.1)
        if data is not None:
            return data
    raise AssertionError("nothing arrived")


def test_a_reader_is_made_and_published_once(zctx: zmq.Context[zmq.Socket[bytes]], hosts: tuple[str, str],
                                             directory: Directory) -> None:
    channels = directory.channels(zctx, hosts[0])
    reader = channels.reader("in", HostId(hosts[1]))
    assert channels.reader("in", HostId(hosts[1])) is reader  # a reloaded unit gets its reader back
    assert directory.published == [("in", reader.address)]
    assert len(directory.allocated) == (1 if hosts[0] == hosts[1] else 0)
    with pytest.raises(ValueError, match="takes records from"):
        channels.reader("in", HostId("host-c"))
    with pytest.raises(ValueError, match="endpoint of the host itself"):
        channels.reader("rpc", HostId(hosts[1]))


def test_a_writer_waits_for_its_reader_and_backs_off_its_lookups(zctx: zmq.Context[zmq.Socket[bytes]],
                                                                 hosts: tuple[str, str], directory: Directory) -> None:
    writer = directory.channels(zctx, hosts[1], ProcId(HostId(hosts[1]), "src", 0)).writer(READER, "in")
    for _ in range(50):
        assert writer.write(b"x") is False  # nobody published "in" yet
    assert directory.lookups == 1  # every write in one retry interval shares one lookup
    reader = directory.channels(zctx, hosts[0]).reader("in", HostId(hosts[1]))
    send(writer, b"first")
    assert take(reader) == b"first"
    assert writer.generation == 1 and writer.address == reader.address


def test_a_writer_follows_its_reader_to_a_new_process(zctx: zmq.Context[zmq.Socket[bytes]],
                                                      hosts: tuple[str, str], directory: Directory) -> None:
    """The reader's process is replaced: the writer notices, looks it up again and connects to the
    new one. Records written in between may be lost; ``generation`` says that it happened."""
    writer = directory.channels(zctx, hosts[1], ProcId(HostId(hosts[1]), "src", 0)).writer(READER, "in")
    old_process = directory.channels(zctx, hosts[0])
    old = old_process.reader("in", HostId(hosts[1]))
    send(writer, b"to the old one")
    assert take(old) == b"to the old one"
    old_process.close()
    send(writer, b"lost", until=False)
    new = directory.channels(zctx, hosts[0]).reader("in", HostId(hosts[1]))
    assert new.address != old.address
    send(writer, b"to the new one")
    assert take(new) == b"to the new one"
    assert writer.generation == 2


def test_a_writer_to_an_address_nobody_listens_on_says_so(zctx: zmq.Context[zmq.Socket[bytes]],
                                                         directory: Directory) -> None:
    """Not ``TimeoutError``: that means a reader that does not keep up."""
    with socket.socket() as s:
        s.bind((BIND, 0))
        port = s.getsockname()[1]
    channels = directory.channels(zctx, "host-a")
    writer = channels.writer_to(StreamAddress(HostId("host-b"), tcp=f"tcp://{BIND}:{port}"))
    assert writer.write(b"x", timeout=0.1) is False
    channels.discard(StreamAddress(HostId("host-b"), tcp=f"tcp://{BIND}:{port}"))

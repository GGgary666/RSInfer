"""The real worker stack (#30's worker units plus ``core.worker``) on a thread with data channels of
its own, under the scheduler test bed of ``tests/units/scheduler/support.py``: a tiny Qwen3.8-shaped
model from an HF checkpoint, on the CPU in fp32."""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import zmq

from core.common.ids import Identity
from core.context.config import ConfigEntry
from core.worker.unit import WorkerLoop
from platforms.torch_eager import TorchEagerPlatform
from tests.units.scheduler.support import SCHED, WORKERS, Bed, Client, Directory, Engine
from tests.units.support.worker import ENGINE, PAGE_SIZE, WORKER_UNITS, spec
from unit.runtime import ProcessPlan, UnitRuntime

UNITS = (*WORKER_UNITS, "core.worker")


class RealWorker:
    """Every worker unit in a unit runtime, stepped on a thread as its host would. ``call`` runs a
    function on that thread between two steps (a reload, as a change would make it)."""

    def __init__(self, directory: Directory, rank: int, hf_config: dict[str, Any], checkpoint: Path,
                 platform: Any = None, **config: Any) -> None:
        self.directory, self.rank, self.proc = directory, rank, WORKERS[rank]
        self.platform = platform
        """The worker's platform (``torch_eager`` on the CPU by default); tests/gpu give a CUDA one."""
        self.values: dict[str, Any] = {
            "model.hf_config": hf_config, "model.dtype": "f32", "state.gdn.conv_dtype": "f32", "weights.path": str(checkpoint),
            "state.kv.page_size": PAGE_SIZE, "state.kv.pages": 64, "state.gdn.slots": 8, "batch.max_rows": 8,
            "batch.max_context_len": 64, "worker.scheduler": {"host": str(SCHED.host), "role": SCHED.role,
                                                               "index": SCHED.index}, "worker.idle_wait_s": 0.002,
            **config}
        self.calls: queue.Queue[tuple[Callable[[RealWorker], Any], queue.Queue[Any]]] = queue.Queue()
        self.ready, self.stopping = threading.Event(), threading.Event()
        self.error: BaseException | None = None
        self.runtime: UnitRuntime | None = None
        self.thread = threading.Thread(target=self.run, name=f"real-worker-{rank}", daemon=True)
        self.thread.start()
        assert self.ready.wait(60) and self.error is None, self.error

    def run(self) -> None:
        zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        channels = self.directory.channels(zctx, self.proc)
        try:
            specs = [spec(u) for u in UNITS]
            entries = {key: ConfigEntry(s.id, decl, self.values.get(key, decl.default))
                       for s in specs for key, decl in s.manifest.config.items()}
            self.runtime = UnitRuntime(self.proc, platform=self.platform or TorchEagerPlatform())
            self.runtime.provide_host("proc.streams@1", channels)
            self.runtime.provide_host("proc.control@1", SimpleNamespace(identity=Identity("v1"), address=""))
            self.runtime.load_process_plan(ProcessPlan(self.proc, Identity("v1"), specs, config=entries,
                                                       code_root=str(ENGINE))).unwrap()
            self.ready.set()
            while not self.stopping.is_set():
                while not self.calls.empty():
                    fn, out = self.calls.get()
                    try:
                        out.put(fn(self))
                    except BaseException as e:  # noqa: BLE001
                        out.put(e)
                self.loop.step()
        except BaseException as e:  # noqa: BLE001
            traceback.print_exc()
            self.error = e
            self.ready.set()
        finally:
            if self.runtime is not None:
                self.runtime.unload_all()
            channels.close()
            zctx.destroy(linger=0)

    @property
    def loop(self) -> WorkerLoop:
        assert self.runtime is not None
        return cast(WorkerLoop, cast(Any, self.runtime.instance("core.worker")).loop)

    def call(self, fn: Callable[[RealWorker], Any]) -> Any:
        out: queue.Queue[Any] = queue.Queue()
        self.calls.put((fn, out))
        got = out.get(timeout=60)
        if isinstance(got, BaseException):
            raise got
        return got

    def close(self) -> None:
        self.stopping.set()
        self.thread.join(30)
        assert not self.thread.is_alive(), "worker thread leaked"
        assert self.error is None, self.error


@contextmanager
def stack(hf_config: dict[str, Any], checkpoint: Path, *, ranks: int = 1, overlap: bool = False,
          token_budget: int = 8, sched: dict[str, Any] | None = None, platform: Any = None, **worker: Any
          ) -> Generator[tuple[Bed, list[RealWorker]]]:
    """The scheduler host's units, ``ranks`` real workers and a client standing in for the frontend.
    ``sched``: config of the scheduler host's units; ``worker``: of the workers'."""
    directory = Directory()
    workers: list[RealWorker] = []
    client: Client | None = None
    engine: Engine | None = None
    try:
        engine = Engine(directory, ranks=ranks, overlap=overlap, max_batch=8, token_budget=token_budget,
                        config={"batch.max_rows": 8, "batch.max_context_len": 64, **(sched or {})})
        workers = [RealWorker(directory, r, hf_config, checkpoint, platform, **worker) for r in range(ranks)]
        client = Client(directory)
        bed = Bed(engine, [], client)
        bed.run_until(lambda: engine.ledger.ready, timeout=60, what="capacity")
        yield bed, workers
    finally:
        problems: list[AssertionError] = []
        for w in workers:
            try:
                w.close()
            except AssertionError as e:
                problems.append(e)
        if client is not None:
            client.close()
        if engine is not None:
            engine.close()
        if problems:
            raise problems[0]


class Oracle:
    """The draft model replaced: at position ``p`` after input ``S[p + 1]`` (the MTP layer's convention:
    the token after the position) it drafts ``S[p + 2]``, the target's greedy token, of whichever
    request ``S`` that is; wrong where ``p % WRONG_EVERY == WRONG_AT``. So steps accept every length.
    Tensor operations only, on ``device``, so that a graph can record it in place of the draft model."""

    WRONG_EVERY, WRONG_AT = 7, 3

    def __init__(self, sequences: dict[int, list[int]], vocab: int, device: str = "cpu") -> None:
        nexts: dict[tuple[int, int], int] = {}
        for s in sequences.values():
            for p in range(len(s) - 2):
                assert nexts.setdefault((p, s[p + 1]), s[p + 2]) == s[p + 2], "two requests meet: pick others"
        keys = sorted(nexts)  # by position, then token: the order of p * vocab + token
        self.keys = torch.tensor([p * vocab + t for p, t in keys], dtype=torch.int64, device=device)
        self.values = torch.tensor([nexts[k] for k in keys], dtype=torch.int64, device=device)
        self.vocab = vocab

    def __call__(self, input_ids: Any, positions: Any, hidden: Any, last_index: Any) -> tuple[Any, Any]:
        p, t = positions[last_index].long(), input_ids[last_index].long()
        q = p * self.vocab + t
        i = torch.searchsorted(self.keys, q).clamp(max=self.keys.numel() - 1)
        draft = torch.where(self.keys[i] == q, self.values[i], torch.zeros_like(q))
        draft = torch.where(p % self.WRONG_EVERY == self.WRONG_AT, (draft + 1) % self.vocab, draft)
        logits = torch.zeros(q.shape[0], self.vocab, device=q.device)
        logits.scatter_(1, draft[:, None], 1.0)
        return logits, hidden[last_index]

    def counts(self, prompt: int, steps: int, draft_tokens: int) -> list[int]:
        """What each step of a request of ``prompt`` tokens accepts: 1 for its prefill, then at ``L``
        the drafts for ``L + 1 ..`` up to the first wrong one (the draft for ``q`` comes from position
        ``q - 2``)."""
        out, at = [1], prompt
        while sum(out) < steps:
            right = 0
            # the draft for q is right if it is not one made wrong, and q is in the sequence the oracle knows
            while right < draft_tokens - 1 and (at + right + 1 - 2) % self.WRONG_EVERY != self.WRONG_AT \
                    and at + right + 1 < prompt + steps:
                right += 1
            out.append(right + 1)
            at += right + 1
        return out

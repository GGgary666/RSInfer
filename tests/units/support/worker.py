"""The stage-2 worker units in one ``UnitRuntime`` on ``torch_eager``, a tiny Qwen3.8-shaped HF
model as the reference, and ``Driver``, a test fixture that writes the step records a scheduler
would (rows, chunks, pages, slots, snapshots). The driver keeps no model state: every token comes
from the executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from core.common.ids import Identity, UnitId
from core.common.manifest import load_manifest
from core.context.config import ConfigEntry
from core.contracts.columns import Column, ragged
from core.contracts.records import AddedRow, BatchPlan, RowDelta, StateAssign, StateCopy, StateOps
from core.contracts.sampling import SamplingParams
from platforms.torch_eager import TorchEagerPlatform
from tests.framework.support import PROC
from unit.runtime import ProcessPlan, UnitRuntime, UnitSpec

ENGINE = Path(__file__).resolve().parents[3] / "engine"
WORKER_UNITS = ("core.kernels", "kernels.reference", "core.state.weights", "state.kv_paged", "state.gdn_slots",
                "worker.batch", "layers.linear", "layers.embedding", "layers.norm", "layers.rope",
                "layers.activation", "layers.sampler", "layers.attn", "layers.gdn", "models.qwen38",
                "executors.pytorch")
PAGE_SIZE = 4


def spec(uid: str) -> UnitSpec:
    return UnitSpec(load_manifest(ENGINE / Path(*uid.split(".")) / "unit.toml").unwrap())


def tiny_config(**kw: Any) -> Qwen3_5TextConfig:
    """3 GDN layers and 1 full-attention layer, every head count different, a partial rotary."""
    args: dict[str, Any] = {
        "vocab_size": 97, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 12,
        "linear_conv_kernel_dim": 4, "max_position_embeddings": 128,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5}}
    return Qwen3_5TextConfig(**{**args, **kw})


def hf_model(cfg: Qwen3_5TextConfig, seed: int = 0) -> Qwen3_5ForCausalLM:
    """HF's initialization, with noise on every vector so that a misplaced norm weight shows."""
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 1:
                p.add_(0.1 * torch.randn_like(p))
    return model


def save_checkpoint(model: Qwen3_5ForCausalLM, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(directory / "model.safetensors"))
    return directory


def hf_greedy(model: Qwen3_5ForCausalLM, prompt: list[int], steps: int) -> tuple[list[int], list[torch.Tensor]]:
    """HF generation, token by token with its cache: the tokens and each step's logits."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=model.config)
    tokens: list[int] = []
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        ids = torch.tensor([prompt])
        for _ in range(steps):
            out = model(ids, past_key_values=cache, use_cache=True)
            step = out.logits[0, -1]
            logits.append(step)
            tokens.append(int(step.argmax()))
            ids = torch.tensor([[tokens[-1]]])
    return tokens, logits


class Worker:
    """The worker units, loaded and started; ``config`` overrides the defaults."""

    def __init__(self, hf_config: dict[str, Any], **config: Any) -> None:
        overrides: dict[str, Any] = {"model.hf_config": hf_config, "model.dtype": "f32", "state.kv.page_size": PAGE_SIZE,
                                     "state.kv.pages": 32, "state.gdn.slots": 6, "batch.max_rows": 8,
                                     "batch.max_context_len": 64, **config}
        self.specs = [spec(u) for u in WORKER_UNITS]
        entries = {key: ConfigEntry(s.id, decl, overrides.get(key, decl.default))
                   for s in self.specs for key, decl in s.manifest.config.items()}
        self.runtime = UnitRuntime(PROC, platform=TorchEagerPlatform())
        self.runtime.load_process_plan(ProcessPlan(PROC, Identity("w"), self.specs, config=entries,
                                                   code_root=str(ENGINE))).unwrap()

    def service(self, name: str) -> Any:
        found = self.runtime.lookup(name)
        assert found is not None, self.runtime.snapshot()
        return found

    def unit(self, uid: str) -> Any:
        return self.runtime.instance(UnitId(uid))

    def close(self) -> None:
        self.runtime.unload_all().unwrap()


@dataclass
class Request:
    row: int
    request: int
    known: list[int]
    """The prompt, then every token sampled once the prompt was computed."""
    sampling: SamplingParams
    prompt_len: int = 0
    computed: int = 0
    pages: list[int] = field(default_factory=list[int])
    slot: int = 0
    in_batch: bool = False
    previous: bool = False
    """The row's last run sampled ``known[-1]``: a decode may name it (``from_previous``)."""

    @property
    def outputs(self) -> list[int]:
        return self.known[self.prompt_len:]


@dataclass
class Driver:
    """Writes each step's ``RowDelta``, ``BatchPlan`` and ``StateOps`` as a scheduler would, and keeps
    the tokens it gets back (a chunk that does not end the prompt samples a token that is dropped).
    Item ids start at 0, as the ledger hands them out; the first delta is full, as a scheduler's is."""

    worker: Worker
    requests: dict[int, Request] = field(default_factory=dict[int, Request])
    next_page: int = 0
    next_slot: int = 0
    epoch: int = 0
    full: bool = True
    removed: list[int] = field(default_factory=list[int])

    def add(self, row: int, prompt: list[int], seed: int = 0, temperature: float = 0.0) -> Request:
        r = Request(row, 100 + row, list(prompt), SamplingParams(temperature=temperature, seed=seed), len(prompt))
        r.slot, self.next_slot = self.next_slot, self.next_slot + 1
        self.requests[row] = r
        return r

    def remove(self, row: int) -> None:
        del self.requests[row]
        self.removed.append(row)

    def rebuilt(self) -> None:
        """The persistent batch was rebuilt: send a full delta, and every token explicitly."""
        self.full = True
        for r in self.requests.values():
            r.in_batch = r.previous = False

    def records(self, chunks: dict[int, int], copies: list[StateCopy] | None = None,
                fresh_slots: bool = True) -> tuple[RowDelta, BatchPlan, StateOps]:
        """``chunks``: row -> tokens to run this step, from where it stopped."""
        reqs = self.requests
        rows = list(chunks)
        added = [AddedRow(r.row, r.request, r.sampling) for r in reqs.values() if self.full or not r.in_batch]
        assign: dict[str, dict[int, list[int]]] = {"kv.paged": {}, "gdn.slots": {}}
        for r in reqs.values():
            if self.full or not r.in_batch:
                if r.pages:
                    assign["kv.paged"][r.row] = list(r.pages)
                if fresh_slots or self.full:
                    assign["gdn.slots"][r.row] = [r.slot]
        tokens: list[int] = []
        prev: list[int] = []
        for row in rows:
            r, n = reqs[row], chunks[row]
            need = -(-(r.computed + n) // PAGE_SIZE) - len(r.pages)
            new = list(range(self.next_page, self.next_page + max(need, 0)))
            self.next_page += len(new)
            r.pages += new
            if new:
                assign["kv.paged"].setdefault(row, []).extend(new)
            use_previous = r.previous and n == 1 and r.computed == len(r.known) - 1
            prev.append(int(use_previous))
            if not use_previous:
                tokens += r.known[r.computed:r.computed + n]
        delta = RowDelta(added, Column.of("i32", [] if self.full else self.removed), self.full)
        state = StateOps([StateAssign(kind, Column.of("i32", list(m)), *ragged("i32", m.values()))
                          for kind, m in assign.items() if m], copies or [])
        decoding = all(chunks[row] == 1 and reqs[row].computed >= len(reqs[row].known) - 1 for row in rows)
        prefilling = all(reqs[row].computed + chunks[row] <= len(reqs[row].known) for row in rows)
        plan = BatchPlan(Column.of("i32", rows), Column.of("i32", list(chunks.values())),
                         Column.of("i32", [reqs[row].computed for row in rows]), Column.of("i64", tokens),
                         Column.of("u8", prev), "decode" if decoding else "prefill" if prefilling else "mixed", 0)
        return delta, plan, state

    def step(self, chunks: dict[int, int], copies: list[StateCopy] | None = None,
             fresh_slots: bool = True) -> Any:
        executor = self.worker.service("executor@1")
        delta, plan, state = self.records(chunks, copies, fresh_slots)
        self.epoch += 1
        result = executor.submit(self.epoch, delta, plan, state)
        if not hasattr(result, "finish"):
            return result  # Resync: nothing ran
        self.full, self.removed = False, []
        sampled = result.finish()
        for r in self.requests.values():
            r.in_batch = True
        for row, token in zip(sampled.rows.values(), sampled.tokens.values(), strict=True):
            r = self.requests[int(row)]
            r.computed += chunks[int(row)]
            ends = r.computed >= len(r.known)
            if ends:  # the scheduler keeps only what ends a prompt or decodes
                r.known.append(int(token))
            r.previous = ends
        return result

    def generate(self, rows: list[int], steps: int, chunk: int | None = None) -> dict[int, list[torch.Tensor]]:
        """Run every row until it has ``steps`` output tokens, prefilling in ``chunk``-token pieces;
        the logits of each output token, per row."""
        out: dict[int, list[torch.Tensor]] = {row: [] for row in rows}
        while any(len(self.requests[row].outputs) < steps for row in rows):
            chunks = {}
            for row in rows:
                r = self.requests[row]
                if len(r.outputs) < steps:
                    left = len(r.known) - r.computed
                    chunks[row] = min(left, chunk) if chunk else left
            result = self.step(chunks)
            for i, row in enumerate(chunks):
                if self.requests[row].previous:
                    out[row].append(result.logits[i])
        return out

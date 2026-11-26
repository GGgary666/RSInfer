"""RSInfer dense Qwen3 agrees with Hugging Face on chunked prefill and decode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import safetensors.torch
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from tests.units.support.worker import Driver, Worker


class DenseWorker(Worker):
    UNITS = tuple(u for u in Worker.UNITS if u not in {"models.qwen38", "state.gdn_slots", "layers.gdn"}) + (
        "models.qwen3",)
    DEFAULTS: dict[str, Any] = {"model.dtype": "f32", "state.kv.page_size": 4, "state.kv.pages": 32,
                                "batch.max_rows": 8, "batch.max_context_len": 64}


def test_dense_qwen3_loads_tied_checkpoint_and_matches_grouped_attention(tmp_path: Path) -> None:
    torch.manual_seed(1)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=12, max_position_embeddings=64, tie_word_embeddings=True,
                      attention_bias=True)
    reference = Qwen3ForCausalLM(cfg).eval()
    safetensors.torch.save_file({name: t.contiguous() for name, t in reference.state_dict().items()
                                if name != "lm_head.weight"}, str(tmp_path / "model.safetensors"))
    prompts = {0: [3, 5, 70, 6, 21, 18, 5], 1: [80, 9, 57, 12, 36, 10, 49, 86, 73]}
    worker = DenseWorker(cfg.to_dict(), **{"weights.path": str(tmp_path)})
    try:
        driver = Driver(worker)
        for row, tokens in prompts.items():
            driver.add(row, tokens)
        logits = driver.generate(list(prompts), 4, chunk=3)
        with torch.no_grad():
            for row, tokens in prompts.items():
                full = tokens + driver.requests[row].outputs
                first = len(tokens) - 1
                want = reference(torch.tensor([full])).logits[0, first:first + 4]
                torch.testing.assert_close(torch.stack(logits[row]), want, rtol=1e-4, atol=1e-4)
    finally:
        worker.close()

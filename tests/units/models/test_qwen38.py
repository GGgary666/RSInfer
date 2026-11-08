"""models.qwen38 and core.state.weights: SGLang's parameters of Qwen3.5, filled from an HF
checkpoint by SGLang's ``load_weights``; dummy weights; ``refit`` in place; every worker unit
passes the manifest rules."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import torch

from core.common.framework import FrameworkModules
from core.manifest.checks import check_tree
from core.state.weights.unit import Refit, WeightStore
from models.qwen38.config import Qwen3_5TextConfig as Config
from models.qwen38.unit import Qwen3_5ForCausalLM, params
from platforms.torch_eager import TorchEagerPlatform
from tests.units.support.worker import ENGINE, WORKER_UNITS, hf_model, save_checkpoint, tiny_config


def store_for(path: str, cfg: Config) -> tuple[WeightStore, Qwen3_5ForCausalLM]:
    store = WeightStore(TorchEagerPlatform(), path, 1234)
    model = Qwen3_5ForCausalLM(cfg, "f32", 4, defaultdict[str, Any](lambda: None))
    store.declare(params(cfg, "f32"))
    store.register(model.load_weights)
    return store, model


def test_load_weights_fuses_the_hf_checkpoint_into_sglangs_parameters(tmp_path: Path) -> None:
    hf_cfg = tiny_config()
    hf = hf_model(hf_cfg)
    cfg = Config.from_hf(hf_cfg.to_dict())
    assert ["attention" if t == "full_attention" else t for t in hf_cfg.layer_types or []] == cfg.layers_block_type
    store, _ = store_for(str(save_checkpoint(hf, tmp_path)), cfg)
    store.load()
    got, want = store.tensors, {k: v.detach() for k, v in hf.state_dict().items()}
    a, g = "model.layers.3.", "model.layers.0.linear_attn."
    assert torch.equal(got[a + "qkv_proj.weight"],
                       torch.cat([want[a + f"self_attn.{p}_proj.weight"] for p in "qkv"]))
    assert torch.equal(got[a + "q_norm.weight"], want[a + "self_attn.q_norm.weight"])
    assert torch.equal(got[g + "in_proj_qkvz.weight"], torch.cat([want[g + "in_proj_qkv.weight"],
                                                                   want[g + "in_proj_z.weight"]]))
    assert torch.equal(got[g + "in_proj_ba.weight"], torch.cat([want[g + "in_proj_b.weight"],
                                                                 want[g + "in_proj_a.weight"]]))
    assert torch.equal(got["model.layers.1.mlp.gate_up_proj.weight"],
                       torch.cat([want["model.layers.1.mlp.gate_proj.weight"], want["model.layers.1.mlp.up_proj.weight"]]))
    assert got[g + "A_log"].dtype == torch.float32 and torch.equal(got["lm_head.weight"], want["lm_head.weight"])
    assert store.version == f"1:{tmp_path}"


def test_a_checkpoint_without_a_declared_weight_is_refused(tmp_path: Path) -> None:
    hf_cfg = tiny_config()
    cfg = Config.from_hf(hf_cfg.to_dict())
    hf = hf_model(hf_cfg)
    del hf.lm_head
    hf.config.tie_word_embeddings = True
    store, _ = store_for(str(save_checkpoint(hf, tmp_path)), cfg)
    with pytest.raises(RuntimeError, match="no weights for"):
        store.load()


def test_dummy_weights_then_refit_in_place_and_abort(tmp_path: Path) -> None:
    hf_cfg = tiny_config()
    cfg = Config.from_hf(hf_cfg.to_dict())
    store, _ = store_for("", cfg)
    store.load()
    w = store.tensors["model.norm.weight"]
    assert store.version == "1:dummy:1234" and bool(((w >= -1e-3) & (w < 1e-3)).all())
    same_size = store.tensors["model.layers.0.input_layernorm.weight"]
    assert torch.equal(w, same_size)  # one seed per parameter: only its size and dtype matter
    addresses = {n: t.data_ptr() for n, t in store.tensors.items()}
    refit = Refit(store)
    refit.prepare({"path": str(save_checkpoint(hf_model(hf_cfg), tmp_path))})
    refit.execute({})
    assert not torch.equal(store.tensors["model.norm.weight"], same_size) and store.version == f"2:{tmp_path}"
    refit.abort()
    assert torch.equal(store.tensors["model.norm.weight"], same_size) and store.version == "1:dummy:1234"
    assert {n: t.data_ptr() for n, t in store.tensors.items()} == addresses and store.generation == 0


def test_the_worker_units_pass_the_manifest_rules() -> None:
    errors, _ = check_tree(ENGINE, None, FrameworkModules.default())
    assert [e for e in errors if e.subject in WORKER_UNITS] == []
